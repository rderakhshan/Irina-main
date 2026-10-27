"""Stage 2: context compaction, durable memory, and skills demonstration.

Concept: run three live checks inside OpenShell:
A. a long five-file task under a 1,500-token context budget;
B. durable memory across a stopped/started workspace;
C. a skill created at runtime and loaded through ``use_skill``.

Design constraints:
* Reuses the existing loop, provider, tool, policy, and OpenShell interfaces.
* All file, memory, and skill operations run through the sandbox backend.
* Compaction runs through the existing ``before_turn`` hook.
* Task state is updated from actual tool events.
* The sandbox is deleted after each check.
* No credentials are printed or passed into the sandbox.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ._env import load_dotenv
except ImportError:
    from _env import load_dotenv

from src.back.harness.context_management import (
    TaskState,
    TaskStateObserver,
    build_system_prompt,
    catalog,
    catalog_prompt,
    compact,
    estimate_tokens,
    remember,
    reset_calibration,
    skill_tools,
)
from src.back.harness.loop.runner import EventType, RunEvent, run_loop
from src.back.harness.loop.termination import TerminationPolicy
from src.back.harness.providers.deepseek import DeepSeekProvider
from src.back.harness.tools_and_guardrails import (
    OpenShellSandbox,
    Policy,
    SandboxError,
    SandboxUnavailable,
    core_tools,
)

CONTEXT_BUDGET = 1_500
TEST_A_TASK = (
    "Create five files one.txt through five.txt, each containing 20 lines of "
    "the word ping. Create one file at a time and read it back after each "
    "write. Then create MANIFEST.md listing each file and its line count, "
    "verified using wc -l."
)
TEST_A_NUDGE = (
    "The five files and MANIFEST.md are verified. Give a concise final summary "
    "now. Do not call more tools."
)
TEST_B_FACT = "project codename: lighthouse-42"
TEST_B_TASK = (
    "What is the project codename? Reply with just the codename."
)
TEST_C_TASK = "Write two short sentences about the project status."
TEST_C_SKILL = (
    "---\n"
    "name: brand-voice\n"
    "description: Friendly pirate voice for project updates.\n"
    "---\n\n"
    "When writing, start with 'Ahoy, matey!' and include the word 'savvy'. "
    "Keep it short and cheerful.\n"
)


def _fixed_tokens(system: str, tools: dict[str, Any]) -> int:
    """Estimate the fixed system prompt and tool-schema context."""
    specs = [tool.spec for tool in tools.values()]
    return estimate_tokens(
        [{"role": "system", "text": system}]
    ) + estimate_tokens(
        [{"role": "user", "text": json.dumps(specs, ensure_ascii=False)}]
    )


def _short(text: str, limit: int = 1600) -> str:
    """Trim console display text."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [display truncated]"


def _format_args(args: dict[str, Any]) -> str:
    """Render tool arguments for the console."""
    return ", ".join(
        f"{key}={_short(str(value), 160)}" for key, value in args.items()
    )


def print_event(event: RunEvent) -> None:
    """Print assistant, tool, compaction, and run events."""
    if event.type is EventType.ASSISTANT:
        if event.text.strip():
            print("ASSISTANT:")
            print(event.text)
            print()
        for call in event.tool_calls:
            print(f"tool_call: {call.name}({_format_args(call.args)})")
        if event.tool_calls:
            print()
    elif event.type is EventType.TOOL_START:
        call = event.tool_call
        if call is not None:
            print(f"tool_start: {call.name}({_format_args(call.args)})")
    elif event.type is EventType.TOOL_END:
        call = event.tool_call
        name = call.name if call is not None else "tool"
        print(f"tool_end: {name} -> {_short(str(event.result))}\n")
    elif event.type is EventType.CONTEXT_COMPACTED:
        print(
            "context_compacted: "
            f"success={event.compaction_succeeded} "
            f"before={event.before_tokens} after={event.after_tokens} "
            f"summarized={event.summarized_messages} "
            f"preserved={event.preserved_messages} "
            f"reason={event.reason}"
        )
        print()
    elif event.type is EventType.RUN_END:
        status = event.status.value if event.status else "unknown"
        print(f"RUN: status={status} reason={event.reason}\n")


def _tool_results(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map tool call IDs to result text."""
    return {
        str(message.get("tool_call_id", "")): str(message.get("text", ""))
        for message in messages
        if message.get("role") == "tool"
    }


def _tool_end_events(
    events: list[RunEvent],
    name: str | None = None,
    text_contains: str | None = None,
) -> list[RunEvent]:
    """Filter tool-end events by tool name and result substring."""
    found = []
    for event in events:
        if event.type is not EventType.TOOL_END or event.tool_call is None:
            continue
        if name is not None and event.tool_call.name != name:
            continue
        result = str(event.result or "")
        if text_contains is not None and text_contains not in result:
            continue
        found.append(event)
    return found


def _wc_evidence(events: list[RunEvent]) -> bool:
    """Return whether a successful wc -l command was observed."""
    for event in _tool_end_events(events, name="bash"):
        call = event.tool_call
        if call is None:
            continue
        command = str(call.args.get("command", ""))
        result = str(event.result or "")
        if "wc" in command and "-l" in command and not result.startswith(
            ("ERROR:", "BLOCKED:")
        ):
            return True
    return False


def _skill_evidence(events: list[RunEvent]) -> bool:
    """Return whether use_skill loaded the pirate instructions."""
    return any(
        "Ahoy, matey!" in str(event.result or "")
        for event in _tool_end_events(events, name="use_skill")
    )


def _tool_integrity(messages: list[dict[str, Any]]) -> bool:
    """Check that the final transcript has no broken tool interaction."""
    calls: dict[str, int] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []) or []:
                call_id = str(call.get("id", ""))
                if not call_id or call_id in calls:
                    return False
                calls[call_id] = 0
    for message in messages:
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id", ""))
            if call_id not in calls:
                return False
            calls[call_id] += 1
    return all(count == 1 for count in calls.values())


def _files_ready(sandbox: OpenShellSandbox) -> tuple[bool, str]:
    """Check the five files and MANIFEST.md contents through the sandbox."""
    try:
        for name in ("one.txt", "two.txt", "three.txt", "four.txt", "five.txt"):
            text = sandbox.read_file(name, max_lines=100)["text"]
            lines = text.splitlines()
            if len(lines) != 20 or any(line != "ping" for line in lines):
                return False, f"{name} does not contain exactly 20 ping lines"
        manifest = sandbox.read_file("MANIFEST.md", max_lines=100)["text"]
        for name in ("one.txt", "two.txt", "three.txt", "four.txt", "five.txt"):
            if name not in manifest or "20" not in manifest:
                return False, "MANIFEST.md is missing a file or line count"
        return True, "files and manifest verified"
    except Exception as exc:
        return False, f"file verification failed: {type(exc).__name__}: {exc}"


def _wc_check(sandbox: OpenShellSandbox) -> tuple[bool, str]:
    """Run an independent wc -l check inside the sandbox."""
    result = sandbox.exec(
        [
            "wc",
            "-l",
            "one.txt",
            "two.txt",
            "three.txt",
            "four.txt",
            "five.txt",
        ],
        workdir=sandbox.workspace,
        timeout=20,
    )
    output = result.stdout + result.stderr
    if result.exit_code != 0:
        return False, f"wc failed: {output.strip()}"
    for name in ("one.txt", "two.txt", "three.txt", "four.txt", "five.txt"):
        matching = [
            line for line in output.splitlines() if line.strip().endswith(name)
        ]
        if not matching or matching[0].split()[0] != "20":
            return False, f"wc did not report 20 for {name}"
    return True, "independent wc check passed"


def _compaction_evidence(events: list[RunEvent]) -> bool:
    """Return whether at least one successful compaction occurred."""
    return any(
        event.type is EventType.CONTEXT_COMPACTED
        and event.compaction_succeeded is True
        for event in events
    )


def _make_event_printer(
    observer: TaskStateObserver,
    events: list[RunEvent],
) -> Callable[[RunEvent], None]:
    """Compose task-state updates, event collection, and console printing."""

    def on_event(event: RunEvent) -> None:
        observer(event)
        events.append(event)
        print_event(event)

    return on_event


def _run_task(
    provider: DeepSeekProvider,
    sandbox: OpenShellSandbox,
    tools: dict[str, Any],
    system: str,
    task: str,
    state: TaskState,
    observer: TaskStateObserver,
    events: list[RunEvent],
    *,
    budget: int | None = None,
    reserve_tokens: int = 0,
    validator: Callable[[Any, list[dict]], bool] | None = None,
    completion_nudge: str | None = None,
    ready: Callable[[], bool] | None = None,
) -> tuple[Any, list[dict]]:
    """Run one Stage 2 task using the existing loop and hooks."""
    on_event = _make_event_printer(observer, events)
    messages: list[dict] = [{"role": "user", "text": task}]
    model = provider.default_model
    before_turn = None
    if budget is not None:
        def before_turn(current: list[dict]) -> list[dict]:
            updated = compact(
                model,
                current,
                budget,
                provider=provider,
                task_state=state,
                on_event=on_event,
                reserve_tokens=reserve_tokens,
            )
            if completion_nudge and ready is not None and ready():
                if not any(
                    message.get("role") == "user"
                    and completion_nudge in str(message.get("text", ""))
                    for message in updated
                ):
                    updated = list(updated)
                    updated.append(
                        {"role": "user", "text": completion_nudge}
                    )
            return updated
    result = run_loop(
        provider=provider,
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        on_event=on_event,
        before_tool=Policy("yolo").check,
        policy=TerminationPolicy(
            max_turns=100,
            max_tool_calls=250,
            max_total_tokens=300_000,
        ),
        completion_validator=validator,
        before_turn=before_turn,
    )
    return result, messages


def test_a(provider: DeepSeekProvider) -> bool:
    """Run the long five-file task with compaction enabled."""
    print("=" * 78)
    print("TEST A: long task with a 1,500-token context budget")
    print("=" * 78)
    sandbox = OpenShellSandbox(workspace="stage2-context")
    try:
        sandbox.start()
        system = build_system_prompt(
            sandbox.workspace,
            workspace=sandbox,
            extra=(
                "Use the workspace tools. Commands run in Bash inside OpenShell. "
                "After writing the files, run `wc -l` and use its output to "
                "create MANIFEST.md. Do not claim success without observed "
                "output. After MANIFEST.md is verified once, stop and give a "
                "concise final summary. Do not repeat verification commands."
            ),
        )
        tools = {
            tool.name: tool
            for tool in core_tools(sandbox.workspace, sandbox)
            if tool.name in {"read_file", "write_file", "bash"}
        }
        reserve = _fixed_tokens(system, tools)
        state = TaskState(goal=TEST_A_TASK)
        observer = TaskStateObserver(state)
        events: list[RunEvent] = []

        def validator(response: Any, messages: list[dict]) -> bool:
            if not response.text.strip():
                return False
            files_ok, _ = _files_ready(sandbox)
            return files_ok and _wc_evidence(events)

        result, messages = _run_task(
            provider,
            sandbox,
            tools,
            system,
            TEST_A_TASK,
            state,
            observer,
            events,
            budget=CONTEXT_BUDGET,
            reserve_tokens=reserve,
            validator=validator,
            completion_nudge=TEST_A_NUDGE,
            ready=lambda: (
                _files_ready(sandbox)[0] and _wc_evidence(events)
            ),
        )

        files_ok, files_reason = _files_ready(sandbox)
        wc_ok, wc_reason = _wc_check(sandbox)
        compaction_ok = _compaction_evidence(events)
        integrity_ok = _tool_integrity(messages)
        checks = {
            "run_completed": result.completed,
            "files_and_manifest": files_ok,
            "wc_evidence": _wc_evidence(events),
            "independent_wc": wc_ok,
            "compaction_event": compaction_ok,
            "tool_integrity": integrity_ok,
            "final_answer": bool(result.text.strip()),
        }
        print("TEST A CHECKS:")
        for name, passed in checks.items():
            print(f"  {name}: {'yes' if passed else 'no'}")
        print(f"  file_state: {files_reason}")
        print(f"  wc_state: {wc_reason}")
        print(f"  context_events: {sum(e.type is EventType.CONTEXT_COMPACTED for e in events)}")
        print(f"  fixed_context_estimate: {reserve}")
        print(f"  final: {result.status.value} / {result.reason}")
        return all(checks.values())
    finally:
        try:
            sandbox.delete()
        except Exception as exc:
            print(f"TEST A cleanup failed: {type(exc).__name__}: {exc}")
        print()


def test_b(provider: DeepSeekProvider) -> bool:
    """Run the durable-memory-across-sessions check."""
    print("=" * 78)
    print("TEST B: durable memory across a fresh session")
    print("=" * 78)
    sandbox = OpenShellSandbox(workspace="stage2-memory", keep=True)
    try:
        sandbox.start()
        remember(
            sandbox.workspace,
            TEST_B_FACT,
            workspace=sandbox,
        )
        first_memory = sandbox.read_file("IRINA.md", max_lines=100)["text"]
        print(f"Initial memory contains fact: {TEST_B_FACT in first_memory}")

        sandbox.stop()
        sandbox.start()
        prompt = build_system_prompt(
            sandbox.workspace,
            workspace=sandbox,
            extra="Answer only from the project memory when it is available.",
        )
        memory_in_prompt = TEST_B_FACT in prompt
        print(f"Fresh prompt contains fact: {memory_in_prompt}")

        messages = [{"role": "user", "text": TEST_B_TASK}]
        result = run_loop(
            provider=provider,
            model=provider.default_model,
            system=prompt,
            messages=messages,
            tools={},
            on_event=print_event,
            policy=TerminationPolicy(max_turns=2),
        )
        answered = TEST_B_FACT.split(": ", 1)[1].lower() in result.text.lower()
        fresh = len([m for m in messages if m.get("role") == "user"]) == 1
        checks = {
            "memory_persisted": TEST_B_FACT in first_memory,
            "prompt_loaded_memory": memory_in_prompt,
            "fresh_conversation": fresh,
            "answer_used_memory": result.completed and answered,
        }
        print("TEST B CHECKS:")
        for name, passed in checks.items():
            print(f"  {name}: {'yes' if passed else 'no'}")
        print(f"  final: {result.text!r}")
        return all(checks.values())
    finally:
        sandbox.keep = False
        try:
            sandbox.delete()
        except Exception as exc:
            print(f"TEST B cleanup failed: {type(exc).__name__}: {exc}")
        print()


def test_c(provider: DeepSeekProvider) -> bool:
    """Run the runtime-skill discovery and loading check."""
    print("=" * 78)
    print("TEST C: skill discovered and loaded without Python changes")
    print("=" * 78)
    sandbox = OpenShellSandbox(workspace="stage2-skills")
    try:
        sandbox.start()
        sandbox.write_file("skills/brand-voice/SKILL.md", TEST_C_SKILL)
        entries = catalog(sandbox.workspace, workspace=sandbox)
        prompt_catalog = catalog_prompt(sandbox.workspace, workspace=sandbox)
        system = build_system_prompt(
            sandbox.workspace,
            workspace=sandbox,
            extra=(
                "If a skill is relevant, load it with use_skill before writing. "
                "Follow the loaded skill exactly."
            ),
        )
        tools = {
            tool.name: tool
            for tool in (
                core_tools(sandbox.workspace, sandbox)
                + skill_tools(sandbox.workspace, workspace=sandbox)
            )
        }
        state = TaskState(goal=TEST_C_TASK)
        observer = TaskStateObserver(state)
        events: list[RunEvent] = []
        result, messages = _run_task(
            provider,
            sandbox,
            tools,
            system,
            TEST_C_TASK,
            state,
            observer,
            events,
        )
        lower = result.text.lower()
        checks = {
            "catalog_discovered": "brand-voice" in entries,
            "catalog_prompt_advertised": "brand-voice" in prompt_catalog,
            "skill_tool_called": bool(_tool_end_events(events, name="use_skill")),
            "skill_body_loaded": _skill_evidence(events),
            "pirate_greeting": "ahoy, matey" in lower,
            "pirate_word": "savvy" in lower,
            "run_completed": result.completed,
        }
        print("TEST C CHECKS:")
        for name, passed in checks.items():
            print(f"  {name}: {'yes' if passed else 'no'}")
        print(f"  catalog: {entries}")
        print(f"  final: {result.text!r}")
        return all(checks.values())
    finally:
        try:
            sandbox.delete()
        except Exception as exc:
            print(f"TEST C cleanup failed: {type(exc).__name__}: {exc}")
        print()


def main() -> int:
    """Run A, B, C, or one selected check."""
    load_dotenv()
    try:
        provider = DeepSeekProvider()
    except RuntimeError as exc:
        print(f"Stage 2 demo skipped: {exc}")
        return 2
    reset_calibration()
    provider = DeepSeekProvider()

    selected = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    runners = {"a": test_a, "b": test_b, "c": test_c}
    if selected not in {"all", *runners}:
        print("usage: python demos/stage2_context_memory_skills.py [a|b|c]")
        return 2

    results: dict[str, bool] = {}
    try:
        for key, runner in runners.items():
            if selected not in {"all", key}:
                continue
            try:
                results[key.upper()] = runner(provider)
            except SandboxUnavailable as exc:
                print(f"OpenShell prerequisite missing: {exc}")
                results[key.upper()] = False
            except (SandboxError, Exception) as exc:
                print(
                    f"TEST {key.upper()} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                results[key.upper()] = False
    finally:
        print("STAGE 2 SUMMARY:")
        for name in ("A", "B", "C"):
            if name in results:
                print(f"  {name}: {'passed' if results[name] else 'failed'}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
