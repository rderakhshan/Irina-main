"""Stage 3: sequential sub-agent demonstration over OpenShell.

Concept: a parent Harness delegates two self-contained file-writing tasks to
fresh child Harnesses through ``spawn_agent``, then runs the generated test
itself and reports the observed result. The provider is scripted so the
demonstration is deterministic; the workspace and sandbox are real.

Design constraints:
* Standard library only.
* No child sessions; exactly one top-level session.
* Children share the parent workspace, policy, and OpenShell sandbox.
* No credentials are read or printed.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.back.harness import Harness
from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.orchestration.session import load
from src.back.harness.tools_and_guardrails import (
    Policy,
    SandboxError,
    SandboxUnavailable,
)

UTILS = (
    "import re\n\n"
    "def slugify(text):\n"
    "    return re.sub(r\"[^a-z0-9]+\", \"-\", text.lower()).strip(\"-\")\n"
)
TESTS = (
    "from utils import slugify\n\n"
    "assert slugify('Hello World') == 'hello-world'\n"
    "assert slugify('  Multiple   Spaces  ') == 'multiple-spaces'\n"
    "assert slugify('Already-good') == 'already-good'\n"
    "assert slugify('Hello, World!') == 'hello-world'\n"
    "assert slugify('A') == 'a'\n"
    "print('5 assertions passed')\n"
)


class ScriptedProvider:
    """Deterministic parent/child response queue."""

    default_model = "stage3-demo"

    def __init__(self) -> None:
        self.responses = [
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="spawn-1",
                        name="spawn_agent",
                        args={
                            "task": (
                                "Write utils.py with a slugify(text) function. "
                                "Do not modify any other file."
                            )
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="child-write-1",
                        name="write_file",
                        args={"path": "utils.py", "content": UTILS},
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="utils.py created with slugify.",
                tool_calls=[],
                finish_reason="stop",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="spawn-2",
                        name="spawn_agent",
                        args={
                            "task": (
                                "Write test_utils.py with five assertions for "
                                "slugify from utils.py. Do not modify utils.py."
                            )
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="child-write-2",
                        name="write_file",
                        args={"path": "test_utils.py", "content": TESTS},
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="test_utils.py created with five assertions.",
                tool_calls=[],
                finish_reason="stop",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="run-tests",
                        name="bash",
                        args={"command": "python3 test_utils.py", "timeout": "30"},
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            ),
            ProviderResponse(
                text=(
                    "Both delegated files were created. I ran "
                    "`python3 test_utils.py` and observed the passing result."
                ),
                tool_calls=[],
                finish_reason="stop",
                usage=TokenUsage(1, 1),
            ),
        ]
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("provider queue exhausted")
        return self.responses.pop(0)


def print_event(event: RunEvent) -> None:
    """Print parent events, including child reports in tool results."""
    if event.type is EventType.ASSISTANT:
        if event.text.strip():
            print(f"ASSISTANT: {event.text}")
        for call in event.tool_calls:
            print(f"tool_call: {call.name}({call.args})")
    elif event.type is EventType.TOOL_END:
        call = event.tool_call
        if call is None:
            return
        result = str(event.result or "")
        if len(result) > 600:
            result = result[:600] + "..."
        print(f"tool_result: {call.name} -> {result}")


def main() -> int:
    demo_root = REPO_ROOT / ".irina" / "stage3-demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    state_dir = demo_root / f"run-{uuid.uuid4().hex[:8]}"
    provider = ScriptedProvider()
    harness = Harness(
        workdir=str(REPO_ROOT),
        model="stage3-demo",
        provider=provider,
        policy=Policy("yolo"),
        on_event=print_event,
        state_dir=state_dir,
        sandbox_workspace="stage3-subagents",
        enable_subagents=True,
        persist=True,
    )
    try:
        print(f"workspace: {harness.workspace}")
        result = harness.run(
            "Use spawn_agent twice: write utils.py, then test_utils.py with "
            "five assertions. Run python3 test_utils.py yourself and report "
            "the observed result."
        )
        utils = harness.tools["read_file"].run(path="utils.py")
        tests = harness.tools["read_file"].run(path="test_utils.py")
        # The parent's user callback already printed the test output; inspect
        # the durable session for the exact tool result.
        data = load(harness.session_path)
        bash_results = [
            message["text"]
            for message in data.messages
            if message.get("role") == "tool"
            and message.get("name") == "bash"
        ]
        spawn_calls = [
            call
            for message in data.messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
            if call.get("name") == "spawn_agent"
        ]
        checks = {
            "run_completed": result.completed,
            "utils_has_slugify": "def slugify" in utils,
            "tests_have_five_assertions": tests.count("assert ") >= 5,
            "two_spawn_calls": len(spawn_calls) == 2,
            "parent_ran_tests": any(
                "5 assertions passed" in text for text in bash_results
            ),
            "one_top_level_session": len(
                list(state_dir.rglob("*.jsonl"))
            ) - len(list(state_dir.rglob("*.recovery.jsonl"))) == 1,
            "session_completed": data.status == "completed",
        }
        print("STAGE 3 SUB-AGENT CHECKS:")
        for name, passed in checks.items():
            print(f"  {name}: {'yes' if passed else 'no'}")
        print(f"session: {harness.session_path}")
        print(f"final: {result.text}")
        return 0 if all(checks.values()) else 1
    except SandboxUnavailable as exc:
        print(f"OpenShell prerequisite missing: {exc}")
        return 2
    except SandboxError as exc:
        print(f"OpenShell sandbox failed: {exc}")
        return 1
    finally:
        try:
            harness.close()
        except Exception as exc:
            print(f"cleanup failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
