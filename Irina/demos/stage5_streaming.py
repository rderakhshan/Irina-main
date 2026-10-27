"""Stage 5: genuine streaming and terminal UI verification.

Run the free synthetic UI tour:

    python demos/stage5_streaming.py

Run the real DeepSeek + OpenShell streaming task:

    python demos/stage5_streaming.py --live --plain
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.back.harness import Harness, Policy
from src.back.harness.loop.runner import EventType, RunEvent, run_loop
from src.back.harness.loop.state import RunStatus
from src.back.harness.orchestration.cli import _configure_streams, _load_dotenv
from src.back.harness.orchestration.session import load
from src.back.harness.providers.base import (
    ProviderStreamEvent,
    StreamEventType,
    TokenUsage,
)
from src.back.harness.providers.deepseek import DeepSeekProvider
from src.back.harness.tools_and_guardrails import LocalWorkspace
from src.back.harness.utils import FleetRenderer, TerminalRenderer, Theme


def _renderer(plain: bool, color: bool | None) -> TerminalRenderer:
    theme = Theme.detect(sys.stdout, color=color, plain=plain)
    return TerminalRenderer(theme, sys.stdout, show_progress=not plain)


def _emit_tool(
    renderer: TerminalRenderer,
    call_id: str,
    name: str,
    args: dict[str, Any],
    result: str,
) -> None:
    from src.back.harness.providers.base import ToolCall

    call = ToolCall(call_id, name, args)
    renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
    renderer(
        RunEvent(EventType.TOOL_END, 1, tool_call=call, result=result)
    )


class _CancelProvider:
    def complete(self, *args: Any) -> Any:
        raise AssertionError("streaming path required")

    def stream(self, *args: Any):
        yield ProviderStreamEvent(
            StreamEventType.TEXT_DELTA, text="This response is streaming"
        )
        raise KeyboardInterrupt()


def _cancel_demo(renderer: TerminalRenderer) -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        workspace = root / "workspace"
        workspace.mkdir()
        harness = Harness(
            workdir=workspace,
            provider=_CancelProvider(),
            policy=Policy("yolo"),
            workspace_backend=LocalWorkspace(workspace),
            state_dir=root / "state",
            stream=True,
        )
        renderer.begin_run()
        try:
            harness.run("cancel this stream")
        except KeyboardInterrupt:
            renderer.stop_progress()
            renderer.error("Stream cancelled by user; no tool was executed.")
        status = harness.session_status
        session = load(harness.session_path) if harness.session_path else None
        print(f"  synthetic cancellation session status: {status}")
        if session is not None:
            print(f"  synthetic cancellation stored status: {session.status}")
        harness.close()


def _ui_demo(plain: bool, color: bool | None) -> int:
    renderer = _renderer(plain, color)
    renderer.banner(
        model="deepseek-v4-pro",
        workspace="~/demo-workspace",
        mode="safe",
        session="demo-session",
    )
    renderer.begin_run()
    markdown = (
        "## Streaming UI\n\n"
        "This text is rendered from genuine delta-shaped events.\n\n"
        "- tool lifecycle\n"
        "- markdown blocks\n"
        "- safe redaction\n\n"
        "```python\nprint('hello')\n```\n"
    )
    for piece in markdown.splitlines(keepends=True):
        renderer(RunEvent(EventType.ASSISTANT_DELTA, 1, text=piece))
    renderer(RunEvent(EventType.ASSISTANT, 1, text=markdown))
    _emit_tool(renderer, "demo-1", "list_files", {"pattern": "**/*"}, "4 entries")
    _emit_tool(
        renderer,
        "demo-2",
        "write_file",
        {"path": "src/app.py", "content": "print(1)"},
        "Wrote 8 chars to src/app.py",
    )
    _emit_tool(
        renderer,
        "demo-3",
        "bash",
        {"command": "python -m unittest -v"},
        "12 tests passed",
    )
    _emit_tool(
        renderer,
        "demo-4",
        "bash",
        {"command": "python -c 'raise SystemExit(1)'"},
        "ERROR: assertion failed in test_add",
    )
    _emit_tool(
        renderer,
        "demo-5",
        "bash",
        {"command": "sudo rm -rf /workspace"},
        "BLOCKED: dangerous command",
    )
    renderer(
        RunEvent(
            EventType.CONTEXT_COMPACTED,
            1,
            before_tokens=18000,
            after_tokens=6400,
            compaction_succeeded=True,
        )
    )
    renderer(
        RunEvent(
            EventType.RUN_END,
            1,
            status=RunStatus.COMPLETED,
            reason="demo",
            usage=TokenUsage(1200, 420),
        )
    )

    print("\nSynthetic cancellation:")
    _cancel_demo(renderer)

    print("\nFleet presentation (simulated concurrent jobs):")
    fleet = FleetRenderer(
        {
            "artisan-coffee": "Building landing page",
            "taskman": "Running tests",
            "viper": "Implementing gameplay",
        },
        renderer.theme,
        sys.stdout,
    )
    fleet.header()
    threads = []

    def run_job(name: str) -> None:
        callback = fleet.callback(name)
        from src.back.harness.providers.base import ToolCall

        for index in range(2):
            call = ToolCall(f"{name}-{index}", "bash", {"command": "run"})
            callback(RunEvent(EventType.TOOL_START, 1, tool_call=call))
            time.sleep(0.02)
            callback(
                RunEvent(EventType.TOOL_END, 1, tool_call=call, result="ok")
            )
        callback(
            RunEvent(
                EventType.RUN_END,
                2,
                status=RunStatus.COMPLETED,
                reason="simulated",
            )
        )

    for name in ("artisan-coffee", "taskman", "viper"):
        thread = threading.Thread(target=run_job, args=(name,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    fleet.finish(
        [
            {"name": "artisan-coffee", "ok": True, "status": "completed", "turns": 2, "tool_calls": 2},
            {"name": "taskman", "ok": True, "status": "completed", "turns": 2, "tool_calls": 2},
            {"name": "viper", "ok": True, "status": "completed", "turns": 2, "tool_calls": 2},
        ]
    )
    print("\nUI demo is synthetic. Use --live for real DeepSeek + OpenShell.")
    return 0


def _text_measure(max_tokens: int) -> int:
    """Measure genuine provider deltas without tools or OpenShell."""
    _load_dotenv()
    provider = DeepSeekProvider(max_tokens=max_tokens)
    prompt = (
        "Write one detailed paragraph of about 70 words explaining why genuine "
        "streaming improves a terminal coding agent. Do not use tools."
    )
    started = time.monotonic()
    times: list[float] = []
    text: list[str] = []
    finish = None
    try:
        for event in provider.stream(
            provider.default_model,
            "Be concise.",
            [{"role": "user", "text": prompt}],
            [],
        ):
            if event.type is StreamEventType.TEXT_DELTA:
                times.append(time.monotonic() - started)
                text.append(event.text)
            else:
                finish = event
    except Exception as exc:
        print(f"Text stream failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not times:
        print("No text deltas arrived.", file=sys.stderr)
        return 1
    print(
        f"text stream: chunks={len(times)} "
        f"first={times[0]:.3f}s last={times[-1]:.3f}s "
        f"total={time.monotonic() - started:.3f}s "
        f"finish={finish.finish_reason if finish else 'missing'} "
        f"usage={finish.usage if finish else 'missing'}"
    )
    print("words=" + str(len("".join(text).split())))
    return 0


def _live(plain: bool, color: bool | None, max_turns: int) -> int:
    _load_dotenv()
    renderer = _renderer(plain, color)
    task = (
        "Use list_files to inspect the workspace, write_file to create "
        "stream_probe.txt with exactly 'stream-ok', then bash to run "
        "cat stream_probe.txt. Report the command output. Keep it concise."
    )
    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "workspace"
        workspace.mkdir()
        first_delta: list[float] = []
        started = time.monotonic()

        def on_event(event: RunEvent) -> None:
            if event.type is EventType.ASSISTANT_DELTA and not first_delta:
                first_delta.append(time.monotonic() - started)
            renderer(event)

        try:
            harness = Harness(
                workdir=workspace,
                policy=Policy("yolo"),
                max_turns=max_turns,
                max_seconds=300,
                stream=True,
                on_event=on_event,
            )
            renderer.banner(
                model=harness.model,
                workspace="OpenShell workspace",
                mode=harness.policy.mode,
                session=None,
            )
            renderer.begin_run()
            result = harness.run(task)
            elapsed = time.monotonic() - started
            renderer.stop_progress()
            print(
                f"measurement: first_delta={first_delta[0]:.3f}s "
                f"elapsed={elapsed:.3f}s status={result.status.value}"
            )
            print(f"session={harness.session_path}")
            harness.close()
            return 0 if result.completed else 1
        except KeyboardInterrupt:
            renderer.stop_progress()
            print("\nLive stream cancelled; session is preserved.", file=sys.stderr)
            return 130
        except Exception as exc:
            renderer.stop_progress()
            print(f"Live stream failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run a real DeepSeek + OpenShell streaming task.",
    )
    parser.add_argument("--plain", action="store_true", help="No ANSI/Unicode UI.")
    parser.add_argument(
        "--color",
        dest="color",
        action="store_true",
        default=None,
        help="Force ANSI color.",
    )
    parser.add_argument("--text", action="store_true", help="Measure a real text-only provider stream.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_streams()
    if args.text:
        return _text_measure(args.max_tokens)
    if args.live:
        return _live(args.plain, args.color, args.max_turns)
    return _ui_demo(args.plain, args.color)


if __name__ == "__main__":
    raise SystemExit(main())
