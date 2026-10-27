"""Stage 4: command-line entry point for the Irina harness.

Concept: the CLI is a thin front door over ``Harness``. It parses configuration,
prints structured events, handles approvals, and maps the actual ``RunResult``
status to a process exit code.

Design constraints:
* Standard library only.
* Safe is the default mode in interactive and headless runs.
* Headless safe mode never auto-approves; use ``--mode yolo`` explicitly.
* API keys and tool arguments are never printed unredacted.
* Ctrl-C preserves the durable session but leaves the owned sandbox alive for
  ``--resume``; a normal exit cleans up owned resources.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from .harness import Harness
from ..loop.runner import EventType, RunEvent
from ..loop.state import RunResult
from .session import SessionError
from ..tools_and_guardrails.security import Policy, redact_secrets
from ..utils import TerminalRenderer, Theme


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="irina",
        description=(
            "Irina, a small standard-library agent harness with OpenShell "
            "execution."
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Run one task without entering the interactive prompt loop.",
    )
    parser.add_argument(
        "-d",
        "--workdir",
        default=".",
        help="Approved project workspace (default: current directory).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model override. Defaults to the provider or IRINA_MODEL.",
    )
    parser.add_argument(
        "--mode",
        choices=("safe", "yolo", "read-only"),
        default="safe",
        help=(
            "Application permission mode. Default: safe. yolo skips approvals "
            "but never weakens OpenShell."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="SESSION",
        help=(
            "Resume the latest resumable session for the workspace. Optionally "
            "pass a trusted session path."
        ),
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream model output as it arrives (default).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Use the non-streaming provider path.",
    )
    parser.add_argument(
        "--color",
        dest="color",
        action="store_true",
        default=None,
        help="Force ANSI color and Unicode symbols.",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable ANSI color.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Automation mode: no color, no spinner, ASCII fallbacks.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=120,
        help="Maximum agent turns (default: 120).",
    )
    return parser


def _configure_streams() -> None:
    """Make Windows console encoding failures non-fatal."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from the nearest .env, if present."""
    candidates = [Path.cwd() / ".env"]
    candidates.extend(parent / ".env" for parent in Path.cwd().parents)
    candidates.append(Path(__file__).resolve().parents[4] / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return


def _clip(text: str, limit: int) -> str:
    """Clip text for one-line terminal display."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_args(args: Any) -> str:
    """Render tool arguments safely and compactly."""
    if not isinstance(args, dict):
        return ""
    parts = []
    for key, value in args.items():
        rendered = redact_secrets(str(value))
        parts.append(f"{key}={_clip(rendered, 120)}")
    return ", ".join(parts)


class EventPrinter:
    """Print harness events without duplicating the transcript."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        try:
            self.ansi = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self.ansi = False

    def _write(self, text: str = "") -> None:
        print(text, file=self.stream)

    def _dim(self, text: str) -> str:
        if self.ansi:
            return f"\033[2m{text}\033[0m"
        return f"  | {text}"

    def __call__(self, event: RunEvent) -> None:
        if event.type is EventType.ASSISTANT:
            if event.text.strip():
                self._write(redact_secrets(event.text))
        elif event.type is EventType.TOOL_START:
            call = event.tool_call
            if call is None:
                return
            self._write(f"  -> {call.name}({_format_args(call.args)})")
        elif event.type is EventType.TOOL_END:
            result = redact_secrets(str(event.result or ""))
            first = result.splitlines()[0] if result.splitlines() else "(no output)"
            self._write(self._dim(_clip(first, 300)))
        elif event.type is EventType.CONTEXT_COMPACTED:
            self._write(
                "  [context compacted: "
                f"before={event.before_tokens} after={event.after_tokens} "
                f"success={bool(event.compaction_succeeded)}]"
            )
        elif event.type is EventType.RUN_END:
            status = event.status.value if event.status else "unknown"
            reason = redact_secrets(str(event.reason or ""))
            self._write(f"  [run {status}: {_clip(reason, 300)}]")


def make_approver(
    *,
    input_fn: Callable[[str], str] = input,
    stream: TextIO | None = None,
    on_prompt: Callable[[], None] | None = None,
) -> Callable[[Any, str], bool]:
    """Create a safe-mode approval callback.

    The tool call itself is printed by the event printer immediately before the
    policy callback. This function adds the reason and asks for explicit
    approval. Enter, EOF, and every non-affirmative answer refuse.
    """

    output = stream if stream is not None else sys.stdout

    def approver(call: Any, reason: str) -> bool:
        if on_prompt is not None:
            on_prompt()
        print(f"Reason: {redact_secrets(str(reason))}", file=output)
        try:
            answer = input_fn("Approve this operation? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print(file=output)
            return False
        return answer.strip().lower() in {"y", "yes"}

    return approver


def build_policy(
    args: argparse.Namespace,
    *,
    interactive: bool,
    on_prompt: Callable[[], None] | None = None,
) -> Policy:
    """Create the application policy for this CLI mode."""
    if args.mode == "safe" and interactive:
        return Policy(
            "safe", approver=make_approver(on_prompt=on_prompt)
        )
    return Policy(args.mode)


def build_harness(
    args: argparse.Namespace,
    *,
    policy: Policy,
    on_event: Callable[[RunEvent], None],
) -> Harness:
    """Construct the Harness from parsed CLI configuration."""
    return Harness(
        workdir=args.workdir,
        model=args.model,
        policy=policy,
        on_event=on_event,
        max_turns=args.max_turns,
        stream=getattr(args, "stream", False),
    )


def _print_result(result: RunResult, stream: TextIO | None = None) -> None:
    """Print the final answer and structured status."""
    output = stream if stream is not None else sys.stdout
    if result.text.strip():
        print("\nFinal answer:", file=output)
        print(redact_secrets(result.text), file=output)
    print(
        f"\nstatus={result.status.value} reason={redact_secrets(result.reason)} "
        f"turns={result.turns} tool_calls={result.tool_calls}",
        file=output,
    )


def _try_resume(harness: Harness, args: argparse.Namespace) -> None:
    """Resume a session when requested; report when none is available."""
    if not args.resume:
        return
    renderer = getattr(args, "renderer", None)
    target = args.resume if isinstance(args.resume, str) else None
    resumed = harness.resume(target)
    if resumed:
        if renderer is not None:
            renderer.resume_notice(str(harness.session_path))
        else:
            print(f"Resumed session: {harness.session_path}")
    elif renderer is not None:
        renderer.info("No resumable session found; starting a new session.")
    else:
        print("No resumable session was found; starting a new session.")


def _run_headless(
    harness: Harness,
    prompt: str,
    args: argparse.Namespace,
) -> int:
    """Run one task and map the actual RunResult to an exit code."""
    renderer = getattr(args, "renderer", None)
    if renderer is None:
        renderer = TerminalRenderer(Theme.detect(sys.stdout))
        args.renderer = renderer
    renderer.begin_run()
    try:
        result = harness.run(prompt)
    except KeyboardInterrupt:
        renderer.stop_progress()
        print(
            "\nRun interrupted. Session log is safe.",
            file=sys.stderr,
        )
        if harness.session_path is not None:
            print(f"Session: {harness.session_path}", file=sys.stderr)
        print("Resume with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        renderer.error(f"Irina failed: {type(exc).__name__}: {exc}")
        return 1

    if harness.session_path is not None:
        print(f"session={harness.session_path}")
    if not result.completed:
        print(
            "The run did not complete. Session history is preserved; "
            "use --resume to continue it.",
            file=sys.stderr,
        )
        return 1
    try:
        harness.close()
    except Exception as exc:
        renderer.error(f"Cleanup failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _run_interactive(
    harness: Harness,
    args: argparse.Namespace,
    renderer: TerminalRenderer | None = None,
) -> int:
    """Run the interactive prompt loop."""
    if renderer is None:
        renderer = getattr(args, "renderer", None)
    if renderer is None:
        renderer = TerminalRenderer(Theme.detect(sys.stdout))
        args.renderer = renderer
    if not getattr(args, "_irina_banner", False):
        renderer.banner(
            model=harness.model,
            workspace=harness.workspace,
            mode=harness.policy.mode,
            session=str(harness.session_path) if harness.session_path else None,
        )
        renderer.info("Type a task, or 'exit' to quit.")
        args._irina_banner = True

    prompt_count = 0
    normal_exit = True
    try:
        while True:
            try:
                line = input(renderer.prompt())
            except EOFError:
                print("\nBye.")
                break
            except KeyboardInterrupt:
                print("\nBye.")
                break
            task = line.strip()
            if not task:
                continue
            if task.lower() in {"exit", "quit"}:
                break

            renderer.begin_run()
            try:
                if prompt_count == 0:
                    result = harness.run(task)
                else:
                    result = harness.continue_task(task)
            except KeyboardInterrupt:
                renderer.stop_progress()
                print(
                    "\nRun interrupted. Session log is safe.",
                    file=sys.stderr,
                )
                if harness.session_path is not None:
                    print(
                        f"Session: {harness.session_path}",
                        file=sys.stderr,
                    )
                print("Resume with --resume.", file=sys.stderr)
                normal_exit = False
                return 130
            except Exception as exc:
                renderer.stop_progress()
                renderer.error(f"Irina failed: {type(exc).__name__}: {exc}")
                normal_exit = False
                return 1

            prompt_count += 1
            if not result.completed:
                renderer.info(
                    "The run did not complete. The session is preserved for "
                    "continuation."
                )
    finally:
        renderer.stop_progress()
        if normal_exit:
            try:
                harness.close()
            except Exception as exc:
                renderer.error(
                    f"Cleanup failed: {type(exc).__name__}: {exc}"
                )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Irina CLI and return a process exit code."""
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")

    _load_dotenv()
    interactive = args.prompt is None
    is_tty = False
    try:
        is_tty = bool(sys.stdout.isatty())
    except (AttributeError, OSError):
        pass
    plain = bool(args.plain or (not is_tty and args.color is not True))
    theme = Theme.detect(
        sys.stdout,
        color=args.color,
        plain=plain,
    )
    renderer = TerminalRenderer(
        theme,
        sys.stdout,
        show_progress=not plain,
    )
    args.renderer = renderer
    policy = build_policy(
        args,
        interactive=interactive,
        on_prompt=renderer.stop_progress,
    )

    try:
        harness = build_harness(args, policy=policy, on_event=renderer)
    except RuntimeError as exc:
        renderer.error(f"Irina cannot start: {exc}")
        return 2
    except Exception as exc:
        renderer.error(
            f"Irina cannot start: {type(exc).__name__}: {exc}"
        )
        return 2

    try:
        _try_resume(harness, args)
    except SessionError as exc:
        renderer.error(f"Resume failed: {exc}")
        return 1
    except Exception as exc:
        renderer.error(
            f"Resume failed: {type(exc).__name__}: {exc}"
        )
        return 1

    if not interactive:
        return _run_headless(harness, args.prompt, args)
    try:
        workspace = harness.workspace
    except Exception as exc:
        renderer.error(
            "Irina cannot start the execution environment: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2
    renderer.banner(
        model=harness.model,
        workspace=workspace,
        mode=harness.policy.mode,
        session=(
            str(harness.session_path) if harness.session_path else None
        ),
    )
    renderer.info("Type a task, or 'exit' to quit.")
    args._irina_banner = True
    return _run_interactive(harness, args)


__all__ = [
    "EventPrinter",
    "build_harness",
    "build_parser",
    "build_policy",
    "main",
    "make_approver",
]