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
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from .harness import Harness
from ..loop.runner import EventType, RunEvent
from ..loop.state import RunResult
from .session import SessionError
from ..tools_and_guardrails.security import Policy, redact_secrets
from ..utils import TerminalRenderer, Theme
from ..utils.paste import (
    PASTE_END,
    PASTE_START,
    PromptComposer,
    compose_stream,
    is_large_paste,
    strip_paste_markers,
)
from ..utils.memory_logging import configure_memory_logging
from ..context_management.memory import (
    DeepSeekMemoryModel,
    MemoryConfig,
    MemoryManager,
)


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
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="Read the task from a host UTF-8 file (headless).",
    )
    parser.add_argument(
        "-d",
        "--workdir",
        default=".",
        help="Approved project workspace (default: current directory).",
    )
    parser.add_argument(
        "--skills-dir",
        default="skills",
        help=(
            "Host directory of skills/<name>/SKILL.md to upload into the "
            "sandbox at startup (default: ./skills). Pass an empty string to "
            "disable."
        ),
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        metavar="PATH",
        help="Host file or directory to upload into input/ before the run.",
    )
    parser.add_argument(
        "--export",
        dest="export_paths",
        action="append",
        default=[],
        metavar="SANDBOX_PATH",
        help="Workspace path to download on clean shutdown.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        metavar="PATH",
        help="Host directory that receives --export paths.",
    )
    parser.add_argument(
        "--export-contents",
        action="store_true",
        help=(
            "Merge exported directory contents into --artifacts-dir instead "
            "of creating a subdirectory."
        ),
    )
    parser.add_argument(
        "--force",
        dest="force_export",
        action="store_true",
        help="Replace an existing exported artifact.",
    )
    parser.add_argument(
        "--auto-export",
        action="store_true",
        help="Export configured artifact paths after each completed run.",
    )
    parser.add_argument(
        "--sandbox-image",
        default=None,
        metavar="IMAGE",
        help=(
            "Approved OpenShell image, e.g. irina/irina-tools:1. "
            "Defaults to the built-in image."
        ),
    )
    parser.add_argument(
        "--sandbox-policy",
        default=None,
        metavar="PATH",
        help=(
            "Approved OpenShell policy YAML. Use "
            "openshell_policy_network.yaml to allow the listed web "
            "destinations; defaults to the strict no-network policy."
        ),
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
        "--memory",
        dest="memory",
        action="store_true",
        default=None,
        help="Enable canonical memory and the configured EvolveLab provider.",
    )
    parser.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="Disable memory even when configuration enables it.",
    )
    parser.add_argument(
        "--memory-provider",
        default=None,
        help="Explicit EvolveLab provider identifier.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show routine memory logs with level-based terminal colors.",
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
        skills_dir=getattr(args, "skills_dir", None),
        inputs=getattr(args, "inputs", None),
        export_paths=getattr(args, "export_paths", None),
        artifacts_dir=getattr(args, "artifacts_dir", None),
        export_contents=getattr(args, "export_contents", False),
        force_export=getattr(args, "force_export", False),
        sandbox_image=getattr(args, "sandbox_image", None),
        sandbox_policy=getattr(args, "sandbox_policy", None),
    )


def _auto_export(harness: Any) -> None:
    """Export artifacts after a run when --auto-export is enabled."""
    if not getattr(harness, "_artifacts_dir", None) or not getattr(
        harness, "_export_paths", None
    ):
        return
    try:
        paths = harness.export_now(
            force=getattr(harness, "_force_export", False)
        )
    except FileExistsError:
        return
    except Exception as exc:
        print(f"artifact export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    for path in paths:
        print(f"artifact={path}", file=sys.stderr)
    harness._transfers_reported = True


def _report_transfers(harness: Any) -> None:
    """Print exported artifacts and nonfatal export failures."""
    if not getattr(harness, "_transfers_reported", False):
        for path in getattr(harness, "exported_artifacts", []) or []:
            print(f"artifact={path}", file=sys.stderr)
    for error in getattr(harness, "export_errors", []) or []:
        print(f"artifact export failed: {error}", file=sys.stderr)


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


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _env_paths(name: str, *, optional: bool = False) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    paths = [part.strip() for part in raw.split(os.pathsep) if part.strip()]
    if optional:
        return [path for path in paths if Path(path).expanduser().exists()]
    return paths


def _apply_env_defaults(args: argparse.Namespace) -> None:
    """Apply IRINA_* defaults for a flag-free ``uv run main.py`` session."""
    required = os.environ.get("IRINA_DEFAULTS_REQUIRE", "").strip()
    if required and not Path(required).expanduser().exists():
        return
    if not args.sandbox_image:
        args.sandbox_image = os.environ.get("IRINA_SANDBOX_IMAGE") or None
    if not args.sandbox_policy:
        args.sandbox_policy = os.environ.get("IRINA_SANDBOX_POLICY") or None
    if args.skills_dir == "skills":
        args.skills_dir = os.environ.get("IRINA_SKILLS_DIR", args.skills_dir)
    if not args.inputs:
        args.inputs = _env_paths("IRINA_INPUTS") + _env_paths(
            "IRINA_OPTIONAL_INPUTS", optional=True
        )
    if not args.export_paths:
        args.export_paths = _env_paths("IRINA_EXPORT_PATHS")
    if not args.artifacts_dir:
        args.artifacts_dir = os.environ.get("IRINA_ARTIFACTS_DIR") or None
    if not args.export_contents:
        args.export_contents = _truthy(os.environ.get("IRINA_EXPORT_CONTENTS"))
    if not args.force_export:
        args.force_export = _truthy(os.environ.get("IRINA_FORCE_EXPORT"))
    if not args.auto_export:
        args.auto_export = _truthy(os.environ.get("IRINA_AUTO_EXPORT"))
    mode = os.environ.get("IRINA_MODE", "").strip().lower()
    if args.mode == "safe" and mode in {"safe", "yolo", "read-only"}:
        args.mode = mode


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

    if getattr(args, "auto_export", False):
        _auto_export(harness)
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
    _report_transfers(harness)
    return 0


MAX_PROMPT_CHARS = 200_000
_BRACKETED_PASTE_START = PASTE_START
_BRACKETED_PASTE_END = PASTE_END
PASTE_TERMINATORS = {'"""', "/end"}
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
_BURST_GAP_SECONDS = 0.02
_ENTER_GAP_SECONDS = 0.15


def _set_bracketed_paste(enabled: bool) -> int | None:
    """Enable or restore Windows virtual-terminal input.

    Returns the previous console mode when it can be changed so the caller can
    restore it. Other platforms and consoles are left untouched.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return None
        previous = int(mode.value)
        if enabled:
            kernel32.SetConsoleMode(handle, previous | ENABLE_VIRTUAL_TERMINAL_INPUT)
        else:
            kernel32.SetConsoleMode(handle, previous & ~ENABLE_VIRTUAL_TERMINAL_INPUT)
        return previous
    except Exception:
        return None


def _flush_burst(
    composer: PromptComposer,
    burst: list[str],
    write: Callable[[str], None],
) -> bool:
    """Flush buffered key events.

    A burst is treated as a paste when it is large; this is the fallback for
    terminals that do not deliver bracketed-paste markers. The caller tracks
    Enter separately, so a flush never ends the prompt by itself.
    """
    if not burst:
        return False
    text = "".join(burst)
    burst.clear()
    if is_large_paste(text):
        write(composer.add_paste(text))
    else:
        write(composer.add_typed(text))
    return False


def _read_windows_console_line(prompt: str) -> str:
    """Read one line with bracketed-paste and burst-paste support."""
    import msvcrt

    composer = PromptComposer()
    previous_mode = _set_bracketed_paste(True)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    write(prompt)
    state = "normal"
    escape = ""
    paste_buffer = ""
    paste_escape = ""
    burst: list[str] = []
    last_key = time.monotonic()
    ends_with_enter = False
    try:
        while True:
            try:
                ready = msvcrt.kbhit()
            except (OSError, ValueError):
                return input(prompt)
            now = time.monotonic()
            if ready:
                try:
                    char = msvcrt.getwch()
                except (OSError, ValueError):
                    return input(prompt)
                if char in {"\x00", "\xe0"}:
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    last_key = now
                    continue
                if char == "\x03":
                    raise KeyboardInterrupt
                if state == "normal":
                    if char == "\x1b":
                        ends_with_enter = _flush_burst(composer, burst, write) or ends_with_enter
                        escape = char
                        state = "escape"
                    elif char in {"\r", "\n"}:
                        if burst and now - last_key < _BURST_GAP_SECONDS:
                            # A newline inside a fast burst is pasted content,
                            # not the Enter key that submits the prompt.
                            burst.append("\n")
                            ends_with_enter = False
                        else:
                            ends_with_enter = _flush_burst(
                                composer, burst, write
                            )
                            composer.add_typed("\n")
                            write("\n")
                            ends_with_enter = True
                    elif char == "\x08":
                        ends_with_enter = _flush_burst(composer, burst, write)
                        removed = composer.backspace()
                        if removed:
                            write("\b \b" * removed)
                        ends_with_enter = False
                    elif char == "\x05":
                        ends_with_enter = _flush_burst(composer, burst, write)
                        if composer.expand_last_paste() is not None:
                            write("\r\x1b[2K" + prompt + composer.render())
                        ends_with_enter = False
                    else:
                        burst.append(char)
                        ends_with_enter = False
                elif state == "escape":
                    escape += char
                    if escape == _BRACKETED_PASTE_START:
                        paste_buffer = ""
                        state = "paste"
                    elif _BRACKETED_PASTE_START.startswith(escape):
                        pass
                    else:
                        # Drop an unrecognized escape sequence (arrows, F-keys).
                        state = "normal"
                        escape = ""
                elif state == "paste":
                    if char == "\x1b":
                        paste_escape = char
                        state = "paste_escape"
                    else:
                        paste_buffer += char
                elif state == "paste_escape":
                    paste_escape += char
                    if paste_escape == _BRACKETED_PASTE_END:
                        write(composer.add_paste(paste_buffer))
                        paste_buffer = ""
                        paste_escape = ""
                        state = "normal"
                        ends_with_enter = False
                    elif _BRACKETED_PASTE_END.startswith(paste_escape):
                        pass
                    else:
                        paste_buffer += paste_escape
                        paste_escape = ""
                        state = "paste"
                last_key = now
                continue

            if burst and now - last_key > _BURST_GAP_SECONDS:
                if _flush_burst(composer, burst, write):
                    ends_with_enter = True
            if ends_with_enter and now - last_key > _ENTER_GAP_SECONDS:
                break
            time.sleep(0.005)
    finally:
        if previous_mode is not None:
            _set_bracketed_paste(False)
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-10), previous_mode
                )
            except Exception:
                pass
    return composer.submit().strip("\n")


def _read_console_line(prompt: str) -> str:
    """Read one console entry, collapsing large paste blocks."""
    try:
        interactive = (
            os.name == "nt"
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )
    except (AttributeError, OSError):
        interactive = False
    if not interactive:
        return _clean_paste(input(prompt))
    return _read_windows_console_line(prompt)


def _clean_paste(text: str) -> str:
    """Remove bracketed markers while preserving pasted content."""
    return strip_paste_markers(text)


def _capture_prompt(line: str) -> str:
    """Turn one console entry into a task string without altering pastes."""
    if _BRACKETED_PASTE_START in line or _BRACKETED_PASTE_END in line:
        text = compose_stream(line).submit()
    else:
        text = line
    probe = text.strip()
    if not probe:
        return ""
    if (
        probe.lower() in {"exit", "quit", "/paste"}
        or probe.lower().startswith(("/file ", "/export"))
    ):
        return probe
    return text.rstrip("\r\n")


def _read_multiline(
    read_line: Callable[[str], str] = _read_console_line,
) -> str:
    """Read a pasted multi-line task until a terminator line."""
    lines: list[str] = []
    while True:
        try:
            line = read_line("... ")
        except (EOFError, KeyboardInterrupt):
            return ""
        if line.strip() in PASTE_TERMINATORS:
            break
        lines.append(line)
    text = "\n".join(lines)
    if not text.strip():
        return ""
    return text if len(text) <= MAX_PROMPT_CHARS else ""


def _task_from_file(path: str) -> str:
    """Read a host file explicitly named with /file."""
    raw = path.strip().strip('"').strip("'")
    try:
        text = Path(raw).expanduser().read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise ValueError(f"cannot read {raw}: {exc}") from None
    if len(text) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"prompt file is larger than {MAX_PROMPT_CHARS} characters"
        )
    return text.strip()


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
        renderer.info(
            "Type a task, /paste for multi-line text, /file <path>, "
            "/export, or 'exit'."
        )
        args._irina_banner = True

    prompt_count = 0
    normal_exit = True
    try:
        while True:
            try:
                line = _read_console_line(renderer.prompt())
            except EOFError:
                print("\nBye.")
                break
            except KeyboardInterrupt:
                print("\nBye.")
                break
            task = _capture_prompt(line)
            if not task:
                continue
            if task.lower() in {"exit", "quit"}:
                break
            if task.lower() in {"/export", "/export force"}:
                force = task.lower().endswith(" force")
                try:
                    paths = harness.export_now(force=force)
                except Exception as exc:
                    renderer.error(
                        f"Export failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                if not paths:
                    renderer.info("No artifacts to export yet.")
                    continue
                for path in paths:
                    print(f"artifact={path}")
                harness._transfers_reported = True
                continue
            if task.lower() == "/paste":
                renderer.info('Paste text, then a line containing only """')
                task = _read_multiline()
                if not task:
                    renderer.info("No pasted text loaded.")
                    continue
                renderer.info(f"Loaded {len(task)} characters.")
            elif task.lower().startswith("/file "):
                try:
                    task = _task_from_file(task[6:])
                except ValueError as exc:
                    renderer.error(str(exc))
                    continue
                if not task:
                    renderer.info("Prompt file is empty.")
                    continue
                renderer.info(f"Loaded {len(task)} characters from file.")

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

            if getattr(args, "auto_export", False):
                _auto_export(harness)
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
            else:
                _report_transfers(harness)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Irina CLI and return a process exit code."""
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.export_paths and not args.artifacts_dir:
        parser.error("--artifacts-dir is required with --export")

    _load_dotenv()
    _apply_env_defaults(args)
    if args.prompt_file:
        if args.prompt is not None:
            parser.error("--prompt-file cannot be combined with --prompt")
        try:
            args.prompt = _task_from_file(args.prompt_file)
        except ValueError as exc:
            parser.error(str(exc))
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
    renderer.memory_verbose = bool(args.verbose)
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

    configure_memory_logging(
        verbose=bool(args.verbose),
        theme=theme,
        stream=sys.stderr,
        on_emit=renderer.stop_progress,
    )

    try:
        memory_config = MemoryConfig.load()
        if args.memory is not None:
            memory_config.enabled = bool(args.memory)
        if args.memory_provider:
            memory_config.active_provider = args.memory_provider
            if args.memory is None:
                memory_config.enabled = True
        if memory_config.enabled:
            memory_manager = MemoryManager(
                memory_config,
                workspace=args.workdir,
                model=DeepSeekMemoryModel(harness.provider, harness.model),
            )
            harness.memory = memory_manager
            args.memory_manager = memory_manager
    except Exception as exc:
        renderer.error(
            "Memory cannot start: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2

    # Providers may create their own child loggers during initialization.
    configure_memory_logging(
        verbose=bool(args.verbose),
        theme=theme,
        stream=sys.stderr,
        on_emit=renderer.stop_progress,
    )

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

    try:
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
        renderer.info(
            "Type a task, /paste for multi-line text, /file <path>, "
            "/export, or 'exit'."
        )
        args._irina_banner = True
        return _run_interactive(harness, args)
    finally:
        memory_manager = getattr(args, "memory_manager", None)
        if memory_manager is not None:
            try:
                memory_manager.close()
            except Exception:
                pass


__all__ = [
    "EventPrinter",
    "build_harness",
    "build_parser",
    "build_policy",
    "main",
    "make_approver",
]