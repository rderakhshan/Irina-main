"""Event-driven terminal renderer for the Irina agent loop."""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from ..loop.runner import EventType, RunEvent
from ..loop.state import RunStatus
from .formatting import (
    StreamingRedactor,
    format_args,
    format_duration,
    format_usage,
    one_line,
    preview_result,
    redacted,
    result_status,
    sanitize_terminal,
)
from .markdown import MarkdownStream
from .theme import Theme


class ProgressIndicator:
    """Small cursor-local spinner; never starts without a TTY."""

    def __init__(
        self,
        theme: Theme,
        stream: TextIO,
        *,
        enabled: bool,
    ) -> None:
        self.theme = theme
        self.stream = stream
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._label = ""
        self._lock = threading.Lock()
        self._frames = "◐◓◑◒" if theme.unicode else "|/-\\"
        self._wrote = False

    def start(self, label: str, *, delay: float = 0.0) -> None:
        if not self.enabled:
            return
        self.stop()
        self._label = label
        self._stop.clear()
        self._wrote = False
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, args=(delay,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if self.enabled and self._wrote:
            try:
                self.stream.write("\r\033[K")
                self.stream.flush()
            except (OSError, ValueError):
                pass
        self._wrote = False

    def _run(self, delay: float) -> None:
        if delay > 0 and self._stop.wait(delay):
            return
        index = 0
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._started
            frame = self._frames[index % len(self._frames)]
            text = (
                f"\r{self.theme.paint('running', frame)} "
                f"{self.theme.paint('dim', self._label)} "
                f"{self.theme.paint('dim', format_duration(elapsed))}"
            )
            try:
                self._wrote = True
                self.stream.write(text)
                self.stream.flush()
            except (OSError, ValueError):
                return
            index += 1
            self._stop.wait(0.12)


class TerminalRenderer:
    """Render loop events progressively without duplicating assistant text."""

    def __init__(
        self,
        theme: Theme | None = None,
        stream: TextIO | None = None,
        *,
        show_progress: bool = True,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.theme = theme or Theme.detect(self.stream)
        try:
            is_tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            is_tty = False
        self.show_progress = show_progress and self.theme.interactive and is_tty
        self._lock = threading.RLock()
        self._progress = ProgressIndicator(
            self.theme, self.stream, enabled=self.show_progress
        )
        self._redactor = StreamingRedactor()
        self._markdown: MarkdownStream | None = None
        self._assistant_turn: int | None = None
        self._saw_delta = False
        self._tool_started: dict[str, float] = {}
        self._tool_ids: set[str] = set()
        self._run_started: float | None = None
        self._at_line_start = True
        self.memory_verbose = False

    # -- basic output -------------------------------------------------
    def _write(self, text: str = "") -> None:
        with self._lock:
            try:
                self.stream.write(text)
                self.stream.flush()
                self._at_line_start = text.endswith("\n")
            except (OSError, ValueError):
                pass

    def _line(self, text: str = "") -> None:
        self._write(text + "\n")

    def _start_markdown(self) -> MarkdownStream:
        if self._markdown is None:
            self._markdown = MarkdownStream(self.theme, self._write)
        return self._markdown

    def _assistant_header(self, turn: int) -> None:
        if self._assistant_turn == turn:
            return
        self._assistant_turn = turn
        self._saw_delta = False
        self._line()
        self._line(
            f"  {self.theme.mark('identity')} "
            f"{self.theme.paint('identity', 'Irina')}"
        )

    def _flush_assistant(self) -> None:
        tail = self._redactor.flush()
        if tail:
            self._start_markdown().feed(tail)
        if self._markdown is not None:
            self._markdown.finish()
            self._markdown = None
        if not self._at_line_start:
            self._write("\n")
        self._redactor = StreamingRedactor()
        self._saw_delta = False

    # -- lifecycle ----------------------------------------------------
    def begin_run(self) -> None:
        self._run_started = time.monotonic()
        self._assistant_turn = None
        self._saw_delta = False
        self._tool_started = {}
        self._tool_ids = set()
        self._redactor = StreamingRedactor()
        self._progress.start("Waiting for model", delay=0.35)

    def stop_progress(self) -> None:
        self._progress.stop()

    def banner(
        self,
        *,
        model: str,
        workspace: str,
        mode: str,
        session: str | None,
        sandbox: str = "NVIDIA OpenShell",
    ) -> None:
        width = 58
        self._line()
        self._line(
            f"  {self.theme.mark('identity')} "
            f"{self.theme.paint('identity', 'IRINA')}"
            f"{self.theme.paint('dim', '  ' + one_line(model, 44))}"
        )
        self._line(self.theme.paint("dim", "  " + self.theme.mark("separator") * width))
        self._line(
            f"  {self.theme.paint('dim', 'Workspace')}  "
            f"{one_line(workspace, width)}"
        )
        self._line(
            f"  {self.theme.paint('dim', 'Mode')}       "
            f"{self.theme.paint('notice', mode)}"
        )
        self._line(
            f"  {self.theme.paint('dim', 'Sandbox')}    "
            f"{one_line(sandbox, width)}"
        )
        if session:
            self._line(
                f"  {self.theme.paint('dim', 'Session')}    "
                f"{one_line(session, width)}"
            )

    def resume_notice(self, session: str) -> None:
        self._line(
            f"  {self.theme.mark('resume')} "
            f"{self.theme.paint('notice', 'Resumed session')}  "
            f"{self.theme.paint('dim', one_line(session, 90))}"
        )

    def prompt(self) -> str:
        """Return a Windows-safe interactive prompt string."""
        return f"{self.theme.mark('prompt')} "

    def info(self, message: str) -> None:
        self._line(self.theme.paint("dim", "  " + one_line(message, 180)))

    def error(self, message: str) -> None:
        self.stop_progress()
        self._line(
            f"  {self.theme.mark('failure')} "
            f"{self.theme.paint('failure', redacted(message))}"
        )

    # -- event handling -----------------------------------------------
    def __call__(self, event: RunEvent) -> None:
        if event.type is EventType.ASSISTANT_DELTA:
            self._progress.stop()
            if not event.text:
                return
            self._assistant_header(event.turn)
            self._saw_delta = True
            self._start_markdown().feed(self._redactor.feed(event.text))
            return

        if event.type is EventType.ASSISTANT:
            self._progress.stop()
            if event.text:
                if not self._saw_delta:
                    self._assistant_header(event.turn)
                    self._start_markdown().feed(event.text)
                self._flush_assistant()
            elif self._saw_delta:
                self._flush_assistant()
            return

        if event.type is EventType.TOOL_START:
            self._progress.stop()
            self._flush_assistant()
            call = event.tool_call
            if call is None:
                return
            self._tool_started[call.id] = time.monotonic()
            self._tool_ids.add(call.id)
            summary = format_args(call.args)
            summary_text = (
                "  " + self.theme.paint("dim", summary) if summary else ""
            )
            tool_name = one_line(call.name, 80)
            self._line(
                f"  {self.theme.paint('running', self.theme.mark('running'))} "
                f"{self.theme.paint('tool', tool_name)}"
                f"{summary_text}"
            )
            self._progress.start(f"Running {one_line(call.name, 80)}", delay=0.35)
            return

        if event.type is EventType.TOOL_END:
            self._progress.stop()
            self._flush_assistant()
            call = event.tool_call
            if call is None:
                return
            result = redacted(event.result or "")
            status = result_status(call.name, result)
            started = self._tool_started.pop(call.id, None)
            self._tool_ids.add(call.id)
            duration = (
                format_duration(time.monotonic() - started)
                if started is not None
                else None
            )
            if call.id not in self._tool_started and started is None:
                self._line(
                    f"  {self.theme.paint(status, self.theme.mark(status))} "
                    f"{self.theme.paint('tool', one_line(call.name, 80))}  "
                    f"{self.theme.paint('dim', format_args(call.args))}"
                )
            icon = self.theme.mark(status)
            preview = preview_result(call.name, result)
            suffix = f" {self.theme.mark('dot')} {duration}" if duration else ""
            self._line(
                f"    {self.theme.paint(status, icon)} "
                f"{self.theme.paint(status if status != 'success' else 'dim', preview)}"
                f"{self.theme.paint('dim', suffix)}"
            )
            return

        if event.type is EventType.MEMORY:
            if not self.memory_verbose and event.memory_ok is not False:
                return
            self._progress.stop()
            self._flush_assistant()
            provider = one_line(event.memory_provider or "memory", 40)
            phase = event.memory_phase or "memory"
            text = redacted(event.text or phase)
            self._line(
                f"  {self.theme.paint('notice', self.theme.mark('notice'))} "
                f"{self.theme.paint('notice', phase)}  "
                f"{self.theme.paint('tool', provider)}  "
                f"{self.theme.paint('dim', text)}"
            )
            return

        if event.type is EventType.CONTEXT_COMPACTED:
            self._progress.stop()
            self._flush_assistant()
            self._line(
                f"  {self.theme.mark('notice')} "
                f"{self.theme.paint('notice', 'Context compacted')}  "
                f"{self.theme.paint('dim', f'{event.before_tokens} -> {event.after_tokens} tokens')}"
            )
            return

        if event.type is EventType.RUN_END:
            self._progress.stop()
            self._flush_assistant()
            status = event.status or RunStatus.RUNNING
            label = {
                RunStatus.COMPLETED: "Completed",
                RunStatus.CANCELLED: "Cancelled",
                RunStatus.PROVIDER_ERROR: "Provider error",
                RunStatus.INVALID_RESPONSE: "Invalid response",
                RunStatus.BLOCKED: "Blocked",
                RunStatus.TIMEOUT: "Timed out",
                RunStatus.MAX_TURNS: "Max turns",
                RunStatus.MAX_TOOL_CALLS: "Max tool calls",
                RunStatus.TOKEN_BUDGET: "Token budget",
                RunStatus.STALLED: "Stalled",
                RunStatus.TOOL_ERROR: "Tool error",
                RunStatus.AWAITING_INPUT: "Awaiting input",
            }.get(status, status.value)
            role = "success" if status is RunStatus.COMPLETED else "failure"
            duration = (
                time.monotonic() - self._run_started
                if self._run_started is not None
                else 0.0
            )
            self._line()
            self._line(self.theme.paint("dim", "  " + self.theme.mark("separator") * 58))
            dot = self.theme.mark("dot")
            detail = (
                f"{event.turn} turns {dot} {len(self._tool_ids)} tools {dot} "
                f"{format_duration(duration)}"
            )
            if event.usage is not None:
                detail += (
                    f" {dot} "
                    + format_usage(
                        event.usage.input_tokens,
                        event.usage.output_tokens,
                    )
                )
            self._line(
                f"  {self.theme.paint(role, self.theme.mark(role))} "
                f"{self.theme.paint(role, label)}   "
                f"{self.theme.paint('dim', detail)}"
            )
            if event.reason and status is not RunStatus.COMPLETED:
                self._line(
                    self.theme.paint("dim", "  " + one_line(event.reason, 180))
                )
            return
