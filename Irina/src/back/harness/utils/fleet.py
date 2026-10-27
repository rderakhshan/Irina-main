"""Thread-safe, serialized presentation for concurrent fleet jobs."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Mapping, Sequence, TextIO

from ..loop.runner import EventType, RunEvent
from ..loop.state import RunStatus
from .formatting import (
    format_args,
    format_duration,
    one_line,
    preview_result,
    result_status,
    sanitize_terminal,
)
from .theme import Theme


class FleetRenderer:
    """Prefix every fleet line with its job and serialize writes."""

    def __init__(
        self,
        jobs: Mapping[str, str] | Sequence[str],
        theme: Theme | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.theme = theme or Theme.detect(self.stream)
        if isinstance(jobs, Mapping):
            self.jobs = [
                (sanitize_terminal(str(name)), sanitize_terminal(str(task)))
                for name, task in jobs.items()
            ]
        else:
            self.jobs = [(sanitize_terminal(str(name)), "") for name in jobs]
        self._lock = threading.Lock()
        self._starts: dict[str, dict[str, float]] = {
            name: {} for name, _ in self.jobs
        }
        self._counts: dict[str, int] = {name: 0 for name, _ in self.jobs}

    def _write(self, text: str = "") -> None:
        with self._lock:
            try:
                self.stream.write(text + "\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass

    def header(self) -> None:
        self._write()
        self._write(
            f"  {self.theme.mark('identity')} "
            f"{self.theme.paint('identity', 'Irina Fleet')}"
            f"{self.theme.paint('dim', f' {self.theme.mark('dot')} {len(self.jobs)} jobs')}"
        )
        for name, task in self.jobs:
            self._write(
                f"  {self.theme.paint('running', self.theme.mark('running'))} "
                f"{self.theme.paint('tool', one_line(name, 24).ljust(24))} "
                f"{self.theme.paint('dim', one_line(task, 80))}"
            )
        self._write()

    def callback(self, job_name: str):
        """Return an event callback for one job."""

        def on_event(event: RunEvent) -> None:
            safe_name = sanitize_terminal(str(job_name))
            prefix = self.theme.paint('tool', safe_name[:28].ljust(28))
            if event.type is EventType.TOOL_START:
                call = event.tool_call
                if call is None:
                    return
                self._starts[job_name][call.id] = time.monotonic()
                self._counts[job_name] += 1
                summary = format_args(call.args)
                self._write(
                    f"  {self.theme.paint('running', self.theme.mark('running'))} "
                    f"{prefix} {self.theme.paint('tool', one_line(call.name, 80))}  "
                    f"{self.theme.paint('dim', one_line(summary, 100))}"
                )
            elif event.type is EventType.TOOL_END:
                call = event.tool_call
                if call is None:
                    return
                status = result_status(call.name, str(event.result or ""))
                started = self._starts[job_name].pop(call.id, None)
                duration = (
                    format_duration(time.monotonic() - started)
                    if started is not None
                    else None
                )
                preview = preview_result(call.name, str(event.result or ""))
                suffix = f" {self.theme.mark('dot')} {duration}" if duration else ""
                self._write(
                    f"  {self.theme.paint(status, self.theme.mark(status))} "
                    f"{prefix} {self.theme.paint('dim', preview + suffix)}"
                )
            elif event.type is EventType.RUN_END:
                status = event.status or RunStatus.RUNNING
                role = (
                    "success"
                    if status is RunStatus.COMPLETED
                    else "failure"
                )
                self._write(
                    f"  {self.theme.paint(role, self.theme.mark(role))} "
                    f"{prefix} {self.theme.paint(role, status.value)}"
                )
            elif event.type is EventType.CONTEXT_COMPACTED:
                self._write(
                    f"  {self.theme.paint('notice', self.theme.mark('notice'))} "
                    f"{prefix} {self.theme.paint('dim', 'context compacted')}"
                )

        return on_event

    def finish(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Print final state in input order, not completion order."""
        completed = sum(1 for result in results if result.get("ok"))
        self._write()
        for result in results:
            role = "success" if result.get("ok") else "failure"
            name = sanitize_terminal(str(result.get("name", "job")))
            status = str(result.get("status", "unknown"))
            turns = int(result.get("turns", 0) or 0)
            tools = int(result.get("tool_calls", 0) or 0)
            self._write(
                f"  {self.theme.paint(role, self.theme.mark(role))} "
                f"{self.theme.paint('tool', name[:24].ljust(24))} "
                f"{self.theme.paint(role, status)}  "
                f"{self.theme.paint('dim', f'{turns} turns {self.theme.mark('dot')} {tools} tools')}"
            )
        self._write(
            f"  {self.theme.paint('dim', f'{completed}/{len(results)} jobs passed')}"
        )
