"""Presentation-safe formatting helpers for the terminal UI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from ..tools_and_guardrails.security import redact_secrets

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_BIDI_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")


def strip_ansi(text: str) -> str:
    """Remove ANSI control sequences from text."""
    return _ANSI_RE.sub("", text)


def sanitize_terminal(text: str) -> str:
    """Remove terminal-control and bidi characters from untrusted display text."""
    text = _ANSI_RE.sub("", str(text))
    text = _BIDI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", text)


def redacted(text: Any) -> str:
    """Sanitize then redact a complete string for display."""
    return redact_secrets(sanitize_terminal(str(text)))


def one_line(text: Any, limit: int = 160) -> str:
    """Collapse whitespace and clip a string to one safe line."""
    collapsed = " ".join(redacted(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)] + "..."


def clip(text: str, limit: int, marker: str = "...") -> str:
    """Clip text by character count."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def format_duration(seconds: float) -> str:
    """Render a compact human-readable duration."""
    if seconds < 0:
        seconds = 0
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def format_tokens(count: int) -> str:
    """Render a compact token count."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}m"


def format_usage(input_tokens: int, output_tokens: int) -> str:
    """Render input/output token usage."""
    return (
        f"{format_tokens(input_tokens)} in / "
        f"{format_tokens(output_tokens)} out"
    )


def pretty_path(path: str | Path, *, cwd: str | Path | None = None) -> str:
    """Make a path compact without changing its identity in tool results."""
    raw = str(path)
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    home = Path.home()
    for root, label in (
        (home, "~"),
        (Path(cwd).resolve() if cwd else Path.cwd().resolve(), "."),
    ):
        try:
            relative = resolved.relative_to(root)
        except (ValueError, OSError):
            continue
        if label == "~":
            return "~" + ("/" + relative.as_posix() if relative.parts else "")
        return "." + ("/" + relative.as_posix() if relative.parts else "")
    return sanitize_terminal(raw).replace("\\", "/")


def format_args(args: Any, *, limit: int = 150) -> str:
    """Render concise, redacted tool arguments."""
    if not isinstance(args, Mapping):
        return ""
    parts: list[str] = []
    for key in ("command", "path", "pattern", "task", "query"):
        if key in args:
            label = sanitize_terminal(str(key))
            parts.append(f"{label}={one_line(args[key], limit)}")
    for key, value in args.items():
        if key in {"command", "path", "pattern", "task", "query"}:
            continue
        key = sanitize_terminal(str(key)) or "arg"
        if key in {"content", "old", "new", "text"}:
            if isinstance(value, str):
                parts.append(f"{key}=<{len(value)} chars>")
            else:
                parts.append(f"{key}=<value>")
            continue
        parts.append(f"{key}={one_line(value, max(24, limit // 3))}")
    return "  ".join(parts[:3])


def result_status(tool_name: str, result: str) -> str:
    """Classify a tool observation for presentation only."""
    text = str(result or "").lstrip()
    if text.startswith("BLOCKED:"):
        return "blocked"
    if text.startswith("ERROR:"):
        return "failure"
    lowered = text.lower()
    if tool_name == "bash" and (
        "exit code 1" in lowered
        or "(exit 1" in lowered
        or "exit_code=1" in lowered
        or "exit status: 1" in lowered
        or "exit status=1" in lowered
        or "traceback (most recent call last)" in lowered
        or "assertionerror" in lowered
    ):
        return "failure"
    return "success"


def preview_result(tool_name: str, result: str, *, limit: int = 220) -> str:
    """Return a short, meaningful preview of a tool result."""
    text = redacted(result).strip()
    if not text:
        return "(no output)"
    if text.startswith("BLOCKED:"):
        return one_line(text[len("BLOCKED:") :].strip(), limit)
    if text.startswith("ERROR:"):
        return one_line(text[len("ERROR:") :].strip(), limit)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if tool_name in {"write_file", "edit_file"}:
        for line in lines:
            lowered = line.lower()
            if lowered.startswith(("wrote", "created", "edited", "updated")):
                return one_line(line, limit)
        return "File updated"
    if tool_name == "list_files":
        if re.match(r"^\d+ entr", text, re.IGNORECASE):
            return one_line(text, limit)
        return f"{len(lines)} entries" if len(lines) != 1 else "1 entry"
    if tool_name == "grep":
        if re.match(r"^\d+ match", text, re.IGNORECASE):
            return one_line(text, limit)
        return f"{len(lines)} matches" if len(lines) != 1 else "1 match"
    if tool_name == "bash":
        return one_line(" / ".join(lines[:2]), limit)
    return one_line(" / ".join(lines[:2]), limit)


def format_summary(
    *,
    status: str,
    turns: int,
    tool_calls: int,
    duration: float,
    tokens: str | None = None,
) -> str:
    """Build a compact completion line."""
    pieces = [f"{turns} turns", f"{tool_calls} tools", format_duration(duration)]
    if tokens:
        pieces.append(tokens)
    return f"{status}   " + " / ".join(pieces)


class StreamingRedactor:
    """Bounded cross-chunk redaction buffer for displayed assistant text.

    Normal prose streams after a small tail window. If a likely credential
    marker appears, the buffer holds from that marker so a split ``sk-`` or
    ``Bearer`` token is redacted before release. This is bounded protection,
    not a guarantee for arbitrary unknown secrets in generated prose.
    """

    _MARKERS = (
        "sk-",
        "bearer ",
        "api_key",
        "api-key",
        "access_key",
        "secret_key",
        "password",
        "passwd",
        "credential",
        "token=",
    )

    def __init__(self, holdback: int = 8) -> None:
        self.holdback = max(1, holdback)
        self._buffer = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += str(text)
        lowered = self._buffer.lower()
        split = max(0, len(self._buffer) - self.holdback)
        for marker in self._MARKERS:
            index = lowered.find(marker)
            if index >= 0:
                split = min(split, index)
                break
        if split <= 0:
            return ""
        safe = self._buffer[:split]
        self._buffer = self._buffer[split:]
        return redacted(safe)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        safe = redacted(self._buffer)
        self._buffer = ""
        return safe
