"""Incremental, line-safe Markdown presentation for streamed assistant text."""

from __future__ import annotations

import re
from typing import Callable

from .formatting import sanitize_terminal
from .theme import Theme

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")


class MarkdownStream:
    """Render Markdown without repainting the terminal.

    Ordinary prose is emitted as it arrives. Lines that begin with Markdown
    structure are held until their newline so the style can be chosen safely.
    Fenced code blocks are held line-by-line while preserving exact text.
    Inline code spans are buffered only while the span is open.
    """

    def __init__(
        self,
        theme: Theme,
        emit: Callable[[str], None],
        *,
        max_partial: int = 240,
    ) -> None:
        self.theme = theme
        self.emit = emit
        self.max_partial = max_partial
        self._line = ""
        self._ordinary = False
        self._special = False
        self._in_code = False
        self._code_span = False
        self._inline_buffer = ""

    def _style_inline(self, text: str) -> str:
        """Style inline code spans in a completed structural line."""
        pieces = text.split("`")
        out: list[str] = []
        for index, piece in enumerate(pieces):
            if index % 2:
                out.append(self.theme.paint("code", piece))
            else:
                out.append(piece)
        return "".join(out)

    def _flush_inline(self) -> None:
        if self._code_span and self._inline_buffer:
            self.emit(sanitize_terminal(self._inline_buffer))
        self._code_span = False
        self._inline_buffer = ""

    def _render_line(self, line: str, *, newline: bool = True) -> None:
        raw = sanitize_terminal(line.rstrip("\n"))
        stripped = raw.strip()
        suffix = "\n" if newline else ""

        if stripped.startswith("```"):
            self._in_code = not self._in_code
            self.emit(self.theme.paint("dim", stripped) + suffix)
            return
        if self._in_code:
            self.emit(self.theme.paint("code", raw) + suffix)
            return

        heading = _HEADING.match(raw)
        if heading:
            self.emit(
                self.theme.paint("heading", self._style_inline(heading.group(2)))
                + suffix
            )
            return
        bullet = _BULLET.match(raw)
        if bullet:
            self.emit(
                "  "
                + self.theme.mark("bullet")
                + " "
                + self._style_inline(bullet.group(1))
                + suffix
            )
            return
        numbered = _NUMBERED.match(raw)
        if numbered:
            self.emit(
                f"  {numbered.group(1)}. "
                + self._style_inline(numbered.group(2))
                + suffix
            )
            return
        if raw.lstrip().startswith(">"):
            self.emit(
                self.theme.paint(
                    "notice", "  " + self.theme.mark("separator") + " "
                )
                + self._style_inline(raw.lstrip()[1:].lstrip())
                + suffix
            )
            return
        if _TABLE.match(raw):
            self.emit(self.theme.paint("dim", raw) + suffix)
            return
        self.emit(self._style_inline(raw) + suffix)

    def feed(self, text: str) -> None:
        """Consume streamed text and batch ordinary characters per delta."""
        pending: list[str] = []

        def flush() -> None:
            if pending:
                self.emit("".join(pending))
                pending.clear()

        for char in text:
            if char == "\n":
                if self._ordinary:
                    if self._code_span:
                        pending.append(sanitize_terminal(self._inline_buffer))
                        self._code_span = False
                        self._inline_buffer = ""
                    pending.append("\n")
                    flush()
                else:
                    flush()
                    self._render_line(self._line)
                self._line = ""
                self._ordinary = False
                self._special = False
                continue

            if self._ordinary:
                if self._code_span:
                    self._inline_buffer += char
                    if char == "`":
                        flush()
                        self.emit(
                            self.theme.paint(
                                "code", self._inline_buffer[1:-1]
                            )
                        )
                        self._code_span = False
                        self._inline_buffer = ""
                    continue
                if char == "`":
                    flush()
                    self._code_span = True
                    self._inline_buffer = "`"
                    continue
                pending.append(sanitize_terminal(char))
                continue

            if not self._line:
                self._special = self._in_code or char in "#-*>`|" or char.isdigit()
                if not self._special:
                    self._ordinary = True
                    pending.append(sanitize_terminal(char))
                    continue

            self._line += char
            if len(self._line) >= self.max_partial:
                flush()
                self._render_line(self._line, newline=True)
                self._line = ""
                self._special = False
        flush()

    def finish(self) -> None:
        """Flush the final partial line and close any visual state."""
        if self._line:
            if self._ordinary:
                self._flush_inline()
                self.emit(sanitize_terminal(self._line))
            else:
                self._render_line(self._line, newline=False)
            self._line = ""
        else:
            self._flush_inline()
        self._ordinary = False
        self._special = False
        self._in_code = False
