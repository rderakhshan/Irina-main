"""Rich-powered Markdown presentation for streamed assistant text.

Assistant responses are rendered as Markdown documents, not decorated line by
line. While streaming, the accumulated text is redrawn through ``rich.live.Live``
at a bounded rate so tables, fenced code, and lists never appear half-formed.
A document taller than the terminal switches to append-only block rendering:
a live region that scrolls past the screen can never be erased, so repainting it
would leave duplicate copies in scrollback. Long responses also switch on size.
The final render is printed once, so the transcript never contains duplicates.

The original assistant text is never modified: it is accumulated verbatim here
and rendered only for display.
"""

from __future__ import annotations

import re
import time
from typing import TextIO

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

# ``ansi_dark`` uses the terminal's basic palette instead of a 16M-colour theme,
# which keeps code readable in PowerShell and other limited terminals.
CODE_THEME = "ansi_dark"

# Redraw at most ~8 times a second while streaming. Deliberately slow enough to
# avoid flicker and full-document reparses on every token.
REDRAW_INTERVAL = 0.12

# Ignore tiny growth between redraws so a fast token stream does not force a
# redraw for every few characters.
MIN_GROWTH = 32

# Above this many characters a full-document redraw costs more than it is worth,
# so the streamer finalises the stable prefix once and appends block by block.
MAX_LIVE_CHARS = 6000

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def build_markdown(text: str) -> Markdown:
    """Return one Rich Markdown renderable for an assistant response."""
    return Markdown(
        text,
        code_theme=CODE_THEME,
        hyperlinks=True,
        justify=None,
    )


def build_console(
    stream: TextIO,
    *,
    terminal: bool,
    color: bool,
    width: int | None = None,
    height: int | None = None,
) -> Console:
    """Create the renderer's single Console bound to an existing stream."""
    return Console(
        file=stream,
        force_terminal=terminal,
        no_color=not color,
        width=width,
        height=height,
        legacy_windows=False,
        highlight=False,
        emoji=False,
        soft_wrap=False,
    )


def stable_boundary(text: str) -> int:
    """Return the end offset of the last complete Markdown block.

    A block ends at a blank line that is outside a fenced code block, so an
    incomplete table, list, or fence is never finalised early.
    """
    boundary = 0
    fence: str | None = None
    index = 0
    length = len(text)
    while index < length:
        newline = text.find("\n", index)
        if newline < 0:
            break
        line = text[index:newline]
        match = _FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
        elif fence is None and not line.strip():
            boundary = newline + 1
        index = newline + 1
    return boundary


class RichMarkdownStream:
    """Accumulate assistant text and present it as Markdown.

    When ``live`` is false (plain output, tests, redirected streams) the
    document is rendered once at ``finish``. When ``live`` is true the visible
    region is repainted from the accumulated buffer, then replaced by one clean
    final render. Past ``max_live_chars`` the streamer stops repainting and
    appends each completed block once instead.
    """

    def __init__(
        self,
        console: Console,
        *,
        live: bool = False,
        redraw_interval: float = REDRAW_INTERVAL,
        min_growth: int = MIN_GROWTH,
        max_live_chars: int = MAX_LIVE_CHARS,
    ) -> None:
        self.console = console
        self.live = live
        self.redraw_interval = max(0.0, redraw_interval)
        self.min_growth = max(0, min_growth)
        self.max_live_chars = max(0, max_live_chars)
        self._text = ""
        self._live: Live | None = None
        self._last_draw = 0.0
        self._drawn_length = 0
        self._append_mode = False
        self._rendered = 0
        self._finished = False

    @property
    def text(self) -> str:
        """Return the exact accumulated assistant text."""
        return self._text

    @property
    def is_live(self) -> bool:
        """Whether a live repaint region is currently active."""
        return self._live is not None

    @property
    def append_mode(self) -> bool:
        """Whether the streamer has switched to append-only block rendering."""
        return self._append_mode

    def _estimated_lines(self) -> int:
        """Approximate how many terminal lines the document renders to."""
        width = max(20, int(self.console.width or 80))
        lines = self._text.count("\n") + 1
        wrapped = sum(
            max(0, (len(line) - 1) // width)
            for line in self._text.split("\n")
        )
        return lines + wrapped + 2

    def _live_height_limit(self) -> int:
        """Lines the live region may occupy before it can no longer be erased.

        A live region taller than the screen scrolls lines into scrollback,
        where Rich can never clear them again; each repaint would leave a
        permanent copy. Keep a couple of lines for the prompt.
        """
        try:
            height = int(self.console.height or 0)
        except Exception:
            height = 0
        return max(0, height - 2)

    def _too_tall_for_live(self) -> bool:
        limit = self._live_height_limit()
        return bool(limit) and self._estimated_lines() > limit

    def feed(self, text: str) -> None:
        """Append a streamed delta and present whatever is safe to show."""
        if not text or self._finished:
            return
        self._text += text
        if not self.live or not self._text.strip():
            return
        if self._append_mode:
            self._append_stable()
            return
        if (
            len(self._text) > self.max_live_chars
            or self._too_tall_for_live()
        ):
            self._switch_to_append()
            return
        now = time.monotonic()
        if self._live is not None:
            if now - self._last_draw < self.redraw_interval:
                return
            if len(self._text) - self._drawn_length < self.min_growth:
                return
        self._draw()

    # -- live repaint -------------------------------------------------
    def _draw(self) -> None:
        renderable = build_markdown(self._text)
        try:
            if self._live is None:
                live = Live(
                    renderable,
                    console=self.console,
                    auto_refresh=False,
                    transient=True,
                    # Never let a repaint exceed the screen: lines that scroll
                    # into scrollback cannot be erased, so they would appear
                    # as duplicated copies of the response.
                    vertical_overflow="ellipsis",
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live = live
                live.start(refresh=True)
            else:
                self._live.update(renderable, refresh=True)
        except (OSError, ValueError):
            # A closed or redirected stream must never break the agent loop.
            self._live = None
            return
        self._last_draw = time.monotonic()
        self._drawn_length = len(self._text)

    # -- append-only blocks -------------------------------------------
    def _stop_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except (OSError, ValueError):
                pass

    def _switch_to_append(self) -> None:
        self._stop_live()
        self._append_mode = True
        boundary = stable_boundary(self._text)
        if boundary > self._rendered:
            self._print_slice(self._rendered, boundary)
            self._rendered = boundary

    def _append_stable(self) -> None:
        boundary = stable_boundary(self._text)
        if boundary <= self._rendered:
            return
        self._print_slice(self._rendered, boundary)
        self._rendered = boundary

    def _print_slice(self, start: int, end: int) -> None:
        chunk = self._text[start:end]
        if not chunk.strip():
            return
        try:
            self.console.print(build_markdown(chunk))
        except (OSError, ValueError):
            pass

    # -- lifecycle -----------------------------------------------------
    def finish(self) -> None:
        """Close any live region and print the finished document once."""
        if self._finished:
            return
        self._finished = True
        self._stop_live()
        if not self._text.strip():
            return
        if self._append_mode:
            self._print_slice(self._rendered, len(self._text))
            self._rendered = len(self._text)
            return
        self._print_slice(0, len(self._text))


__all__ = [
    "CODE_THEME",
    "MAX_LIVE_CHARS",
    "MIN_GROWTH",
    "REDRAW_INTERVAL",
    "RichMarkdownStream",
    "build_console",
    "build_markdown",
    "stable_boundary",
]
