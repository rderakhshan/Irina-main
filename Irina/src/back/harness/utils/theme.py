"""Semantic terminal theme and capability detection."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TextIO

RESET = "\033[0m"
STYLES = {
    "identity": "\033[1;36m",
    "user": "\033[1m",
    "assistant": "",
    "tool": "\033[36m",
    "running": "\033[33m",
    "success": "\033[32m",
    "blocked": "\033[33m",
    "failure": "\033[31m",
    "dim": "\033[2m",
    "notice": "\033[35m",
    "heading": "\033[1;36m",
    "code": "\033[36m",
    "link": "\033[4;36m",
    "warning": "\033[33m",
}


@dataclass(frozen=True)
class Symbols:
    """Unicode and ASCII fallbacks for semantic UI marks."""

    identity: str = "◆"
    prompt: str = "❯"
    running: str = "◌"
    success: str = "✓"
    failure: str = "✗"
    blocked: str = "!"
    notice: str = "◇"
    resume: str = "↻"
    bullet: str = "•"
    separator: str = "─"
    dot: str = "·"

    @classmethod
    def ascii(cls) -> "Symbols":
        return cls(
            identity="*",
            prompt=">",
            running="o",
            success="+",
            failure="x",
            blocked="!",
            notice="~",
            resume="~",
            bullet="-",
            separator="-",
            dot="-",
        )


@dataclass(frozen=True)
class Theme:
    """Centralized styling. No other UI module should emit raw ANSI."""

    color: bool = True
    unicode: bool = True
    symbols: Symbols = Symbols()

    def paint(self, role: str, text: str) -> str:
        """Apply one semantic style if color is enabled."""
        if not self.color or not text:
            return text
        code = STYLES.get(role, "")
        if not code:
            return text
        return f"{code}{text}{RESET}"

    def mark(self, name: str) -> str:
        """Return a symbol by semantic name."""
        return getattr(self.symbols, name)

    @classmethod
    def detect(
        cls,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        plain: bool = False,
    ) -> "Theme":
        """Detect capabilities without assuming a specific terminal."""
        target = stream if stream is not None else sys.stdout
        is_tty = False
        encoding = "utf-8"
        try:
            is_tty = bool(target.isatty())
            encoding = str(getattr(target, "encoding", "utf-8") or "utf-8")
        except (AttributeError, OSError):
            pass

        if plain:
            resolved_color = False
        elif color is not None:
            resolved_color = color
        else:
            no_color = os.environ.get("NO_COLOR")
            force_color = os.environ.get("FORCE_COLOR")
            dumb = os.environ.get("TERM", "").lower() == "dumb"
            resolved_color = (
                is_tty
                and not no_color
                and not dumb
                and (force_color is not None or is_tty)
            )

        use_unicode = not plain
        if use_unicode:
            try:
                "◆❯◌✓✗─•".encode(encoding)
            except (LookupError, UnicodeEncodeError):
                use_unicode = False
        symbols = Symbols() if use_unicode else Symbols.ascii()
        return cls(color=resolved_color, unicode=use_unicode, symbols=symbols)

    @property
    def interactive(self) -> bool:
        """Whether a caller should use cursor-animated progress."""
        return self.color
