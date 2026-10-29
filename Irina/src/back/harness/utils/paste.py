"""Presentation-only collapsing for pasted terminal input.

The terminal prompt keeps the complete original text for submission and only
changes how large paste blocks are drawn. A paste becomes a compact placeholder
once it exceeds 800 characters or three lines; ordinary typing is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
COLLAPSE_CHARS = 800
COLLAPSE_LINES = 3


def paste_line_count(text: str) -> int:
    """Return the number of visible lines in pasted text."""
    if not text:
        return 0
    return len(text.splitlines()) or 1


def is_large_paste(text: str) -> bool:
    """Return whether a paste should collapse in the prompt display."""
    return len(text) > COLLAPSE_CHARS or paste_line_count(text) > COLLAPSE_LINES


def paste_placeholder(text: str) -> str:
    """Return the compact display form for one pasted block."""
    lines = paste_line_count(text)
    noun = "line" if lines == 1 else "lines"
    return f"[Pasted text · {lines} {noun}]"


@dataclass
class PromptPart:
    """One typed or pasted segment of the current prompt."""

    kind: str
    text: str
    collapsed: bool = False


@dataclass
class PromptComposer:
    """Compose the visible prompt while preserving the submitted text."""

    parts: list[PromptPart] = field(default_factory=list)

    def add_typed(self, text: str) -> str:
        """Add typed characters and return the text to echo."""
        if not text:
            return ""
        if self.parts and self.parts[-1].kind == "text":
            self.parts[-1].text += text
        else:
            self.parts.append(PromptPart("text", text))
        return text

    def add_paste(self, text: str) -> str:
        """Add a paste block and return the visible representation."""
        if not text:
            return ""
        collapsed = is_large_paste(text)
        self.parts.append(PromptPart("paste", text, collapsed))
        return paste_placeholder(text) if collapsed else text

    def backspace(self) -> int:
        """Remove the last visible unit.

        Returns:
            Approximately how many displayed characters were removed. A
            collapsed paste is removed as a whole.
        """
        if not self.parts:
            return 0
        part = self.parts[-1]
        if part.kind == "paste":
            self.parts.pop()
            return len(paste_placeholder(part.text)) if part.collapsed else len(part.text)
        if len(part.text) > 1:
            part.text = part.text[:-1]
            return 1
        self.parts.pop()
        return 1

    def expand_last_paste(self) -> str | None:
        """Reveal the most recent collapsed paste, returning its text."""
        for part in reversed(self.parts):
            if part.kind == "paste" and part.collapsed:
                part.collapsed = False
                return part.text
        return None

    def render(self) -> str:
        """Return the visible prompt text."""
        return "".join(
            paste_placeholder(part.text) if part.kind == "paste" and part.collapsed else part.text
            for part in self.parts
        )

    def submit(self) -> str:
        """Return the complete original prompt text."""
        return "".join(part.text for part in self.parts)

    @property
    def raw(self) -> str:
        """Alias for :meth:`submit` used by short input loops."""
        return self.submit()


def compose_stream(raw: str) -> PromptComposer:
    """Build a composer from complete input that may contain paste markers.

    This is the fallback for terminals whose ``input()`` returns bracketed
    markers inline. Text outside markers is treated as typed input.
    """
    composer = PromptComposer()
    index = 0
    while index < len(raw):
        start = raw.find(PASTE_START, index)
        if start < 0:
            composer.add_typed(raw[index:])
            break
        if start > index:
            composer.add_typed(raw[index:start])
        end = raw.find(PASTE_END, start + len(PASTE_START))
        if end < 0:
            composer.add_typed(raw[start + len(PASTE_START):])
            break
        composer.add_paste(raw[start + len(PASTE_START):end])
        index = end + len(PASTE_END)
    return composer


def strip_paste_markers(raw: str) -> str:
    """Remove bracketed paste markers without changing the pasted content."""
    return raw.replace(PASTE_START, "").replace(PASTE_END, "")


__all__ = [
    "COLLAPSE_CHARS",
    "COLLAPSE_LINES",
    "PASTE_END",
    "PASTE_START",
    "PromptComposer",
    "PromptPart",
    "compose_stream",
    "is_large_paste",
    "paste_line_count",
    "paste_placeholder",
    "strip_paste_markers",
]
