"""Terminal presentation utilities for Irina."""

from .fleet import FleetRenderer
from .formatting import (
    StreamingRedactor,
    format_args,
    format_duration,
    format_tokens,
    format_usage,
    pretty_path,
    preview_result,
    redacted,
    result_status,
    sanitize_terminal,
)
from .renderer import ProgressIndicator, TerminalRenderer
from .rich_markdown import RichMarkdownStream, build_markdown
from .paste import PromptComposer, compose_stream, paste_placeholder
from .theme import Symbols, Theme

__all__ = [
    "FleetRenderer",
    "ProgressIndicator",
    "PromptComposer",
    "RichMarkdownStream",
    "StreamingRedactor",
    "Symbols",
    "TerminalRenderer",
    "Theme",
    "build_markdown",
    "compose_stream",
    "format_args",
    "format_duration",
    "format_tokens",
    "format_usage",
    "paste_placeholder",
    "pretty_path",
    "preview_result",
    "redacted",
    "result_status",
    "sanitize_terminal",
]
