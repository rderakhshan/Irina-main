"""Terminal presentation for EvolveLab/MemEvolve loggers.

Memory providers either log through Python's standard ``logging`` module or
print directly. By default Irina keeps routine messages out of the terminal;
``--verbose`` routes them through the existing :class:`Theme` with level-based
colors. No upstream provider algorithm or message is changed.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Iterator, TextIO

from .theme import Theme

MEMORY_LOGGER_ROOTS = ("EvolveLab", "MemEvolve", "FlashOAgents", "base_agent")

# Loggers created by the vendored providers. Pre-creating them lets Irina
# install its own handler before a provider's __init__ decides to add one.
MEMORY_LOGGER_NAMES = MEMORY_LOGGER_ROOTS + (
    "EvolveLab.providers.lightweight_memory_provider",
    "EvolveLab.providers.lightweight_memory_provider.Lightweight",
    "EvolveLab.providers.skillweaver_provider",
    "EvolveLab.providers.agent_workflow_memory_provider",
    "EvolveLab.providers.evolver_memory_provider",
)

_LEVEL_ROLES = {
    "DEBUG": "dim",
    "INFO": "tool",
    "WARNING": "warning",
    "ERROR": "failure",
    "CRITICAL": "failure",
}

_SUCCESS_HINTS = (
    "success",
    "succeeded",
    "complete",
    "completed",
    "saved",
    "absorbed",
    "provided",
    "loaded",
    "added",
    "initialized",
)
_ERROR_HINTS = (
    "error",
    "failed",
    "failure",
    "traceback",
    "exception",
    "critical",
)
_WARNING_HINTS = ("warning", "warn", "⚠")

_ACTIVE: dict[str, Any] | None = None
_CAPTURE_LOCK = threading.RLock()


def _display_provider(name: str) -> str:
    cleaned = name.replace("_provider", "").replace("_", " ").strip()
    return cleaned.title()[:28]


def _provider_name(logger_name: str) -> str:
    parts = [
        part
        for part in logger_name.split(".")
        if part not in {"EvolveLab", "MemEvolve", "FlashOAgents", "providers"}
    ]
    if not parts:
        return ""
    name = parts[-1]
    if name.endswith("_provider"):
        name = name[: -len("_provider")]
    if "_" in name:
        name = name.replace("_", " ").title()
    return name[:28]


def _is_success(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _SUCCESS_HINTS)


def _classify(message: str) -> str:
    lowered = message.lower()
    if any(hint in lowered for hint in _ERROR_HINTS):
        return "ERROR"
    if any(hint in lowered for hint in _WARNING_HINTS):
        return "WARNING"
    if _is_success(lowered):
        return "SUCCESS"
    if "debug" in lowered:
        return "DEBUG"
    return "INFO"


def format_memory_line(theme: Theme, provider: str, level: str, message: str) -> str:
    """Format one memory line with semantic colors from the active theme."""
    display_level = "INFO" if level == "SUCCESS" else level
    role = _LEVEL_ROLES.get(display_level, "dim")
    if level == "SUCCESS" or (level == "INFO" and _is_success(message)):
        role = "success"
    pieces = []
    if provider:
        pieces.append(theme.paint("tool", provider))
    pieces.append(theme.paint(role, display_level))
    pieces.append(theme.paint(role, message))
    return "  " + "  ".join(pieces)


class MemoryLogFormatter(logging.Formatter):
    """Format one memory log record with Irina's semantic theme."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme

    def format(self, record: logging.LogRecord) -> str:
        return format_memory_line(
            self.theme,
            _provider_name(record.name),
            record.levelname,
            record.getMessage(),
        )


def _write_line(
    *,
    theme: Theme,
    stream: TextIO | None,
    on_emit: Callable[[], Any] | None,
    provider: str,
    level: str,
    message: str,
) -> None:
    try:
        if on_emit is not None:
            on_emit()
        target = stream or sys.stderr
        target.write(format_memory_line(theme, provider, level, message) + "\n")
        target.flush()
    except Exception:
        pass


class MemoryLogHandler(logging.Handler):
    """Write whole lines to a stream without fighting the live renderer."""

    def __init__(
        self,
        theme: Theme,
        stream: TextIO | None = None,
        on_emit: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.setFormatter(MemoryLogFormatter(theme))
        self.theme = theme
        self.stream = stream or sys.stderr
        self.on_emit = on_emit

    def emit(self, record: logging.LogRecord) -> None:
        _write_line(
            theme=self.theme,
            stream=self.stream,
            on_emit=self.on_emit,
            provider=_provider_name(record.name),
            level=record.levelname,
            message=record.getMessage(),
        )


def _memory_loggers() -> list[logging.Logger]:
    logger_dict = logging.Logger.manager.loggerDict
    names = set(MEMORY_LOGGER_NAMES)
    for name, value in logger_dict.items():
        if not isinstance(value, logging.Logger):
            continue
        if any(name == root or name.startswith(root + ".") for root in MEMORY_LOGGER_ROOTS):
            names.add(name)
    return [logging.getLogger(name) for name in sorted(names)]


def configure_memory_logging(
    *,
    verbose: bool,
    theme: Theme,
    stream: TextIO | None = None,
    on_emit: Callable[[], Any] | None = None,
) -> None:
    """Route upstream memory logs according to ``verbose``.

    Default mode suppresses routine INFO/DEBUG/WARNING records but still shows
    ERROR/CRITICAL. Verbose mode shows every level, colored by the active theme
    (which already honors ``--no-color`` and ``--plain``).
    """
    global _ACTIVE
    target_stream = stream or sys.stderr
    _ACTIVE = {
        "verbose": bool(verbose),
        "theme": theme,
        "stream": target_stream,
        "on_emit": on_emit,
    }
    handler = MemoryLogHandler(theme, stream=target_stream, on_emit=on_emit)
    handler.setLevel(logging.DEBUG if verbose else logging.ERROR)
    for logger in _memory_loggers():
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
            try:
                existing.close()
            except Exception:
                pass
        logger.propagate = False
        logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
        logger.addHandler(handler)


def reset_memory_logging() -> None:
    """Clear the active presentation policy (used by tests)."""
    global _ACTIVE
    _ACTIVE = None


@contextmanager
def capture_memory_output(*, provider: str = "") -> Iterator[None]:
    """Capture provider ``print`` output for the active presentation policy.

    No-op until ``configure_memory_logging`` has run. Provider calls are short
    and synchronous, so the process-wide stream redirect is bounded to one
    memory operation.
    """
    policy = _ACTIVE
    if policy is None:
        yield
        return
    buffer = io.StringIO()
    with _CAPTURE_LOCK:
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                yield
        finally:
            text = buffer.getvalue()
            name = _display_provider(provider)
            for line in text.splitlines():
                message = line.strip()
                if not message:
                    continue
                level = _classify(message)
                if not policy["verbose"] and level != "ERROR":
                    continue
                _write_line(
                    theme=policy["theme"],
                    stream=policy["stream"],
                    on_emit=policy["on_emit"],
                    provider=name,
                    level=level,
                    message=message,
                )
