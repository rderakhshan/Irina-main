"""Day 1: explicit execution state and run outcomes.

Concept: a run is not just a string. The harness returns a structured outcome
that distinguishes success from interruption, failure, and incomplete work.

Design constraints:
* Internal counters live in an explicit dataclass, not ad-hoc dictionaries.
* Monotonic time is used for elapsed-time accounting.
* Only ``RunStatus.COMPLETED`` means the task was successfully completed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any

from ..providers.base import TokenUsage


class RunStatus(str, Enum):
    """Terminal classification of an execution loop run."""

    RUNNING = "running"
    COMPLETED = "completed"
    AWAITING_INPUT = "awaiting_input"
    MAX_TURNS = "max_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    TIMEOUT = "timeout"
    TOKEN_BUDGET = "token_budget"
    STALLED = "stalled"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    INVALID_RESPONSE = "invalid_response"


@dataclass
class RunResult:
    """Structured result of one execution loop run.

    Attributes:
        status: Terminal run classification.
        text: Final assistant text, or the last available assistant text for an
            incomplete run.
        messages: The transcript. This is the caller's list object when one was
            supplied.
        turns: Number of provider completion calls made.
        tool_calls: Number of model-requested tool calls accepted for
            processing, including unknown, blocked, and malformed calls.
        usage: Token usage accumulated across every provider call.
        reason: Human-readable explanation of the terminal status.
    """

    status: RunStatus
    text: str
    messages: list[dict[str, Any]]
    turns: int
    tool_calls: int
    usage: TokenUsage
    reason: str

    @property
    def completed(self) -> bool:
        """Return whether the run reached verified completion."""
        return self.status is RunStatus.COMPLETED


def tool_call_signature(name: str, args: dict[str, Any]) -> str:
    """Return a deterministic signature for repeated-call detection.

    Args:
        name: Tool name.
        args: Parsed arguments.

    Returns:
        Normalised JSON with sorted keys and compact separators.
    """
    return json.dumps(
        {"name": name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


@dataclass
class ExecutionState:
    """Mutable counters and transcript for one run.

    Attributes:
        messages: The live transcript, shared with the caller.
        turns: Provider calls made.
        tool_calls: Model-requested tool calls processed.
        usage: Accumulated token usage.
        consecutive_empty: Consecutive empty assistant responses.
        consecutive_tool_errors: Consecutive failed or blocked tool results.
        last_text: Most recent non-empty assistant text.
        last_tool_signature: Previous normalised tool-call signature.
        repeated_tool_calls: Consecutive count of that signature.
        started_at: Monotonic start timestamp.
    """

    messages: list[dict[str, Any]]
    turns: int = 0
    tool_calls: int = 0
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    consecutive_empty: int = 0
    consecutive_tool_errors: int = 0
    last_text: str = ""
    last_tool_signature: str | None = None
    repeated_tool_calls: int = 0
    started_at: float = field(default_factory=monotonic)

    @property
    def total_tokens(self) -> int:
        """Return accumulated input plus output tokens."""
        return self.usage.total_tokens

    def add_usage(self, usage: TokenUsage) -> None:
        """Accumulate usage from one provider request."""
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
        )

    def elapsed(self, now: float | None = None) -> float:
        """Return elapsed monotonic seconds since the run started."""
        current = monotonic() if now is None else now
        return current - self.started_at

    def record_tool_signature(self, signature: str) -> int:
        """Track consecutive identical tool calls and return the new count."""
        if signature == self.last_tool_signature:
            self.repeated_tool_calls += 1
        else:
            self.last_tool_signature = signature
            self.repeated_tool_calls = 1
        return self.repeated_tool_calls

    def record_tool_result(self, failed: bool) -> None:
        """Update the consecutive tool-failure counter."""
        if failed:
            self.consecutive_tool_errors += 1
        else:
            self.consecutive_tool_errors = 0
