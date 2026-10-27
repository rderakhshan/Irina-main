"""Day 1: provider contract for the Irina harness.

Concept: the execution loop depends on a narrow provider interface rather than
a vendor SDK. A provider converts neutral transcript messages into its wire
format and normalises the reply into explicit dataclasses.

Design constraints:
* Standard library only, synchronous, and no vendor-specific fields (such as
  Gemini thought signatures) in the neutral model.
* Malformed tool arguments are kept as recoverable data so the loop can emit a
  tool error instead of crashing the run.
* Typed provider exceptions let the loop distinguish invalid responses,
  permanent HTTP failures, and exhausted retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by a provider for one request.

    Attributes:
        input_tokens: Prompt tokens consumed by the request.
        output_tokens: Completion tokens produced by the request.
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Return the combined input and output token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    ``args`` always contains a dictionary so callers can inspect it safely. If
    the provider supplied malformed or non-object JSON, ``args`` is empty,
    ``args_error`` explains the validation failure, and the loop must not run
    the tool. ``raw_arguments`` preserves the original payload so a transcript
    can echo it back without changing the model's own output.

    Attributes:
        id: Provider-assigned call identifier. It must round-trip unchanged.
        name: Requested tool name.
        args: Parsed keyword arguments, or an empty dictionary on parse failure.
        args_error: Human-readable validation failure, if parsing failed.
        raw_arguments: Original JSON string from the provider, when available.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    args_error: str | None = None
    raw_arguments: str | None = None


@dataclass
class ProviderResponse:
    """Normalised result of one model completion request.

    Attributes:
        text: Assistant text. Empty when the model only requested tools.
        tool_calls: Tool invocations requested by the model, in order.
        finish_reason: Provider finish reason, preserved verbatim.
        usage: Token usage reported for this request.
    """

    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: TokenUsage


class StreamEventType(str, Enum):
    """Provider-neutral streaming event kinds."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    FINISH = "finish"
    ERROR = "stream_error"


@dataclass(frozen=True)
class ProviderStreamEvent:
    """One normalised streaming event from a provider.

    Providers may split text, tool names, IDs, and JSON arguments across any
    number of events. Consumers must accumulate by ``tool_index`` and must not
    execute a tool until a final :class:`ProviderResponse` has been assembled.

    Attributes:
        type: Event kind.
        text: Text delta for ``TEXT_DELTA``.
        tool_index: Stable index for a tool call within one response.
        tool_id: Provider-assigned call ID, when known.
        tool_name: Function name, when known.
        arguments_delta: Partial JSON argument text for a tool call.
        finish_reason: Provider finish reason for ``FINISH``.
        usage: Provider-reported usage for ``FINISH``, or ``None`` when absent.
        error: Credential-free error description for ``ERROR``.
    """

    type: StreamEventType
    text: str = ""
    tool_index: int | None = None
    tool_id: str = ""
    tool_name: str = ""
    arguments_delta: str = ""
    finish_reason: str = ""
    usage: TokenUsage | None = None
    error: str = ""


class StreamingProvider(Protocol):
    """Optional streaming extension for a provider."""

    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Iterator[ProviderStreamEvent]:
        """Yield provider-neutral stream events in wire order."""
        ...


class ProviderError(Exception):
    """Base class for classified provider failures."""


class ProviderHTTPError(ProviderError):
    """A permanent HTTP failure that must not be retried automatically.

    Attributes:
        status: HTTP status code.
        description: Truncated, credential-free response description.
    """

    def __init__(self, status: int, description: str) -> None:
        self.status = status
        self.description = description
        super().__init__(f"provider returned HTTP {status}: {description}")


class ProviderRetryExhausted(ProviderError):
    """A transient provider failure that survived every retry.

    Distinct from :class:`ProviderHTTPError` so callers can tell a retry budget
    exhaustion apart from a permanent failure such as authentication.

    Attributes:
        status: Last HTTP status, or ``None`` for a network failure.
        description: Truncated, credential-free failure description.
        attempts: Number of HTTP attempts made, when known.
    """

    def __init__(
        self,
        status: int | None,
        description: str,
        attempts: int | None = None,
    ) -> None:
        self.status = status
        self.description = description
        self.attempts = attempts
        prefix = f"after {attempts} attempts " if attempts else ""
        super().__init__(
            f"provider retries exhausted {prefix}(status={status}): {description}"
        )


class ProviderInvalidResponse(ProviderError):
    """A response that was structurally invalid or unsupported.

    Examples include missing choices, missing tool-call identifiers, missing
    usage data, and unknown finish reasons.
    """


class ProviderStreamError(ProviderError):
    """A streaming connection failed or ended before a valid finish."""


class Provider(Protocol):
    """Synchronous completion interface consumed by the execution loop."""

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Run one model completion.

        Args:
            model: Model identifier understood by the provider.
            system: System prompt for this request.
            messages: Neutral transcript messages (user, assistant, tool).
            tools: Neutral tool specifications, each shaped as
                ``{"schema": {"name": ..., "description": ..., "parameters": ...}}``.

        Returns:
            A normalised :class:`ProviderResponse`.

        Raises:
            ProviderError: If the request fails after classification and retry.
        """
        ...
