"""Day 1: bounded agent execution loop.

Concept: the runner owns the turn cycle, tool dispatch, transcript integrity,
and every terminal decision. Completion is a harness decision based on the
provider response, the finish reason, and an optional task-specific validator.

Design constraints:
* Depends only on the provider contract, never on DeepSeek directly.
* Returns a structured :class:`RunResult`; a run ending without tool calls is
  not automatically successful.
* Provider-specific behaviour stays out of the loop.
* Tool exceptions are caught at the tool boundary. There is no broad exception
  handler around the whole loop.
* Every accepted assistant tool call receives exactly one tool result before
  the next provider request, including unknown, blocked, and malformed calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from ..providers.base import (
    Provider,
    ProviderError,
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderRetryExhausted,
    ProviderStreamEvent,
    ProviderStreamError,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from .state import ExecutionState, RunResult, RunStatus, tool_call_signature
from .termination import TerminationPolicy

EMPTY_RESPONSE_NUDGE = (
    "Your previous response was empty. Reply with content or call a tool."
)
COMPLETION_REJECTED_NUDGE = (
    "Completion was rejected by the task policy because required evidence is "
    "missing. Continue working and provide the missing evidence before finishing."
)


class EventType(str, Enum):
    """Kinds of structured execution events."""

    ASSISTANT = "assistant"
    ASSISTANT_DELTA = "assistant_delta"
    MODEL = "model"
    MEMORY = "memory"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    RUN_END = "run_end"
    CONTEXT_COMPACTED = "context_compacted"


@dataclass
class RunEvent:
    """Structured event emitted by the execution loop.

    Attributes:
        type: Event kind.
        turn: Provider turn number associated with the event.
        text: Assistant text for assistant and run-end events.
        tool_calls: Assistant-requested calls for assistant events.
        tool_call: The specific call for tool events.
        result: Tool result text for tool-end events.
        status: Terminal status for run-end events.
        reason: Human-readable explanation for run-end events.
        before_tokens: Estimated tokens before context compaction.
        after_tokens: Estimated tokens after context compaction.
        summarized_messages: Original messages replaced by a summary.
        preserved_messages: Original messages kept after compaction.
        compaction_succeeded: Whether a compaction attempt succeeded.
        usage: Accumulated token usage for a run-end event, when available.
        memory_provider: Provider identifier for memory lifecycle events.
        memory_phase: Memory phase, such as retrieve or update.
        memory_count: Number of retrieved/updated items, when measured.
        memory_ok: Whether the memory operation succeeded, when known.
        provider: Selected provider name for ``MODEL`` events.
        model: Selected provider model for ``MODEL`` events.
        model_id: Catalog identifier for ``MODEL`` events.
        routing_mode: ``auto``, ``manual``, or ``default``.
        routing_reason: Credential-free routing explanation.
    """

    type: EventType
    turn: int
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call: ToolCall | None = None
    result: str | None = None
    status: RunStatus | None = None
    reason: str = ""
    before_tokens: int | None = None
    after_tokens: int | None = None
    summarized_messages: int = 0
    preserved_messages: int = 0
    compaction_succeeded: bool | None = None
    usage: TokenUsage | None = None
    memory_provider: str = ""
    memory_phase: str = ""
    memory_count: int | None = None
    memory_ok: bool | None = None
    provider: str = ""
    model: str = ""
    model_id: str = ""
    routing_mode: str = ""
    routing_reason: str = ""


def _emit(
    on_event: Callable[[RunEvent], None] | None,
    event: RunEvent,
) -> None:
    """Deliver an event to the caller's callback, if one exists."""
    if on_event is not None:
        on_event(event)


def _collect_tool_specs(tools: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the tool map and return neutral specifications for the provider.

    Raises:
        ValueError: If a tool lacks a schema, mismatches its dictionary key, or
            is not callable through ``run(**kwargs)``.
    """
    specs: list[dict[str, Any]] = []
    for name, tool in tools.items():
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, dict):
            raise ValueError(f"tool {name!r} must expose a dictionary spec")
        schema = spec.get("schema", spec)
        if not isinstance(schema, dict):
            raise ValueError(f"tool {name!r} spec['schema'] must be a dictionary")
        if schema.get("name") != name:
            raise ValueError(
                f"tool key {name!r} must match its schema name "
                f"{schema.get('name')!r}"
            )
        if not callable(getattr(tool, "run", None)):
            raise ValueError(f"tool {name!r} must expose a callable run method")
        specs.append(spec)
    return specs


def _append_assistant(
    messages: list[dict[str, Any]],
    response: ProviderResponse,
) -> None:
    """Append the neutral assistant message for this provider response."""
    message: dict[str, Any] = {"role": "assistant", "text": response.text}
    if response.continuation:
        message["provider_state"] = response.continuation
    if response.tool_calls:
        entries: list[dict[str, Any]] = []
        for call in response.tool_calls:
            entry: dict[str, Any] = {
                "id": call.id,
                "name": call.name,
                "args": dict(call.args),
            }
            if call.args_error is not None:
                entry["args_error"] = call.args_error
                if call.raw_arguments is not None:
                    entry["raw_arguments"] = call.raw_arguments
            entries.append(entry)
        message["tool_calls"] = entries
    messages.append(message)


def _append_tool_result(
    messages: list[dict[str, Any]],
    call: ToolCall,
    text: str,
) -> None:
    """Append the single tool result owed to one assistant tool call."""
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "text": text,
        }
    )


def _execute_tool(
    call: ToolCall,
    tools: Mapping[str, Any],
    before_tool: Callable[[ToolCall], str | None] | None,
) -> tuple[str, bool]:
    """Execute one tool call and return ``(result_text, failed)``.

    Tool exceptions, unknown tools, malformed arguments, and authorization
    rejections become result strings. They never escape the tool boundary.
    """
    tool = tools.get(call.name)
    if tool is None:
        return f"ERROR: unknown tool {call.name}", True
    if call.args_error is not None:
        return f"ERROR: invalid tool arguments: {call.args_error}", True

    if before_tool is not None:
        try:
            decision = before_tool(call)
        except Exception as exc:
            return (
                "BLOCKED: authorization callback failed: "
                f"{type(exc).__name__}: {exc}",
                True,
            )
        if decision is not None:
            if isinstance(decision, str) and decision.strip():
                reason = decision
            else:
                reason = "blocked by before_tool policy"
            return f"BLOCKED: {reason}", True

    try:
        value = tool.run(**call.args)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}", True
    return str(value), False


def _block_calls(
    state: ExecutionState,
    calls: list[ToolCall],
    reason: str,
    on_event: Callable[[RunEvent], None] | None,
    turn: int,
) -> None:
    """Resolve calls that will not execute with explicit blocked results."""
    for call in calls:
        text = f"BLOCKED: {reason}"
        _append_tool_result(state.messages, call, text)
        _emit(
            on_event,
            RunEvent(
                type=EventType.TOOL_END,
                turn=turn,
                tool_call=call,
                result=text,
            ),
        )


def _process_tool_batch(
    state: ExecutionState,
    calls: list[ToolCall],
    tools: Mapping[str, Any],
    policy: TerminationPolicy,
    turn: int,
    on_event: Callable[[RunEvent], None] | None,
    before_tool: Callable[[ToolCall], str | None] | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[RunStatus | None, str]:
    """Execute a tool batch in order, returning a terminal status if any.

    The batch is never left with unresolved call identifiers: calls that will
    not run receive explicit blocked results before the function returns.
    """
    for index, call in enumerate(calls):
        remaining = calls[index:]

        if should_cancel is not None and should_cancel():
            _block_calls(state, remaining, "run cancelled", on_event, turn)
            return RunStatus.CANCELLED, "run cancelled before all tool calls ran"

        if state.elapsed() >= policy.max_seconds:
            _block_calls(
                state, remaining, "time budget exhausted", on_event, turn
            )
            return RunStatus.TIMEOUT, "time budget exhausted during tool execution"

        if state.tool_calls >= policy.max_tool_calls:
            _block_calls(
                state, remaining, "tool call limit reached", on_event, turn
            )
            return RunStatus.MAX_TOOL_CALLS, (
                f"maximum tool calls ({policy.max_tool_calls}) reached"
            )

        if policy.enable_stall_detection:
            signature = tool_call_signature(call.name, call.args)
            repeats = state.record_tool_signature(signature)
            if repeats >= policy.max_repeated_calls:
                _block_calls(
                    state,
                    remaining,
                    "repeated identical tool call detected",
                    on_event,
                    turn,
                )
                return RunStatus.STALLED, (
                    "identical tool call repeated "
                    f"{repeats} times (limit {policy.max_repeated_calls})"
                )

        state.tool_calls += 1
        _emit(
            on_event,
            RunEvent(
                type=EventType.TOOL_START,
                turn=turn,
                tool_call=call,
            ),
        )
        result, failed = _execute_tool(call, tools, before_tool)
        _append_tool_result(state.messages, call, result)
        state.record_tool_result(failed)
        _emit(
            on_event,
            RunEvent(
                type=EventType.TOOL_END,
                turn=turn,
                tool_call=call,
                result=result,
            ),
        )

        if failed and state.consecutive_tool_errors >= policy.max_consecutive_tool_errors:
            rest = calls[index + 1 :]
            _block_calls(
                state,
                rest,
                "consecutive tool error limit reached",
                on_event,
                turn,
            )
            return RunStatus.TOOL_ERROR, (
                "consecutive tool failures reached "
                f"{state.consecutive_tool_errors} "
                f"(limit {policy.max_consecutive_tool_errors})"
            )

    return None, ""


def _finish(
    state: ExecutionState,
    status: RunStatus,
    reason: str,
    on_event: Callable[[RunEvent], None] | None,
) -> RunResult:
    """Build and emit the terminal result for a run."""
    result = RunResult(
        status=status,
        text=state.last_text,
        messages=state.messages,
        turns=state.turns,
        tool_calls=state.tool_calls,
        usage=state.usage,
        reason=reason,
    )
    _emit(
        on_event,
        RunEvent(
            type=EventType.RUN_END,
            turn=state.turns,
            text=result.text,
            status=status,
            reason=reason,
            usage=result.usage,
        ),
    )
    return result


def _completion_decision(
    response: ProviderResponse,
    messages: list[dict[str, Any]],
    completion_validator: (
        Callable[[ProviderResponse, list[dict[str, Any]]], bool | RunStatus] | None
    ),
) -> RunStatus | None:
    """Decide whether a candidate final answer is complete.

    Returns ``COMPLETED``, ``AWAITING_INPUT``, or ``None`` when the answer was
    rejected and the loop should continue with corrective feedback.
    """
    if completion_validator is None:
        return RunStatus.COMPLETED
    decision = completion_validator(response, messages)
    if decision is True or decision is RunStatus.COMPLETED:
        return RunStatus.COMPLETED
    if decision is RunStatus.AWAITING_INPUT:
        return RunStatus.AWAITING_INPUT
    return None


def _streamed_tool_call(
    index: int,
    accumulator: dict[str, str],
) -> ToolCall:
    """Validate one fully assembled streamed tool call."""
    call_id = accumulator.get("id", "")
    name = accumulator.get("name", "")
    raw_arguments = accumulator.get("arguments", "")
    if not call_id:
        raise ProviderInvalidResponse(
            f"streamed tool call {index} is missing an id"
        )
    if not name:
        raise ProviderInvalidResponse(
            f"streamed tool call {call_id!r} is missing a name"
        )
    args: dict[str, Any] = {}
    args_error: str | None = None
    try:
        parsed = json.loads(raw_arguments or "{}")
    except (json.JSONDecodeError, ValueError) as exc:
        args_error = f"invalid JSON arguments: {exc}"
    else:
        if isinstance(parsed, dict):
            args = parsed
        else:
            args_error = "tool arguments must decode to a JSON object"
    return ToolCall(
        id=call_id,
        name=name,
        args=args,
        args_error=args_error,
        raw_arguments=raw_arguments,
    )


def _consume_stream(
    provider: Provider,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    turn: int,
    on_event: Callable[[RunEvent], None] | None,
) -> ProviderResponse:
    """Consume one provider stream and assemble a normal response.

    Text deltas are emitted immediately. Tool calls remain local until the
    stream finishes and their JSON arguments validate. No tool is executed
    here.
    """
    stream_method = getattr(provider, "stream", None)
    if not callable(stream_method):
        return provider.complete(model, system, messages, specs)

    text_parts: list[str] = []
    accumulators: dict[int, dict[str, str]] = {}
    finish_reason = ""
    usage: TokenUsage | None = None
    continuation: dict[str, Any] | None = None

    for event in stream_method(model, system, messages, specs):
        if not isinstance(event, ProviderStreamEvent):
            raise ProviderInvalidResponse(
                "provider stream yielded a non-ProviderStreamEvent"
            )
        if event.type is StreamEventType.TEXT_DELTA:
            if not isinstance(event.text, str):
                raise ProviderInvalidResponse("stream text delta must be a string")
            if event.text:
                text_parts.append(event.text)
                _emit(
                    on_event,
                    RunEvent(
                        type=EventType.ASSISTANT_DELTA,
                        turn=turn,
                        text=event.text,
                    ),
                )
        elif event.type in (
            StreamEventType.TOOL_CALL_START,
            StreamEventType.TOOL_CALL_DELTA,
        ):
            if event.tool_index is None:
                raise ProviderInvalidResponse(
                    "streamed tool event is missing a tool index"
                )
            accumulator = accumulators.setdefault(
                event.tool_index,
                {"id": "", "name": "", "arguments": ""},
            )
            if event.tool_id:
                accumulator["id"] = event.tool_id
            if event.tool_name:
                accumulator["name"] = event.tool_name
            if event.arguments_delta:
                accumulator["arguments"] += event.arguments_delta
        elif event.type is StreamEventType.FINISH:
            finish_reason = event.finish_reason
            usage = event.usage
            continuation = event.continuation
        elif event.type is StreamEventType.ERROR:
            raise ProviderStreamError(event.error or "provider stream failed")
        else:
            raise ProviderInvalidResponse(
                f"unknown provider stream event {event.type!r}"
            )

    if not finish_reason:
        raise ProviderInvalidResponse(
            "provider stream returned no finish_reason"
        )
    tool_calls = [
        _streamed_tool_call(index, accumulators[index])
        for index in sorted(accumulators)
    ]
    return ProviderResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage or TokenUsage(0, 0),
        continuation=continuation,
    )


def run_loop(
    provider: Provider,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Mapping[str, Any],
    on_event: Callable[[RunEvent], None] | None = None,
    before_tool: Callable[[ToolCall], str | None] | None = None,
    policy: TerminationPolicy | None = None,
    completion_validator: (
        Callable[[ProviderResponse, list[dict[str, Any]]], bool | RunStatus] | None
    ) = None,
    before_turn: (
        Callable[[list[dict[str, Any]]], list[dict[str, Any]] | None] | None
    ) = None,
    should_cancel: Callable[[], bool] | None = None,
    stream: bool = False,
) -> RunResult:
    """Run the bounded agent loop until completion or an explicit stop.

    Args:
        provider: Provider implementing the neutral completion contract.
        model: Model identifier passed to the provider.
        system: System prompt passed to the provider.
        messages: Neutral transcript. Mutated in place and returned in the
            result, preserving the caller's list identity.
        tools: Mapping of tool name to an object exposing ``.spec`` and
            ``.run(**kwargs)``.
        on_event: Optional callback receiving structured :class:`RunEvent`s.
        before_tool: Authorization boundary. Return ``None`` to allow a call or
            a reason string to block it.
        policy: Termination ceilings. Defaults to :class:`TerminationPolicy`.
        completion_validator: Optional task policy called for candidate final
            answers. Return ``True`` to accept completion, ``False`` to demand
            more evidence, or ``RunStatus.AWAITING_INPUT`` to request
            clarification. With no validator, non-empty conversational answers
            are accepted.
        before_turn: Optional callback receiving the live transcript. A
            returned list replaces the transcript in place.
        should_cancel: Optional cancellation probe checked between turns and
            between tool calls.
        stream: Use ``provider.stream`` when available. If the provider has no
            streaming method, this falls back to ``complete``.

    Returns:
        A :class:`RunResult` whose status distinguishes completion from every
        interruption and failure mode.

    Raises:
        ValueError: If the tool map or a tool specification is invalid.
    """
    policy = policy or TerminationPolicy()
    specs = _collect_tool_specs(tools)
    state = ExecutionState(messages=messages)

    while True:
        if should_cancel is not None and should_cancel():
            return _finish(
                state,
                RunStatus.CANCELLED,
                "run cancelled before the next turn",
                on_event,
            )
        if state.elapsed() >= policy.max_seconds:
            return _finish(
                state,
                RunStatus.TIMEOUT,
                f"time budget of {policy.max_seconds}s exhausted",
                on_event,
            )
        if state.turns >= policy.max_turns:
            return _finish(
                state,
                RunStatus.MAX_TURNS,
                f"maximum model turns ({policy.max_turns}) reached before "
                "verified completion",
                on_event,
            )
        if state.tool_calls >= policy.max_tool_calls:
            return _finish(
                state,
                RunStatus.MAX_TOOL_CALLS,
                f"maximum tool calls ({policy.max_tool_calls}) reached",
                on_event,
            )
        if state.total_tokens >= policy.max_total_tokens:
            return _finish(
                state,
                RunStatus.TOKEN_BUDGET,
                f"token budget of {policy.max_total_tokens} exhausted",
                on_event,
            )

        if before_turn is not None:
            replacement = before_turn(messages)
            if replacement is not None:
                if not isinstance(replacement, list):
                    raise TypeError("before_turn must return a list or None")
                messages[:] = replacement

        try:
            if stream:
                response = _consume_stream(
                    provider,
                    model,
                    system,
                    messages,
                    specs,
                    state.turns + 1,
                    on_event,
                )
            else:
                response = provider.complete(model, system, messages, specs)
        except ProviderInvalidResponse as exc:
            return _finish(
                state, RunStatus.INVALID_RESPONSE, str(exc), on_event
            )
        except ProviderRetryExhausted as exc:
            return _finish(
                state,
                RunStatus.PROVIDER_ERROR,
                f"provider retries exhausted: {exc}",
                on_event,
            )
        except ProviderError as exc:
            return _finish(
                state, RunStatus.PROVIDER_ERROR, str(exc), on_event
            )

        state.turns += 1
        state.add_usage(response.usage)
        _append_assistant(state.messages, response)
        if response.text.strip():
            state.last_text = response.text
        _emit(
            on_event,
            RunEvent(
                type=EventType.ASSISTANT,
                turn=state.turns,
                text=response.text,
                tool_calls=list(response.tool_calls),
            ),
        )

        if state.total_tokens >= policy.max_total_tokens:
            _block_calls(
                state,
                response.tool_calls,
                "token budget exhausted",
                on_event,
                state.turns,
            )
            return _finish(
                state,
                RunStatus.TOKEN_BUDGET,
                f"token budget of {policy.max_total_tokens} exhausted",
                on_event,
            )

        finish_reason = response.finish_reason
        if finish_reason == "length":
            _block_calls(
                state,
                response.tool_calls,
                "output truncated by the provider",
                on_event,
                state.turns,
            )
            return _finish(
                state,
                RunStatus.INVALID_RESPONSE,
                "provider stopped at its output limit (finish_reason=length); "
                "the response is potentially truncated",
                on_event,
            )
        if finish_reason == "content_filter":
            _block_calls(
                state,
                response.tool_calls,
                "provider content filter",
                on_event,
                state.turns,
            )
            return _finish(
                state,
                RunStatus.BLOCKED,
                "provider reported content filtering (finish_reason=content_filter)",
                on_event,
            )
        if finish_reason == "insufficient_system_resource":
            _block_calls(
                state,
                response.tool_calls,
                "provider resource exhaustion",
                on_event,
                state.turns,
            )
            return _finish(
                state,
                RunStatus.PROVIDER_ERROR,
                "provider reported insufficient system resource",
                on_event,
            )
        if finish_reason not in ("stop", "tool_calls"):
            _block_calls(
                state,
                response.tool_calls,
                f"unsupported finish reason {finish_reason}",
                on_event,
                state.turns,
            )
            return _finish(
                state,
                RunStatus.INVALID_RESPONSE,
                f"provider returned unsupported finish_reason {finish_reason!r}",
                on_event,
            )

        if finish_reason == "tool_calls" and not response.tool_calls:
            return _finish(
                state,
                RunStatus.INVALID_RESPONSE,
                "provider finish_reason was tool_calls but no tool calls were "
                "returned",
                on_event,
            )

        if response.tool_calls:
            batch_status, batch_reason = _process_tool_batch(
                state,
                response.tool_calls,
                tools,
                policy,
                state.turns,
                on_event,
                before_tool,
                should_cancel,
            )
            if batch_status is not None:
                return _finish(
                    state, batch_status, batch_reason, on_event
                )
            state.consecutive_empty = 0
            continue

        if not response.text.strip():
            state.consecutive_empty += 1
            if state.consecutive_empty >= policy.max_empty_responses:
                return _finish(
                    state,
                    RunStatus.STALLED,
                    "model returned "
                    f"{state.consecutive_empty} consecutive empty responses",
                    on_event,
                )
            state.messages.append(
                {"role": "user", "text": EMPTY_RESPONSE_NUDGE}
            )
            continue

        state.consecutive_empty = 0
        decision = _completion_decision(
            response, state.messages, completion_validator
        )
        if decision is RunStatus.COMPLETED:
            reason = (
                "completion validator accepted the response"
                if completion_validator is not None
                else "model produced a final conversational answer"
            )
            return _finish(state, RunStatus.COMPLETED, reason, on_event)
        if decision is RunStatus.AWAITING_INPUT:
            return _finish(
                state,
                RunStatus.AWAITING_INPUT,
                "completion policy requested clarification from the user",
                on_event,
            )

        state.messages.append(
            {"role": "user", "text": COMPLETION_REJECTED_NUDGE}
        )
