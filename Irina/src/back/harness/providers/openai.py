"""OpenAI Responses API provider adapter.

The adapter reuses Irina's neutral provider contract and keeps all OpenAI wire
details (Responses input items, function calls, reasoning items, and SSE event
names) inside this module.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from ._http import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    iter_sse,
    open_stream,
    post_json,
    truncate,
)
from .base import (
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderStreamError,
    ProviderStreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)

API_ROOT = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_TOOL_ARGUMENT_CHARS = 200_000


def _parse_arguments(raw: Any, call_id: str) -> tuple[dict[str, Any], str | None, str | None]:
    """Parse function arguments without discarding malformed payloads."""
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        return {}, "tool arguments must be a JSON string", None
    if len(raw) > MAX_TOOL_ARGUMENT_CHARS:
        return {}, "tool arguments are too large", raw[:MAX_TOOL_ARGUMENT_CHARS]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, f"invalid JSON arguments: {exc}", raw
    if not isinstance(parsed, dict):
        return {}, "tool arguments must decode to a JSON object", raw
    return parsed, None, raw


def _tool_to_wire(schema: dict[str, Any]) -> dict[str, Any]:
    name = schema.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProviderInvalidResponse("tool is missing a name")
    description = schema.get("description", "")
    parameters = schema.get("parameters", {})
    if not isinstance(description, str):
        raise ProviderInvalidResponse(f"tool {name!r} description must be a string")
    if not isinstance(parameters, dict):
        raise ProviderInvalidResponse(f"tool {name!r} parameters must be an object")
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def _tools_to_wire(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            raise ProviderInvalidResponse("tools must contain dictionaries")
        schema = entry.get("schema", entry)
        if not isinstance(schema, dict):
            raise ProviderInvalidResponse("tool schema must be an object")
        declarations.append(_tool_to_wire(schema))
    return declarations


def _to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the neutral transcript into Responses API input items."""
    items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ProviderInvalidResponse(f"message {index} must be an object")
        role = message.get("role")
        state = message.get("provider_state")
        if role == "user":
            items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(message.get("text", ""))}],
                }
            )
        elif role == "assistant":
            if isinstance(state, dict):
                for item in state.get("reasoning_items", []) or []:
                    if isinstance(item, dict):
                        items.append(item)
            text = str(message.get("text", ""))
            if text:
                items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    raise ProviderInvalidResponse("assistant tool call must be an object")
                call_id = call.get("id")
                name = call.get("name")
                if not isinstance(call_id, str) or not call_id:
                    raise ProviderInvalidResponse("assistant tool call is missing an id")
                if not isinstance(name, str) or not name:
                    raise ProviderInvalidResponse("assistant tool call is missing a name")
                raw = call.get("raw_arguments")
                if raw is None:
                    raw = json.dumps(call.get("args") or {}, ensure_ascii=False, sort_keys=True)
                if not isinstance(raw, str):
                    raise ProviderInvalidResponse("assistant tool arguments must be a string")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": raw,
                    }
                )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderInvalidResponse(f"tool message {index} is missing tool_call_id")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(message.get("text", "")),
                }
            )
        else:
            raise ProviderInvalidResponse(
                f"message {index} has unsupported role {role!r}"
            )
    return items


def _parse_output_item(item: dict[str, Any]) -> tuple[str, list[ToolCall], list[dict[str, Any]]]:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    reasoning: list[dict[str, Any]] = []
    item_type = item.get("type")
    if item_type == "message":
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text_parts.append(str(content.get("text", "")))
    elif item_type == "function_call":
        call_id = item.get("call_id") or item.get("id")
        name = item.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise ProviderInvalidResponse("function_call is missing a call_id")
        if not isinstance(name, str) or not name.strip():
            raise ProviderInvalidResponse(f"function_call {call_id!r} is missing a name")
        args, args_error, raw = _parse_arguments(item.get("arguments"), call_id)
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                args=args,
                args_error=args_error,
                raw_arguments=raw,
            )
        )
    elif item_type == "reasoning":
        reasoning.append(item)
    return "".join(text_parts), calls, reasoning


def _finish_reason(payload: dict[str, Any], has_calls: bool) -> str:
    status = payload.get("status")
    if status == "incomplete":
        detail = payload.get("incomplete_details") or {}
        reason = str(detail.get("reason", "")).lower()
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
        if reason == "content_filter":
            return "content_filter"
        return "length"
    if status == "failed":
        error = payload.get("error") or {}
        raise ProviderInvalidResponse(
            "OpenAI response failed: " + truncate(error.get("message", "unknown error"))
        )
    if has_calls:
        return "tool_calls"
    return "stop"


def _usage(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage(0, 0)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
        input_tokens = 0
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
        output_tokens = 0
    return TokenUsage(input_tokens, output_tokens)


def _parse_response(payload: dict[str, Any]) -> ProviderResponse:
    error = payload.get("error")
    if isinstance(error, dict):
        raise ProviderInvalidResponse(
            "OpenAI returned an error: " + truncate(error.get("message", "unknown error"))
        )
    output = payload.get("output")
    if output is None:
        output = []
    if not isinstance(output, list):
        raise ProviderInvalidResponse("OpenAI output must be a list")
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    reasoning: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise ProviderInvalidResponse("OpenAI output item must be an object")
        text, item_calls, item_reasoning = _parse_output_item(item)
        if text:
            text_parts.append(text)
        calls.extend(item_calls)
        reasoning.extend(item_reasoning)
    finish = _finish_reason(payload, bool(calls))
    continuation: dict[str, Any] = {}
    if isinstance(payload.get("id"), str) and payload["id"]:
        continuation["response_id"] = payload["id"]
    if reasoning:
        continuation["reasoning_items"] = reasoning
    return ProviderResponse(
        text="".join(text_parts),
        tool_calls=calls,
        finish_reason=finish,
        usage=_usage(payload),
        continuation=continuation or None,
    )


class OpenAIProvider:
    """OpenAI Responses API adapter.

    Configuration is read from ``OPENAI_API_KEY`` (required),
    ``OPENAI_BASE_URL`` (optional), ``OPENAI_MODEL`` (optional default), with
    constructor arguments taking precedence.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.4,
        max_output_tokens: int = 8192,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before selecting the "
                "OpenAI provider."
            )
        resolved_base = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        resolved_model = model if model is not None else os.environ.get("OPENAI_MODEL")
        self._api_key = resolved_key.strip()
        self.base_url = (resolved_base or API_ROOT).rstrip("/")
        self.default_model = (resolved_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.retries = retries

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _body(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        model_name = (model or self.default_model).strip() or self.default_model
        body: dict[str, Any] = {
            "model": model_name,
            "input": _to_input(messages),
            "store": False,
            "max_output_tokens": self.max_output_tokens,
        }
        if system:
            if not isinstance(system, str):
                raise TypeError("system prompt must be a string")
            body["instructions"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        declarations = _tools_to_wire(tools)
        if declarations:
            body["tools"] = declarations
            body["tool_choice"] = "auto"
        if stream:
            body["stream"] = True
        return body

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Run one Responses API request and normalise the result."""
        body = self._body(model, system, messages, tools, stream=False)
        payload = post_json(
            f"{self.base_url}/responses",
            body,
            headers=self._headers(),
            retries=self.retries,
            timeout=self.timeout,
        )
        return _parse_response(payload)

    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Iterator[ProviderStreamEvent]:
        """Yield genuine Responses API SSE events."""
        body = self._body(model, system, messages, tools, stream=True)
        response = open_stream(
            f"{self.base_url}/responses",
            body,
            headers=self._headers(),
            retries=self.retries,
            timeout=self.timeout,
        )
        finish_reason = ""
        usage: TokenUsage | None = None
        continuation: dict[str, Any] | None = None
        saw_tool_call = False
        argument_deltas: set[int] = set()
        try:
            with response:
                for data in iter_sse(response):
                    if not data:
                        continue
                    if data.strip() == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ProviderInvalidResponse(
                            "OpenAI stream returned invalid JSON: " + truncate(exc)
                        ) from None
                    if not isinstance(payload, dict):
                        raise ProviderInvalidResponse("OpenAI stream event must be an object")
                    event_type = str(payload.get("type", ""))
                    if event_type == "error":
                        error = payload.get("error") or {}
                        raise ProviderStreamError(
                            truncate(error.get("message", "OpenAI stream error"))
                        )
                    if event_type == "response.output_text.delta":
                        delta = payload.get("delta")
                        if isinstance(delta, str) and delta:
                            yield ProviderStreamEvent(
                                type=StreamEventType.TEXT_DELTA, text=delta
                            )
                    elif event_type == "response.output_item.added":
                        item = payload.get("item") or {}
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            index = payload.get("output_index")
                            if not isinstance(index, int):
                                index = 0
                            initial_arguments = str(item.get("arguments") or "")
                            if initial_arguments:
                                argument_deltas.add(index)
                            yield ProviderStreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_index=index,
                                tool_id=str(item.get("call_id") or item.get("id") or ""),
                                tool_name=str(item.get("name") or ""),
                                arguments_delta=initial_arguments,
                            )
                            saw_tool_call = True
                    elif event_type == "response.function_call_arguments.delta":
                        delta = payload.get("delta")
                        index = payload.get("output_index")
                        if isinstance(delta, str) and delta:
                            argument_deltas.add(index if isinstance(index, int) else 0)
                            yield ProviderStreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_index=index if isinstance(index, int) else 0,
                                arguments_delta=delta,
                            )
                    elif event_type == "response.function_call_arguments.done":
                        index = payload.get("output_index")
                        resolved_index = index if isinstance(index, int) else 0
                        arguments = payload.get("arguments")
                        if (
                            resolved_index not in argument_deltas
                            and isinstance(arguments, str)
                            and arguments
                        ):
                            argument_deltas.add(resolved_index)
                            yield ProviderStreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_index=resolved_index,
                                arguments_delta=arguments,
                            )
                    elif event_type == "response.output_item.done":
                        item = payload.get("item") or {}
                        index = payload.get("output_index")
                        resolved_index = index if isinstance(index, int) else 0
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                            and resolved_index not in argument_deltas
                        ):
                            arguments = item.get("arguments")
                            if isinstance(arguments, str) and arguments:
                                argument_deltas.add(resolved_index)
                                yield ProviderStreamEvent(
                                    type=StreamEventType.TOOL_CALL_DELTA,
                                    tool_index=resolved_index,
                                    arguments_delta=arguments,
                                )
                    elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                        final = payload.get("response")
                        if not isinstance(final, dict):
                            raise ProviderInvalidResponse(
                                "OpenAI stream finish event is missing response"
                            )
                        parsed = _parse_response(final)
                        finish_reason = parsed.finish_reason
                        usage = parsed.usage
                        continuation = parsed.continuation
                        saw_tool_call = saw_tool_call or bool(parsed.tool_calls)
        except (ProviderStreamError, ProviderInvalidResponse):
            raise
        if not finish_reason:
            if saw_tool_call:
                finish_reason = "tool_calls"
            else:
                raise ProviderInvalidResponse(
                    "OpenAI stream ended without a completion event"
                )
        yield ProviderStreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=finish_reason,
            usage=usage,
            continuation=continuation,
        )


__all__ = ["API_ROOT", "DEFAULT_MODEL", "OpenAIProvider"]
