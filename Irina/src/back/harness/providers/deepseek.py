"""Day 1: DeepSeek Chat Completions provider.

Concept: a concrete provider adapter translates the harness's neutral
transcript into DeepSeek's OpenAI-compatible wire format and parses the reply
back into explicit dataclasses. It also owns HTTP retry classification.

Design constraints:
* Standard library only; ``urllib.request`` performs HTTP.
* Credentials come from the environment and are never logged or included in
  exception messages.
* Transient failures are retried with bounded exponential backoff and jitter;
  authentication and malformed requests are never retried.
* Tool-call identifiers round-trip exactly. Malformed arguments are represented
  as recoverable validation failures instead of crashing or silently becoming
  ``{}`` executions.
"""

from __future__ import annotations

import json
import os
import random
import time
import http.client
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterator

from .base import (
    ProviderHTTPError,
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderRetryExhausted,
    ProviderStreamError,
    ProviderStreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)

API_ROOT = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 5
BASE_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 30.0
MAX_ERROR_DESCRIPTION_CHARS = 500

KNOWN_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "insufficient_system_resource",
    }
)


def _truncate(text: str, limit: int = MAX_ERROR_DESCRIPTION_CHARS) -> str:
    """Return ``text`` collapsed to a bounded single-line description."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _http_error_description(error: urllib.error.HTTPError) -> str:
    """Read a credential-free description from an HTTP error response."""
    try:
        raw = error.read()
    except Exception:
        raw = b""
    if isinstance(raw, bytes):
        body = raw.decode("utf-8", "replace")
    else:
        body = str(raw)
    if not body:
        body = str(error.reason or "")
    return _truncate(body)


def _header_value(headers: Any, name: str) -> str | None:
    """Read a header case-insensitively from urllib or plain mappings."""
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is not None:
        return str(value)
    items = getattr(headers, "items", None)
    if items is not None:
        lowered = name.lower()
        for key, candidate in items():
            if str(key).lower() == lowered:
                return str(candidate)
    return None


def _parse_retry_after(value: str | None) -> float | None:
    """Convert a ``Retry-After`` value into a bounded delay in seconds."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if seconds < 0:
        return 0.0
    return min(seconds, MAX_RETRY_DELAY_SECONDS)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Compute bounded exponential backoff with jitter, honouring Retry-After."""
    honoured = _parse_retry_after(retry_after)
    if honoured is not None:
        return honoured
    ceiling = min(MAX_RETRY_DELAY_SECONDS, BASE_RETRY_DELAY_SECONDS * (2**attempt))
    return random.uniform(0.0, ceiling)


def _decode_json(raw: bytes) -> dict[str, Any]:
    """Decode a JSON object body or raise a classified provider error."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderInvalidResponse(
            f"provider returned invalid JSON: {_truncate(str(exc))}"
        ) from None
    if not isinstance(payload, dict):
        raise ProviderInvalidResponse("provider response must be a JSON object")
    return payload


def urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """Open one HTTP request through ``urllib.request``.

    This small indirection keeps the network boundary mockable at the module
    level while still delegating to the standard library.
    """
    return urllib.request.urlopen(request, timeout=timeout)


def _post(
    url: str,
    body: dict[str, Any],
    retries: int = DEFAULT_RETRIES,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST a JSON body and return the decoded JSON object.

    ``retries`` counts retry attempts after the initial request, so at most
    ``retries + 1`` HTTP requests are made.

    Args:
        url: Absolute endpoint URL.
        body: JSON-serialisable request body.
        retries: Maximum retries after the initial attempt.
        headers: Extra request headers, such as authorization.
        timeout: Per-request socket timeout in seconds.

    Returns:
        The decoded JSON object.

    Raises:
        ProviderHTTPError: For a permanent HTTP failure.
        ProviderRetryExhausted: When a transient failure survives all retries.
        ProviderInvalidResponse: When a successful response is not valid JSON.
    """
    retry_budget = max(0, int(retries))
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    attempts = retry_budget + 1
    last_status: int | None = None
    last_description = "no response"

    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=request_headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return _decode_json(raw)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            description = _http_error_description(exc)
            last_status = status
            last_description = description
            if status in RETRYABLE_STATUS_CODES:
                if attempt < retry_budget:
                    retry_after = _header_value(exc.headers, "Retry-After")
                    time.sleep(_retry_delay(attempt, retry_after))
                    continue
                raise ProviderRetryExhausted(
                    status, description, attempts=attempts
                ) from None
            raise ProviderHTTPError(status, description) from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            description = _truncate(f"{type(reason).__name__}: {reason}")
            last_status = None
            last_description = description
            if attempt < retry_budget:
                time.sleep(_retry_delay(attempt, None))
                continue
            raise ProviderRetryExhausted(
                None, description, attempts=attempts
            ) from None

    raise ProviderRetryExhausted(last_status, last_description, attempts=attempts)


def _encode_tool_call(tool_call: Any) -> dict[str, Any]:
    """Encode one neutral assistant tool call as a DeepSeek function call."""
    if isinstance(tool_call, ToolCall):
        call_id = tool_call.id
        name = tool_call.name
        args = tool_call.args
        raw_arguments = tool_call.raw_arguments
    elif isinstance(tool_call, dict):
        call_id = tool_call.get("id", "")
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        raw_arguments = tool_call.get("raw_arguments")
    else:
        raise ProviderInvalidResponse(
            "assistant tool call must be a ToolCall or a dictionary"
        )

    if not isinstance(call_id, str) or not call_id:
        raise ProviderInvalidResponse("assistant tool call is missing an id")
    if not isinstance(name, str) or not name:
        raise ProviderInvalidResponse("assistant tool call is missing a name")
    if not isinstance(args, dict):
        raise ProviderInvalidResponse(
            f"assistant tool call {call_id!r} has non-object arguments"
        )
    if raw_arguments is None:
        raw_arguments = json.dumps(args, ensure_ascii=False, sort_keys=True)
    if not isinstance(raw_arguments, str):
        raise ProviderInvalidResponse(
            f"assistant tool call {call_id!r} has non-string raw arguments"
        )
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw_arguments},
    }


def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral transcript messages to DeepSeek wire messages.

    Args:
        messages: Neutral messages with ``role`` set to ``user``, ``assistant``,
            or ``tool``. Tool messages must carry ``tool_call_id``.

    Returns:
        OpenAI-compatible message dictionaries.

    Raises:
        ProviderInvalidResponse: If a message has an unsupported role or shape.
    """
    wire: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ProviderInvalidResponse(f"message {index} must be a dictionary")
        role = message.get("role")
        if role == "user":
            wire.append(
                {"role": "user", "content": str(message.get("text", ""))}
            )
        elif role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": str(message.get("text", "")),
            }
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ProviderInvalidResponse(
                        f"assistant message {index} has non-list tool_calls"
                    )
                entry["tool_calls"] = [
                    _encode_tool_call(tool_call) for tool_call in tool_calls
                ]
            wire.append(entry)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderInvalidResponse(
                    f"tool message {index} is missing tool_call_id"
                )
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(message.get("text", "")),
                }
            )
        else:
            raise ProviderInvalidResponse(
                f"message {index} has unsupported role {role!r}"
            )
    return wire


def _tools_to_wire(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral tool specifications into DeepSeek function declarations."""
    if not tools:
        return []
    if not isinstance(tools, list):
        raise ProviderInvalidResponse("tools must be a list")
    declarations: list[dict[str, Any]] = []
    for index, entry in enumerate(tools):
        if not isinstance(entry, dict):
            raise ProviderInvalidResponse(f"tool {index} must be a dictionary")
        schema = entry.get("schema", entry)
        if not isinstance(schema, dict):
            raise ProviderInvalidResponse(f"tool {index} schema must be a dictionary")
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProviderInvalidResponse(f"tool {index} is missing a name")
        description = schema.get("description", "")
        parameters = schema.get("parameters", {})
        if not isinstance(description, str):
            raise ProviderInvalidResponse(
                f"tool {name!r} description must be a string"
            )
        if not isinstance(parameters, dict):
            raise ProviderInvalidResponse(
                f"tool {name!r} parameters must be a dictionary"
            )
        declarations.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return declarations


def _parse_tool_call(raw: Any, index: int) -> ToolCall:
    """Parse one wire tool call, preserving malformed arguments as data."""
    if not isinstance(raw, dict):
        raise ProviderInvalidResponse(f"tool call {index} must be a dictionary")
    call_id = raw.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ProviderInvalidResponse(f"tool call {index} is missing an id")
    function = raw.get("function")
    if not isinstance(function, dict):
        raise ProviderInvalidResponse(f"tool call {call_id!r} is missing a function")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProviderInvalidResponse(f"tool call {call_id!r} is missing a name")

    raw_arguments = function.get("arguments")
    args: dict[str, Any] = {}
    args_error: str | None = None
    preserved_arguments: str | None = None
    if raw_arguments is None:
        raw_arguments = ""
    if not isinstance(raw_arguments, str):
        args_error = "tool arguments must be a JSON string"
    else:
        preserved_arguments = raw_arguments
        try:
            parsed = json.loads(raw_arguments)
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
        raw_arguments=preserved_arguments,
    )


def _parse_usage(value: Any) -> TokenUsage | None:
    """Parse usage without inventing missing counters."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProviderInvalidResponse("provider stream usage must be an object")
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return None
    for label, number in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
    ):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ProviderInvalidResponse(
                f"provider stream usage {label} must be a non-negative integer"
            )
    return TokenUsage(
        input_tokens=prompt_tokens or 0,
        output_tokens=completion_tokens or 0,
    )


def _iter_sse(response: Any) -> Iterator[str]:
    """Yield SSE ``data`` payloads, including multi-line data fields."""
    data: list[str] = []
    for raw in response:
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", "replace")
        else:
            text = str(raw)
        for line in text.splitlines():
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def _parse_response(payload: dict[str, Any]) -> ProviderResponse:
    """Parse a DeepSeek chat completion payload into a provider response.

    Raises:
        ProviderInvalidResponse: For missing choices, malformed tool calls,
            missing usage, or unsupported finish reasons.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderInvalidResponse("provider response is missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderInvalidResponse("provider choice must be a dictionary")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ProviderInvalidResponse("provider choice is missing a message")

    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        raise ProviderInvalidResponse("provider message content must be a string")

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise ProviderInvalidResponse("provider message tool_calls must be a list")
    tool_calls = [
        _parse_tool_call(raw, index) for index, raw in enumerate(raw_tool_calls)
    ]

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ProviderInvalidResponse("provider response is missing finish_reason")
    if finish_reason not in KNOWN_FINISH_REASONS:
        raise ProviderInvalidResponse(
            f"provider returned unsupported finish_reason {finish_reason!r}"
        )

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ProviderInvalidResponse("provider response is missing usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    for label, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProviderInvalidResponse(f"provider usage {label} must be a non-negative integer")

    return ProviderResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        ),
    )


class DeepSeekProvider:
    """DeepSeek Chat Completions adapter.

    Configuration is read from ``DEEPSEEK_API_KEY`` (required),
    ``IRINA_MODEL`` (optional model default), and ``DEEPSEEK_BASE_URL``
    (optional endpoint override), with constructor arguments taking precedence.

    Args:
        api_key: API key override. Defaults to ``DEEPSEEK_API_KEY``.
        base_url: Endpoint root override. Defaults to ``DEEPSEEK_BASE_URL`` or
            :data:`API_ROOT`.
        model: Default model override. Defaults to ``IRINA_MODEL`` or
            :data:`DEFAULT_MODEL`.
        temperature: Sampling temperature sent with every request.
        max_tokens: Output token ceiling sent with every request.
        timeout: Per-request socket timeout in seconds.
        retries: Retries after the initial request for transient failures.

    Raises:
        RuntimeError: If no API key is configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 8192,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Export the key before creating "
                "DeepSeekProvider; credentials are never read from source files."
            )
        resolved_base = base_url if base_url is not None else os.environ.get("DEEPSEEK_BASE_URL")
        resolved_model = model if model is not None else os.environ.get("IRINA_MODEL")

        self._api_key = resolved_key.strip()
        self.base_url = (resolved_base or API_ROOT).rstrip("/")
        self.default_model = (resolved_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Run one DeepSeek Chat Completions request and normalise the reply."""
        model_name = (model or self.default_model).strip() or self.default_model
        wire_messages: list[dict[str, Any]] = []
        if system:
            if not isinstance(system, str):
                raise TypeError("system prompt must be a string")
            wire_messages.append({"role": "system", "content": system})
        wire_messages.extend(_to_wire(messages))

        body: dict[str, Any] = {
            "model": model_name,
            "messages": wire_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        declarations = _tools_to_wire(tools)
        if declarations:
            body["tools"] = declarations
            body["tool_choice"] = "auto"

        payload = _post(
            f"{self.base_url}/chat/completions",
            body,
            retries=self.retries,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.timeout,
        )
        return _parse_response(payload)

    def _open_stream(self, body: dict[str, Any]) -> Any:
        """Open a streaming HTTP response, retrying only before events flow."""
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
        }
        retry_budget = max(0, int(self.retries))
        attempts = retry_budget + 1
        last_status: int | None = None
        last_description = "no response"
        for attempt in range(attempts):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                return urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                description = _http_error_description(exc)
                last_status = status
                last_description = description
                if status in RETRYABLE_STATUS_CODES:
                    if attempt < retry_budget:
                        retry_after = _header_value(exc.headers, "Retry-After")
                        time.sleep(_retry_delay(attempt, retry_after))
                        continue
                    raise ProviderRetryExhausted(
                        status, description, attempts=attempts
                    ) from None
                raise ProviderHTTPError(status, description) from None
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                description = _truncate(f"{type(reason).__name__}: {reason}")
                last_status = None
                last_description = description
                if attempt < retry_budget:
                    time.sleep(_retry_delay(attempt, None))
                    continue
                raise ProviderRetryExhausted(
                    None, description, attempts=attempts
                ) from None
        raise ProviderRetryExhausted(
            last_status, last_description, attempts=attempts
        )

    def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Iterator[ProviderStreamEvent]:
        """Yield genuine DeepSeek stream events as they arrive.

        The stream is parsed from the server-sent-event response. Tool-call
        arguments are emitted as deltas; this method never executes a tool.
        """
        model_name = (model or self.default_model).strip() or self.default_model
        wire_messages: list[dict[str, Any]] = []
        if system:
            if not isinstance(system, str):
                raise TypeError("system prompt must be a string")
            wire_messages.append({"role": "system", "content": system})
        wire_messages.extend(_to_wire(messages))

        body: dict[str, Any] = {
            "model": model_name,
            "messages": wire_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        declarations = _tools_to_wire(tools)
        if declarations:
            body["tools"] = declarations
            body["tool_choice"] = "auto"

        response = self._open_stream(body)
        finish_reason = ""
        usage: TokenUsage | None = None
        try:
            with response:
                for data in _iter_sse(response):
                    if not data:
                        continue
                    if data.strip() == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ProviderInvalidResponse(
                            "provider stream returned invalid JSON: "
                            f"{_truncate(str(exc))}"
                        ) from None
                    if not isinstance(payload, dict):
                        raise ProviderInvalidResponse(
                            "provider stream payload must be an object"
                        )
                    error = payload.get("error")
                    if isinstance(error, dict):
                        message = error.get("message") or "stream error"
                        raise ProviderStreamError(_truncate(str(message)))

                    parsed_usage = _parse_usage(payload.get("usage"))
                    if parsed_usage is not None:
                        usage = parsed_usage

                    choices = payload.get("choices")
                    if choices is None:
                        continue
                    if not isinstance(choices, list):
                        raise ProviderInvalidResponse(
                            "provider stream choices must be a list"
                        )
                    if not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise ProviderInvalidResponse(
                            "provider stream choice must be an object"
                        )

                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if content is not None:
                            if not isinstance(content, str):
                                raise ProviderInvalidResponse(
                                    "provider stream content must be a string"
                                )
                            if content:
                                yield ProviderStreamEvent(
                                    type=StreamEventType.TEXT_DELTA,
                                    text=content,
                                )

                        raw_calls = delta.get("tool_calls")
                        if raw_calls is not None:
                            if not isinstance(raw_calls, list):
                                raise ProviderInvalidResponse(
                                    "provider stream tool_calls must be a list"
                                )
                            for position, raw_call in enumerate(raw_calls):
                                if not isinstance(raw_call, dict):
                                    raise ProviderInvalidResponse(
                                        "provider stream tool call must be an object"
                                    )
                                index = raw_call.get("index", position)
                                if (
                                    isinstance(index, bool)
                                    or not isinstance(index, int)
                                    or index < 0
                                ):
                                    raise ProviderInvalidResponse(
                                        "provider stream tool index must be a "
                                        "non-negative integer"
                                    )
                                call_id = raw_call.get("id")
                                if call_id is not None and not isinstance(call_id, str):
                                    raise ProviderInvalidResponse(
                                        "provider stream tool id must be a string"
                                    )
                                function = raw_call.get("function")
                                name = ""
                                arguments = ""
                                if function is not None:
                                    if not isinstance(function, dict):
                                        raise ProviderInvalidResponse(
                                            "provider stream function must be an object"
                                        )
                                    raw_name = function.get("name")
                                    if raw_name is not None:
                                        if not isinstance(raw_name, str):
                                            raise ProviderInvalidResponse(
                                                "provider stream tool name must be a string"
                                            )
                                        name = raw_name
                                    raw_arguments = function.get("arguments")
                                    if raw_arguments is not None:
                                        if not isinstance(raw_arguments, str):
                                            raise ProviderInvalidResponse(
                                                "provider stream tool arguments must be a string"
                                            )
                                        arguments = raw_arguments
                                if call_id or name:
                                    yield ProviderStreamEvent(
                                        type=StreamEventType.TOOL_CALL_START,
                                        tool_index=index,
                                        tool_id=call_id or "",
                                        tool_name=name,
                                        arguments_delta=arguments,
                                    )
                                elif arguments:
                                    yield ProviderStreamEvent(
                                        type=StreamEventType.TOOL_CALL_DELTA,
                                        tool_index=index,
                                        arguments_delta=arguments,
                                    )

                    raw_finish = choice.get("finish_reason")
                    if raw_finish is not None:
                        if not isinstance(raw_finish, str) or not raw_finish:
                            raise ProviderInvalidResponse(
                                "provider stream finish_reason must be a string"
                            )
                        finish_reason = raw_finish
        except (urllib.error.URLError, OSError, http.client.IncompleteRead) as exc:
            raise ProviderStreamError(
                "stream connection failed: "
                f"{_truncate(f'{type(exc).__name__}: {exc}')}"
            ) from None

        if not finish_reason:
            raise ProviderInvalidResponse(
                "provider stream ended without a finish_reason"
            )
        yield ProviderStreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=finish_reason,
            usage=usage,
        )
