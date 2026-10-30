"""Google Gemini generateContent provider adapter.

Gemini has a different conversation shape from OpenAI: contents are role/parts
objects, function calls and responses are parts, and thought signatures must be
echoed back. All of that translation stays in this module.
"""

from __future__ import annotations

import json
import os
import uuid
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

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash"


def _tools_to_wire(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            raise ProviderInvalidResponse("tools must contain dictionaries")
        schema = entry.get("schema", entry)
        if not isinstance(schema, dict):
            raise ProviderInvalidResponse("tool schema must be an object")
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProviderInvalidResponse("tool is missing a name")
        description = schema.get("description", "")
        parameters = schema.get("parameters", {})
        if not isinstance(description, str):
            raise ProviderInvalidResponse(f"tool {name!r} description must be a string")
        if not isinstance(parameters, dict):
            raise ProviderInvalidResponse(f"tool {name!r} parameters must be an object")
        declarations.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return [{"functionDeclarations": declarations}] if declarations else []


def _to_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the neutral transcript into Gemini contents."""
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise ProviderInvalidResponse(f"message {index} must be an object")
        role = message.get("role")
        if role == "user":
            contents.append(
                {"role": "user", "parts": [{"text": str(message.get("text", ""))}]}
            )
        elif role == "assistant":
            state = message.get("provider_state")
            if isinstance(state, dict) and isinstance(state.get("gemini_parts"), list):
                parts = [dict(part) for part in state["gemini_parts"] if isinstance(part, dict)]
            else:
                parts = []
                text = str(message.get("text", ""))
                if text:
                    parts.append({"text": text})
                for call in message.get("tool_calls", []) or []:
                    if not isinstance(call, dict):
                        raise ProviderInvalidResponse("assistant tool call must be an object")
                    call_id = call.get("id")
                    name = call.get("name")
                    if not isinstance(call_id, str) or not call_id:
                        raise ProviderInvalidResponse("assistant tool call is missing an id")
                    if not isinstance(name, str) or not name:
                        raise ProviderInvalidResponse("assistant tool call is missing a name")
                    call_names[call_id] = name
                    args = call.get("args")
                    if not isinstance(args, dict):
                        raw = call.get("raw_arguments")
                        if isinstance(raw, str):
                            try:
                                parsed = json.loads(raw)
                            except (json.JSONDecodeError, ValueError):
                                parsed = {}
                            args = parsed if isinstance(parsed, dict) else {}
                        else:
                            args = {}
                    parts.append({"functionCall": {"name": name, "args": args}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            parts = []
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_message = messages[index]
                call_id = tool_message.get("tool_call_id")
                name = tool_message.get("name") or call_names.get(str(call_id), "")
                if not isinstance(name, str) or not name:
                    raise ProviderInvalidResponse(
                        f"tool message {index} has no callable name"
                    )
                parts.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {"result": str(tool_message.get("text", ""))},
                        }
                    }
                )
                index += 1
            contents.append({"role": "user", "parts": parts})
            continue
        else:
            raise ProviderInvalidResponse(
                f"message {index} has unsupported role {role!r}"
            )
        index += 1
    return contents


def _usage(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return TokenUsage(0, 0)
    prompt = usage.get("promptTokenCount", 0)
    output = usage.get("candidatesTokenCount", 0)
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
        prompt = 0
    if not isinstance(output, int) or isinstance(output, bool) or output < 0:
        output = 0
    return TokenUsage(prompt, output)


def _finish_reason(raw: Any, has_calls: bool) -> str:
    reason = str(raw or "").upper()
    if reason in {"MAX_TOKENS", "MAX_OUTPUT_TOKENS"}:
        return "length"
    if reason in {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    }:
        return "content_filter"
    if reason in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
        raise ProviderInvalidResponse(
            f"Gemini reported {reason.lower().replace('_', ' ')}"
        )
    if has_calls:
        return "tool_calls"
    return "stop"


def _parse_candidate(payload: dict[str, Any]) -> tuple[str, list[ToolCall], list[dict[str, Any]], str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        error = payload.get("error")
        if isinstance(error, dict):
            raise ProviderInvalidResponse(
                "Gemini returned an error: " + truncate(error.get("message", "unknown"))
            )
        raise ProviderInvalidResponse("Gemini response is missing candidates")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ProviderInvalidResponse("Gemini candidate must be an object")
    content = candidate.get("content") or {}
    if not isinstance(content, dict):
        raise ProviderInvalidResponse("Gemini content must be an object")
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise ProviderInvalidResponse("Gemini parts must be a list")
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for position, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            name = function_call.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ProviderInvalidResponse("Gemini function call is missing a name")
            call_id = function_call.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"gemini-call-{uuid.uuid4().hex[:12]}-{position}"
            args = function_call.get("args")
            args_error = None
            raw_arguments = None
            if args is None:
                args = {}
            if not isinstance(args, dict):
                args_error = "Gemini function arguments must be an object"
                args = {}
            try:
                raw_arguments = json.dumps(args, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                raw_arguments = None
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    args=args,
                    args_error=args_error,
                    raw_arguments=raw_arguments,
                )
            )
    finish = _finish_reason(candidate.get("finishReason"), bool(calls))
    return "".join(text_parts), calls, parts, finish


def _parse_response(payload: dict[str, Any]) -> ProviderResponse:
    text, calls, parts, finish = _parse_candidate(payload)
    continuation = {"gemini_parts": parts} if parts else None
    return ProviderResponse(
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        usage=_usage(payload),
        continuation=continuation,
    )


class GeminiProvider:
    """Gemini generateContent adapter.

    Configuration is read from ``GEMINI_API_KEY`` (required),
    ``GEMINI_BASE_URL`` (optional), and ``GEMINI_MODEL`` (optional default).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            resolved_key = os.environ.get("GOOGLE_API_KEY")
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before selecting the "
                "Gemini provider."
            )
        resolved_base = base_url if base_url is not None else os.environ.get("GEMINI_BASE_URL")
        resolved_model = model if model is not None else os.environ.get("GEMINI_MODEL")
        self._api_key = resolved_key.strip()
        self.base_url = (resolved_base or API_ROOT).rstrip("/")
        self.default_model = (resolved_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.retries = retries

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def _body(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if system and not isinstance(system, str):
            raise TypeError("system prompt must be a string")
        body: dict[str, Any] = {
            "contents": _to_contents(messages),
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        declarations = _tools_to_wire(tools)
        if declarations:
            body["tools"] = declarations
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        return body

    def _url(self, model: str, method: str) -> str:
        model_name = (model or self.default_model).strip() or self.default_model
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]
        return f"{self.base_url}/models/{model_name}:{method}"

    def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        """Run one generateContent request."""
        payload = post_json(
            self._url(model, "generateContent"),
            self._body(system, messages, tools),
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
        """Yield genuine Gemini SSE chunks."""
        response = open_stream(
            self._url(model, "streamGenerateContent") + "?alt=sse",
            self._body(system, messages, tools),
            headers=self._headers(),
            retries=self.retries,
            timeout=self.timeout,
        )
        finish_reason = ""
        usage: TokenUsage | None = None
        collected_parts: list[dict[str, Any]] = []
        tool_index = 0
        try:
            with response:
                for data in iter_sse(response):
                    if not data or data.strip() == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ProviderInvalidResponse(
                            "Gemini stream returned invalid JSON: " + truncate(exc)
                        ) from None
                    if not isinstance(payload, dict):
                        raise ProviderInvalidResponse("Gemini stream chunk must be an object")
                    error = payload.get("error")
                    if isinstance(error, dict):
                        raise ProviderStreamError(
                            truncate(error.get("message", "Gemini stream error"))
                        )
                    parsed_usage = _usage(payload)
                    if parsed_usage.total_tokens:
                        usage = parsed_usage
                    candidates = payload.get("candidates")
                    if not isinstance(candidates, list) or not candidates:
                        continue
                    candidate = candidates[0]
                    if not isinstance(candidate, dict):
                        continue
                    content = candidate.get("content") or {}
                    parts = content.get("parts") if isinstance(content, dict) else None
                    for part in parts or []:
                        if not isinstance(part, dict):
                            continue
                        collected_parts.append(part)
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            yield ProviderStreamEvent(
                                type=StreamEventType.TEXT_DELTA, text=text
                            )
                        function_call = part.get("functionCall")
                        if isinstance(function_call, dict):
                            name = function_call.get("name")
                            if not isinstance(name, str) or not name.strip():
                                raise ProviderInvalidResponse(
                                    "Gemini stream function call is missing a name"
                                )
                            call_id = function_call.get("id")
                            if not isinstance(call_id, str) or not call_id:
                                call_id = (
                                    f"gemini-stream-{uuid.uuid4().hex[:12]}-"
                                    f"{tool_index}"
                                )
                            args = function_call.get("args")
                            if not isinstance(args, dict):
                                args = {}
                            yield ProviderStreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_index=tool_index,
                                tool_id=call_id,
                                tool_name=name,
                                arguments_delta=json.dumps(
                                    args, ensure_ascii=False, sort_keys=True
                                ),
                            )
                            tool_index += 1
                    raw_finish = candidate.get("finishReason")
                    if raw_finish:
                        finish_reason = _finish_reason(raw_finish, False)
        except (ProviderStreamError, ProviderInvalidResponse):
            raise
        if not finish_reason:
            finish_reason = "tool_calls" if tool_index else "stop"
        elif tool_index and finish_reason == "stop":
            finish_reason = "tool_calls"
        continuation = {"gemini_parts": collected_parts} if collected_parts else None
        yield ProviderStreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=finish_reason,
            usage=usage,
            continuation=continuation,
        )


__all__ = ["API_ROOT", "DEFAULT_MODEL", "GeminiProvider"]
