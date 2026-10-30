"""Shared standard-library HTTP helpers for provider adapters.

Only the new OpenAI and Gemini adapters use this module. The existing DeepSeek
adapter keeps its original implementation so its behavior and tests stay
unchanged.
"""

from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterator

from .base import (
    ProviderHTTPError,
    ProviderInvalidResponse,
    ProviderRetryExhausted,
    ProviderStreamError,
)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 5
BASE_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 30.0
MAX_ERROR_DESCRIPTION_CHARS = 500


def truncate(text: str, limit: int = MAX_ERROR_DESCRIPTION_CHARS) -> str:
    """Collapse text into a bounded, credential-free single line."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def http_error_description(error: urllib.error.HTTPError) -> str:
    """Read a bounded description from an HTTP error response."""
    try:
        raw = error.read()
    except Exception:
        raw = b""
    body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    if not body:
        body = str(error.reason or "")
    return truncate(body)


def header_value(headers: Any, name: str) -> str | None:
    """Read a header case-insensitively."""
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is not None:
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


def parse_retry_after(value: str | None) -> float | None:
    """Convert a Retry-After header into a bounded delay."""
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
    return min(max(0.0, seconds), MAX_RETRY_DELAY_SECONDS)


def retry_delay(attempt: int, retry_after: str | None = None) -> float:
    """Return bounded exponential backoff with jitter."""
    honoured = parse_retry_after(retry_after)
    if honoured is not None:
        return honoured
    ceiling = min(MAX_RETRY_DELAY_SECONDS, BASE_RETRY_DELAY_SECONDS * (2**attempt))
    return random.uniform(0.0, ceiling)


def decode_json(raw: bytes) -> dict[str, Any]:
    """Decode a JSON object or raise a classified provider error."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderInvalidResponse(
            f"provider returned invalid JSON: {truncate(exc)}"
        ) from None
    if not isinstance(payload, dict):
        raise ProviderInvalidResponse("provider response must be a JSON object")
    return payload


def urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """The single network indirection shared by the new adapters."""
    return urllib.request.urlopen(request, timeout=timeout)


def _request_headers(
    headers: dict[str, str] | None,
    *,
    accept: str = "application/json",
) -> dict[str, str]:
    request_headers = {"Content-Type": "application/json", "Accept": accept}
    if headers:
        request_headers.update(headers)
    return request_headers


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST JSON with bounded retries and classified failures."""
    retry_budget = max(0, int(retries))
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = _request_headers(headers)
    attempts = retry_budget + 1
    last_status: int | None = None
    last_description = "no response"
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=encoded, headers=request_headers, method="POST"
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return decode_json(response.read())
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            description = http_error_description(exc)
            last_status, last_description = status, description
            if status in RETRYABLE_STATUS_CODES:
                if attempt < retry_budget:
                    time.sleep(
                        retry_delay(attempt, header_value(exc.headers, "Retry-After"))
                    )
                    continue
                raise ProviderRetryExhausted(
                    status, description, attempts=attempts
                ) from None
            raise ProviderHTTPError(status, description) from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            description = truncate(f"{type(reason).__name__}: {reason}")
            last_status, last_description = None, description
            if attempt < retry_budget:
                time.sleep(retry_delay(attempt))
                continue
            raise ProviderRetryExhausted(
                None, description, attempts=attempts
            ) from None
    raise ProviderRetryExhausted(last_status, last_description, attempts=attempts)


def open_stream(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Open one streaming HTTP response, retrying only before events flow."""
    retry_budget = max(0, int(retries))
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_headers = _request_headers(headers, accept="text/event-stream")
    attempts = retry_budget + 1
    last_status: int | None = None
    last_description = "no response"
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=encoded, headers=request_headers, method="POST"
        )
        try:
            return urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            description = http_error_description(exc)
            last_status, last_description = status, description
            if status in RETRYABLE_STATUS_CODES:
                if attempt < retry_budget:
                    time.sleep(
                        retry_delay(attempt, header_value(exc.headers, "Retry-After"))
                    )
                    continue
                raise ProviderRetryExhausted(
                    status, description, attempts=attempts
                ) from None
            raise ProviderHTTPError(status, description) from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            description = truncate(f"{type(reason).__name__}: {reason}")
            last_status, last_description = None, description
            if attempt < retry_budget:
                time.sleep(retry_delay(attempt))
                continue
            raise ProviderRetryExhausted(
                None, description, attempts=attempts
            ) from None
    raise ProviderRetryExhausted(last_status, last_description, attempts=attempts)


def iter_sse(response: Any) -> Iterator[str]:
    """Yield SSE data payloads, assembling multi-line data fields."""
    data: list[str] = []
    try:
        for raw in response:
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
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
    except (urllib.error.URLError, OSError, http.client.IncompleteRead) as exc:
        raise ProviderStreamError(
            "stream connection failed: "
            f"{truncate(f'{type(exc).__name__}: {exc}')}"
        ) from None


__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "decode_json",
    "header_value",
    "http_error_description",
    "iter_sse",
    "open_stream",
    "parse_retry_after",
    "post_json",
    "retry_delay",
    "truncate",
    "urlopen",
]
