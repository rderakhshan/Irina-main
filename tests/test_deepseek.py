"""Day 1: offline tests for the DeepSeek provider.

Concept: verify wire conversion, response normalisation, credential handling,
and retry classification without touching the network.

Design constraints:
* ``unittest`` only; HTTP and sleep are mocked.
* No API credentials are required or read from disk.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from email.message import Message
from unittest import mock
from urllib.error import HTTPError, URLError

from src.back.harness.providers import deepseek
from src.back.harness.providers.base import (
    ProviderHTTPError,
    ProviderInvalidResponse,
    ProviderRetryExhausted,
    TokenUsage,
)
from src.back.harness.providers.deepseek import DeepSeekProvider, _to_wire

API_URL = "https://api.deepseek.com/chat/completions"


class FakeResponse:
    """Minimal context-managed stand-in for ``urlopen`` responses."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def make_payload(
    text: str = "hello",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    usage: dict | None = None,
) -> dict:
    """Build a DeepSeek-shaped chat completion payload."""
    message: dict = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if usage is None:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def http_error(
    status: int,
    body: bytes = b'{"error": "failure"}',
    retry_after: str | None = None,
) -> HTTPError:
    """Build an ``HTTPError`` with optional ``Retry-After`` metadata."""
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(API_URL, status, "failure", headers, io.BytesIO(body))


def make_provider(**overrides: object) -> DeepSeekProvider:
    """Create a provider with a fake key and predictable retry settings."""
    options: dict = {"api_key": "test-key", "retries": 3}
    options.update(overrides)
    return DeepSeekProvider(**options)


class CredentialTests(unittest.TestCase):
    def test_missing_api_key_raises_clear_runtime_error(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key != "DEEPSEEK_API_KEY"
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                DeepSeekProvider()
        self.assertIn("DEEPSEEK_API_KEY", str(caught.exception))

    def test_blank_api_key_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            DeepSeekProvider(api_key="   ")

    def test_model_and_base_url_can_come_from_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "env-key",
                "IRINA_MODEL": "env-model",
                "DEEPSEEK_BASE_URL": "https://example.test/",
            },
            clear=True,
        ):
            provider = DeepSeekProvider()
        self.assertEqual(provider.default_model, "env-model")
        self.assertEqual(provider.base_url, "https://example.test")


class MessageConversionTests(unittest.TestCase):
    def test_user_message_serialises_to_content(self) -> None:
        wire = _to_wire([{"role": "user", "text": "Hello"}])
        self.assertEqual(wire, [{"role": "user", "content": "Hello"}])

    def test_assistant_tool_calls_serialise_with_function_arguments(self) -> None:
        wire = _to_wire(
            [
                {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "name": "roll_dice",
                            "args": {"count": "3"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            wire,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "roll_dice",
                                "arguments": '{"count": "3"}',
                            },
                        }
                    ],
                }
            ],
        )

    def test_tool_result_preserves_tool_call_id(self) -> None:
        wire = _to_wire(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "name": "roll_dice",
                    "text": "[4, 5, 3]",
                }
            ]
        )
        self.assertEqual(
            wire,
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "[4, 5, 3]",
                }
            ],
        )

    def test_raw_arguments_are_preserved_when_present(self) -> None:
        wire = _to_wire(
            [
                {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "name": "roll_dice",
                            "args": {},
                            "raw_arguments": "{broken",
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            wire[0]["tool_calls"][0]["function"]["arguments"], "{broken"
        )


class RequestTests(unittest.TestCase):
    def test_complete_builds_expected_body_and_headers(self) -> None:
        tool_spec = {
            "schema": {
                "name": "roll_dice",
                "description": "Roll six-sided dice",
                "parameters": {},
            }
        }
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            return_value=FakeResponse(make_payload()),
        ) as opened:
            provider = make_provider()
            provider.complete(
                "deepseek-v4-pro",
                "You are Irina.",
                [{"role": "user", "text": "Hello"}],
                [tool_spec],
            )

        request = opened.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, API_URL)
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-key"
        )
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "You are Irina."},
                {"role": "user", "content": "Hello"},
            ],
        )
        self.assertEqual(body["temperature"], 0.4)
        self.assertEqual(body["max_tokens"], 8192)
        self.assertFalse(body["stream"])
        self.assertEqual(
            body["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "roll_dice",
                        "description": "Roll six-sided dice",
                        "parameters": {},
                    },
                }
            ],
        )
        self.assertEqual(body["tool_choice"], "auto")

    def test_tools_are_omitted_when_none_are_provided(self) -> None:
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            return_value=FakeResponse(make_payload()),
        ) as opened:
            provider = make_provider()
            provider.complete("m", "s", [{"role": "user", "text": "hi"}], [])

        body = json.loads(opened.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)


class ResponseTests(unittest.TestCase):
    def _complete(self, payload: dict):
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            return_value=FakeResponse(payload),
        ):
            return make_provider().complete(
                "m", "s", [{"role": "user", "text": "hi"}], []
            )

    def test_multiple_tool_calls_are_parsed_in_order(self) -> None:
        response = self._complete(
            make_payload(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "roll_dice",
                            "arguments": '{"count": 2}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "flip_coin",
                            "arguments": "{}",
                        },
                    },
                ],
            )
        )
        self.assertEqual([call.id for call in response.tool_calls], ["call_1", "call_2"])
        self.assertEqual([call.name for call in response.tool_calls], ["roll_dice", "flip_coin"])
        self.assertEqual(response.tool_calls[0].args, {"count": 2})
        self.assertIsNone(response.tool_calls[0].args_error)

    def test_invalid_json_arguments_become_recoverable_errors(self) -> None:
        response = self._complete(
            make_payload(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "roll_dice",
                            "arguments": "{not json",
                        },
                    }
                ],
            )
        )
        call = response.tool_calls[0]
        self.assertEqual(call.args, {})
        self.assertIsNotNone(call.args_error)
        self.assertEqual(call.raw_arguments, "{not json")

    def test_non_object_arguments_are_recoverable_errors(self) -> None:
        response = self._complete(
            make_payload(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_list",
                        "type": "function",
                        "function": {"name": "roll_dice", "arguments": "[1, 2]"},
                    }
                ],
            )
        )
        self.assertEqual(response.tool_calls[0].args, {})
        self.assertIn("JSON object", response.tool_calls[0].args_error or "")

    def test_token_usage_and_finish_reason_are_preserved(self) -> None:
        response = self._complete(
            make_payload(
                text="done",
                finish_reason="length",
                prompt_tokens=42,
                completion_tokens=7,
            )
        )
        self.assertEqual(response.text, "done")
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.usage, TokenUsage(42, 7))

    def test_missing_choices_is_invalid(self) -> None:
        with self.assertRaises(ProviderInvalidResponse):
            self._complete({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def test_missing_tool_call_id_is_invalid(self) -> None:
        with self.assertRaises(ProviderInvalidResponse):
            self._complete(
                make_payload(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {"name": "roll_dice", "arguments": "{}"},
                        }
                    ],
                )
            )

    def test_unsupported_finish_reason_is_invalid(self) -> None:
        with self.assertRaises(ProviderInvalidResponse):
            self._complete(make_payload(finish_reason="teleported"))

    def test_missing_usage_is_invalid(self) -> None:
        with self.assertRaises(ProviderInvalidResponse):
            self._complete(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )


class RetryTests(unittest.TestCase):
    def _complete_with(self, provider: DeepSeekProvider):
        return provider.complete(
            "m", "s", [{"role": "user", "text": "hi"}], []
        )

    def test_transient_http_failure_is_retried(self) -> None:
        provider = make_provider(retries=3)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[http_error(500), FakeResponse(make_payload("recovered"))],
        ) as opened, mock.patch.object(deepseek.time, "sleep") as slept:
            response = self._complete_with(provider)

        self.assertEqual(response.text, "recovered")
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(slept.call_count, 1)

    def test_transient_network_failure_is_retried(self) -> None:
        provider = make_provider(retries=1)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[
                URLError(TimeoutError("timed out")),
                FakeResponse(make_payload("recovered")),
            ],
        ) as opened, mock.patch.object(deepseek.time, "sleep"):
            response = self._complete_with(provider)

        self.assertEqual(response.text, "recovered")
        self.assertEqual(opened.call_count, 2)

    def test_permanent_http_failure_is_not_retried(self) -> None:
        provider = make_provider(retries=3)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[http_error(401, b'{"error": "bad key"}')],
        ) as opened, mock.patch.object(deepseek.time, "sleep") as slept:
            with self.assertRaises(ProviderHTTPError) as caught:
                self._complete_with(provider)

        self.assertEqual(caught.exception.status, 401)
        self.assertIn("bad key", caught.exception.description)
        self.assertEqual(opened.call_count, 1)
        slept.assert_not_called()

    def test_retry_exhaustion_is_distinguishable(self) -> None:
        provider = make_provider(retries=2)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[http_error(503) for _ in range(3)],
        ) as opened, mock.patch.object(deepseek.time, "sleep"):
            with self.assertRaises(ProviderRetryExhausted) as caught:
                self._complete_with(provider)

        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(opened.call_count, 3)

    def test_retry_after_is_honoured(self) -> None:
        provider = make_provider(retries=1)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[
                http_error(429, retry_after="2"),
                FakeResponse(make_payload("recovered")),
            ],
        ), mock.patch.object(deepseek.time, "sleep") as slept:
            self._complete_with(provider)

        self.assertEqual(slept.call_args.args[0], 2.0)

    def test_error_messages_never_include_the_api_key(self) -> None:
        provider = make_provider(retries=0)
        with mock.patch.object(
            deepseek.urllib.request,
            "urlopen",
            side_effect=[http_error(400, b'{"error": "bad request"}')],
        ):
            with self.assertRaises(ProviderHTTPError) as caught:
                self._complete_with(provider)
        self.assertNotIn("test-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
