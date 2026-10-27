"""Day 1: offline tests for the bounded execution loop.

Concept: verify completion semantics, tool dispatch, protocol integrity, and
every configured termination mode using a scripted provider.

Design constraints:
* ``unittest`` only; no network access or credentials.
* The provider is a fake implementing the neutral contract.
* Tests assert structured statuses, not just final strings.
"""

from __future__ import annotations

import copy
import unittest
from collections import Counter

from src.back.harness.loop.runner import EventType, run_loop
from src.back.harness.loop.state import RunStatus
from src.back.harness.loop.termination import TerminationPolicy
from src.back.harness.providers.base import (
    ProviderHTTPError,
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderRetryExhausted,
    TokenUsage,
    ToolCall,
)


class ScriptedProvider:
    """Provider that returns pre-scripted responses and records calls."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        if not self.responses:
            raise AssertionError("provider was called more times than scripted")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingTool:
    """Tool that records keyword arguments and returns a fixed value."""

    def __init__(
        self,
        name: str = "echo",
        result: str = "ok",
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.calls: list[dict] = []
        self.spec = {
            "schema": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {},
            }
        }

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def make_call(
    call_id: str = "call_1",
    name: str = "echo",
    args: dict | None = None,
    args_error: str | None = None,
) -> ToolCall:
    """Build a tool call with optional validation failure."""
    return ToolCall(
        id=call_id,
        name=name,
        args={} if args is None else dict(args),
        args_error=args_error,
    )


def make_response(
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str | None = None,
    usage: tuple[int, int] = (1, 1),
) -> ProviderResponse:
    """Build a provider response with sensible finish-reason defaults."""
    calls = list(tool_calls or [])
    if finish_reason is None:
        finish_reason = "tool_calls" if calls else "stop"
    return ProviderResponse(
        text=text,
        tool_calls=calls,
        finish_reason=finish_reason,
        usage=TokenUsage(*usage),
    )


def default_policy(**overrides: object) -> TerminationPolicy:
    """Return a policy with generous limits for tests about other behaviour."""
    options: dict = {
        "max_turns": 20,
        "max_tool_calls": 100,
        "max_seconds": 60,
        "max_total_tokens": 1_000_000,
        "max_repeated_calls": 10,
        "max_empty_responses": 3,
        "max_consecutive_tool_errors": 10,
    }
    options.update(overrides)
    return TerminationPolicy(**options)


def execute(provider, tools=None, **kwargs):
    """Run the loop with test defaults and return result, messages, events."""
    messages = kwargs.pop("messages", None)
    if messages is None:
        messages = [{"role": "user", "text": "hi"}]
    events = kwargs.pop("events", None)
    if events is None:
        events = []
    policy = kwargs.pop("policy", None)
    if policy is None:
        policy = default_policy()
    result = run_loop(
        provider=provider,
        model="test-model",
        system="test-system",
        messages=messages,
        tools=tools or {},
        on_event=events.append,
        policy=policy,
        **kwargs,
    )
    return result, messages, events


def tool_messages(messages: list[dict]) -> list[dict]:
    """Return only tool-role transcript messages."""
    return [message for message in messages if message.get("role") == "tool"]


class CompletionTests(unittest.TestCase):
    def test_plain_text_completes_a_conversational_task(self) -> None:
        provider = ScriptedProvider([make_response("Hello!", usage=(10, 2))])
        result, messages, events = execute(provider)

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "Hello!")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(result.usage, TokenUsage(10, 2))
        self.assertEqual(messages[-1], {"role": "assistant", "text": "Hello!"})
        self.assertEqual(
            [event.type for event in events],
            [EventType.ASSISTANT, EventType.RUN_END],
        )
        self.assertIs(events[-1].status, RunStatus.COMPLETED)

    def test_completion_validator_can_reject_premature_completion(self) -> None:
        attempts: list[str] = []

        def validator(response, messages):
            attempts.append(response.text)
            return response.text == "DONE"

        provider = ScriptedProvider(
            [make_response("not enough evidence"), make_response("DONE")]
        )
        result, messages, _ = execute(provider, completion_validator=validator)

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(attempts, ["not enough evidence", "DONE"])
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(
            any(
                message.get("role") == "user"
                and "Completion was rejected" in message.get("text", "")
                for message in messages
            )
        )

    def test_completion_policy_can_request_user_input(self) -> None:
        def validator(response, messages):
            return RunStatus.AWAITING_INPUT

        provider = ScriptedProvider(
            [make_response("Which city should the page target?")]
        )
        result, _, _ = execute(provider, completion_validator=validator)

        self.assertIs(result.status, RunStatus.AWAITING_INPUT)
        self.assertEqual(result.text, "Which city should the page target?")

    def test_length_finish_reason_is_not_completion(self) -> None:
        provider = ScriptedProvider(
            [make_response("partial answer", finish_reason="length")]
        )
        result, _, _ = execute(provider)

        self.assertIs(result.status, RunStatus.INVALID_RESPONSE)
        self.assertNotEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.text, "partial answer")
        self.assertIn("truncated", result.reason)

    def test_tool_calls_finish_reason_without_calls_is_invalid(self) -> None:
        provider = ScriptedProvider(
            [make_response("", finish_reason="tool_calls")]
        )
        result, _, _ = execute(provider)
        self.assertIs(result.status, RunStatus.INVALID_RESPONSE)

    def test_content_filter_is_classified_as_blocked(self) -> None:
        provider = ScriptedProvider(
            [make_response("filtered", finish_reason="content_filter")]
        )
        result, _, _ = execute(provider)
        self.assertIs(result.status, RunStatus.BLOCKED)

    def test_empty_responses_are_retried_within_the_limit(self) -> None:
        provider = ScriptedProvider([make_response(""), make_response("Answer")])
        result, messages, _ = execute(
            provider, policy=default_policy(max_empty_responses=2)
        )

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.text, "Answer")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(
            any(
                message.get("role") == "user"
                and "previous response was empty" in message.get("text", "")
                for message in messages
            )
        )

    def test_empty_response_limit_terminates_as_stalled(self) -> None:
        provider = ScriptedProvider([make_response(""), make_response("")])
        result, _, _ = execute(
            provider, policy=default_policy(max_empty_responses=2)
        )
        self.assertIs(result.status, RunStatus.STALLED)
        self.assertEqual(result.turns, 2)


class ToolTests(unittest.TestCase):
    def test_tool_call_is_executed_and_result_appended(self) -> None:
        tool = RecordingTool("echo", result="42")
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call(name="echo", args={"x": 1})]),
                make_response("done"),
            ]
        )
        result, messages, events = execute(provider, tools={"echo": tool})

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(tool.calls, [{"x": 1}])
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.turns, 2)
        self.assertEqual(tool_messages(messages)[0]["text"], "42")
        self.assertIn(EventType.TOOL_START, [event.type for event in events])
        self.assertIn(EventType.TOOL_END, [event.type for event in events])

    def test_multiple_tool_calls_execute_in_order(self) -> None:
        first = RecordingTool("first")
        second = RecordingTool("second")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call("call_1", "first", {"n": 1}),
                        make_call("call_2", "second", {"n": 2}),
                    ]
                ),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(
            provider, tools={"first": first, "second": second}
        )

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(first.calls, [{"n": 1}])
        self.assertEqual(second.calls, [{"n": 2}])
        results = tool_messages(messages)
        self.assertEqual([message["tool_call_id"] for message in results], ["call_1", "call_2"])

    def test_tool_results_are_appended_before_the_next_request(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call("call_a", "echo", {"v": 7})]),
                make_response("done"),
            ]
        )
        execute(provider, tools={"echo": tool})

        second_request = provider.calls[1]["messages"]
        assistant = second_request[1]
        tool_result = second_request[2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_a")
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(tool_result["tool_call_id"], "call_a")
        self.assertEqual(tool_result["text"], "ok")

    def test_unknown_tool_does_not_crash_the_loop(self) -> None:
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call("call_x", "missing")]),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(provider, tools={"echo": RecordingTool()})

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            tool_messages(messages)[0]["text"], "ERROR: unknown tool missing"
        )

    def test_tool_exception_does_not_crash_the_loop(self) -> None:
        tool = RecordingTool("echo", error=ValueError("boom"))
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call("call_e", "echo")]),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(provider, tools={"echo": tool})

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            tool_messages(messages)[0]["text"], "ERROR: ValueError: boom"
        )

    def test_blocked_tool_is_not_executed(self) -> None:
        tool = RecordingTool("danger")
        seen: list[ToolCall] = []

        def before_tool(call):
            seen.append(call)
            return "not allowed"

        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call("call_d", "danger")]),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(
            provider, tools={"danger": tool}, before_tool=before_tool
        )

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(tool.calls, [])
        self.assertEqual(seen[0].id, "call_d")
        self.assertEqual(
            tool_messages(messages)[0]["text"], "BLOCKED: not allowed"
        )

    def test_invalid_arguments_never_reach_run(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call(
                            "call_bad",
                            "echo",
                            args={},
                            args_error="invalid JSON arguments: Expecting value",
                        )
                    ]
                ),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(provider, tools={"echo": tool})

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(tool.calls, [])
        self.assertTrue(
            tool_messages(messages)[0]["text"].startswith(
                "ERROR: invalid tool arguments:"
            )
        )

    def test_every_tool_call_has_exactly_one_result(self) -> None:
        echo = RecordingTool("echo")
        guarded = RecordingTool("guarded")

        def before_tool(call):
            return "denied" if call.name == "guarded" else None

        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call("c1", "echo", {"ok": True}),
                        make_call("c2", "unknown"),
                        make_call("c3", "guarded"),
                        make_call(
                            "c4",
                            "echo",
                            args={},
                            args_error="invalid JSON arguments",
                        ),
                    ]
                ),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": echo, "guarded": guarded},
            before_tool=before_tool,
        )

        self.assertIs(result.status, RunStatus.COMPLETED)
        requested = Counter(
            entry["id"]
            for message in messages
            if message.get("role") == "assistant"
            for entry in message.get("tool_calls", [])
        )
        answered = Counter(
            message["tool_call_id"] for message in tool_messages(messages)
        )
        self.assertEqual(requested, answered)
        self.assertEqual(guarded.calls, [])


class LimitTests(unittest.TestCase):
    def test_loop_terminates_at_the_turn_limit(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    "thinking", tool_calls=[make_call("c1", "echo", {"n": 1})]
                ),
                make_response(
                    "thinking", tool_calls=[make_call("c2", "echo", {"n": 2})]
                ),
                make_response(
                    "thinking", tool_calls=[make_call("c3", "echo", {"n": 3})]
                ),
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": tool},
            policy=default_policy(max_turns=2),
        )

        self.assertIs(result.status, RunStatus.MAX_TURNS)
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(result.text, "thinking")
        self.assertIn("turns", result.reason)
        self.assertTrue(messages)

    def test_loop_terminates_at_the_tool_call_limit(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call("c1", "echo", {"n": 1}),
                        make_call("c2", "echo", {"n": 2}),
                    ]
                ),
                make_response("done"),
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": tool},
            policy=default_policy(max_tool_calls=1),
        )

        self.assertIs(result.status, RunStatus.MAX_TOOL_CALLS)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(tool.calls), 1)
        results = tool_messages(messages)
        self.assertEqual(results[0]["text"], "ok")
        self.assertEqual(
            results[1]["text"], "BLOCKED: tool call limit reached"
        )

    def test_loop_terminates_at_the_token_budget(self) -> None:
        provider = ScriptedProvider([make_response("hello", usage=(6, 6))])
        result, _, _ = execute(
            provider, policy=default_policy(max_total_tokens=10)
        )

        self.assertIs(result.status, RunStatus.TOKEN_BUDGET)
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.usage, TokenUsage(6, 6))

    def test_loop_terminates_at_the_time_budget(self) -> None:
        provider = ScriptedProvider([make_response("never used")])
        result, _, _ = execute(
            provider, policy=default_policy(max_seconds=0)
        )

        self.assertIs(result.status, RunStatus.TIMEOUT)
        self.assertEqual(provider.calls, [])

    def test_repeated_calls_trigger_stall_detection(self) -> None:
        tool = RecordingTool("echo")
        repeated = make_call("c1", "echo", {"n": 1})
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[repeated]),
                make_response(tool_calls=[make_call("c2", "echo", {"n": 1})]),
                make_response(tool_calls=[make_call("c3", "echo", {"n": 1})]),
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": tool},
            policy=default_policy(max_repeated_calls=3),
        )

        self.assertIs(result.status, RunStatus.STALLED)
        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(result.turns, 3)
        self.assertEqual(
            tool_messages(messages)[-1]["text"],
            "BLOCKED: repeated identical tool call detected",
        )

    def test_stall_detection_can_be_disabled(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(tool_calls=[make_call("c1", "echo", {"n": 1})]),
                make_response(tool_calls=[make_call("c2", "echo", {"n": 1})]),
                make_response("done"),
            ]
        )
        result, _, _ = execute(
            provider,
            tools={"echo": tool},
            policy=default_policy(
                max_repeated_calls=2, enable_stall_detection=False
            ),
        )

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(tool.calls), 2)

    def test_cancellation_stops_further_tool_execution(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call("c1", "echo", {"n": 1}),
                        make_call("c2", "echo", {"n": 2}),
                    ]
                )
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": tool},
            should_cancel=lambda: len(tool.calls) >= 1,
        )

        self.assertIs(result.status, RunStatus.CANCELLED)
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(
            tool_messages(messages)[-1]["text"], "BLOCKED: run cancelled"
        )

    def test_consecutive_tool_errors_terminate_the_run(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[
                        make_call("c1", "missing"),
                        make_call("c2", "missing"),
                        make_call("c3", "missing"),
                    ]
                )
            ]
        )
        result, messages, _ = execute(
            provider,
            tools={"echo": tool},
            policy=default_policy(max_consecutive_tool_errors=2),
        )

        self.assertIs(result.status, RunStatus.TOOL_ERROR)
        self.assertEqual(len(tool_messages(messages)), 3)
        self.assertEqual(
            tool_messages(messages)[-1]["text"],
            "BLOCKED: consecutive tool error limit reached",
        )


class ErrorAndAccountingTests(unittest.TestCase):
    def test_provider_error_produces_explicit_failure(self) -> None:
        provider = ScriptedProvider([ProviderHTTPError(401, "bad key")])
        result, _, _ = execute(provider)

        self.assertIs(result.status, RunStatus.PROVIDER_ERROR)
        self.assertIn("401", result.reason)
        self.assertEqual(result.turns, 0)

    def test_retry_exhaustion_is_reported_as_provider_error(self) -> None:
        provider = ScriptedProvider(
            [ProviderRetryExhausted(503, "unavailable", attempts=6)]
        )
        result, _, _ = execute(provider)

        self.assertIs(result.status, RunStatus.PROVIDER_ERROR)
        self.assertIn("retries exhausted", result.reason)

    def test_invalid_provider_response_maps_to_invalid_response(self) -> None:
        provider = ScriptedProvider([ProviderInvalidResponse("missing choices")])
        result, _, _ = execute(provider)

        self.assertIs(result.status, RunStatus.INVALID_RESPONSE)
        self.assertIn("missing choices", result.reason)

    def test_usage_accumulates_across_model_calls(self) -> None:
        tool = RecordingTool("echo")
        provider = ScriptedProvider(
            [
                make_response(
                    tool_calls=[make_call("c1", "echo")], usage=(10, 5)
                ),
                make_response("done", usage=(3, 4)),
            ]
        )
        result, _, _ = execute(provider, tools={"echo": tool})

        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.usage, TokenUsage(13, 9))

    def test_before_turn_replaces_transcript_in_place(self) -> None:
        messages = [{"role": "user", "text": "original"}]
        transitions: list[list[dict]] = []

        def before_turn(current):
            transitions.append(list(current))
            if len(transitions) == 1:
                return [{"role": "user", "text": "replaced"}]
            return None

        provider = ScriptedProvider([make_response("ok")])
        result, returned_messages, _ = execute(
            provider, messages=messages, before_turn=before_turn
        )

        self.assertIs(result.messages, messages)
        self.assertIs(returned_messages, messages)
        self.assertEqual(
            messages,
            [
                {"role": "user", "text": "replaced"},
                {"role": "assistant", "text": "ok"},
            ],
        )
        self.assertEqual(provider.calls[0]["messages"][0]["text"], "replaced")


if __name__ == "__main__":
    unittest.main()
