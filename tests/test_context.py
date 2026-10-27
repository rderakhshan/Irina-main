"""Stage 2: deterministic tests for context compaction and task state."""

from __future__ import annotations

import json
import math
import unittest

from src.back.harness.context_management.context import (
    CHARS_PER_TOKEN,
    COMPACTION_INSTRUCTION,
    KEEP_RECENT,
    SUMMARY_PREFIX,
    CompactionError,
    TaskState,
    TaskStateObserver,
    calibrate,
    calibration_ratio,
    compact,
    estimate_tokens,
    reset_calibration,
)
from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.loop.state import RunStatus
from src.back.harness.providers.base import (
    ProviderError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


class FakeProvider:
    """Provider that returns one scripted summary and records calls."""

    def __init__(
        self,
        text: str = "Summary of the earlier work.",
        finish_reason: str = "stop",
        error: BaseException | None = None,
        usage: tuple[int, int] = (100, 20),
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.error = error
        self.usage = usage
        self.calls: list[dict] = []

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
            }
        )
        if self.error is not None:
            raise self.error
        return ProviderResponse(
            text=self.text,
            tool_calls=[],
            finish_reason=self.finish_reason,
            usage=TokenUsage(*self.usage),
        )


def message_pair(index: int, name: str = "write_file") -> list[dict]:
    """Create one complete assistant tool call and result pair."""
    call_id = f"call-{index}"
    if name == "write_file":
        args = {"path": f"file-{index}.txt", "content": "ping\n" * 20}
        result = f"Wrote 100 chars to file-{index}.txt"
    else:
        args = {"command": "echo ping"}
        result = "ping\n"
    return [
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"id": call_id, "name": name, "args": args}],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "text": result,
        },
    ]


def long_messages(pairs: int = 5) -> list[dict]:
    """Build a task plus several tool pairs and a final answer."""
    messages = [{"role": "user", "text": "Create the files and verify them."}]
    for index in range(pairs):
        messages.extend(message_pair(index))
    messages.append({"role": "assistant", "text": "Done."})
    return messages


def assert_tool_integrity(
    testcase: unittest.TestCase,
    messages: list[dict],
    require_calls: bool = True,
) -> None:
    """Assert every assistant tool call has exactly one result and vice versa."""
    calls = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                testcase.assertNotIn(call["id"], calls)
                calls[call["id"]] = 0
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            testcase.assertIn(call_id, calls)
            calls[call_id] += 1
    if require_calls:
        testcase.assertTrue(calls)
    for call_id, count in calls.items():
        testcase.assertEqual(count, 1, call_id)


class EstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_calibration()

    def tearDown(self) -> None:
        reset_calibration()

    def test_baseline_is_serialized_characters_over_four_rounded_up(self) -> None:
        messages = [{"role": "user", "text": "hello"}]
        serialized = json.dumps(
            messages,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        expected = math.ceil(len(serialized) / CHARS_PER_TOKEN)
        self.assertEqual(estimate_tokens(messages), expected)

    def test_calibration_raises_the_conservative_estimate(self) -> None:
        messages = [{"role": "user", "text": "hello"}]
        baseline = estimate_tokens(messages)
        calibrate(baseline, baseline * 2)
        self.assertEqual(calibration_ratio(), 2.0)
        self.assertEqual(estimate_tokens(messages), baseline * 2)
        reset_calibration()
        self.assertEqual(estimate_tokens(messages), baseline)


class CompactionTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_calibration()

    def tearDown(self) -> None:
        reset_calibration()

    def test_short_context_is_returned_unchanged_without_a_provider(self) -> None:
        messages = long_messages(1)
        provider = FakeProvider()
        events: list[RunEvent] = []
        result = compact(
            "model",
            messages,
            10_000,
            provider=provider,
            on_event=events.append,
        )
        self.assertEqual(result, messages)
        self.assertEqual(provider.calls, [])
        self.assertEqual(events, [])

    def test_compaction_preserves_task_recent_groups_and_state(self) -> None:
        messages = long_messages(20)
        state = TaskState(goal="Create the files and verify them.")
        state.add_completed("wrote file-0.txt")
        state.add_modified("file-0.txt")
        provider = FakeProvider("Older work summarized.")
        events: list[RunEvent] = []

        result = compact(
            "model",
            messages,
            1500,
            provider=provider,
            task_state=state,
            on_event=events.append,
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["system"], COMPACTION_INSTRUCTION)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(result[0], messages[0])
        self.assertTrue(result[1]["text"].startswith(SUMMARY_PREFIX))
        self.assertIn("Older work summarized.", result[1]["text"])
        self.assertTrue(result[2]["text"].startswith("[Task state]"))
        self.assertIn("file-0.txt", result[2]["text"])
        assert_tool_integrity(self, result)
        self.assertEqual(estimate_tokens(result), events[-1].after_tokens)
        self.assertLessEqual(events[-1].after_tokens or 0, 1500)
        self.assertTrue(events[-1].compaction_succeeded)
        self.assertGreater(events[-1].summarized_messages, 0)
        self.assertGreaterEqual(
            events[-1].preserved_messages, KEEP_RECENT
        )

    def test_recent_slice_never_starts_with_a_bare_tool_result(self) -> None:
        messages = long_messages(20)
        result = compact(
            "model",
            messages,
            1500,
            provider=FakeProvider(),
            task_state=TaskState(goal="Create the files and verify them."),
        )
        # The original task and the two synthetic user messages come first.
        recent = result[3:]
        self.assertEqual(recent[0]["role"], "assistant")
        assert_tool_integrity(self, result)

    def test_preservation_boundary_moves_when_all_groups_are_recent(self) -> None:
        messages = long_messages(2)
        result = compact(
            "model",
            messages,
            186,
            provider=FakeProvider(),
            task_state=TaskState(goal="Create the files and verify them."),
        )
        summary = next(
            message
            for message in result
            if message.get("text", "").startswith(SUMMARY_PREFIX)
        )
        self.assertIn("earlier", summary["text"])
        assert_tool_integrity(self, result, require_calls=False)

    def test_orphaned_tool_result_stops_compaction(self) -> None:
        messages = [
            {"role": "user", "text": "Do work."},
            {
                "role": "tool",
                "tool_call_id": "missing",
                "name": "write_file",
                "text": "Wrote 1 char to x",
            },
        ]
        events: list[RunEvent] = []
        provider = FakeProvider()
        with self.assertRaises(CompactionError):
            compact(
                "model",
                messages,
                1,
                provider=provider,
                on_event=events.append,
            )
        self.assertEqual(provider.calls, [])
        self.assertFalse(events[-1].compaction_succeeded)

    def test_pending_tool_call_stops_compaction(self) -> None:
        messages = [
            {"role": "user", "text": "Do work."},
            {
                "role": "assistant",
                "text": "",
                "tool_calls": [
                    {"id": "pending", "name": "write_file", "args": {}}
                ],
            },
        ]
        provider = FakeProvider()
        with self.assertRaises(CompactionError):
            compact("model", messages, 1, provider=provider)
        self.assertEqual(provider.calls, [])

    def test_provider_error_emits_failure_once_and_raises(self) -> None:
        messages = long_messages(5)
        provider = FakeProvider(error=ProviderError("boom"))
        events: list[RunEvent] = []
        with self.assertRaises(CompactionError) as caught:
            compact(
                "model",
                messages,
                400,
                provider=provider,
                on_event=events.append,
            )
        self.assertIn("summarizer", str(caught.exception))
        self.assertEqual(len(provider.calls), 1)
        self.assertFalse(events[-1].compaction_succeeded)
        self.assertIn("summarizer", events[-1].reason)

    def test_oversized_summary_is_reduced_without_recalling_provider(self) -> None:
        messages = long_messages(5)
        provider = FakeProvider("x" * 100_000)
        events: list[RunEvent] = []
        result = compact(
            "model",
            messages,
            400,
            provider=provider,
            on_event=events.append,
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(events[-1].compaction_succeeded)
        self.assertLessEqual(events[-1].after_tokens or 0, 400)
        self.assertTrue(
            any("summary truncated" in item.get("text", "") for item in result)
        )

    def test_fixed_context_reserve_counts_against_the_budget(self) -> None:
        messages = long_messages(5)
        provider = FakeProvider()
        events: list[RunEvent] = []
        compact(
            "model",
            messages,
            500,
            provider=provider,
            reserve_tokens=100,
            on_event=events.append,
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertGreaterEqual(events[-1].before_tokens or 0, 500)
        self.assertLessEqual(events[-1].after_tokens or 0, 500)
        self.assertTrue(events[-1].compaction_succeeded)

    def test_reserve_alone_over_budget_fails_without_provider(self) -> None:
        provider = FakeProvider()
        events: list[RunEvent] = []
        with self.assertRaises(CompactionError):
            compact(
                "model",
                [{"role": "user", "text": "hi"}],
                100,
                provider=provider,
                reserve_tokens=100,
                on_event=events.append,
            )
        self.assertEqual(provider.calls, [])
        self.assertFalse(events[-1].compaction_succeeded)

    def test_impossible_budget_fails_clearly(self) -> None:
        messages = long_messages(5)
        provider = FakeProvider()
        events: list[RunEvent] = []
        with self.assertRaises(CompactionError):
            compact(
                "model",
                messages,
                30,
                provider=provider,
                on_event=events.append,
            )
        self.assertFalse(events[-1].compaction_succeeded)
        self.assertEqual(len(provider.calls), 1)

    def test_empty_summary_and_non_stop_finish_are_rejected(self) -> None:
        messages = long_messages(5)
        for provider in (
            FakeProvider(""),
            FakeProvider("summary", finish_reason="length"),
        ):
            with self.subTest(provider=provider):
                events: list[RunEvent] = []
                with self.assertRaises(CompactionError):
                    compact(
                        "model",
                        messages,
                        400,
                        provider=provider,
                        on_event=events.append,
                    )
                self.assertFalse(events[-1].compaction_succeeded)
                self.assertEqual(len(provider.calls), 1)

    def test_secret_text_is_redacted_before_summary(self) -> None:
        messages = [
            {"role": "user", "text": "Use DEEPSEEK_API_KEY=sk-abcdefghijklmnopqrstuvwx"},
        ]
        for index in range(5):
            messages.extend(message_pair(index))
        messages.append({"role": "assistant", "text": "Done."})
        provider = FakeProvider("Summary.")
        compact("model", messages, 400, provider=provider)
        prompt = provider.calls[0]["messages"][0]["text"]
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", prompt)
        self.assertIn("[REDACTED]", prompt)


class TaskStateTests(unittest.TestCase):
    def test_observer_records_verified_files_and_errors(self) -> None:
        state = TaskState(goal="Build it.")
        observer = TaskStateObserver(state)
        write_call = ToolCall(
            id="w1",
            name="write_file",
            args={"path": "a.txt", "content": "x"},
        )
        observer(
            RunEvent(
                type=EventType.TOOL_END,
                turn=1,
                tool_call=write_call,
                result="Wrote 1 chars to a.txt",
            )
        )
        error_call = ToolCall(
            id="b1",
            name="bash",
            args={"command": "false"},
        )
        observer(
            RunEvent(
                type=EventType.TOOL_END,
                turn=2,
                tool_call=error_call,
                result="ERROR: RuntimeError: failed",
            )
        )
        observer(
            RunEvent(
                type=EventType.RUN_END,
                turn=2,
                status=RunStatus.MAX_TURNS,
                reason="turns",
            )
        )
        data = state.as_dict()
        self.assertEqual(data["goal"], "Build it.")
        self.assertIn("a.txt", data["modified_files"])
        self.assertTrue(any("write_file" in step for step in data["completed_steps"]))
        self.assertTrue(any("ERROR" in error for error in data["unresolved_errors"]))
        self.assertTrue(any("max_turns" in error for error in data["unresolved_errors"]))

    def test_task_state_is_bounded_redacted_and_renderable(self) -> None:
        state = TaskState(goal="Goal")
        for index in range(60):
            state.add_completed(f"step {index}")
        for index in range(30):
            state.add_error(f"error {index} with password=hunter2")
        self.assertLessEqual(len(state.completed_steps), 50)
        self.assertLessEqual(len(state.unresolved_errors), 20)
        rendered = state.render()
        self.assertIn("Goal: Goal", rendered)
        self.assertIn("Completed:", rendered)
        self.assertNotIn("hunter2", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_empty_goal_renders_a_placeholder(self) -> None:
        self.assertIn("(not set)", TaskState().render())


if __name__ == "__main__":
    unittest.main()
