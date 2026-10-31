"""Current-run experience boundaries and bounded-record tests.

These tests use synthetic transcripts and fake providers only. They never read
the user's session directory or contact a provider.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.back.harness.context_management.context import (
    SUMMARY_PREFIX,
    compact,
)
from src.back.harness.context_management.memory.experience_store import (
    MAX_RECORD_CHARS,
    ExperienceRecord,
    ExperienceStore,
    _bounded_record,
)
from src.back.harness.context_management.memory.trajectory_bridge import (
    record_from_result,
    upstream_payload,
)
from src.back.harness.loop.runner import EventType, run_loop
from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.orchestration.harness import Harness
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy
from src.back.harness.tools_and_guardrails.tools import Tool

PROBE_SPEC = {
    "name": "probe",
    "description": "Return a deterministic observation.",
    "parameters": {
        "type": "object",
        "properties": {"tag": {"type": "string"}},
        "required": ["tag"],
    },
}


def probe(tag: str) -> str:
    return f"OBS-{tag}"


PROBE = Tool(name="probe", spec={"schema": PROBE_SPEC}, run=probe)


class ScriptedProvider:
    """Ask for one probe call, then answer once the tool result arrives."""

    def complete(self, model, system, messages, tools):
        last = messages[-1] if messages else {}
        if last.get("role") == "user":
            tag = str(last.get("text", "")).strip().split()[-1] or "0"
            return ProviderResponse(
                "",
                [ToolCall(f"call-{tag}", "probe", {"tag": tag})],
                "tool_calls",
                TokenUsage(1, 1),
            )
        return ProviderResponse("done", [], "stop", TokenUsage(1, 1))


class TextProvider:
    """Always answer without calling tools."""

    def complete(self, model, system, messages, tools):
        return ProviderResponse("just text", [], "stop", TokenUsage(1, 1))


class RecordingMemory:
    """Capture experiences instead of running MemEvolve."""

    def __init__(self, fail: bool = False) -> None:
        self.records: list[ExperienceRecord] = []
        self.fail = fail

    def before_task(self, task: str) -> str:
        return ""

    def after_task(self, record: ExperienceRecord):
        if self.fail:
            raise RuntimeError("memory backend unavailable")
        self.records.append(record)
        return None


def make_harness(
    temp: str | Path,
    memory: RecordingMemory,
    *,
    provider=None,
) -> Harness:
    workspace = Path(temp) / "workspace"
    workspace.mkdir()
    return Harness(
        workdir=workspace,
        provider=provider or ScriptedProvider(),
        policy=Policy("yolo"),
        workspace_backend=LocalWorkspace(workspace),
        state_dir=Path(temp) / "state",
        memory_manager=memory,
        extra_tools=[PROBE],
        enable_subagents=False,
    )


def make_result(
    messages: list[dict],
    *,
    status: RunStatus = RunStatus.COMPLETED,
    reason: str = "done",
) -> RunResult:
    return RunResult(
        status=status,
        text="",
        messages=messages,
        turns=1,
        tool_calls=0,
        usage=TokenUsage(1, 1),
        reason=reason,
    )


def task_turn(tag: str) -> list[dict]:
    return [
        {"role": "user", "text": f"task {tag}"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"id": f"call-{tag}", "name": "probe", "args": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": f"call-{tag}",
            "name": "probe",
            "text": f"OBS-{tag}",
        },
        {"role": "assistant", "text": "done"},
    ]


class CurrentRunBoundaryTests(unittest.TestCase):
    def test_current_task_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = RecordingMemory()
            harness = make_harness(temp, memory)
            harness.run("task 1")
            harness.continue_task("task 2")
            harness.continue_task("task 3")
            harness.close()

        self.assertEqual(len(memory.records), 3)
        self.assertEqual(
            [o["text"] for o in memory.records[0].tool_observations],
            ["OBS-1"],
        )
        self.assertEqual(
            [o["text"] for o in memory.records[2].tool_observations],
            ["OBS-3"],
        )

    def test_continued_session_excludes_previous_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = RecordingMemory()
            harness = make_harness(temp, memory)
            harness.run("task 1")
            harness.continue_task("task 2")
            path = harness.session_path
            harness.close()

            self.assertIsNotNone(path)
            second = memory.records[1]
            texts = {o["text"] for o in second.tool_observations}
            self.assertEqual(texts, {"OBS-2"})
            self.assertEqual(second.task_description, "task 2")

    def test_restored_session_keeps_current_run_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = RecordingMemory()
            harness = make_harness(temp, memory)
            harness.run("task 1")
            path = harness.session_path
            harness.close()

            restored_memory = RecordingMemory()
            restored = Harness(
                workdir=Path(temp) / "workspace",
                provider=ScriptedProvider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(Path(temp) / "workspace"),
                state_dir=Path(temp) / "state",
                session_path=path,
                memory_manager=restored_memory,
                extra_tools=[PROBE],
                enable_subagents=False,
            )
            restored.continue_task("task 2")
            restored.close()

        texts = [o["text"] for o in restored_memory.records[0].tool_observations]
        self.assertEqual(texts, ["OBS-2"])

    def test_no_tool_execution_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = RecordingMemory()
            harness = make_harness(
                temp, memory, provider=TextProvider()
            )
            result = harness.run("just answer")
            harness.close()

        self.assertTrue(result.completed)
        record = memory.records[0]
        self.assertEqual(record.tool_actions, [])
        self.assertEqual(record.tool_observations, [])

    def test_compaction_does_not_leak_history(self) -> None:
        class Summarizer:
            def complete(self, model, system, messages, tools):
                return ProviderResponse(
                    "SUMMARY of earlier turns", [], "stop", TokenUsage(1, 1)
                )

        class BigProbeProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, model, system, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return ProviderResponse(
                        "",
                        [ToolCall("c1", "probe", {"tag": "B" * 5_000})],
                        "tool_calls",
                        TokenUsage(1, 1),
                    )
                return ProviderResponse("done", [], "stop", TokenUsage(1, 1))

        events: list = []

        def before_turn(current):
            return compact(
                "m",
                current,
                budget_tokens=200,
                provider=Summarizer(),
                on_event=events.append,
            )

        messages = [{"role": "user", "text": "task 1"}]
        run_messages: list[dict] = []
        result = run_loop(
            provider=BigProbeProvider(),
            model="m",
            system="",
            messages=messages,
            tools={"probe": PROBE},
            before_turn=before_turn,
            run_messages=run_messages,
        )
        self.assertEqual(result.status, RunStatus.COMPLETED)

        # A compaction really happened (first turn copied, later turns compacted).
        self.assertTrue(
            any(
                getattr(event, "type", None) is EventType.CONTEXT_COMPACTED
                for event in events
            )
        )
        self.assertTrue(
            any(
                str(message.get("text", "")).startswith(SUMMARY_PREFIX)
                for message in messages
            )
        )

        record = record_from_result(
            task="task 1",
            result=result,
            workspace="C:/workspace",
            run_messages=run_messages,
        )
        self.assertEqual(record.execution_outcome, RunStatus.COMPLETED.value)
        self.assertFalse(
            any(
                str(message.get("text", "")).startswith(SUMMARY_PREFIX)
                for message in run_messages
            )
        )
        texts = [o["text"] for o in record.tool_observations]
        self.assertNotIn("OBS-2", texts)
        self.assertTrue(all(t.startswith("OBS-") for t in texts))

    def test_long_session_does_not_grow_records(self) -> None:
        def history(pairs: int) -> list[dict]:
            messages: list[dict] = []
            for index in range(pairs):
                messages.extend(task_turn(f"h{index}"))
            return messages

        tail = task_turn("current")
        short = make_result(history(2) + tail)
        long = make_result(history(100) + tail)
        run_messages = [m for m in tail if m.get("role") != "user"]

        short_record = record_from_result(
            task="task current",
            result=short,
            workspace="C:/workspace",
            session_id="session-1",
            run_index=1,
            run_messages=run_messages,
        )
        long_record = record_from_result(
            task="task current",
            result=long,
            workspace="C:/workspace",
            session_id="session-1",
            run_index=1,
            run_messages=run_messages,
        )
        self.assertEqual(
            [o["text"] for o in long_record.tool_observations], ["OBS-current"]
        )
        short_data = _bounded_record(short_record)
        long_data = _bounded_record(long_record)
        short_data.pop("timestamp", None)
        long_data.pop("timestamp", None)
        self.assertEqual(long_data, short_data)
        self.assertLess(
            len(json.dumps(long_data, ensure_ascii=False, sort_keys=True)),
            MAX_RECORD_CHARS,
        )

    def test_memory_failure_does_not_fail_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = RecordingMemory(fail=True)
            harness = make_harness(temp, memory)
            result = harness.run("task 1")
            harness.close()
        self.assertTrue(result.completed)


class BoundedRecordTests(unittest.TestCase):
    def make_record(self, observations: list[str]) -> ExperienceRecord:
        return ExperienceRecord(
            experience_id="exp-1",
            session_id="s",
            task_description="the task",
            relevant_context="context",
            execution_outcome="completed",
            run_status="completed",
            tool_observations=[
                {"tool_call_id": f"c{i}", "name": "bash", "text": text}
                for i, text in enumerate(observations)
            ],
        )

    def test_single_large_observation_is_bounded(self) -> None:
        record = self.make_record(["x" * 50_000])
        bounded = _bounded_record(record)
        self.assertLessEqual(
            len(json.dumps(bounded, ensure_ascii=False, sort_keys=True)),
            MAX_RECORD_CHARS,
        )
        self.assertTrue(bounded["truncated"])
        self.assertIn("tool_observations", bounded["truncated_fields"])
        self.assertEqual(bounded["task_description"], "the task")
        self.assertEqual(bounded["execution_outcome"], "completed")

    def test_multiple_large_observations_fit(self) -> None:
        record = self.make_record(["y" * 20_000 for _ in range(10)])
        bounded = _bounded_record(record)
        self.assertLessEqual(
            len(json.dumps(bounded, ensure_ascii=False, sort_keys=True)),
            MAX_RECORD_CHARS,
        )
        self.assertTrue(bounded["truncated"])

    def test_duplicate_observations_are_dropped(self) -> None:
        record = self.make_record(["same", "same", "different"])
        bounded = _bounded_record(record)
        texts = [o["text"] for o in bounded["tool_observations"]]
        self.assertEqual(texts, ["same", "different"])
        self.assertIn("tool_observations", bounded["truncated_fields"])

    def test_memevolve_payload_uses_current_run(self) -> None:
        result = make_result(task_turn("old") + task_turn("new"))
        record = record_from_result(
            task="task new",
            result=result,
            workspace="C:/workspace",
            run_messages=[
                m for m in task_turn("new") if m.get("role") != "user"
            ],
        )
        payload = upstream_payload(record)
        self.assertEqual(payload["query"], "task new")
        self.assertEqual(payload["result"], "completed")
        self.assertIn("OBS-new", json.dumps(payload))
        self.assertNotIn("OBS-old", json.dumps(payload))

    def test_old_jsonl_records_still_load(self) -> None:
        legacy = {
            "experience_id": "legacy",
            "schema_version": 1,
            "task_description": "old task",
            "execution_outcome": "completed",
        }
        record = ExperienceRecord.from_dict(legacy)
        self.assertFalse(record.truncated)
        self.assertEqual(record.truncated_fields, [])

    def test_truncation_metadata_round_trips_through_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            store.add(self.make_record(["z" * 20_000]))
            loaded = ExperienceStore(temp, "C:/workspace").load()
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].truncated)
        self.assertIn("tool_observations", loaded[0].truncated_fields)


if __name__ == "__main__":
    unittest.main()
