"""Stage 3: tests for durable sessions, JSONL repair, and recording."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.back.harness.context_management.context import TaskState
from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.providers.base import TokenUsage, ToolCall
from src.back.harness.orchestration.session import (
    SessionError,
    SessionLocked,
    SessionRecorder,
    append,
    latest,
    load,
    new_session,
    recovery_path,
    record_resume,
    workspace_session_dir,
)


class SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.state_dir = self.root / "state"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def make_session(self, label: str = "test") -> Path:
        return new_session(
            self.workspace,
            label,
            state_dir=self.state_dir,
            sandbox_name=None,
            sandbox_workspace=str(self.workspace),
        )

    def append_pair(
        self,
        path: Path,
        index: int = 0,
        *,
        finish: bool = True,
    ) -> None:
        call_id = f"call-{index}"
        append(
            path,
            {
                "type": "message",
                "message": {"role": "user", "text": f"task {index}"},
            },
        )
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": "write_file",
                            "args": {"path": f"file-{index}.txt"},
                        }
                    ],
                },
            },
        )
        append(
            path,
            {
                "type": "tool_start",
                "call_id": call_id,
                "name": "write_file",
                "args": {"path": f"file-{index}.txt"},
            },
        )
        if finish:
            append(
                path,
                {
                    "type": "tool_end",
                    "call_id": call_id,
                    "name": "write_file",
                    "result": f"Wrote 1 chars to file-{index}.txt",
                },
            )
            append(
                path,
                {
                    "type": "message",
                    "message": {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "write_file",
                        "text": f"Wrote 1 chars to file-{index}.txt",
                    },
                },
            )


class SessionCreationTests(SessionTestCase):
    def test_new_session_uses_slug_and_trusted_directory(self) -> None:
        path = new_session(
            self.workspace,
            "Build Python Utility!",
            state_dir=self.state_dir,
        )
        self.assertTrue(path.is_file())
        self.assertIn("build-python-utility", path.name)
        self.assertEqual(path.parent, workspace_session_dir(
            self.workspace, state_dir=self.state_dir
        ))
        records = load(path).records
        self.assertEqual(records[0]["type"], "session_start")
        self.assertEqual(records[0]["workspace"], str(self.workspace.resolve()))

    def test_empty_label_uses_fallback_and_never_overwrites(self) -> None:
        first = new_session(self.workspace, " ", state_dir=self.state_dir)
        self.assertIn("session", first.name)
        with self.assertRaises(SessionError):
            new_session(
                self.workspace,
                "other",
                state_dir=self.state_dir,
                path=first,
            )

    def test_explicit_path_outside_trusted_store_is_rejected(self) -> None:
        outside = self.root / "outside.jsonl"
        with self.assertRaises(SessionError):
            new_session(
                self.workspace,
                "test",
                state_dir=self.state_dir,
                path=outside,
            )

    def test_workspace_directories_are_isolated(self) -> None:
        other = self.root / "other"
        other.mkdir()
        first = new_session(self.workspace, "one", state_dir=self.state_dir)
        second = new_session(other, "two", state_dir=self.state_dir)
        self.assertNotEqual(first.parent, second.parent)
        self.assertEqual(
            latest(self.workspace, state_dir=self.state_dir), first
        )
        self.assertEqual(latest(other, state_dir=self.state_dir), second)


class SessionRecordTests(SessionTestCase):
    def test_append_load_and_status_are_machine_readable(self) -> None:
        path = self.make_session()
        self.append_pair(path, 0)
        state = TaskState(goal="task 0", modified_files=["file-0.txt"])
        append(
            path,
            {"type": "task_state", "state": state.as_dict()},
        )
        append(
            path,
            {
                "type": "run_end",
                "status": "completed",
                "reason": "done",
                "text": "Finished.",
                "turns": 1,
                "tool_calls": 1,
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )
        append(
            path,
            {"type": "task_state", "state": state.as_dict()},
        )

        data = load(path)
        self.assertEqual(data.status, "completed")
        self.assertEqual(data.text, "Finished.")
        self.assertEqual(data.turns, 1)
        self.assertEqual(data.tool_calls, 1)
        self.assertEqual(data.usage, TokenUsage(10, 4))
        self.assertEqual(data.task_state.goal, "task 0")
        self.assertEqual(data.recovery_events, [])
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)

    def test_secrets_are_redacted_in_session_records(self) -> None:
        path = self.make_session()
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "text": "api key: sk-abcdefghijklmnopqrstuvwx",
                },
            },
        )
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", raw)
        self.assertIn("[REDACTED]", raw)

    def test_lock_prevents_concurrent_append(self) -> None:
        path = self.make_session()
        lock = path.with_name(path.name + ".lock")
        lock.write_text("other", encoding="utf-8")
        try:
            with self.assertRaises(SessionLocked):
                append(
                    path,
                    {"type": "message", "message": {"role": "user", "text": "x"}},
                )
        finally:
            lock.unlink()


class RecoveryTests(SessionTestCase):
    def test_torn_final_record_is_ignored_and_truncated(self) -> None:
        path = self.make_session()
        self.append_pair(path, 0)
        with path.open("ab") as handle:
            handle.write(b'{"type": "message", "message": {')
        data = load(path)
        self.assertEqual(data.messages[0]["text"], "task 0")
        self.assertEqual(data.status, "interrupted")
        self.assertEqual(
            len(path.read_text(encoding="utf-8").splitlines()), 6
        )
        load(path)

    def test_middle_corruption_is_not_ignored(self) -> None:
        path = self.make_session()
        self.append_pair(path, 0)
        with path.open("ab") as handle:
            handle.write(b"{broken}\n")
        with self.assertRaises(SessionError):
            load(path)

    def test_missing_tool_result_becomes_unknown_with_matching_id(self) -> None:
        path = self.make_session()
        append(
            path,
            {"type": "message", "message": {"role": "user", "text": "work"}},
        )
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {"id": "call-a", "name": "write_file", "args": {}},
                        {"id": "call-b", "name": "bash", "args": {"command": "x"}},
                    ],
                },
            },
        )
        append(
            path,
            {
                "type": "tool_start",
                "call_id": "call-a",
                "name": "write_file",
                "args": {},
            },
        )
        append(
            path,
            {
                "type": "tool_end",
                "call_id": "call-a",
                "name": "write_file",
                "result": "Wrote 1 chars to x",
            },
        )
        data = load(path)
        results = [m for m in data.messages if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], ["call-a", "call-b"])
        self.assertIn("Wrote 1 chars", results[0]["text"])
        self.assertIn("may or may not have completed", results[1]["text"])
        self.assertIn("tool_call_id: call-b", results[1]["text"])
        self.assertTrue(
            any(
                event.get("call_id") == "call-b"
                and event.get("outcome") == "unknown"
                for event in data.recovery_events
            )
        )
        self.assertTrue(
            any("call-b" in error for error in data.task_state.unresolved_errors)
        )

    def test_confirmed_tool_end_without_message_is_restored(self) -> None:
        path = self.make_session()
        append(
            path,
            {"type": "message", "message": {"role": "user", "text": "work"}},
        )
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {"id": "call-a", "name": "write_file", "args": {}}
                    ],
                },
            },
        )
        append(
            path,
            {
                "type": "tool_end",
                "call_id": "call-a",
                "name": "write_file",
                "result": "Wrote 1 chars to x",
            },
        )
        data = load(path)
        tool = next(m for m in data.messages if m.get("role") == "tool")
        self.assertEqual(tool["tool_call_id"], "call-a")
        self.assertEqual(tool["text"], "Wrote 1 chars to x")
        self.assertFalse(data.recovery_events)

    def test_unconfirmed_tool_start_is_not_replayed(self) -> None:
        path = self.make_session()
        append(
            path,
            {"type": "message", "message": {"role": "user", "text": "work"}},
        )
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {"id": "call-a", "name": "bash", "args": {"command": "x"}}
                    ],
                },
            },
        )
        append(
            path,
            {
                "type": "tool_start",
                "call_id": "call-a",
                "name": "bash",
                "args": {"command": "x"},
            },
        )
        data = load(path)
        tool = next(m for m in data.messages if m.get("role") == "tool")
        self.assertIn("start_recorded: yes", tool["text"])
        self.assertIn("may or may not", tool["text"])
        self.assertEqual(
            [event["outcome"] for event in data.recovery_events], ["unknown"]
        )

    def test_orphaned_or_duplicate_results_raise(self) -> None:
        path = self.make_session()
        append(
            path,
            {
                "type": "message",
                "message": {
                    "role": "tool",
                    "tool_call_id": "none",
                    "name": "x",
                    "text": "x",
                },
            },
        )
        with self.assertRaises(SessionError):
            load(path)

    def test_resume_event_is_separate_from_original_history(self) -> None:
        path = self.make_session()
        self.append_pair(path, 0)
        record_resume(path)
        self.assertTrue(recovery_path(path).exists())
        data = load(path)
        self.assertTrue(
            any(event.get("outcome") == "resumed" for event in data.recovery_events)
        )
        self.assertEqual(
            data.records[-1]["type"], "message"
        )


class LatestTests(SessionTestCase):
    def test_latest_selects_newest_and_skips_child_sessions(self) -> None:
        first = self.make_session("first")
        self.append_pair(first, 0)
        time.sleep(0.02)
        child = new_session(
            self.workspace,
            "child",
            state_dir=self.state_dir,
            metadata={"depth": 1, "parent": "parent-id"},
        )
        time.sleep(0.02)
        second = self.make_session("second")
        self.assertEqual(
            latest(self.workspace, state_dir=self.state_dir), second
        )

    def test_resumable_only_skips_completed_session(self) -> None:
        older = self.make_session("older")
        self.append_pair(older, 0)
        time.sleep(0.02)
        completed = self.make_session("completed")
        self.append_pair(completed, 1)
        append(
            completed,
            {
                "type": "run_end",
                "status": "completed",
                "reason": "done",
                "text": "done",
                "turns": 1,
                "tool_calls": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        self.assertEqual(
            latest(self.workspace, state_dir=self.state_dir), completed
        )
        self.assertEqual(
            latest(
                self.workspace,
                state_dir=self.state_dir,
                resumable_only=True,
            ),
            older,
        )


class RecorderTests(SessionTestCase):
    def test_recorder_persists_assistant_tools_compaction_and_completion(self) -> None:
        path = self.make_session()
        state = TaskState(goal="work")
        recorder = SessionRecorder(path, state)
        recorder.record_user("work")
        recorder.record_event(
            RunEvent(
                type=EventType.ASSISTANT,
                turn=1,
                text="",
                tool_calls=[
                    ToolCall(id="call-1", name="write_file", args={"path": "x"})
                ],
            )
        )
        recorder.record_event(
            RunEvent(
                type=EventType.TOOL_START,
                turn=1,
                tool_call=ToolCall(
                    id="call-1", name="write_file", args={"path": "x"}
                ),
            )
        )
        state.add_modified("x")
        state.add_completed("write_file: x")
        recorder.record_event(
            RunEvent(
                type=EventType.TOOL_END,
                turn=1,
                tool_call=ToolCall(
                    id="call-1", name="write_file", args={"path": "x"}
                ),
                result="Wrote 1 chars to x",
            )
        )
        recorder.record_event(
            RunEvent(
                type=EventType.CONTEXT_COMPACTED,
                turn=2,
                before_tokens=100,
                after_tokens=50,
                summarized_messages=4,
                preserved_messages=2,
                compaction_succeeded=True,
                reason="compacted",
            )
        )
        result = RunResult(
            status=RunStatus.COMPLETED,
            text="done",
            messages=[],
            turns=1,
            tool_calls=1,
            usage=TokenUsage(10, 2),
            reason="done",
        )
        recorder.record_run_end(result)

        data = load(path)
        self.assertEqual(data.status, "completed")
        self.assertEqual(
            [m["role"] for m in data.messages],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(data.messages[2]["tool_call_id"], "call-1")
        self.assertEqual(data.messages[2]["text"], "Wrote 1 chars to x")
        self.assertEqual(data.task_state.modified_files, ["x"])
        self.assertEqual(data.usage, TokenUsage(10, 2))
        self.assertTrue(
            any(record["type"] == "context_compacted" for record in data.records)
        )

    def test_sync_user_messages_skips_compaction_summaries(self) -> None:
        path = self.make_session()
        recorder = SessionRecorder(path, TaskState(goal="work"))
        recorder.record_user("work")
        recorder.sync_user_messages(
            [
                {"role": "user", "text": "work"},
                {"role": "user", "text": "[Conversation so far, compacted]\nold"},
                {"role": "user", "text": "[Task state]\ngoal"},
                {"role": "user", "text": "Your previous response was empty."},
            ]
        )
        data = load(path)
        user_texts = [
            message["text"]
            for message in data.messages
            if message.get("role") == "user"
        ]
        self.assertEqual(
            user_texts,
            ["work", "Your previous response was empty."],
        )


if __name__ == "__main__":
    unittest.main()
