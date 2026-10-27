"""Stage 3: tests for Harness composition, recovery, and delegation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from src.back.harness import Harness, HarnessError
from src.back.harness.context_management.context import (
    COMPACTION_INSTRUCTION,
)
from src.back.harness.loop.state import RunStatus
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.session import (
    SessionError,
    append,
    latest,
    load,
    new_session,
    recovery_path,
)
from src.back.harness.tools_and_guardrails import (
    LocalWorkspace,
    Policy,
    SandboxUnavailable,
    Tool,
)


class ScriptedProvider:
    """Provider with a queue of structured responses."""

    default_model = "fake-model"

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
            raise AssertionError("provider called more times than scripted")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def response(
    text: str = "",
    calls: list[ToolCall] | None = None,
    usage: tuple[int, int] = (1, 1),
) -> ProviderResponse:
    """Build a scripted provider response."""
    calls = list(calls or [])
    return ProviderResponse(
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=TokenUsage(*usage),
    )


def call(
    call_id: str,
    name: str,
    args: dict | None = None,
) -> ToolCall:
    """Build a tool call."""
    return ToolCall(id=call_id, name=name, args=args or {})


class UnavailableBackend:
    """Workspace that fails like an unavailable OpenShell runtime."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = str(workspace)
        self.platform_name = "Linux"

    def _fail(self, *args, **kwargs):
        raise SandboxUnavailable("sandbox is unavailable")

    read_file = _fail
    write_file = _fail
    list_files = _fail
    exec = _fail


class HarnessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.state = self.root / "state"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def make_harness(self, provider, **overrides) -> Harness:
        options = {
            "workdir": str(self.workspace),
            "model": "fake-model",
            "provider": provider,
            "policy": Policy("yolo"),
            "workspace_backend": LocalWorkspace(self.workspace),
            "state_dir": self.state,
        }
        options.update(overrides)
        return Harness(**options)


class ConstructionTests(HarnessTestCase):
    def test_registers_existing_tools_and_defaults_to_safe(self) -> None:
        harness = Harness(
            workdir=str(self.workspace),
            model="fake-model",
            provider=ScriptedProvider([]),
            workspace_backend=LocalWorkspace(self.workspace),
            state_dir=self.state,
        )
        names = set(harness.tools)
        self.assertTrue(
            {
                "read_file",
                "write_file",
                "edit_file",
                "bash",
                "list_files",
                "grep",
                "remember",
                "use_skill",
                "spawn_agent",
            }.issubset(names)
        )
        self.assertEqual(harness.policy.mode, "safe")
        self.assertIn("You are Irina", harness.system_prompt)
        self.assertIn(self.workspace.as_posix(), harness.system_prompt.replace("\\", "/"))

    def test_extra_tools_merge_but_duplicates_are_rejected(self) -> None:
        custom = Tool(
            name="custom_tool",
            spec={
                "schema": {
                    "name": "custom_tool",
                    "description": "custom",
                    "parameters": {"type": "object", "properties": {}},
                }
            },
            run=lambda: "ok",
        )
        harness = self.make_harness(
            ScriptedProvider([]), extra_tools=[custom]
        )
        self.assertIn("custom_tool", harness.tools)

        duplicate = Tool(
            name="read_file",
            spec={
                "schema": {
                    "name": "read_file",
                    "description": "duplicate",
                    "parameters": {"type": "object", "properties": {}},
                }
            },
            run=lambda path: "x",
        )
        broken = self.make_harness(
            ScriptedProvider([]), extra_tools=[duplicate]
        )
        with self.assertRaises(HarnessError):
            _ = broken.tools


class RunAndSessionTests(HarnessTestCase):
    def test_plain_text_run_creates_completed_session(self) -> None:
        provider = ScriptedProvider([response("hello", usage=(2, 3))])
        harness = self.make_harness(provider)
        result = harness.run("say hello")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.text, "hello")
        self.assertIsNotNone(harness.session_path)
        self.assertFalse(
            str(harness.session_path).startswith(str(self.workspace))
        )
        data = load(harness.session_path)
        self.assertEqual(data.status, "completed")
        self.assertEqual(data.usage, TokenUsage(2, 3))
        self.assertEqual(
            [message["role"] for message in data.messages],
            ["user", "assistant"],
        )
        self.assertEqual(latest(self.workspace, state_dir=self.state), harness.session_path)
        self.assertFalse(harness.resume())
        harness.close()

    def test_tool_call_is_executed_and_session_keeps_ids(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call(
                            "call-1",
                            "write_file",
                            {"path": "a.txt", "content": "hello"},
                        )
                    ]
                ),
                response("done"),
            ]
        )
        harness = self.make_harness(provider)
        result = harness.run("write a.txt")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(
            (self.workspace / "a.txt").read_text(encoding="utf-8"), "hello"
        )
        data = load(harness.session_path)
        self.assertEqual(data.status, "completed")
        assistant = next(
            message for message in data.messages if message["role"] == "assistant"
        )
        tool = next(
            message for message in data.messages if message["role"] == "tool"
        )
        self.assertEqual(assistant["tool_calls"][0]["id"], "call-1")
        self.assertEqual(tool["tool_call_id"], "call-1")
        self.assertIn("a.txt", data.task_state.modified_files)
        harness.close()

    def test_persist_false_creates_no_session(self) -> None:
        provider = ScriptedProvider([response("hello")])
        harness = self.make_harness(provider, persist=False)
        harness.run("hello")
        self.assertIsNone(harness.session_path)
        self.assertFalse(self.state.exists() and list(self.state.rglob("*.jsonl")))
        harness.close()


class ResumeTests(HarnessTestCase):
    def _interrupted_session(self) -> Path:
        path = new_session(
            self.workspace,
            "interrupted",
            state_dir=self.state,
            sandbox_workspace=str(self.workspace),
        )
        append(
            path,
            {"type": "message", "message": {"role": "user", "text": "create part1"}},
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
                            "id": "call-1",
                            "name": "write_file",
                            "args": {"path": "part1.txt", "content": "partial"},
                        }
                    ],
                },
            },
        )
        append(
            path,
            {
                "type": "tool_start",
                "call_id": "call-1",
                "name": "write_file",
                "args": {"path": "part1.txt", "content": "partial"},
            },
        )
        append(
            path,
            {
                "type": "task_state",
                "state": {
                    "goal": "create part1",
                    "completed_steps": [],
                    "pending_steps": [],
                    "modified_files": [],
                    "unresolved_errors": [],
                },
            },
        )
        return path

    def test_resume_repairs_unknown_outcome_and_continues(self) -> None:
        path = self._interrupted_session()
        provider = ScriptedProvider([response("continued")])
        harness = self.make_harness(provider)
        self.assertTrue(harness.resume(path))
        self.assertEqual(harness.session_status, "interrupted")
        tool_messages = [
            message for message in harness.messages if message.get("role") == "tool"
        ]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call-1")
        self.assertIn("may or may not have completed", tool_messages[0]["text"])
        self.assertTrue(harness.task_state.unresolved_errors)
        self.assertFalse((self.workspace / "part1.txt").exists())

        result = harness.run("Continue the task.")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.text, "continued")
        self.assertFalse((self.workspace / "part1.txt").exists())
        self.assertTrue(recovery_path(path).exists())
        data = load(path)
        self.assertEqual(data.status, "completed")
        self.assertTrue(
            any(event.get("outcome") == "unknown" for event in data.recovery_events)
        )
        harness.close()

    def test_resume_latest_and_wrong_workspace(self) -> None:
        path = self._interrupted_session()
        provider = ScriptedProvider([response("continued")])
        harness = self.make_harness(provider)
        self.assertTrue(harness.resume())
        harness.close()

        other = self.root / "other"
        other.mkdir()
        wrong = Harness(
            workdir=str(other),
            model="fake-model",
            provider=ScriptedProvider([]),
            policy=Policy("yolo"),
            workspace_backend=LocalWorkspace(other),
            state_dir=self.state,
        )
        with self.assertRaises(SessionError):
            wrong.resume(path)
        self.assertFalse(wrong.resume())
        wrong.close()

    def test_completed_session_is_not_resumed_again(self) -> None:
        provider = ScriptedProvider([response("hello")])
        harness = self.make_harness(provider)
        harness.run("hello")
        with self.assertRaises(SessionError):
            harness.run("again")
        harness.close()


class SubagentHarnessTests(HarnessTestCase):
    def test_two_calls_use_one_parent_session_and_fresh_children(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call(
                            "spawn-1",
                            "spawn_agent",
                            {"task": "Write utils.py with slugify."},
                        )
                    ]
                ),
                response(
                    calls=[
                        call(
                            "write-1",
                            "write_file",
                            {
                                "path": "utils.py",
                                "content": "def slugify(text):\n    return text.lower()\n",
                            },
                        )
                    ]
                ),
                response("utils.py created"),
                response("parent integrated"),
            ]
        )
        harness = self.make_harness(provider)
        result = harness.run("Delegate utils.py.")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.text, "parent integrated")
        self.assertTrue((self.workspace / "utils.py").exists())
        sessions = list(self.state.rglob("*.jsonl"))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(provider.calls[1]["messages"][0]["text"], "Write utils.py with slugify.")
        self.assertEqual(provider.calls[3]["messages"][0]["text"], "Delegate utils.py.")
        data = load(harness.session_path)
        spawn_result = next(
            message["text"]
            for message in data.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "spawn-1"
        )
        self.assertIn("utils.py created", spawn_result)
        harness.close()

    def test_child_inherits_restrictive_policy(self) -> None:
        seen: list[str] = []

        def approver(tool_call, reason):
            seen.append(tool_call.name)
            return tool_call.name == "spawn_agent"

        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call(
                            "spawn-1",
                            "spawn_agent",
                            {"task": "Write blocked.txt."},
                        )
                    ]
                ),
                response(
                    calls=[
                        call(
                            "write-1",
                            "write_file",
                            {"path": "blocked.txt", "content": "no"},
                        )
                    ]
                ),
                response("child was blocked"),
                response("parent noted the block"),
            ]
        )
        harness = self.make_harness(
            provider, policy=Policy("safe", approver=approver)
        )
        result = harness.run("Delegate a write.")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertFalse((self.workspace / "blocked.txt").exists())
        self.assertIn("write_file", seen)
        self.assertIn("spawn_agent", seen)
        data = load(harness.session_path)
        spawn_result = next(
            message["text"]
            for message in data.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "spawn-1"
        )
        self.assertIn("child was blocked", spawn_result)
        harness.close()

    def test_unavailable_backend_returns_tool_error_not_host_fallback(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call(
                            "write-1",
                            "write_file",
                            {"path": "host-fallback.txt", "content": "no"},
                        )
                    ]
                ),
                response("observed failure"),
            ]
        )
        harness = self.make_harness(
            provider,
            workspace_backend=UnavailableBackend(self.workspace),
        )
        result = harness.run("Try a write.")
        self.assertIs(result.status, RunStatus.COMPLETED)
        self.assertFalse((self.workspace / "host-fallback.txt").exists())
        data = load(harness.session_path)
        tool = next(message for message in data.messages if message["role"] == "tool")
        self.assertIn("ERROR: SandboxUnavailable", tool["text"])
        harness.close()


class ChildLimitTests(HarnessTestCase):
    def test_child_turn_limit_is_reported_in_parent_session(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call("spawn-1", "spawn_agent", {"task": "loop forever"})
                    ]
                ),
                response(
                    calls=[
                        call("child-read", "read_file", {"path": "missing.txt"})
                    ]
                ),
                response("parent completed"),
            ]
        )
        harness = self.make_harness(provider, child_max_turns=1)
        result = harness.run("delegate")
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "parent completed")
        data = load(harness.session_path)
        spawn = next(
            message
            for message in data.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "spawn-1"
        )
        self.assertIn("ERROR: sub-agent finished", spawn["text"])
        self.assertIn("max_turns", spawn["text"])
        self.assertEqual(len(provider.calls), 3)
        harness.close()

    def test_child_time_limit_stops_before_child_provider_call(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call("spawn-1", "spawn_agent", {"task": "slow task"})
                    ]
                ),
                response("parent completed"),
            ]
        )
        harness = self.make_harness(provider, child_max_seconds=0)
        result = harness.run("delegate")
        self.assertTrue(result.completed)
        self.assertEqual(len(provider.calls), 2)
        data = load(harness.session_path)
        spawn = next(
            message
            for message in data.messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "spawn-1"
        )
        self.assertIn("timeout", spawn["text"])
        harness.close()

    def test_total_delegated_task_limit_blocks_second_child(self) -> None:
        provider = ScriptedProvider(
            [
                response(
                    calls=[
                        call("spawn-1", "spawn_agent", {"task": "first"})
                    ]
                ),
                response("first child done"),
                response(
                    calls=[
                        call("spawn-2", "spawn_agent", {"task": "second"})
                    ]
                ),
                response("parent done"),
            ]
        )
        harness = self.make_harness(provider, max_subagent_tasks=1)
        result = harness.run("delegate twice")
        self.assertTrue(result.completed)
        data = load(harness.session_path)
        results = {
            message["tool_call_id"]: message["text"]
            for message in data.messages
            if message.get("role") == "tool"
        }
        self.assertIn("[sub-agent completed]", results["spawn-1"])
        self.assertIn("task limit", results["spawn-2"])
        self.assertEqual(len(provider.calls), 4)
        harness.close()


class CompactionResumeTests(HarnessTestCase):
    def test_compaction_keeps_durable_history_and_task_state_on_resume(self) -> None:
        path = new_session(
            self.workspace,
            "long",
            state_dir=self.state,
            sandbox_workspace=str(self.workspace),
        )
        append(
            path,
            {"type": "message", "message": {"role": "user", "text": "long task"}},
        )
        for index in range(40):
            call_id = f"c{index}"
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
                                "args": {
                                    "path": f"f{index}.txt",
                                    "content": "ping\n" * 20,
                                },
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
                    "args": {"path": f"f{index}.txt"},
                },
            )
            append(
                path,
                {
                    "type": "tool_end",
                    "call_id": call_id,
                    "name": "write_file",
                    "result": f"Wrote 100 chars to f{index}.txt",
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
                        "text": f"Wrote 100 chars to f{index}.txt",
                    },
                },
            )
        append(
            path,
            {
                "type": "task_state",
                "state": {
                    "goal": "long task",
                    "completed_steps": ["many", "more"],
                    "pending_steps": [],
                    "modified_files": [],
                    "unresolved_errors": [],
                },
            },
        )

        provider = ScriptedProvider(
            [
                response("compacted summary", usage=(10, 5)),
                response("done", usage=(5, 2)),
            ]
        )
        harness = self.make_harness(provider, budget_tokens=4200)
        self.assertTrue(harness.resume(path))
        result = harness.run("Continue the task.")
        self.assertTrue(result.completed)
        self.assertEqual(result.text, "done")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            provider.calls[0]["system"], COMPACTION_INSTRUCTION
        )
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertLess(len(harness.messages), 83)
        data = load(path)
        self.assertEqual(data.status, "completed")
        self.assertEqual(data.task_state.goal, "long task")
        self.assertTrue(
            any(
                record["type"] == "context_compacted"
                and record["success"]
                for record in data.records
            )
        )
        self.assertGreaterEqual(len(data.messages), 83)
        self.assertTrue(
            any(
                message.get("role") == "user"
                and message.get("text") == "long task"
                for message in data.messages
            )
        )
        harness.close()


if __name__ == "__main__":
    unittest.main()
