"""Integration tests for canonical memory, upstream providers, migration, and safety."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.back.harness.context_management.memory import (
    ArchitectureError,
    ArchitectureRecord,
    ArchitectureRegistry,
    ExperienceRecord,
    ExperienceStore,
    MemoryConfig,
    MemoryManager,
    MemoryQuery,
    ReplayManager,
)
from src.back.harness.context_management.memory.adapter import (
    EvolveLabBackend,
    MemoryUpdateResult,
)
from src.back.harness.context_management.memory.evaluation_bridge import (
    CodingTask,
    default_coding_tasks,
    evaluate,
    run_fixture,
)
from src.back.harness.context_management.memory.evolution_bridge import (
    EvolutionBridge,
    EvolutionUnavailable,
    dependency_status,
    export_trajectories,
)
from src.back.harness.context_management.memory.provider_registry import (
    UPSTREAM_COMMIT,
    PROVIDERS,
    probe_all,
    probe_provider,
    upstream_info,
)
from src.back.harness.context_management.memory.retrieval_bridge import (
    RetrievedMemory,
    render_context,
)
from src.back.harness.context_management.memory.security import (
    isolated_import,
    static_validate,
)
from src.back.harness.context_management.memory.trajectory_bridge import (
    record_from_result,
    upstream_payload,
)
from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.orchestration.harness import Harness
from src.back.harness.providers.base import ProviderResponse, TokenUsage, ToolCall
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy


class FakeMemoryModel:
    def __call__(self, messages):
        text = ""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )
            text += str(content)

        class Response:
            pass

        response = Response()
        if "Output (JSON)" in text:
            response.content = (
                '{"strategic":["Inspect the exact failing assertion before editing."],'
                '"operational":["Run the test suite after each fix."]}'
            )
        else:
            response.content = (
                "Historical experience: inspect the failing assertion, patch "
                "the implementation, and rerun the tests."
            )
        return response


def make_result(
    status: RunStatus = RunStatus.COMPLETED,
    *,
    text: str = "done",
    messages: list[dict] | None = None,
) -> RunResult:
    return RunResult(
        status=status,
        text=text,
        messages=messages
        or [
            {"role": "user", "text": "fix the test"},
            {"role": "assistant", "text": "", "tool_calls": [
                {"id": "call_1", "name": "bash", "args": {"command": "python -m unittest"}}
            ]},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "bash",
                "text": "Ran 2 tests\nOK",
            },
            {"role": "assistant", "text": "fixed"},
        ],
        turns=2,
        tool_calls=1,
        usage=TokenUsage(10, 5),
        reason="model produced a final conversational answer",
    )


class CanonicalStoreTests(unittest.TestCase):
    def make_record(self, id_: str = "exp-1", workspace: str = "C:/a") -> ExperienceRecord:
        return ExperienceRecord(
            experience_id=id_,
            session_id="session-1",
            workspace_id="ws",
            timestamp="2026-01-01T00:00:00+00:00",
            task_description="Fix a test",
            execution_outcome="completed",
            verified_success=True,
            run_status="completed",
        )

    def test_idempotent_append_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            self.assertTrue(store.add(self.make_record()))
            self.assertFalse(store.add(self.make_record()))
            self.assertEqual(store.count(), 1)
            reloaded = ExperienceStore(temp, "C:/workspace")
            self.assertEqual(reloaded.count(), 1)
            self.assertEqual(reloaded.load()[0].experience_id, "exp-1")

    def test_torn_final_line_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            store.add(self.make_record())
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write('{"broken":')
            self.assertEqual(store.count(), 1)
            self.assertTrue(store.add(self.make_record("exp-2")))
            self.assertEqual(store.count(), 2)

    def test_workspace_isolation_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = ExperienceStore(temp, "C:/workspace-a")
            second = ExperienceStore(temp, "C:/workspace-b")
            record = self.make_record()
            record.task_description = "token=sk-" + "a" * 32
            first.add(record)
            self.assertEqual(first.count(), 1)
            self.assertEqual(second.count(), 0)
            raw = first.path.read_text(encoding="utf-8")
            self.assertNotIn("sk-" + "a" * 32, raw)

    def test_middle_corruption_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            store.add(self.make_record("a"))
            valid = json.dumps(self.make_record("b").to_dict())
            store.path.write_text(
                store.path.read_text(encoding="utf-8")
                + "not json\n"
                + valid
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                ExperienceStore(temp, "C:/workspace").load()


class TrajectoryTests(unittest.TestCase):
    def test_success_requires_observed_command_evidence(self) -> None:
        record = record_from_result(
            task="fix the test",
            result=make_result(),
            workspace="C:/workspace",
            session_path=Path("session.jsonl"),
        )
        self.assertTrue(record.verified_success)
        self.assertTrue(record.experience_id)
        self.assertEqual(record.tool_actions[0]["id"], "call_1")
        self.assertEqual(record.tool_observations[0]["tool_call_id"], "call_1")
        self.assertEqual(record.test_results[0]["passed"], True)
        payload = upstream_payload(record)
        self.assertTrue(payload["metadata"]["is_correct"])
        self.assertIn("call_1", json.dumps(payload))

    def test_failed_and_unverified_runs_are_labeled(self) -> None:
        failed = record_from_result(
            task="fix",
            result=make_result(RunStatus.TOOL_ERROR, messages=[
                {"role": "user", "text": "fix"},
                {"role": "tool", "tool_call_id": "x", "name": "bash", "text": "FAILED"},
            ]),
            workspace="C:/workspace",
        )
        self.assertFalse(failed.verified_success)
        self.assertEqual(failed.failure_reason, "model produced a final conversational answer")
        unverified = record_from_result(
            task="describe",
            result=make_result(messages=[{"role": "assistant", "text": "done"}]),
            workspace="C:/workspace",
        )
        self.assertFalse(unverified.verified_success)

    def test_retrieval_is_bounded_and_lower_trust(self) -> None:
        text = render_context(
            [
                RetrievedMemory("1", "Use pytest -q."),
                RetrievedMemory("2", "Use pytest -q."),
                RetrievedMemory("3", "Read the assertion."),
            ],
            max_tokens=100,
            provider="lightweight_memory",
        )
        self.assertIn("lower-trust", text)
        self.assertEqual(text.count("Use pytest -q."), 1)
        self.assertIn("Read the assertion.", text)


class ProviderIntegrationTests(unittest.TestCase):
    def test_registry_reports_all_sources_and_dependency_blockers(self) -> None:
        self.assertEqual(len(PROVIDERS), 13)
        statuses = {status.provider_id: status for status in probe_all()}
        self.assertTrue(all(status.source_present for status in statuses.values()))
        lightweight = statuses["lightweight_memory"]
        self.assertTrue(lightweight.import_ok)
        self.assertTrue(lightweight.initialize_ok)
        self.assertTrue(lightweight.as_dict()["source_present"])
        # This environment intentionally lacks optional research dependencies.
        self.assertFalse(statuses["expel"].integration_ok)
        self.assertIn("expel", statuses)
        info = upstream_info()
        self.assertEqual(info["commit"], UPSTREAM_COMMIT)
        self.assertTrue(info["manifest_present"])

    def test_lightweight_upstream_provider_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            backend = EvolveLabBackend(
                "lightweight_memory",
                storage_dir=Path(temp) / "store",
                config={
                    "enable_longterm_provision": True,
                    "top_k_longterm": 3,
                },
                model=FakeMemoryModel(),
            )
            record = ExperienceRecord(
                experience_id="exp-1",
                task_description="Fix the failing test by inspecting the assertion.",
                verified_success=True,
                run_status="completed",
                execution_outcome="completed",
                tool_actions=[{"id": "c1", "name": "bash", "args": {"command": "python -m unittest"}}],
                tool_observations=[{"tool_call_id": "c1", "name": "bash", "text": "OK"}],
                test_results=[{"passed": True}],
            )
            update = backend.add_experience(record)
            self.assertTrue(update.ok, update.error)
            memories = backend.retrieve(MemoryQuery(query="fix a failing test"))
            self.assertTrue(memories)
            self.assertTrue(any("test" in memory.content.lower() for memory in memories))
            backend.close()


    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None,
        "ExpeL requires the optional memory-research dependencies",
    )
    def test_expel_upstream_provider_with_coding_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            backend = EvolveLabBackend(
                "expel",
                storage_dir=Path(temp) / "expel",
                config={"top_k": 3},
                model=FakeMemoryModel(),
            )
            record = ExperienceRecord(
                experience_id="expel-1",
                task_description="Fix a failing unit test after inspecting the assertion.",
                verified_success=True,
                run_status="completed",
                execution_outcome="completed",
                tool_actions=[
                    {"id": "c1", "name": "bash", "args": {"command": "pytest -q"}}
                ],
                tool_observations=[
                    {"tool_call_id": "c1", "name": "bash", "text": "2 passed"}
                ],
                test_results=[{"kind": "command", "passed": True}],
            )
            update = backend.add_experience(record)
            self.assertTrue(update.ok, update.error)
            memories = backend.retrieve(
                MemoryQuery(query="failing unit test assertion")
            )
            self.assertTrue(memories)
            self.assertTrue(
                any("assertion" in memory.content.lower() for memory in memories)
            )
            backend.close()


class ManagerTests(unittest.TestCase):
    def test_manager_stores_canonical_history_and_retrieves(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            config = MemoryConfig(
                enabled=True,
                active_provider="lightweight_memory",
                state_dir=str(Path(temp) / "memory"),
                provider_config={
                    "lightweight_memory": {
                        "enable_longterm_provision": True,
                        "top_k_longterm": 3,
                    }
                },
            )
            manager = MemoryManager(
                config,
                workspace=workspace,
                model=FakeMemoryModel(),
            )
            record = ExperienceRecord(
                experience_id="exp-1",
                task_description="Fix a failing test.",
                verified_success=True,
                run_status="completed",
                execution_outcome="completed",
            )
            update = manager.after_task(record)
            self.assertIsNotNone(update)
            self.assertTrue(update.ok, update.error)
            self.assertEqual(manager.store.count(), 1)
            context = manager.before_task("fix a Python test")
            self.assertIn("Historical experience", context)
            manager.close()

    def test_manager_architecture_activation_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            config = MemoryConfig(
                enabled=True,
                active_provider="lightweight_memory",
                state_dir=str(Path(temp) / "memory"),
            )
            manager = MemoryManager(
                config,
                workspace=workspace,
                model=FakeMemoryModel(),
            )
            baseline = manager.active_architecture
            candidate = ArchitectureRecord(
                architecture_id="candidate-v2",
                architecture_version=2,
                storage_namespace=str(Path(temp) / "candidate-v2"),
                configuration={"provider": "lightweight_memory"},
                validation_status="validated",
                state="validated",
            )
            manager.registry.register(candidate)
            activated = manager.activate_architecture("candidate-v2")
            self.assertEqual(activated.architecture_id, "candidate-v2")
            self.assertEqual(manager.active_architecture.architecture_id, "candidate-v2")
            rolled = manager.rollback_architecture()
            self.assertEqual(rolled.architecture_id, baseline.architecture_id)
            manager.close()

    def test_harness_memory_hooks(self) -> None:
        class CompleteProvider:
            def complete(self, model, system, messages, tools):
                return ProviderResponse("done", [], "stop", TokenUsage(1, 1))

        class FakeManager:
            def __init__(self):
                self.before = []
                self.after = []

            def before_task(self, task):
                self.before.append(task)
                return "Historical experience: inspect the assertion."

            def after_task(self, record):
                self.after.append(record)
                return None

        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            memory = FakeManager()
            harness = Harness(
                workdir=workspace,
                provider=CompleteProvider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace),
                state_dir=Path(temp) / "state",
                memory_manager=memory,
            )
            result = harness.run("fix a test")
            self.assertTrue(result.completed)
            self.assertEqual(memory.before, ["fix a test"])
            self.assertEqual(len(memory.after), 1)
            self.assertIn("Historical experience", harness.system_prompt)
            harness.close()


class RegistryMigrationTests(unittest.TestCase):
    def test_architecture_activation_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            registry = ArchitectureRegistry(temp)
            baseline = registry.ensure_baseline(
                "lightweight_memory", storage_namespace=str(Path(temp) / "v1")
            )
            candidate = ArchitectureRecord(
                architecture_id="candidate",
                architecture_version=2,
                implementation_path=str(Path(temp) / "candidate.py"),
                storage_namespace=str(Path(temp) / "v2"),
                validation_status="validated",
                state="validated",
            )
            registry.register(candidate)
            active = registry.activate("candidate")
            self.assertEqual(active.architecture_id, "candidate")
            self.assertEqual(registry.active().architecture_id, "candidate")
            rolled = registry.rollback()
            self.assertEqual(rolled.architecture_id, baseline.architecture_id)
            self.assertEqual(
                registry.active().architecture_id, baseline.architecture_id
            )

    def test_activation_rejects_unvalidated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            registry = ArchitectureRegistry(temp)
            registry.ensure_baseline("lightweight_memory", storage_namespace="v1")
            registry.register(
                ArchitectureRecord(
                    architecture_id="bad",
                    state="created",
                    storage_namespace="v2",
                )
            )
            with self.assertRaises(ArchitectureError):
                registry.activate("bad")

    def test_replay_does_not_modify_source_and_reports_failures(self) -> None:
        class FakeBackend:
            provider_id = "fake"

            def __init__(self, fail: bool = False):
                self.records = []
                self.fail = fail

            def add_experience(self, record):
                if self.fail:
                    return MemoryUpdateResult(
                        False, "", "fake", record.experience_id, error="boom"
                    )
                self.records.append(record)
                return MemoryUpdateResult(
                    True, "ok", "fake", record.experience_id
                )

            def retrieve(self, query):
                return []

            def health_check(self):
                return {"ok": True}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            store.add(
                ExperienceRecord(
                    experience_id="1", task_description="one", timestamp="2026-01-01"
                )
            )
            store.add(
                ExperienceRecord(
                    experience_id="2", task_description="two", timestamp="2026-01-02"
                )
            )
            registry = ArchitectureRegistry(temp)
            manager = ReplayManager(store, registry)
            backend = FakeBackend()
            result = manager.replay(
                source_architecture="v1",
                target_architecture="v2",
                target_backend=backend,
            )
            self.assertEqual(result.migrated_experience_count, 2)
            self.assertEqual(store.count(), 2)
            self.assertEqual(len(backend.records), 2)
            failed = manager.replay(
                source_architecture="v1",
                target_architecture="v3",
                target_backend=FakeBackend(fail=True),
            )
            self.assertEqual(failed.failed_experience_count, 2)
            self.assertEqual(store.count(), 2)


class SafetyEvaluationEvolutionTests(unittest.TestCase):
    def test_candidate_static_and_isolated_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            safe = Path(temp) / "safe.py"
            safe.write_text("VALUE = 3\n", encoding="utf-8")
            self.assertTrue(static_validate(safe).ok)
            self.assertTrue(isolated_import(safe).ok)
            unsafe = Path(temp) / "unsafe.py"
            unsafe.write_text("import subprocess\nsubprocess.run(['x'])\n", encoding="utf-8")
            self.assertFalse(static_validate(unsafe).ok)

    def test_evaluation_fixture_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            task = CodingTask(
                name="fixture",
                prompt="do nothing",
                fixture_files={
                    "test_ok.py": "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n"
                },
                validation_command=("python", "-m", "unittest", "-v"),
            )
            row = run_fixture(task, lambda task, root: {"turns": 1, "tool_calls": 0})
            self.assertTrue(row["verified_success"])
            metrics = evaluate([task], lambda task, root: {"turns": 1, "tool_calls": 0})
            self.assertEqual(metrics.task_success_rate, 1.0)
            self.assertEqual(len(default_coding_tasks()), 4)

    def test_evolution_status_is_honest_and_exports_trajectories(self) -> None:
        status = dependency_status()
        self.assertEqual(status.upstream_commit, UPSTREAM_COMMIT)
        if not status.available:
            self.assertTrue(
                status.missing_dependencies or status.import_error
            )
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(EvolutionUnavailable):
                    EvolutionBridge(work_dir=temp).run(temp)
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp, "C:/workspace")
            store.add(ExperienceRecord(experience_id="1", task_description="one"))
            written = export_trajectories(store, Path(temp) / "logs")
            self.assertEqual(len(written), 1)
            self.assertTrue((Path(temp) / "logs" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
