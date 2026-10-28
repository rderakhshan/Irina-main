"""Finalization tests: isolation, registry atomicity, evolution, CLI contract."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.back.harness.context_management.memory.architecture_registry import (
    ArchitectureRecord,
    ArchitectureRegistry,
)
from src.back.harness.context_management.memory.configuration import MemoryConfig
from src.back.harness.context_management.memory.evolution_bridge import (
    EvolutionBridge,
)
from src.back.harness.context_management.memory.experience_store import (
    ExperienceRecord,
)
from src.back.harness.context_management.memory.manager import MemoryManager
from src.back.harness.context_management.memory.provider_registry import (
    upstream_imports,
)


class FakeMemoryModel:
    def __call__(self, messages):
        text = " ".join(
            str(message.get("content", "")) for message in messages
        )

        class Response:
            pass

        response = Response()
        if "Output (JSON)" in text:
            response.content = (
                '{"strategic":["Inspect the exact failing assertion."],'
                '"operational":["Run the test suite after each fix."]}'
            )
        elif "PROJECT-ALPHA" in text:
            response.content = "PROJECT-ALPHA: inspect the assertion."
        else:
            response.content = "Historical experience: inspect the assertion."
        return response


def _record(marker: str = "PROJECT-ALPHA") -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=f"exp-{marker}",
        task_description=f"Fix the failing test for {marker}.",
        verified_success=True,
        run_status="completed",
        execution_outcome="completed",
        tool_actions=[{"id": "c1", "name": "bash", "args": "pytest -q"}],
        tool_observations=[
            {"tool_call_id": "c1", "name": "bash", "text": "OK"}
        ],
        test_results=[{"kind": "command", "passed": True}],
    )


class IsolationTests(unittest.TestCase):
    def test_provider_storage_is_workspace_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "memory"
            alpha = Path(temp) / "alpha"
            beta = Path(temp) / "beta"
            alpha.mkdir()
            beta.mkdir()
            config = MemoryConfig(enabled=True, state_dir=str(root))
            first = MemoryManager(config, workspace=alpha, model=FakeMemoryModel())
            self.assertTrue(first.after_task(_record()).ok)
            first.close()

            second = MemoryManager(config, workspace=beta, model=FakeMemoryModel())
            self.assertEqual(second.store.count(), 0)
            self.assertNotEqual(
                first._workspace_storage("lightweight_memory"),
                second._workspace_storage("lightweight_memory"),
            )
            alpha_file = (
                first._workspace_storage("lightweight_memory")
                / "longterm_memory.json"
            )
            beta_file = (
                second._workspace_storage("lightweight_memory")
                / "longterm_memory.json"
            )
            lesson = "Inspect the exact failing assertion."
            self.assertIn(
                lesson,
                alpha_file.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                lesson,
                beta_file.read_text(encoding="utf-8"),
            )
            second.close()


class RegistryAtomicityTests(unittest.TestCase):
    def test_two_instances_observe_each_others_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = ArchitectureRegistry(temp)
            baseline = first.ensure_baseline(
                "lightweight_memory",
                storage_namespace=str(Path(temp) / "v1"),
            )
            second = ArchitectureRegistry(temp)
            second.register(
                ArchitectureRecord(
                    architecture_id="candidate",
                    storage_namespace=str(Path(temp) / "v2"),
                    configuration={"provider": "lightweight_memory"},
                    validation_status="validated",
                    state="validated",
                )
            )
            first.activate("candidate")
            self.assertEqual(
                ArchitectureRegistry(temp).active().architecture_id,
                "candidate",
            )
            second.rollback()
            self.assertEqual(
                ArchitectureRegistry(temp).active().architecture_id,
                baseline.architecture_id,
            )

    def test_activation_survives_manager_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            config = MemoryConfig(
                enabled=True,
                active_provider="lightweight_memory",
                state_dir=str(Path(temp) / "memory"),
            )
            manager = MemoryManager(
                config, workspace=workspace, model=FakeMemoryModel()
            )
            baseline = manager.active_architecture
            manager.registry.register(
                ArchitectureRecord(
                    architecture_id="approved",
                    storage_namespace=str(Path(temp) / "approved"),
                    configuration={"provider": "lightweight_memory"},
                    validation_status="validated",
                    state="validated",
                )
            )
            manager.activate_architecture("approved")
            manager.close()

            restarted = MemoryManager(
                MemoryConfig(
                    enabled=True,
                    active_provider="lightweight_memory",
                    state_dir=str(Path(temp) / "memory"),
                ),
                workspace=workspace,
                model=FakeMemoryModel(),
            )
            self.assertEqual(
                restarted.active_architecture.architecture_id, "approved"
            )
            self.assertEqual(
                (Path(temp) / "memory" / "selection.json").is_file(), True
            )
            restarted.close()
            self.assertNotEqual(baseline.architecture_id, "approved")


_GENERATED_RESPONSE = '''**Class Name**: AdapterMemoryProvider
**Module Name**: adapter_memory_provider

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryType,
    TrajectoryData,
)


class AdapterMemoryProvider(BaseMemoryProvider):
    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(MemoryType.ADAPTER_MEMORY, config)
        base = Path((config or {}).get("storage_dir", "."))
        self.path = base / "adapter_memory.json"
        self.items: list[dict[str, Any]] = []

    def initialize(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
        return True

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        self.items.append({"query": trajectory_data.query})
        self.path.write_text(json.dumps(self.items), encoding="utf-8")
        return True, f"stored {len(self.items)}"

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        found = [
            MemoryItem(id=str(index), content=item["query"], metadata={})
            for index, item in enumerate(self.items)
            if request.query.lower() in item["query"].lower()
        ]
        return MemoryResponse(
            memories=found,
            memory_type=self.memory_type,
            total_count=len(found),
        )
```

**Enum Name**: ADAPTER_MEMORY
**Enum Value**: adapter_memory
**Storage Format**: json
'''


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, **kwargs):
        return _FakeResponse(_GENERATED_RESPONSE)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class EvolutionFixtureTests(unittest.TestCase):
    def test_generate_create_validate_and_register_without_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            state_root = Path(temp) / "memory"
            logs = Path(temp) / "logs"
            logs.mkdir()
            (logs / "0001.json").write_text(
                json.dumps({"question": "fix a failing test", "is_correct": True}),
                encoding="utf-8",
            )
            registry = ArchitectureRegistry(state_root)
            baseline = registry.ensure_baseline(
                "lightweight_memory",
                storage_namespace=str(
                    state_root / "providers" / "lightweight_memory" / "seed"
                ),
            )
            previous_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "fixture-key"
            try:
                with upstream_imports():
                    import MemEvolve.phases.phase_analyzer as analyzer

                    original_agent = analyzer.AnalysisAgent

                    class FixtureAnalysisAgent:
                        def __init__(self, **kwargs) -> None:
                            pass

                        def __call__(self, prompt):
                            return {
                                "agent_result": (
                                    "Trajectories show successful JSON writes; "
                                    "prefer deterministic storage."
                                )
                            }

                    analyzer.AnalysisAgent = FixtureAnalysisAgent
                    try:
                        bridge = EvolutionBridge(
                            work_dir=Path(temp) / "evolution",
                            registry=registry,
                            state_root=state_root,
                            workspace=workspace,
                            model_client=_FakeClient(),
                        )
                        result = bridge.run(
                            logs,
                            default_provider="lightweight_memory",
                            max_analysis_steps=1,
                        )
                    finally:
                        analyzer.AnalysisAgent = original_agent
            finally:
                if previous_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous_key

            self.assertTrue(result.success, result.error)
            self.assertEqual(len(result.registered), 1)
            self.assertFalse(result.activated)
            candidate = result.registered[0]["architecture_id"]
            record = registry.get(candidate)
            self.assertEqual(record.state, "validated")
            self.assertEqual(record.validation_status, "validated")
            self.assertEqual(
                registry.active().architecture_id, baseline.architecture_id
            )
            self.assertTrue(
                (Path(temp) / "evolution" / "build" / "EvolveLab" / "providers"
                 / "adapter_memory_provider.py").is_file()
            )

            registry.activate(candidate)
            self.assertEqual(registry.active().architecture_id, candidate)
            rolled = registry.rollback()
            self.assertEqual(rolled.architecture_id, baseline.architecture_id)


_ADAPTER_PROVIDER = '''
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryType,
    TrajectoryData,
)


class AdapterMemoryProvider(BaseMemoryProvider):
    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(MemoryType.LIGHTWEIGHT_MEMORY, config)
        base = Path((config or {}).get("storage_dir", "."))
        self.path = base / "adapter_items.json"
        self.items: list[str] = []

    def initialize(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
        return True

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        self.items.append(str(trajectory_data.query))
        self.path.write_text(json.dumps(self.items), encoding="utf-8")
        return True, f"stored {len(self.items)}"

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        found = [
            MemoryItem(id=str(index), content=value, metadata={})
            for index, value in enumerate(self.items)
            if "adapter-lesson" in value.lower()
        ]
        return MemoryResponse(
            memories=found,
            memory_type=self.memory_type,
            total_count=len(found),
        )
'''


class EvolvedProviderActivationTests(unittest.TestCase):
    def test_generated_provider_can_be_activated_and_rolled_back(self) -> None:
        import shutil

        from src.back.harness.context_management.memory.provider_registry import (
            PROVIDERS,
            UPSTREAM_ROOT,
        )

        with tempfile.TemporaryDirectory() as temp:
            try:
                workspace = Path(temp) / "workspace"
                workspace.mkdir()
                build = Path(temp) / "build"
                shutil.copytree(UPSTREAM_ROOT / "EvolveLab", build / "EvolveLab")
                provider_file = (
                    build / "EvolveLab" / "providers" / "adapter_test_provider.py"
                )
                provider_file.write_text(_ADAPTER_PROVIDER, encoding="utf-8")
                state_root = Path(temp) / "memory"
                config = MemoryConfig(
                    enabled=True,
                    state_dir=str(state_root),
                    provider_config={
                        "lightweight_memory": {"enable_longterm_provision": True}
                    },
                )
                manager = MemoryManager(
                    config, workspace=workspace, model=FakeMemoryModel()
                )
                baseline = manager.active_architecture
                manager.registry.register(
                    ArchitectureRecord(
                        architecture_id="evolved-adapter",
                        storage_namespace=str(
                            state_root
                            / "providers"
                            / "adapter_test"
                            / "evolved-adapter"
                        ),
                        configuration={
                            "provider": "adapter_test",
                            "module": "EvolveLab.providers.adapter_test_provider",
                            "class_name": "AdapterMemoryProvider",
                            "implementation_root": str(build),
                        },
                        validation_status="validated",
                        state="validated",
                    )
                )
                manager.activate_architecture("evolved-adapter")
                self.assertEqual(
                    manager.active_architecture.architecture_id, "evolved-adapter"
                )
                update = manager.after_task(
                    ExperienceRecord(
                        experience_id="adapter-1",
                        task_description="ADAPTER-LESSON: fix the failing test.",
                        verified_success=True,
                        run_status="completed",
                        execution_outcome="completed",
                    )
                )
                self.assertTrue(update.ok, update.error)
                context = manager.before_task("adapter-lesson")
                self.assertIn("ADAPTER-LESSON", context)
                rolled = manager.rollback_architecture()
                self.assertEqual(rolled.architecture_id, baseline.architecture_id)
                manager.close()
            finally:
                PROVIDERS.pop("adapter_test", None)


if __name__ == "__main__":
    unittest.main()

