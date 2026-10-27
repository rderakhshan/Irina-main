"""Stage 6: inspect providers and exercise original EvolveLab memory safely."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.back.harness.context_management.memory import (
    DeepSeekMemoryModel,
    ExperienceRecord,
    MemoryConfig,
    MemoryManager,
    probe_all,
    upstream_info,
)
from src.back.harness.context_management.memory.evolution_bridge import (
    EvolutionBridge,
    dependency_status,
)
from src.back.harness.orchestration.cli import _configure_streams, _load_dotenv
from src.back.harness.providers.deepseek import DeepSeekProvider


class FakeModel:
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
                '{"strategic":["Inspect the failing assertion before editing."],'
                '"operational":["Rerun the test suite after the fix."]}'
            )
        else:
            response.content = (
                "Inspect the failure, patch the implementation, and rerun tests."
            )
        return response


def _record() -> ExperienceRecord:
    return ExperienceRecord(
        experience_id="demo-experience-1",
        session_id="demo-session",
        task_id="demo-task",
        workspace_id="demo-workspace",
        timestamp="2026-10-27T00:00:00+00:00",
        task_description=(
            "Fix a failing pytest assertion by reading the assertion, patching "
            "the implementation, and rerunning pytest."
        ),
        tool_actions=[
            {
                "id": "call-1",
                "name": "bash",
                "args": '{"command": "pytest -q"}',
            }
        ],
        tool_observations=[
            {
                "tool_call_id": "call-1",
                "name": "bash",
                "text": "2 passed in 0.12s",
            }
        ],
        execution_outcome="completed",
        test_results=[{"kind": "command", "passed": True}],
        verified_success=True,
        run_status="completed",
        source_event_references=[{"fixture": True}],
    )


def _providers() -> int:
    print(json.dumps(upstream_info(), indent=2))
    for status in probe_all():
        row = status.as_dict()
        print(
            f"{row['provider_id']:<26} {row['status']:<14} "
            f"source={row['source_present']} import={row['import_ok']} "
            f"init={row['initialize_ok']} integration={row['integration_ok']}"
        )
        if row["missing_dependencies"]:
            print("  missing: " + ", ".join(row["missing_dependencies"]))
        if row["error"]:
            print("  error: " + row["error"])
    return 0


def _memory(*, provider: str, fake: bool) -> int:
    _load_dotenv()
    model = FakeModel() if fake else DeepSeekMemoryModel(DeepSeekProvider())
    with tempfile.TemporaryDirectory() as temp:
        config = MemoryConfig(
            enabled=True,
            active_provider=provider,
            state_dir=str(Path(temp) / "memory"),
        )
        manager = MemoryManager(config, workspace=Path(temp) / "workspace", model=model)
        update = manager.after_task(_record())
        print(
            "update:",
            None if update is None else update.ok,
            None if update is None else update.error,
        )
        print("canonical_count:", manager.store.count())
        context = manager.before_task(
            "A pytest assertion is failing. What should I inspect first?"
        )
        print("retrieval_chars:", len(context))
        print(context[:1200])
        manager.close()
    return 0


def _evolution() -> int:
    status = dependency_status()
    print(json.dumps(status.as_dict(), indent=2))
    bridge = EvolutionBridge(work_dir=ROOT / ".irina" / "memory" / "evolution")
    if not status.available:
        print("evolution is unavailable; original source remains vendored.")
        return 2
    print("use `python main.py memory evolve <task_logs> --work-dir ... --approve`")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--evolution", action="store_true")
    parser.add_argument("--provider", default="expel")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_streams()
    if args.providers:
        return _providers()
    if args.memory:
        return _memory(provider=args.provider, fake=args.fake)
    if args.evolution:
        return _evolution()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
