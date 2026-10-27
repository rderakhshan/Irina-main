"""Bridge to the original MemEvolve analysis/generation/creation pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experience_store import ExperienceRecord, ExperienceStore
from .provider_registry import UPSTREAM_COMMIT, UPSTREAM_ROOT, upstream_imports


class EvolutionUnavailable(RuntimeError):
    """The original MemEvolve engine or one of its dependencies is unavailable."""


@dataclass
class EvolutionStatus:
    available: bool
    upstream_root: str
    upstream_commit: str
    missing_dependencies: list[str]
    detail: str = ""
    import_ok: bool | None = None
    import_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionResult:
    success: bool
    work_dir: str
    stages: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def dependency_status() -> EvolutionStatus:
    missing = [
        module
        for module in (
            "openai",
            "yaml",
            "dotenv",
            "jinja2",
            "rich",
            "json_repair",
        )
        if importlib.util.find_spec(module) is None
    ]
    import_ok = None
    import_error = None
    if UPSTREAM_ROOT.is_dir() and not missing:
        try:
            with upstream_imports():
                import MemEvolve  # noqa: F401
            import_ok = True
        except Exception as exc:
            import_ok = False
            import_error = f"{type(exc).__name__}: {exc}"
    return EvolutionStatus(
        available=UPSTREAM_ROOT.is_dir() and not missing and import_ok is not False,
        upstream_root=str(UPSTREAM_ROOT),
        upstream_commit=UPSTREAM_COMMIT,
        missing_dependencies=missing,
        detail=(
            "MemEvolve requires optional research dependencies: "
            + ", ".join(missing)
            if missing
            else import_error
            or "upstream source and declared Python dependencies are present"
        ),
        import_ok=import_ok,
        import_error=import_error,
    )


def export_trajectories(store: ExperienceStore, destination: str | Path) -> list[Path]:
    """Write canonical experiences as the JSON task logs used by the analyzer."""
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, record in enumerate(
        sorted(store.load(), key=lambda item: (item.timestamp or "", item.experience_id)),
        1,
    ):
        payload = {
            "task_id": record.task_id or record.experience_id,
            "session_id": record.session_id,
            "workspace_id": record.workspace_id,
            "question": record.task_description,
            "status": record.run_status,
            "is_correct": bool(record.verified_success),
            "verified_success": bool(record.verified_success),
            "test_results": record.test_results,
            "tool_actions": record.tool_actions,
            "tool_observations": record.tool_observations,
            "execution_outcome": record.execution_outcome,
            "failure_reason": record.failure_reason,
            "experience_id": record.experience_id,
        }
        path = target / f"{index:04d}_{record.experience_id[:12]}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "upstream_commit": UPSTREAM_COMMIT,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "experience_count": len(written),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return written


class EvolutionBridge:
    """Invoke the original four-stage MemoryEvolver, without rewriting it."""

    def __init__(
        self,
        *,
        work_dir: str | Path,
        analysis_model: str | None = None,
        generation_model: str | None = None,
    ) -> None:
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.analysis_model = analysis_model
        self.generation_model = generation_model

    def status(self) -> EvolutionStatus:
        return dependency_status()

    def run(
        self,
        task_logs_dir: str | Path,
        *,
        default_provider: str = "lightweight_memory",
        max_analysis_steps: int = 20,
        creativity_index: float = 0.5,
    ) -> EvolutionResult:
        status = self.status()
        if not status.available:
            raise EvolutionUnavailable(status.detail)
        with upstream_imports():
            try:
                from MemEvolve.core.memory_evolver import MemoryEvolver
            except Exception as exc:
                raise EvolutionUnavailable(
                    f"could not import original MemEvolve: {type(exc).__name__}: {exc}"
                ) from None
            evolver = MemoryEvolver(
                work_dir=str(self.work_dir),
                analysis_model_id=self.analysis_model,
                gen_model_id=self.generation_model,
            )
            stages: dict[str, Any] = {}
            try:
                stages["analysis"] = evolver.analyze(
                    str(task_logs_dir),
                    default_provider=default_provider or "lightweight_memory",
                    max_steps=max_analysis_steps,
                )
                stages["generation"] = evolver.generate(
                    creativity_index=creativity_index
                )
                stages["creation"] = evolver.create()
                stages["validation"] = evolver.validate()
            except Exception as exc:
                return EvolutionResult(
                    False,
                    str(self.work_dir),
                    stages,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return EvolutionResult(True, str(self.work_dir), stages)
