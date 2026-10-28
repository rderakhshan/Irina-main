"""Bridge to the original MemEvolve analysis/generation/creation pipeline."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .architecture_registry import ArchitectureError, ArchitectureRecord, ArchitectureRegistry
from .experience_store import ExperienceStore
from .provider_registry import UPSTREAM_COMMIT, UPSTREAM_ROOT, upstream_imports
from .security import validate_candidate


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
    registered: list[dict[str, Any]] = field(default_factory=list)
    activated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Packages required by the original engine and the agent core it imports.
# Web/media tools are imported lazily, so requests/crawl4ai/pandas/biopython
# are only needed when those specific tools are used.
_ENGINE_DEPENDENCIES = (
    "openai",
    "yaml",
    "dotenv",
    "jinja2",
    "rich",
    "json_repair",
    "huggingface_hub",
    "PIL",
)


def dependency_status() -> EvolutionStatus:
    missing = [
        module
        for module in _ENGINE_DEPENDENCIES
        if importlib.util.find_spec(module) is None
    ]
    import_ok = None
    import_error = None
    if UPSTREAM_ROOT.is_dir() and not missing:
        try:
            with upstream_imports():
                import MemEvolve.core.memory_evolver  # noqa: F401
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


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "candidate"


class EvolutionBridge:
    """Invoke the original four-stage MemoryEvolver, without rewriting it."""

    def __init__(
        self,
        *,
        work_dir: str | Path,
        analysis_model: str | None = None,
        generation_model: str | None = None,
        registry: ArchitectureRegistry | None = None,
        state_root: str | Path | None = None,
        workspace: str | Path | None = None,
        model_client: Any | None = None,
    ) -> None:
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.analysis_model = analysis_model
        self.generation_model = generation_model
        self.registry = registry
        self.state_root = Path(state_root).expanduser().resolve() if state_root else None
        self.workspace = str(workspace) if workspace is not None else None
        self.model_client = model_client

    def status(self) -> EvolutionStatus:
        return dependency_status()

    def _prepare_build_tree(self) -> Path:
        """Copy the upstream provider package so create() never edits vendored source."""
        build = self.work_dir / "build"
        package = build / "EvolveLab"
        if not package.is_dir():
            build.mkdir(parents=True, exist_ok=True)
            shutil.copytree(UPSTREAM_ROOT / "EvolveLab", package)
        return build

    def run(
        self,
        task_logs_dir: str | Path,
        *,
        default_provider: str = "lightweight_memory",
        max_analysis_steps: int = 20,
        creativity_index: float = 0.5,
        validation_dataset: str | None = None,
    ) -> EvolutionResult:
        status = self.status()
        if not status.available:
            raise EvolutionUnavailable(status.detail)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        build = self._prepare_build_tree()
        with upstream_imports():
            try:
                from MemEvolve.core.memory_evolver import MemoryEvolver
            except Exception as exc:
                raise EvolutionUnavailable(
                    f"could not import original MemEvolve: {type(exc).__name__}: {exc}"
                ) from None
            try:
                evolver = MemoryEvolver(
                    work_dir=str(self.work_dir),
                    analysis_model_id=self.analysis_model,
                    gen_model_id=self.generation_model,
                )
            except Exception as exc:
                raise EvolutionUnavailable(
                    "could not start original MemEvolve: "
                    f"{type(exc).__name__}: {exc}. Configure OPENAI_API_KEY "
                    "(and OPENAI_BASE_URL for a compatible endpoint)."
                ) from None
            if self.model_client is not None:
                evolver.openai_client = self.model_client
            stages: dict[str, Any] = {}
            try:
                stages["analysis"] = evolver.analyze(
                    str(task_logs_dir),
                    default_provider=default_provider or "lightweight_memory",
                    max_steps=max_analysis_steps,
                )
                analysis_error = self._analysis_error(stages["analysis"])
                if analysis_error:
                    return EvolutionResult(
                        False,
                        str(self.work_dir),
                        stages,
                        error=f"analysis failed: {analysis_error}",
                    )
                stages["generation"] = evolver.generate(
                    creativity_index=creativity_index
                )
                generation = stages["generation"] or {}
                if not generation.get("success") or not generation.get("config"):
                    return EvolutionResult(
                        False,
                        str(self.work_dir),
                        stages,
                        error=(
                            "generation failed: the original generator did not "
                            "produce a parsable system configuration"
                        ),
                    )
                stages["creation"] = evolver.create(base_dir=str(build))
            except Exception as exc:
                return EvolutionResult(
                    False,
                    str(self.work_dir),
                    stages,
                    error=f"{type(exc).__name__}: {exc}",
                )
            config_path = evolver.state["phases"]["generate"]["output"]
            created = list(stages["creation"].get("created") or [])
            if validation_dataset:
                try:
                    stages["validation"] = evolver.validate(
                        dataset_name=validation_dataset,
                        config_path=config_path,
                        created_systems=created or None,
                    )
                except Exception as exc:
                    stages["validation"] = {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            else:
                stages["validation"] = {
                    "success": True,
                    "skipped": True,
                    "reason": (
                        "no validation dataset configured; upstream dataset runners "
                        "are not vendored with Irina"
                    ),
                }

            evaluation = self._evaluate_systems(config_path, build, created)
            stages["evaluation"] = evaluation
            registered = self._register_systems(
                config_path,
                evaluation,
                build,
            )
            stages["registration"] = {
                "registered": registered,
                "activated": False,
            }
            return EvolutionResult(
                True,
                str(self.work_dir),
                stages,
                registered=registered,
                activated=False,
            )

    def _analysis_error(self, analysis: dict[str, Any]) -> str | None:
        """Return a reason when upstream analysis finished without a report.

        Upstream marks the phase complete even when the analysis agent returns
        an error object. Do not generate from an empty analysis.
        """
        report_path = (analysis or {}).get("report_path")
        if not report_path:
            return "analysis produced no report"
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return f"analysis report is unreadable: {type(exc).__name__}: {exc}"
        result = report.get("agent_result")
        if isinstance(result, dict):
            error = result.get("error")
            if error:
                return str(error)
            result = result.get("agent_result")
        if not isinstance(result, str) or not result.strip():
            return "analysis agent returned no text"
        return None

    def _load_configs(self, config_path: str | Path) -> list[dict[str, Any]]:
        value = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]

    def _evaluate_systems(
        self,
        config_path: str | Path,
        build: Path,
        created: list[str],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for config in self._load_configs(config_path):
            try:
                name = config["memory_type_info"]["enum_value"]
                module = config["provider_code"]["module_name"]
            except (KeyError, TypeError):
                continue
            if created and name not in created:
                continue
            provider_path = build / "EvolveLab" / "providers" / f"{module}.py"
            validation = validate_candidate(
                provider_path,
                python_paths=[str(build), str(UPSTREAM_ROOT)],
            )
            results[name] = {
                "ok": validation.ok,
                "provider_path": str(provider_path),
                "module": module,
                "class_name": config["provider_code"].get("class_name"),
                "validation": validation.as_dict(),
            }
        return results

    def _register_systems(
        self,
        config_path: str | Path,
        evaluation: dict[str, Any],
        build: Path,
    ) -> list[dict[str, Any]]:
        if self.registry is None or self.state_root is None or self.workspace is None:
            return []
        registered: list[dict[str, Any]] = []
        for config in self._load_configs(config_path):
            try:
                name = config["memory_type_info"]["enum_value"]
                module = config["provider_code"]["module_name"]
                class_name = config["provider_code"]["class_name"]
            except (KeyError, TypeError):
                continue
            item = evaluation.get(name) or {}
            ok = bool(item.get("ok"))
            architecture_id = self._unique_architecture_id(name)
            namespace = (
                self.state_root
                / "providers"
                / name
                / architecture_id
            )
            record = ArchitectureRecord(
                architecture_id=architecture_id,
                origin="evolved",
                upstream_commit=UPSTREAM_COMMIT,
                parent_architecture=(
                    self.registry.active().architecture_id
                    if self.registry.active() is not None
                    else None
                ),
                implementation_path=str(build),
                storage_namespace=str(namespace),
                configuration={
                    "provider": name,
                    "module": f"EvolveLab.providers.{module}",
                    "class_name": class_name,
                    "implementation_root": str(build),
                },
                validation_status="validated" if ok else "failed",
                evaluation_metrics=item.get("validation", {}),
                state="validated" if ok else "rejected",
            )
            try:
                self.registry.register(record)
            except ArchitectureError:
                continue
            registered.append(
                {
                    "architecture_id": architecture_id,
                    "provider": name,
                    "ok": ok,
                    "state": record.state,
                    "activated": False,
                }
            )
        return registered

    def _unique_architecture_id(self, name: str) -> str:
        base = f"evolved-{_slug(name)}"
        existing = {
            record.architecture_id
            for record in (self.registry.list() if self.registry else [])
        }
        if base not in existing:
            return base
        index = 2
        while f"{base}-v{index}" in existing:
            index += 1
        return f"{base}-v{index}"
