"""Discovery and honest probing of the vendored EvolveLab providers."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream" / "memevolve"
UPSTREAM_COMMIT = "6035d5659d7a092dbfa6a87b1a32a3cee652ba54"

PROVIDERS: dict[str, dict[str, Any]] = {
    "agent_kb": {
        "display_name": "Agent-KB",
        "module": "EvolveLab.providers.agent_kb_provider",
        "class": "AgentKBProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "skillweaver": {
        "display_name": "SkillWeaver",
        "module": "EvolveLab.providers.skillweaver_provider",
        "class": "SkillWeaverProvider",
        "origin": "upstream-baseline",
        "dependencies": [],
    },
    "mobilee": {
        "display_name": "Mobile-E",
        "module": "EvolveLab.providers.mobilee_provider",
        "class": "MobileEProvider",
        "origin": "upstream-baseline",
        "dependencies": ["jieba"],
    },
    "expel": {
        "display_name": "ExpeL",
        "module": "EvolveLab.providers.expel_provider",
        "class": "ExpeLProvider",
        "origin": "upstream-baseline",
        "dependencies": ["sklearn"],
    },
    "voyager": {
        "display_name": "Voyager",
        "module": "EvolveLab.providers.voyager_memory_provider",
        "class": "VoyagerMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "dilu": {
        "display_name": "DiLu",
        "module": "EvolveLab.providers.dilu_memory_provider",
        "class": "DiluMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "generative": {
        "display_name": "Generative",
        "module": "EvolveLab.providers.generative_memory_provider",
        "class": "GenerativeMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "memp": {
        "display_name": "MemP",
        "module": "EvolveLab.providers.memp_memory_provider",
        "class": "MempMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "dynamic_cheatsheet": {
        "display_name": "Dynamic Cheatsheet",
        "module": "EvolveLab.providers.dynamic_cheatsheet_provider",
        "class": "DynamicCheatsheetProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn", "sentence_transformers"],
    },
    "agent_workflow_memory": {
        "display_name": "Agent Workflow Memory",
        "module": "EvolveLab.providers.agent_workflow_memory_provider",
        "class": "AgentWorkflowMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sentence_transformers"],
    },
    "evolver": {
        "display_name": "Evolver",
        "module": "EvolveLab.providers.evolver_memory_provider",
        "class": "EvolverMemoryProvider",
        "origin": "upstream-baseline",
        "dependencies": ["numpy", "sklearn"],
    },
    "lightweight_memory": {
        "display_name": "Lightweight Memory",
        "module": "EvolveLab.providers.lightweight_memory_provider",
        "class": "LightweightMemoryProvider",
        "origin": "upstream-evolved",
        "dependencies": [],
    },
    "cerebra_fusion_memory": {
        "display_name": "Cerebra Fusion Memory",
        "module": "EvolveLab.providers.cerebra_fusion_memory_provider",
        "class": "CerebraFusionMemoryProvider",
        "origin": "upstream-evolved",
        "dependencies": ["numpy", "sklearn"],
    },
}

_PATH_LOCK = threading.RLock()


@contextmanager
def upstream_imports() -> Iterator[Path]:
    """Temporarily expose the vendored upstream root to Python imports."""
    root = str(UPSTREAM_ROOT)
    added = root not in sys.path
    with _PATH_LOCK:
        if added:
            sys.path.insert(0, root)
    try:
        yield UPSTREAM_ROOT
    finally:
        if added:
            with _PATH_LOCK:
                try:
                    sys.path.remove(root)
                except ValueError:
                    pass


@dataclass
class ProviderStatus:
    provider_id: str
    display_name: str
    origin: str
    source_present: bool
    import_ok: bool
    initialize_ok: bool | None = None
    store_ok: bool | None = None
    retrieve_ok: bool | None = None
    integration_ok: bool = False
    status: str = "unknown"
    dependencies: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    storage_location: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_present(spec: dict[str, Any]) -> bool:
    module = spec["module"].replace(".", "/") + ".py"
    return (UPSTREAM_ROOT / module).is_file()


def missing_dependencies(provider_id: str) -> list[str]:
    spec = PROVIDERS[provider_id]
    return [
        name
        for name in spec["dependencies"]
        if importlib.util.find_spec(name) is None
    ]


def probe_provider(provider_id: str, *, storage_dir: str | Path | None = None) -> ProviderStatus:
    """Probe one provider without claiming an untested end-to-end result."""
    provider_id = provider_id.lower()
    if provider_id not in PROVIDERS:
        raise KeyError(f"unknown memory provider {provider_id!r}")
    spec = PROVIDERS[provider_id]
    missing = missing_dependencies(provider_id)
    status = ProviderStatus(
        provider_id=provider_id,
        display_name=spec["display_name"],
        origin=spec["origin"],
        source_present=_source_present(spec),
        import_ok=False,
        dependencies=list(spec["dependencies"]),
        missing_dependencies=missing,
        storage_location=str(storage_dir) if storage_dir else None,
        status="blocked" if missing else "unknown",
        error="missing dependencies: " + ", ".join(missing) if missing else None,
    )
    if not status.source_present:
        status.status = "missing-source"
        status.error = "vendored source file is missing"
        return status
    if missing:
        return status
    try:
        with upstream_imports():
            module = importlib.import_module(spec["module"])
            cls = getattr(module, spec["class"])
        status.import_ok = True
        status.status = "importable"
    except Exception as exc:
        status.status = "import-error"
        status.error = f"{type(exc).__name__}: {exc}"
        return status

    if provider_id == "lightweight_memory":
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as temp:
                config = {
                    "storage_dir": str(Path(temp) / "storage"),
                    "enable_longterm_provision": False,
                }
                instance = cls(config)
                initialized = bool(instance.initialize())
                status.initialize_ok = initialized
                status.status = "available" if initialized else "init-failed"
                if not initialized:
                    status.error = "initialize() returned false"
        except Exception as exc:
            status.initialize_ok = False
            status.status = "init-error"
            status.error = f"{type(exc).__name__}: {exc}"
    return status


def probe_all(*, storage_dir: str | Path | None = None) -> list[ProviderStatus]:
    return [
        probe_provider(provider_id, storage_dir=storage_dir)
        for provider_id in PROVIDERS
    ]


def register_provider(
    provider_id: str,
    *,
    module: str,
    class_name: str,
    implementation_root: str | Path,
    origin: str = "evolved",
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    """Register generated provider code for this process.

    The provider is loaded from ``implementation_root`` by file path, never by
    mutating the vendored package. Registration makes an approved evolved
    architecture loadable by the ordinary adapter.
    """
    provider_id = provider_id.strip().lower()
    spec = {
        "display_name": provider_id,
        "module": module,
        "class": class_name,
        "origin": origin,
        "dependencies": list(dependencies or []),
        "implementation_root": str(Path(implementation_root).resolve()),
    }
    PROVIDERS[provider_id] = spec
    return spec


@contextmanager
def _generated_package(root: str | Path) -> Iterator[Path]:
    """Expose a generated ``EvolveLab`` tree without touching the vendored one."""
    root_path = Path(root).resolve()
    with _PATH_LOCK:
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == "EvolveLab" or name.startswith("EvolveLab.")
        }
        for name in saved:
            del sys.modules[name]
        sys.path.insert(0, str(root_path))
        try:
            yield root_path
        finally:
            try:
                sys.path.remove(str(root_path))
            except ValueError:
                pass
            for name in [
                name
                for name in sys.modules
                if name == "EvolveLab" or name.startswith("EvolveLab.")
            ]:
                del sys.modules[name]
            sys.modules.update(saved)


def _load_generated_provider(
    provider_id: str,
    spec: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    root = Path(spec["implementation_root"])
    module_name = spec["module"]
    relative = module_name.replace(".", "/") + ".py"
    path = root / relative
    if not path.is_file():
        raise RuntimeError(
            f"generated provider module is missing: {path}"
        )
    types: dict[str, Any] = {}
    with _generated_package(root):
        import uuid

        name = f"_irina_evolved_{provider_id}_{uuid.uuid4().hex[:8]}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot load generated provider: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        cls = getattr(module, spec["class"])
        types_module = sys.modules.get("EvolveLab.memory_types")
        if types_module is not None:
            types = {
                attr: getattr(types_module, attr)
                for attr in (
                    "TrajectoryData",
                    "MemoryRequest",
                    "MemoryResponse",
                    "MemoryItem",
                    "MemoryItemType",
                    "MemoryStatus",
                )
                if hasattr(types_module, attr)
            }
        instance = cls(config or {})
    instance._irina_types = types
    instance._irina_provider_id = provider_id
    return instance


def load_provider(provider_id: str, config: dict[str, Any] | None = None) -> Any:
    """Load an actual upstream provider class and initialize one instance."""
    provider_id = provider_id.lower()
    if provider_id not in PROVIDERS:
        raise KeyError(f"unknown memory provider {provider_id!r}")
    spec = PROVIDERS[provider_id]
    if spec.get("implementation_root"):
        instance = _load_generated_provider(provider_id, spec, config or {})
    else:
        with upstream_imports():
            module = importlib.import_module(spec["module"])
            cls = getattr(module, spec["class"])
            instance = cls(config or {})
    if not instance.initialize():
        raise RuntimeError(f"upstream provider {provider_id!r} failed to initialize")
    return instance


def upstream_info() -> dict[str, Any]:
    manifest = UPSTREAM_ROOT / "UPSTREAM_MANIFEST.json"
    return {
        "root": str(UPSTREAM_ROOT),
        "commit": UPSTREAM_COMMIT,
        "manifest_present": manifest.is_file(),
        "manifest": str(manifest) if manifest.is_file() else None,
    }
