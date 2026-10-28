"""Memory administration commands: status, providers, registry, rollback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .architecture_registry import ArchitectureError, ArchitectureRegistry
from .configuration import MemoryConfig, MemoryConfigError
from .evolution_bridge import (
    EvolutionBridge,
    EvolutionUnavailable,
    dependency_status,
    export_trajectories,
)
from .adapter import DeepSeekMemoryModel
from .experience_store import ExperienceStore
from .manager import MemoryManager
from ...providers.deepseek import DeepSeekProvider
from .provider_registry import (
    PROVIDERS,
    UPSTREAM_COMMIT,
    UPSTREAM_ROOT,
    probe_all,
    upstream_info,
)
from .security import validate_candidate


def _config(args: argparse.Namespace) -> MemoryConfig:
    config = MemoryConfig.load(args.config)
    if args.provider:
        config.active_provider = args.provider
    if args.state_dir:
        config.state_dir = args.state_dir
        config.apply_selection()
    return config


def _status(args: argparse.Namespace) -> int:
    config = _config(args)
    root = config.root
    registry = ArchitectureRegistry(root)
    payload = {
        "config": config.to_dict(),
        "root": str(root),
        "architecture": registry.status(),
        "upstream": upstream_info(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _providers(args: argparse.Namespace) -> int:
    config = _config(args)
    rows = [status.as_dict() for status in probe_all(storage_dir=config.root / "providers")]
    for row in rows:
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


def _inspect(args: argparse.Namespace) -> int:
    config = _config(args)
    store = ExperienceStore(config.root, args.workdir)
    records = store.load()
    print(json.dumps(
        {
            "workspace_id": store.workspace_id,
            "path": str(store.path),
            "count": len(records),
            "latest": [
                {
                    "experience_id": record.experience_id,
                    "task_id": record.task_id,
                    "run_status": record.run_status,
                    "verified_success": record.verified_success,
                    "timestamp": record.timestamp,
                }
                for record in records[-10:]
            ],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


def _select(args: argparse.Namespace) -> int:
    config = _config(args)
    config.active_provider = args.provider
    registry = ArchitectureRegistry(config.root)
    namespace = config.root / "providers" / args.provider
    try:
        activated = registry.switch_provider(
            args.provider,
            storage_namespace=str(namespace),
            upstream_commit=UPSTREAM_COMMIT,
        )
    except ArchitectureError as exc:
        print(f"Provider selection failed: {exc}", file=sys.stderr)
        return 1
    path = config.persist_selection()
    print(
        f"Selected {config.active_provider} "
        f"(architecture {activated.architecture_id}); wrote {path}"
    )
    return 0


def _replay(args: argparse.Namespace) -> int:
    config = _config(args)
    config.enabled = True
    try:
        model = DeepSeekMemoryModel(DeepSeekProvider())
    except Exception as exc:
        print(f"Replay requires the configured provider model: {exc}", file=sys.stderr)
        return 2
    manager = MemoryManager(config, workspace=args.workdir, model=model)
    try:
        result = manager.replay(
            target_architecture=args.target_architecture,
            target_provider=args.target_provider,
            source_architecture=args.source_architecture,
        )
    finally:
        manager.close()
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.failed_experience_count == 0 else 1


def _candidates(args: argparse.Namespace) -> int:
    config = _config(args)
    registry = ArchitectureRegistry(config.root)
    print(json.dumps(
        {
            "active": registry.status()["active"],
            "previous": registry.status()["previous"],
            "architectures": [record.to_dict() for record in registry.list()],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


def _promote(args: argparse.Namespace) -> int:
    config = _config(args)
    registry = ArchitectureRegistry(config.root)
    if not args.approve:
        print(
            "Promotion requires --approve. The candidate remains inactive.",
            file=sys.stderr,
        )
        return 2
    if config.enabled:
        manager = MemoryManager(config, workspace=args.workdir)
        try:
            record = manager.activate_architecture(args.candidate)
        except Exception as exc:
            print(f"Promotion failed: {exc}", file=sys.stderr)
            return 1
        finally:
            manager.close()
    else:
        try:
            record = registry.activate(args.candidate)
        except ArchitectureError as exc:
            print(f"Promotion failed: {exc}", file=sys.stderr)
            return 1
    print(f"Activated {record.architecture_id}")
    return 0


def _rollback(args: argparse.Namespace) -> int:
    config = _config(args)
    registry = ArchitectureRegistry(config.root)
    if config.enabled:
        manager = MemoryManager(config, workspace=args.workdir)
        try:
            record = manager.rollback_architecture()
        finally:
            manager.close()
    else:
        record = registry.rollback()
    if record is None:
        print("No previous architecture is available.", file=sys.stderr)
        return 1
    print(f"Rolled back to {record.architecture_id}")
    return 0


def _python_root(path: Path) -> str:
    for parent in (path.parent, *path.parents):
        if parent.name == "EvolveLab":
            return str(parent.parent)
    return str(path.parent)


def _evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    registry = ArchitectureRegistry(config.root)
    try:
        record = registry.get(args.candidate)
    except ArchitectureError as exc:
        print(f"Unknown candidate: {exc}", file=sys.stderr)
        return 1
    implementation = Path(record.implementation_path)
    known_module = str(record.configuration.get("module") or "")
    known_root = record.configuration.get("implementation_root")
    if known_module and known_root:
        implementation = (
            Path(known_root)
            / Path(*known_module.split(".")).with_suffix(".py")
        )
    elif implementation.is_dir():
        ecosystem = implementation / "EvolveLab" / "providers"
        candidates = sorted(ecosystem.glob("*.py")) if ecosystem.is_dir() else []
        modules = [
            path for path in candidates
            if path.name not in {"__init__.py"} and "provider" in path.name
        ]
        if not modules:
            print(
                "Candidate has no importable provider module under "
                f"{ecosystem}",
                file=sys.stderr,
            )
            return 1
        implementation = modules[-1]
    validation = validate_candidate(
        implementation,
        python_paths=[_python_root(implementation), str(UPSTREAM_ROOT)],
    )
    try:
        updated = registry.evaluate(
            args.candidate,
            metrics=validation.as_dict(),
            ok=validation.ok,
        )
    except ArchitectureError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    payload = {
        "candidate": updated.architecture_id,
        "state": updated.state,
        "validation_status": updated.validation_status,
        "implementation_path": str(implementation),
        "validation": validation.as_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if validation.ok else 1


def _evolve(args: argparse.Namespace) -> int:
    config = _config(args)
    status = dependency_status()
    if not status.available:
        detail = (
            "missing: " + ", ".join(status.missing_dependencies)
            if status.missing_dependencies
            else status.detail
        )
        print(
            "Original MemEvolve is vendored but cannot start: " + detail,
            file=sys.stderr,
        )
        return 2
    logs: str | None = args.task_logs
    if args.from_canonical or not logs:
        store = ExperienceStore(config.root, args.workdir)
        destination = Path(args.work_dir) / "task_logs"
        written = export_trajectories(store, destination)
        if not written:
            print(
                "No canonical experiences are available to evolve from.",
                file=sys.stderr,
            )
            return 1
        logs = str(destination)
        print(
            f"Exported {len(written)} canonical experience(s) to {destination}",
            file=sys.stderr,
        )
    registry = None if args.no_register else ArchitectureRegistry(config.root)
    bridge = EvolutionBridge(
        work_dir=args.work_dir,
        analysis_model=args.analysis_model,
        generation_model=args.generation_model,
        registry=registry,
        state_root=config.root,
        workspace=args.workdir,
    )
    try:
        result = bridge.run(
            logs,
            default_provider=args.provider or config.active_provider,
            max_analysis_steps=args.max_steps,
            validation_dataset=args.validation_dataset,
        )
    except EvolutionUnavailable as exc:
        print(f"Evolution unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irina memory")
    parser.add_argument("--config", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--workdir", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.set_defaults(func=_status)
    providers = sub.add_parser("providers")
    providers.set_defaults(func=_providers)
    inspect = sub.add_parser("inspect")
    inspect.set_defaults(func=_inspect)
    select = sub.add_parser("select")
    select.add_argument("provider", choices=sorted(PROVIDERS))
    select.set_defaults(func=_select)
    replay = sub.add_parser("replay")
    replay.add_argument("target_architecture")
    replay.add_argument("--target-provider", required=True, choices=sorted(PROVIDERS))
    replay.add_argument("--source-architecture", default=None)
    replay.set_defaults(func=_replay)
    candidates = sub.add_parser("candidates")
    candidates.set_defaults(func=_candidates)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("candidate")
    evaluate.set_defaults(func=_evaluate)
    promote = sub.add_parser("promote")
    promote.add_argument("candidate")
    promote.add_argument("--approve", action="store_true")
    promote.set_defaults(func=_promote)
    rollback = sub.add_parser("rollback")
    rollback.set_defaults(func=_rollback)
    evolve = sub.add_parser("evolve")
    evolve.add_argument("task_logs", nargs="?", default=None)
    evolve.add_argument("--work-dir", required=True)
    evolve.add_argument("--from-canonical", action="store_true")
    evolve.add_argument("--no-register", action="store_true")
    evolve.add_argument("--validation-dataset", default=None)
    evolve.add_argument("--analysis-model", default=None)
    evolve.add_argument("--generation-model", default=None)
    evolve.add_argument("--max-steps", type=int, default=20)
    evolve.set_defaults(func=_evolve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    # Upstream evolution stages print Unicode diagnostics; keep Windows
    # consoles from turning a successful run into a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass
    # Memory administration commands need the same provider credentials as
    # normal execution (replay constructs the configured memory model).
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except MemoryConfigError as exc:
        print(f"Memory configuration error: {exc}", file=sys.stderr)
        return 2
