"""Memory administration commands: status, providers, registry, rollback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .architecture_registry import ArchitectureError, ArchitectureRegistry
from .configuration import MemoryConfig, MemoryConfigError
from .evolution_bridge import EvolutionBridge, EvolutionUnavailable, dependency_status
from .adapter import DeepSeekMemoryModel
from .experience_store import ExperienceStore
from .manager import MemoryManager
from ...providers.deepseek import DeepSeekProvider
from .provider_registry import PROVIDERS, probe_all, upstream_info


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
    path = config.persist_selection()
    print(f"Selected {config.active_provider}; wrote {path}")
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


def _evolve(args: argparse.Namespace) -> int:
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
    bridge = EvolutionBridge(
        work_dir=args.work_dir,
        analysis_model=args.analysis_model,
        generation_model=args.generation_model,
    )
    try:
        result = bridge.run(
            args.task_logs,
            default_provider=args.provider,
            max_analysis_steps=args.max_steps,
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
    promote = sub.add_parser("promote")
    promote.add_argument("candidate")
    promote.add_argument("--approve", action="store_true")
    promote.set_defaults(func=_promote)
    rollback = sub.add_parser("rollback")
    rollback.set_defaults(func=_rollback)
    evolve = sub.add_parser("evolve")
    evolve.add_argument("task_logs")
    evolve.add_argument("--work-dir", required=True)
    evolve.add_argument("--analysis-model", default=None)
    evolve.add_argument("--generation-model", default=None)
    evolve.add_argument("--max-steps", type=int, default=20)
    evolve.set_defaults(func=_evolve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except MemoryConfigError as exc:
        print(f"Memory configuration error: {exc}", file=sys.stderr)
        return 2
