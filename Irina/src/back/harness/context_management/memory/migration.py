"""Canonical replay for moving memory architectures without losing history."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .adapter import MemoryBackend
from .architecture_registry import ArchitectureError, ArchitectureRegistry
from .experience_store import ExperienceRecord, ExperienceStore


class MigrationError(RuntimeError):
    """A replay or migration contract failed."""


@dataclass
class MigrationResult:
    source_architecture: str
    target_architecture: str
    canonical_schema_version: int
    source_experience_count: int
    migrated_experience_count: int
    failed_experience_count: int
    migration_started_at: str
    migration_completed_at: str
    validation_status: str
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayManager:
    """Replay canonical records into an inactive target backend."""

    def __init__(
        self,
        store: ExperienceStore,
        registry: ArchitectureRegistry,
        *,
        migration_dir: str | Path | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.migration_dir = Path(
            migration_dir or (store.root / "migrations")
        ).resolve()
        self.migration_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, target_architecture: str) -> Path:
        safe = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in target_architecture
        )
        return self.migration_dir / f"{safe}.json"

    def _write_metadata(self, result: MigrationResult) -> None:
        path = self._metadata_path(result.target_architecture)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def replay(
        self,
        *,
        source_architecture: str,
        target_architecture: str,
        target_backend: MemoryBackend,
        start_index: int = 0,
        validate: Callable[[MemoryBackend, list[ExperienceRecord]], bool] | None = None,
    ) -> MigrationResult:
        """Replay all canonical records in deterministic order.

        The source store and active backend are never modified. A target
        backend is expected to use an isolated storage namespace.
        """
        records = sorted(
            self.store.load(),
            key=lambda record: (record.timestamp or "", record.experience_id),
        )
        if start_index < 0 or start_index > len(records):
            raise MigrationError("start_index is outside the canonical history")
        started = datetime.now(timezone.utc).isoformat()
        failed: list[str] = []
        migrated = 0
        for record in records[start_index:]:
            try:
                outcome = target_backend.add_experience(record)
                if not outcome.ok:
                    failed.append(
                        f"{record.experience_id}: {outcome.error or 'update failed'}"
                    )
                else:
                    migrated += 1
            except Exception as exc:
                failed.append(
                    f"{record.experience_id}: {type(exc).__name__}: {exc}"
                )
        validation_status = "not_validated"
        if validate is not None and not failed:
            try:
                validation_status = (
                    "validated"
                    if validate(target_backend, records[start_index:])
                    else "validation_failed"
                )
            except Exception as exc:
                validation_status = f"validation_error: {type(exc).__name__}: {exc}"
        result = MigrationResult(
            source_architecture=source_architecture,
            target_architecture=target_architecture,
            canonical_schema_version=1,
            source_experience_count=len(records),
            migrated_experience_count=migrated,
            failed_experience_count=len(failed),
            migration_started_at=started,
            migration_completed_at=datetime.now(timezone.utc).isoformat(),
            validation_status=validation_status,
            errors=failed,
        )
        self._write_metadata(result)
        if failed:
            try:
                self.registry.set_state(target_architecture, "rejected")
            except ArchitectureError:
                pass
        return result
