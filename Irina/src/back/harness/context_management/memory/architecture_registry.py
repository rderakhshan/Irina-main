"""Versioned memory architecture registry with atomic activation/rollback.

The registry is shared by concurrent Irina processes (the running agent and CLI
administration commands). Every mutation reloads the on-disk state while holding
a cross-process lock, then writes the whole document to a temporary file and
atomically replaces the registry. A failed write leaves the previous registry
untouched.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

STATES = {
    "discovered",
    "created",
    "validated",
    "evaluated",
    "eligible",
    "active",
    "rejected",
    "archived",
}

LOCK_STALE_SECONDS = 60


class ArchitectureError(RuntimeError):
    """Invalid architecture state or registry operation."""


@dataclass
class ArchitectureRecord:
    architecture_id: str
    architecture_version: int = 1
    origin: str = "upstream"
    upstream_commit: str | None = None
    parent_architecture: str | None = None
    implementation_path: str = ""
    storage_namespace: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    validation_status: str = "discovered"
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    activated_at: str | None = None
    state: str = "created"

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ArchitectureError(f"invalid architecture state {self.state!r}")
        if not self.architecture_id:
            raise ArchitectureError("architecture_id must be non-empty")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArchitectureRegistry:
    """Single authoritative pointer to the active architecture."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "architectures.json"
        self.lock_path = self.root / "architectures.lock"
        self._lock = threading.RLock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"active": None, "previous": None, "architectures": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchitectureError(f"cannot read architecture registry: {exc}") from None
        if not isinstance(data, dict):
            raise ArchitectureError("architecture registry must be an object")
        data.setdefault("active", None)
        data.setdefault("previous", None)
        data.setdefault("architectures", {})
        return data

    def _reload(self) -> None:
        self._data = self._load()

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def _acquire_file_lock(self) -> None:
        deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                except OSError:
                    age = 0
                if age > LOCK_STALE_SECONDS:
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise ArchitectureError("architecture registry is locked")
                time.sleep(0.05)

    def _release_file_lock(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ArchitectureError(f"could not release registry lock: {exc}")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._lock:
            self._acquire_file_lock()
            try:
                self._reload()
                yield
            finally:
                self._release_file_lock()

    def _active_record(self) -> ArchitectureRecord | None:
        active_id = self._data.get("active")
        if not active_id:
            return None
        raw = self._data["architectures"].get(active_id)
        return ArchitectureRecord(**raw) if raw else None

    def register(self, record: ArchitectureRecord) -> ArchitectureRecord:
        with self._guard():
            if record.architecture_id in self._data["architectures"]:
                raise ArchitectureError(
                    f"architecture {record.architecture_id!r} already exists"
                )
            self._data["architectures"][record.architecture_id] = record.to_dict()
            self._save()
            return record

    def get(self, architecture_id: str) -> ArchitectureRecord:
        with self._lock:
            self._reload()
            raw = self._data["architectures"].get(architecture_id)
            if raw is None:
                raise ArchitectureError(f"unknown architecture {architecture_id!r}")
            return ArchitectureRecord(**raw)

    def list(self) -> list[ArchitectureRecord]:
        with self._lock:
            self._reload()
            return [
                ArchitectureRecord(**raw)
                for raw in self._data["architectures"].values()
            ]

    def active(self) -> ArchitectureRecord | None:
        with self._lock:
            self._reload()
            return self._active_record()

    def set_state(self, architecture_id: str, state: str) -> ArchitectureRecord:
        if state not in STATES:
            raise ArchitectureError(f"invalid state {state!r}")
        with self._guard():
            record = self._get_locked(architecture_id)
            if record.state == "active" and state != "active":
                raise ArchitectureError("deactivate the active architecture with rollback")
            record.state = state
            if state == "validated":
                record.validation_status = "validated"
            self._data["architectures"][architecture_id] = record.to_dict()
            self._save()
            return record

    def evaluate(
        self,
        architecture_id: str,
        *,
        metrics: dict[str, Any],
        ok: bool,
    ) -> ArchitectureRecord:
        """Record evaluation evidence without activating the candidate."""
        with self._guard():
            record = self._get_locked(architecture_id)
            if record.state == "active":
                raise ArchitectureError("the active architecture cannot be re-evaluated")
            record.evaluation_metrics = dict(metrics)
            if ok:
                record.validation_status = "evaluated"
                if record.state in {"discovered", "created"}:
                    record.state = "evaluated"
            else:
                record.validation_status = "failed"
                record.state = "rejected"
            self._data["architectures"][architecture_id] = record.to_dict()
            self._save()
            return record

    def _get_locked(self, architecture_id: str) -> ArchitectureRecord:
        raw = self._data["architectures"].get(architecture_id)
        if raw is None:
            raise ArchitectureError(f"unknown architecture {architecture_id!r}")
        return ArchitectureRecord(**raw)

    def _next_baseline_id(self, provider_id: str) -> str:
        baseline_id = f"baseline-{provider_id}"
        if baseline_id in self._data["architectures"]:
            index = 2
            while f"{baseline_id}-v{index}" in self._data["architectures"]:
                index += 1
            baseline_id = f"{baseline_id}-v{index}"
        return baseline_id

    def ensure_baseline(
        self,
        provider_id: str,
        *,
        storage_namespace: str,
        upstream_commit: str | None = None,
    ) -> ArchitectureRecord:
        """Return the active architecture, creating a baseline only if none exists.

        An existing active architecture always wins so that a restart cannot
        silently archive an approved candidate.
        """
        with self._guard():
            active = self._active_record()
            if active is not None:
                return active
            return self._create_baseline_locked(
                provider_id,
                storage_namespace=storage_namespace,
                upstream_commit=upstream_commit,
                previous_id=None,
            )

    def switch_provider(
        self,
        provider_id: str,
        *,
        storage_namespace: str,
        upstream_commit: str | None = None,
    ) -> ArchitectureRecord:
        """Explicitly activate the baseline for a different provider."""
        with self._guard():
            active = self._active_record()
            if (
                active is not None
                and active.configuration.get("provider") == provider_id
            ):
                return active
            previous_id = active.architecture_id if active is not None else None
            if active is not None:
                active.state = "archived"
                self._data["architectures"][active.architecture_id] = (
                    active.to_dict()
                )
            return self._create_baseline_locked(
                provider_id,
                storage_namespace=storage_namespace,
                upstream_commit=upstream_commit,
                previous_id=previous_id,
            )

    def _create_baseline_locked(
        self,
        provider_id: str,
        *,
        storage_namespace: str,
        upstream_commit: str | None,
        previous_id: str | None,
    ) -> ArchitectureRecord:
        record = ArchitectureRecord(
            architecture_id=self._next_baseline_id(provider_id),
            architecture_version=1,
            origin="upstream",
            upstream_commit=upstream_commit,
            implementation_path="upstream/memevolve/EvolveLab/providers",
            storage_namespace=storage_namespace,
            configuration={"provider": provider_id},
            dependencies=[],
            validation_status="validated",
            state="active",
            activated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._data["architectures"][record.architecture_id] = record.to_dict()
        self._data["active"] = record.architecture_id
        self._data["previous"] = previous_id
        self._save()
        return record

    def activate(self, architecture_id: str) -> ArchitectureRecord:
        with self._guard():
            candidate = self._get_locked(architecture_id)
            if candidate.state not in {"validated", "evaluated", "eligible"}:
                raise ArchitectureError(
                    f"architecture {architecture_id!r} is not eligible "
                    f"(state={candidate.state})"
                )
            previous_id = self._data.get("active")
            if previous_id:
                previous = self._get_locked(previous_id)
                previous.state = "archived"
                self._data["architectures"][previous_id] = previous.to_dict()
            self._data["previous"] = previous_id
            candidate.state = "active"
            candidate.activated_at = datetime.now(timezone.utc).isoformat()
            self._data["architectures"][architecture_id] = candidate.to_dict()
            self._data["active"] = architecture_id
            self._save()
            return candidate

    def rollback(self) -> ArchitectureRecord | None:
        with self._guard():
            previous_id = self._data.get("previous")
            if not previous_id:
                return None
            previous = self._get_locked(previous_id)
            current_id = self._data.get("active")
            if current_id:
                current = self._get_locked(current_id)
                current.state = "archived"
                self._data["architectures"][current_id] = current.to_dict()
            previous.state = "active"
            previous.activated_at = datetime.now(timezone.utc).isoformat()
            self._data["architectures"][previous_id] = previous.to_dict()
            self._data["active"] = previous_id
            self._data["previous"] = current_id
            self._save()
            return previous

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reload()
            active = self._active_record()
            return {
                "active": active.to_dict() if active else None,
                "previous": self._data.get("previous"),
                "count": len(self._data["architectures"]),
                "path": str(self.path),
            }
