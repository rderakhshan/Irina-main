"""Versioned memory architecture registry with atomic activation/rollback."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def register(self, record: ArchitectureRecord) -> ArchitectureRecord:
        with self._lock:
            if record.architecture_id in self._data["architectures"]:
                raise ArchitectureError(
                    f"architecture {record.architecture_id!r} already exists"
                )
            self._data["architectures"][record.architecture_id] = record.to_dict()
            self._save()
            return record

    def get(self, architecture_id: str) -> ArchitectureRecord:
        raw = self._data["architectures"].get(architecture_id)
        if raw is None:
            raise ArchitectureError(f"unknown architecture {architecture_id!r}")
        return ArchitectureRecord(**raw)

    def list(self) -> list[ArchitectureRecord]:
        return [
            ArchitectureRecord(**raw)
            for raw in self._data["architectures"].values()
        ]

    def active(self) -> ArchitectureRecord | None:
        active_id = self._data.get("active")
        return self.get(active_id) if active_id else None

    def set_state(self, architecture_id: str, state: str) -> ArchitectureRecord:
        if state not in STATES:
            raise ArchitectureError(f"invalid state {state!r}")
        with self._lock:
            record = self.get(architecture_id)
            if record.state == "active" and state != "active":
                raise ArchitectureError("deactivate the active architecture with rollback")
            record.state = state
            if state == "validated":
                record.validation_status = "validated"
            self._data["architectures"][architecture_id] = record.to_dict()
            self._save()
            return record

    def ensure_baseline(
        self,
        provider_id: str,
        *,
        storage_namespace: str,
        upstream_commit: str | None = None,
    ) -> ArchitectureRecord:
        with self._lock:
            active = self.active()
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
            baseline_id = f"baseline-{provider_id}"
            if baseline_id in self._data["architectures"]:
                index = 2
                while f"{baseline_id}-v{index}" in self._data["architectures"]:
                    index += 1
                baseline_id = f"{baseline_id}-v{index}"
            record = ArchitectureRecord(
                architecture_id=baseline_id,
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
            self._data["architectures"][baseline_id] = record.to_dict()
            self._data["active"] = baseline_id
            self._data["previous"] = previous_id
            self._save()
            return record

    def activate(self, architecture_id: str) -> ArchitectureRecord:
        with self._lock:
            candidate = self.get(architecture_id)
            if candidate.state not in {"validated", "evaluated", "eligible"}:
                raise ArchitectureError(
                    f"architecture {architecture_id!r} is not eligible "
                    f"(state={candidate.state})"
                )
            previous_id = self._data.get("active")
            if previous_id:
                previous = self.get(previous_id)
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
        with self._lock:
            previous_id = self._data.get("previous")
            if not previous_id:
                return None
            previous = self.get(previous_id)
            current_id = self._data.get("active")
            if current_id:
                current = self.get(current_id)
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
        active = self.active()
        return {
            "active": active.to_dict() if active else None,
            "previous": self._data.get("previous"),
            "count": len(self._data["architectures"]),
            "path": str(self.path),
        }
