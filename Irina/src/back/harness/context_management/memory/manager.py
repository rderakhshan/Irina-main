"""Irina memory manager: canonical history, active provider, retrieval, updates."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapter import EvolveLabBackend, MemoryBackend, MemoryUpdateResult
from .architecture_registry import (
    ArchitectureError,
    ArchitectureRecord,
    ArchitectureRegistry,
)
from .configuration import MemoryConfig
from .experience_store import ExperienceRecord, ExperienceStore
from .migration import ReplayManager
from .provider_registry import UPSTREAM_COMMIT, PROVIDERS
from .retrieval_bridge import MemoryQuery, RetrievedMemory, render_context


class MemoryManager:
    """One active provider over a durable, architecture-independent history."""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        workspace: str | Path,
        model: Any | None = None,
        backend: MemoryBackend | None = None,
        state_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.workspace = str(Path(workspace).resolve())
        self.root = Path(state_root or config.root).expanduser().resolve()
        self.store = ExperienceStore(self.root, self.workspace)
        self.registry = ArchitectureRegistry(self.root)
        self._lock = threading.RLock()
        self._backend = backend
        self._model = model
        self._last_retrieval: list[RetrievedMemory] = []
        self._update_path = self.root / "updates.jsonl"
        self._updated = self._load_updated_ids()
        self._active = self.registry.ensure_baseline(
            config.active_provider,
            storage_namespace=str(
                self.root / "providers" / config.active_provider
            ),
            upstream_commit=UPSTREAM_COMMIT,
        )
        if self.config.enabled and self._backend is None:
            self._backend = self._load_backend(config.active_provider)

    def _load_updated_ids(self) -> set[str]:
        if not self._update_path.exists():
            return set()
        result: set[str] = set()
        for line in self._update_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("ok") and value.get("experience_id"):
                result.add(str(value["experience_id"]))
        return result

    def _record_update(self, result: MemoryUpdateResult) -> None:
        payload = {
            "experience_id": result.experience_id,
            "provider_id": result.provider_id,
            "ok": result.ok,
            "description": result.description,
            "error": result.error,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with self._update_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if result.ok:
            self._updated.add(result.experience_id)

    def _load_backend(
        self,
        provider_id: str,
        *,
        storage_dir: str | Path | None = None,
    ) -> MemoryBackend:
        provider_id = provider_id.lower()
        if provider_id not in PROVIDERS:
            raise ValueError(f"unknown provider {provider_id!r}")
        settings = self.config.provider_settings(provider_id)
        if storage_dir is not None:
            settings["storage_dir"] = str(Path(storage_dir))
            if provider_id == "expel":
                base = Path(settings["storage_dir"])
                settings["insights_file_path"] = str(base / "insights.json")
                settings["success_trajectories_file_path"] = str(
                    base / "success_trajectories.json"
                )
        if self._model is None:
            configured = settings.get("model")
            self._model = configured if callable(configured) else None
        return EvolveLabBackend(
            provider_id,
            storage_dir=Path(settings["storage_dir"]),
            config=settings,
            model=self._model,
        )

    @property
    def backend(self) -> MemoryBackend | None:
        return self._backend

    @property
    def last_retrieval(self) -> list[RetrievedMemory]:
        return list(self._last_retrieval)

    @property
    def active_architecture(self) -> Any:
        return self._active

    def before_task(
        self,
        task: str,
        *,
        context: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Retrieve relevant lower-trust memory context for a task."""
        if not self.config.enabled or not self.config.retrieve_before_task:
            return ""
        if self._backend is None:
            return ""
        query = MemoryQuery(
            query=task,
            context=context,
            max_tokens=max_tokens or self.config.max_context_tokens,
        )
        try:
            memories = self._backend.retrieve(query)
        except Exception:
            memories = []
        self._last_retrieval = memories
        return render_context(
            memories,
            max_tokens=query.max_tokens,
            provider=getattr(self._backend, "provider_id", ""),
        )

    def after_task(self, record: ExperienceRecord) -> MemoryUpdateResult | None:
        """Persist canonical history, then update the active provider once."""
        if not self.config.enabled:
            return None
        added = self.store.add(record)
        if not self.config.learn_after_task:
            return None
        if self._backend is None:
            result = MemoryUpdateResult(
                False,
                "no active memory backend",
                self.config.active_provider,
                record.experience_id,
                error="no active memory backend",
            )
            self._record_update(result)
            return result
        if record.experience_id in self._updated and not added:
            return None
        with self._lock:
            result = self._backend.add_experience(record)
            self._record_update(result)
            return result

    def activate_architecture(self, architecture_id: str) -> Any:
        """Activate a validated architecture only after a backend is available."""
        with self._lock:
            record = self.registry.get(architecture_id)
            if record.validation_status not in {"validated", "evaluated"}:
                raise ValueError(
                    "architecture has not been validated/evaluated"
                )
            provider_id = str(
                record.configuration.get("provider", self.config.active_provider)
            ).lower()
            new_backend = self._load_backend(
                provider_id,
                storage_dir=record.storage_namespace
                or self.root / "providers" / provider_id,
            )
            old = self._backend
            self._backend = new_backend
            try:
                activated = self.registry.activate(architecture_id)
            except Exception:
                self._backend = old
                new_backend.close()
                raise
            self.config.active_provider = provider_id
            self._active = activated
            if old is not None:
                old.close()
            return activated

    def rollback_architecture(self) -> Any:
        """Restore the previous architecture and its isolated storage."""
        with self._lock:
            previous = self.registry.get(self.registry.status()["previous"]) if self.registry.status()["previous"] else None
            if previous is None:
                return None
            provider_id = str(
                previous.configuration.get("provider", self.config.active_provider)
            ).lower()
            new_backend = self._load_backend(
                provider_id,
                storage_dir=previous.storage_namespace
                or self.root / "providers" / provider_id,
            )
            old = self._backend
            self._backend = new_backend
            try:
                activated = self.registry.rollback()
            except Exception:
                self._backend = old
                new_backend.close()
                raise
            self.config.active_provider = provider_id
            self._active = activated
            if old is not None:
                old.close()
            return activated

    def select_provider(self, provider_id: str) -> None:
        """Explicitly switch providers; this never runs automatically."""
        provider_id = provider_id.strip().lower()
        if provider_id not in PROVIDERS:
            raise ValueError(f"unknown provider {provider_id!r}")
        with self._lock:
            new_backend = self._load_backend(provider_id)
            old = self._backend
            self._backend = new_backend
            self.config.active_provider = provider_id
            if old is not None:
                old.close()
            self._active = self.registry.ensure_baseline(
                provider_id,
                storage_namespace=str(self.root / "providers" / provider_id),
                upstream_commit=UPSTREAM_COMMIT,
            )

    def health_check(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "active_provider": self.config.active_provider,
            "canonical": self.store.health(),
            "architecture": self.registry.status(),
            "backend": (
                self._backend.health_check() if self._backend is not None else None
            ),
        }

    def status(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "active_architecture": self._active.to_dict(),
            "canonical": self.store.health(),
            "last_retrieval_count": len(self._last_retrieval),
            "providers": [
                {"id": key, "display_name": value["display_name"]}
                for key, value in PROVIDERS.items()
            ],
        }

    def replay(
        self,
        *,
        target_architecture: str,
        target_provider: str,
        source_architecture: str | None = None,
    ) -> Any:
        """Replay canonical history into a separate target namespace."""
        target_settings = self.config.provider_settings(target_provider)
        target_settings["storage_dir"] = str(
            self.root / "providers" / target_provider / target_architecture
        )
        try:
            self.registry.get(target_architecture)
        except ArchitectureError:
            self.registry.register(
                ArchitectureRecord(
                    architecture_id=target_architecture,
                    origin="evolved",
                    parent_architecture=(
                        source_architecture or self._active.architecture_id
                    ),
                    implementation_path=target_settings["storage_dir"],
                    storage_namespace=target_settings["storage_dir"],
                    configuration={"provider": target_provider},
                    validation_status="not_validated",
                    state="created",
                )
            )
        backend = EvolveLabBackend(
            target_provider,
            storage_dir=Path(target_settings["storage_dir"]),
            config=target_settings,
            model=self._model,
        )
        try:
            result = ReplayManager(self.store, self.registry).replay(
                source_architecture=(
                    source_architecture or self._active.architecture_id
                ),
                target_architecture=target_architecture,
                target_backend=backend,
            )
            if result.validation_status == "validated":
                try:
                    self.registry.set_state(target_architecture, "validated")
                except ArchitectureError:
                    pass
            return result
        finally:
            backend.close()

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
