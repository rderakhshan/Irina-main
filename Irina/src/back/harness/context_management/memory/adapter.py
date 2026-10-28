"""Adapter from Irina's stable memory contract to original EvolveLab providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...tools_and_guardrails.security import redact_secrets
from .experience_store import ExperienceRecord
from .provider_registry import load_provider, upstream_imports
from .retrieval_bridge import MemoryQuery, RetrievedMemory
from .trajectory_bridge import upstream_payload


@runtime_checkable
class MemoryBackend(Protocol):
    """Stable contract consumed by the harness and migration manager."""

    provider_id: str

    def add_experience(self, experience: ExperienceRecord) -> "MemoryUpdateResult": ...

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]: ...

    def health_check(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class MemoryUpdateResult:
    """Outcome of one provider update; never inferred from a model claim."""

    ok: bool
    description: str
    provider_id: str
    experience_id: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryModelResponse:
    """Small response object understood by upstream provider extractors."""

    def __init__(self, content: str) -> None:
        self.content = content


class DeepSeekMemoryModel:
    """Expose Irina's DeepSeek provider as the callable upstream expects."""

    def __init__(self, provider: Any, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    def _flatten(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict):
                    pieces.append(str(item.get("text", "")))
                else:
                    pieces.append(str(item))
            return "\n".join(piece for piece in pieces if piece)
        return str(content)

    def __call__(self, messages: list[dict[str, Any]]) -> MemoryModelResponse:
        neutral = []
        for message in messages:
            role = str(message.get("role", "user"))
            text = self._flatten(message.get("content", ""))
            if role not in {"user", "assistant"}:
                role = "user"
            neutral.append({"role": role, "text": text})
        response = self.provider.complete(
            self.model or getattr(self.provider, "default_model", ""),
            "",
            neutral,
            [],
        )
        return MemoryModelResponse(redact_secrets(response.text))


class EvolveLabBackend:
    """Concrete adapter around one original EvolveLab provider."""

    def __init__(
        self,
        provider_id: str,
        *,
        storage_dir: str | Path,
        config: dict[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        settings = dict(config or {})
        settings.setdefault("storage_dir", str(self.storage_dir))
        if provider_id == "expel":
            settings.setdefault(
                "insights_file_path",
                str(self.storage_dir / "insights.json"),
            )
            settings.setdefault(
                "success_trajectories_file_path",
                str(self.storage_dir / "success_trajectories.json"),
            )
        if model is not None:
            settings["model"] = model
        self.config = settings
        self.provider = load_provider(provider_id, settings)
        self._closed = False

    def add_experience(self, experience: ExperienceRecord) -> MemoryUpdateResult:
        if self._closed:
            return MemoryUpdateResult(
                False,
                "backend is closed",
                self.provider_id,
                experience.experience_id,
                error="backend is closed",
            )
        payload = upstream_payload(experience)
        types = getattr(self.provider, "_irina_types", {}) or {}
        try:
            TrajectoryData = types.get("TrajectoryData")
            if TrajectoryData is None:
                with upstream_imports():
                    from EvolveLab.memory_types import (
                        TrajectoryData as _TrajectoryData,
                    )

                TrajectoryData = _TrajectoryData
            data = TrajectoryData(**payload)
            ok, description = self.provider.take_in_memory(data)
            return MemoryUpdateResult(
                bool(ok),
                str(description),
                self.provider_id,
                experience.experience_id,
                error=None if ok else str(description),
                metadata={
                    "verified_success": experience.verified_success,
                    "run_status": experience.run_status,
                },
            )
        except Exception as exc:
            return MemoryUpdateResult(
                False,
                "",
                self.provider_id,
                experience.experience_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        if self._closed:
            return []
        types = getattr(self.provider, "_irina_types", {}) or {}
        try:
            MemoryRequest = types.get("MemoryRequest")
            MemoryStatus = types.get("MemoryStatus")
            if MemoryRequest is None or MemoryStatus is None:
                with upstream_imports():
                    from EvolveLab.memory_types import (
                        MemoryRequest as _MemoryRequest,
                        MemoryStatus as _MemoryStatus,
                    )

                MemoryRequest = _MemoryRequest
                MemoryStatus = _MemoryStatus
            request = MemoryRequest(
                query=query.query,
                context=query.context,
                status=MemoryStatus.BEGIN,
                additional_params=query.metadata or {},
            )
            response = self.provider.provide_memory(request)
            results = []
            for item in response.memories:
                content = getattr(item, "content", None)
                if not content:
                    continue
                results.append(
                    RetrievedMemory(
                        memory_id=str(getattr(item, "id", "")),
                        content=str(content),
                        score=(
                            float(item.score)
                            if getattr(item, "score", None) is not None
                            else None
                        ),
                        provider=self.provider_id,
                        metadata=dict(getattr(item, "metadata", {}) or {}),
                    )
                )
            return results
        except Exception:
            # Retrieval is nonfatal when the provider is unavailable.
            return []

    def health_check(self) -> dict[str, Any]:
        return {
            "ok": not self._closed,
            "provider_id": self.provider_id,
            "storage_dir": str(self.storage_dir),
            "initialized": not self._closed,
        }

    def close(self) -> None:
        self._closed = True
        self.provider = None
