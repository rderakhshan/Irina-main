"""Memory configuration with safe defaults for the Irina runtime."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KNOWN_PROVIDERS = frozenset(
    {
        "agent_kb",
        "skillweaver",
        "mobilee",
        "expel",
        "voyager",
        "dilu",
        "generative",
        "memp",
        "dynamic_cheatsheet",
        "agent_workflow_memory",
        "evolver",
        "lightweight_memory",
        "cerebra_fusion_memory",
    }
)


class MemoryConfigError(ValueError):
    """Invalid memory configuration."""


@dataclass
class EvolutionConfig:
    """MemEvolve controls. Evolution is off unless explicitly enabled."""

    enabled: bool = False
    auto_promote: bool = False
    require_approval: bool = True
    analysis_model: str | None = None
    generation_model: str | None = None
    work_dir: str | None = None


@dataclass
class MemoryConfig:
    """Irina memory configuration.

    Ordinary execution defaults to disabled so existing Harness users do not
    unexpectedly create memory state. Enable it explicitly in a config file or
    with ``IRINA_MEMORY_ENABLED=1``.
    """

    enabled: bool = False
    backend: str = "evolvelab"
    active_provider: str = "lightweight_memory"
    learn_after_task: bool = True
    retrieve_before_task: bool = True
    max_context_tokens: int = 3_000
    state_dir: str | None = None
    provider_config: dict[str, Any] = field(default_factory=dict)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    def __post_init__(self) -> None:
        self.active_provider = str(self.active_provider).strip().lower()
        if self.active_provider not in KNOWN_PROVIDERS:
            raise MemoryConfigError(
                f"unknown memory provider {self.active_provider!r}"
            )
        if self.backend not in {"evolvelab"}:
            raise MemoryConfigError(f"unknown memory backend {self.backend!r}")
        if self.max_context_tokens <= 0:
            raise MemoryConfigError("max_context_tokens must be positive")
        if not isinstance(self.provider_config, dict):
            raise MemoryConfigError("provider_config must be an object")

    @property
    def root(self) -> Path:
        raw = self.state_dir or os.environ.get("IRINA_MEMORY_STATE_DIR")
        return (
            Path(raw).expanduser().resolve()
            if raw
            else (Path.home() / ".irina" / "memory").resolve()
        )

    def provider_settings(self, provider_id: str | None = None) -> dict[str, Any]:
        provider = provider_id or self.active_provider
        values = dict(self.provider_config.get(provider, {}))
        values.setdefault("storage_dir", str(self.root / "providers" / provider))
        if provider == "lightweight_memory":
            # Irina uses the upstream evolved provider as a learning baseline;
            # long-term provision is off by default upstream.
            values.setdefault("enable_longterm_provision", True)
        elif provider == "expel":
            base = Path(values["storage_dir"])
            values.setdefault(
                "insights_file_path", str(base / "insights.json")
            )
            values.setdefault(
                "success_trajectories_file_path",
                str(base / "success_trajectories.json"),
            )
        return values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MemoryConfig":
        if not isinstance(value, dict):
            raise MemoryConfigError("memory config must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known - {"memory"}
        if unknown:
            raise MemoryConfigError(
                "unknown memory config keys: " + ", ".join(sorted(unknown))
            )
        data = dict(value)
        evolution = data.pop("evolution", {})
        if not isinstance(evolution, dict):
            raise MemoryConfigError("memory.evolution must be an object")
        try:
            evo = EvolutionConfig(**evolution)
            return cls(evolution=evo, **data)
        except TypeError as exc:
            raise MemoryConfigError(str(exc)) from None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MemoryConfig":
        if path is None:
            env_path = os.environ.get("IRINA_MEMORY_CONFIG")
            path = Path(env_path) if env_path else None
        if path is not None:
            candidate = Path(path).expanduser()
            if candidate.exists():
                if candidate.suffix.lower() == ".toml":
                    value = tomllib.loads(candidate.read_text(encoding="utf-8"))
                else:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                if "memory" in value:
                    value = value["memory"]
                return cls.from_mapping(value)
        config = cls.from_env()
        config.apply_selection()
        return config

    def apply_selection(self) -> None:
        """Apply an explicit provider selection from the memory state root."""
        selection = self.root / "selection.json"
        if not selection.is_file():
            return
        try:
            payload = json.loads(selection.read_text(encoding="utf-8"))
            provider = str(payload.get("active_provider", "")).lower()
            if provider in KNOWN_PROVIDERS:
                self.active_provider = provider
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    def persist_selection(self) -> Path:
        """Atomically persist the active provider selection."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "selection.json"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                {"active_provider": self.active_provider},
                handle,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        enabled = os.environ.get("IRINA_MEMORY_ENABLED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        provider = os.environ.get(
            "IRINA_MEMORY_PROVIDER", "lightweight_memory"
        ).lower()
        tokens_raw = os.environ.get("IRINA_MEMORY_MAX_CONTEXT_TOKENS", "3000")
        try:
            tokens = int(tokens_raw)
        except ValueError:
            raise MemoryConfigError(
                "IRINA_MEMORY_MAX_CONTEXT_TOKENS must be an integer"
            ) from None
        evolution_enabled = os.environ.get(
            "IRINA_MEMORY_EVOLUTION_ENABLED", ""
        ).lower() in {"1", "true", "yes", "on"}
        auto_promote = os.environ.get(
            "IRINA_MEMORY_AUTO_PROMOTE", ""
        ).lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            active_provider=provider,
            max_context_tokens=tokens,
            state_dir=os.environ.get("IRINA_MEMORY_STATE_DIR"),
            evolution=EvolutionConfig(
                enabled=evolution_enabled,
                auto_promote=auto_promote,
                require_approval=not auto_promote,
            ),
        )
