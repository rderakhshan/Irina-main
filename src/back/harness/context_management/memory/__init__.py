"""Irina memory: legacy IRINA.md plus canonical experience and EvolveLab integration."""

from .adapter import (
    DeepSeekMemoryModel,
    EvolveLabBackend,
    MemoryBackend,
    MemoryUpdateResult,
)
from .architecture_registry import (
    ArchitectureError,
    ArchitectureRecord,
    ArchitectureRegistry,
)
from .configuration import EvolutionConfig, MemoryConfig, MemoryConfigError
from .evolution_bridge import (
    EvolutionBridge,
    EvolutionResult,
    EvolutionStatus,
    EvolutionUnavailable,
    export_trajectories,
)
from .experience_store import (
    ExperienceRecord,
    ExperienceStore,
    ExperienceStoreError,
    stable_experience_id,
    workspace_id,
)
from .legacy import (
    BASE_SYSTEM_PROMPT,
    MEMORY_FILE,
    build_system_prompt,
    memory_tools,
    read_memory,
    remember,
)
from .manager import MemoryManager
from .migration import MigrationError, MigrationResult, ReplayManager
from .provider_registry import (
    PROVIDERS,
    UPSTREAM_COMMIT,
    ProviderStatus,
    load_provider,
    probe_all,
    probe_provider,
    upstream_imports,
    upstream_info,
)
from .retrieval_bridge import (
    MemoryQuery,
    RetrievedMemory,
    render_context,
)
from .security import CandidateValidation, validate_candidate
from .trajectory_bridge import record_from_result, upstream_payload

__all__ = [
    "ArchitectureError",
    "ArchitectureRecord",
    "ArchitectureRegistry",
    "BASE_SYSTEM_PROMPT",
    "CandidateValidation",
    "DeepSeekMemoryModel",
    "EvolveLabBackend",
    "EvolutionBridge",
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionStatus",
    "EvolutionUnavailable",
    "ExperienceRecord",
    "ExperienceStore",
    "ExperienceStoreError",
    "export_trajectories",
    "MEMORY_FILE",
    "MemoryBackend",
    "MemoryConfig",
    "MemoryConfigError",
    "MemoryManager",
    "MemoryQuery",
    "MemoryUpdateResult",
    "MigrationError",
    "MigrationResult",
    "PROVIDERS",
    "ProviderStatus",
    "ReplayManager",
    "RetrievedMemory",
    "UPSTREAM_COMMIT",
    "build_system_prompt",
    "load_provider",
    "memory_tools",
    "probe_all",
    "probe_provider",
    "read_memory",
    "record_from_result",
    "remember",
    "render_context",
    "stable_experience_id",
    "upstream_imports",
    "upstream_info",
    "upstream_payload",
    "validate_candidate",
    "workspace_id",
]
