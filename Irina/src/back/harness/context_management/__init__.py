"""Stage 2: context compaction, durable memory, and workspace skills.

Concept: keep long tasks within a message budget, remember stable project facts
across sessions, and discover reusable instructions without changing Python code.

Design constraints:
* Context compaction runs in the trusted harness; file-backed memory and skills
  use the existing workspace backend (OpenShell in production).
* Memory and skills are lower-trust project context, never permission grants.
* No second agent loop, provider, or tool registry is introduced.
"""

from .context import (
    CHARS_PER_TOKEN,
    COMPACTION_INSTRUCTION,
    KEEP_RECENT,
    SUMMARY_PREFIX,
    CompactionError,
    TaskState,
    TaskStateObserver,
    calibrate,
    calibration_ratio,
    compact,
    estimate_tokens,
    reset_calibration,
)
from .memory import (
    BASE_SYSTEM_PROMPT,
    MEMORY_FILE,
    build_system_prompt,
    memory_tools,
    read_memory,
    remember,
)
from .skills import (
    SKILLS_DIR,
    USE_SKILL_TOOL,
    catalog,
    catalog_prompt,
    read_skill,
    skill_tools,
)

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "CHARS_PER_TOKEN",
    "COMPACTION_INSTRUCTION",
    "CompactionError",
    "KEEP_RECENT",
    "MEMORY_FILE",
    "SKILLS_DIR",
    "SUMMARY_PREFIX",
    "TaskState",
    "TaskStateObserver",
    "USE_SKILL_TOOL",
    "build_system_prompt",
    "calibrate",
    "calibration_ratio",
    "catalog",
    "catalog_prompt",
    "compact",
    "estimate_tokens",
    "memory_tools",
    "read_memory",
    "read_skill",
    "remember",
    "reset_calibration",
    "skill_tools",
]
