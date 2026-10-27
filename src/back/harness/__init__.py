"""Stage 3: Irina harness composition, durable sessions, and sub-agents.

Concept: the harness is still a small synchronous execution kernel. Stage 3 adds
one composition class, trusted host-side session history, truthful crash repair,
and sequential child agents.

Design constraints:
* Standard library only and synchronous.
* Provider-specific behaviour stays inside ``providers``.
* Sessions live outside the agent-writable workspace.
* The public surface re-exports the composition and existing tool contracts.
"""

from .harness import Harness, HarnessError
from .session import (
    SESSION_DIR,
    SessionData,
    SessionError,
    SessionLocked,
    SessionRecorder,
    append,
    latest,
    load,
    new_session,
    record_resume,
    recovery_path,
    workspace_session_dir,
)
from .subagent import DelegationBudget, subagent_tool
from .tools_and_guardrails import Policy, Tool, tool

__all__ = [
    "DelegationBudget",
    "Harness",
    "HarnessError",
    "Policy",
    "SESSION_DIR",
    "SessionData",
    "SessionError",
    "SessionLocked",
    "SessionRecorder",
    "Tool",
    "append",
    "latest",
    "load",
    "new_session",
    "record_resume",
    "recovery_path",
    "subagent_tool",
    "tool",
    "workspace_session_dir",
]
