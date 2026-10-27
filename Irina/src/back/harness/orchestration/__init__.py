"""Control-plane orchestration for the Irina harness.

This package owns composition, durable session orchestration, sub-agent
delegation, the CLI front door, and parallel fleet execution.
"""

from .fleet import FleetError, FleetJob, run_fleet
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

__all__ = [
    "DelegationBudget",
    "FleetError",
    "FleetJob",
    "Harness",
    "HarnessError",
    "SESSION_DIR",
    "SessionData",
    "SessionError",
    "SessionLocked",
    "SessionRecorder",
    "append",
    "latest",
    "load",
    "new_session",
    "record_resume",
    "recovery_path",
    "run_fleet",
    "subagent_tool",
    "workspace_session_dir",
]
