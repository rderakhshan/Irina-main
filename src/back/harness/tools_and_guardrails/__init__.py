"""Day 2: workspace tools, application guardrails, and OpenShell integration.

Concept: give Irina a small set of file and command tools, decide whether each
request is permitted, and run every operation inside an OpenShell sandbox.

Design constraints:
* Tool schemas match the existing provider contract.
* Application permissions never replace sandbox isolation.
* OpenShell is the only runtime for agent-controlled file and command work.
* The public exports are the stable surface used by the harness and demos.
"""

from .sandbox import (
    DEFAULT_IMAGE,
    OpenShellSandbox,
    SandboxError,
    SandboxResult,
    SandboxUnavailable,
    parse_worker_response,
)
from .sandbox_worker import resolve_path
from .security import (
    DENY_PATTERNS,
    READ_TOOLS,
    Policy,
    dangerous_pattern,
    is_dangerous_command,
)
from .shell import DEFAULT_SHELL, ShellConfig, ShellResult, run_command
from .tools import LocalWorkspace, Tool, core_tools, tool

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_SHELL",
    "DENY_PATTERNS",
    "LocalWorkspace",
    "OpenShellSandbox",
    "Policy",
    "READ_TOOLS",
    "SandboxError",
    "SandboxResult",
    "SandboxUnavailable",
    "ShellConfig",
    "ShellResult",
    "Tool",
    "core_tools",
    "dangerous_pattern",
    "is_dangerous_command",
    "parse_worker_response",
    "resolve_path",
    "run_command",
    "tool",
]
