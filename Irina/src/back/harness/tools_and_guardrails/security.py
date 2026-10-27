"""Day 2: application-level permission policy and command guardrails.

Concept: decide whether Irina may request an operation before a tool runs. This
is a policy layer above the runtime; it never replaces OpenShell's filesystem,
network, process, or credential enforcement.

Design constraints:
* Dangerous-command patterns are defensive heuristics, not a sandbox.
* ``read-only``, ``safe``, and ``yolo`` modes cannot weaken OpenShell.
* Approvals are explicit: a missing approver or a non-``True`` answer refuses.
* The same command checks apply to Bash and PowerShell-style commands.
"""

from __future__ import annotations

import re
from typing import Any, Callable

READ_TOOLS = frozenset({"read_file", "list_files", "grep"})
COMMAND_TOOLS = frozenset({"bash", "shell", "powershell", "pwsh"})
MODES = frozenset({"read-only", "safe", "yolo"})

DENY_PATTERNS: tuple[str, ...] = (
    # rm with both recursive and force flags, targeting root or home. Kept
    # deliberately broad so ``bash -c "rm -rf /"`` is caught too.
    r"\brm\b(?=[^\n]*\s(?:-[A-Za-z]*r[A-Za-z]*|--recursive)\b)"
    r"(?=[^\n]*\s(?:-[A-Za-z]*f[A-Za-z]*|--force)\b)"
    r"[^\n]*\s[\"']?(?:/\*|/|~|\$HOME|/home(?:/[^/\s\"']+)?)"
    r"[\"']?(?=\s|$|;|&)",
    r"(?:^|[;&|(])\s*sudo\b",
    r"(?:^|[;&|(])\s*mkfs(?:\.[a-z0-9]+)?\b",
    r"(?:^|[;&|(])\s*dd\b[^\n]*\bif\s*=",
    r"(?:^|[;&|(])\s*(?:curl|wget)\b[^\n|]*\|"
    r"\s*(?:sudo\s+)?(?:sh|bash|zsh|dash)\b",
    r"(?:^|[;&|(])\s*git\s+push\b[^\n]*(?:--force(?:-with-lease)?|-f)\b",
    r"(?:>{1,2}|\btee\b)\s*/dev/sd[a-z]",
    # PowerShell and Windows command variants.
    r"\bRemove-Item\b[^\n]*(?:-Recurse|-Force|\s-[rR]\b)",
    r"\bRemove-Item\b[^\n]*(?:[A-Za-z]:\\|\\\*|~|\$HOME)",
    r"\b(?:Clear-Disk|Format-Volume|Initialize-Disk)\b",
    r"\b(?:rmdir|del)\b[^\n]*(?:/s|/q|/f)"
    r"[^\n]*(?:[A-Za-z]:\\|~|\$HOME|\\\*)",
)

_COMPILED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in DENY_PATTERNS
)


def dangerous_pattern(command: str) -> str | None:
    """Return the first dangerous-command pattern matching ``command``.

    Args:
        command: Shell command text.

    Returns:
        The matching pattern, or ``None`` when no rule matches.
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(command):
            return pattern.pattern
    return None


def is_dangerous_command(command: str) -> bool:
    """Return whether ``command`` matches a dangerous-command rule."""
    return dangerous_pattern(command) is not None


def _call_name(call: Any) -> str | None:
    """Read a tool name from a ToolCall-like object or dictionary."""
    name = getattr(call, "name", None)
    if name is None and isinstance(call, dict):
        name = call.get("name")
    return name if isinstance(name, str) else None


def _call_args(call: Any) -> dict[str, Any]:
    """Read tool arguments from a ToolCall-like object or dictionary."""
    args = getattr(call, "args", None)
    if args is None and isinstance(call, dict):
        args = call.get("args")
    return args if isinstance(args, dict) else {}


class Policy:
    """Application-level permission policy for tool requests.

    Args:
        mode: ``read-only``, ``safe``, or ``yolo``.
        approver: Optional callback ``approver(call, reason) -> bool`` used by
            ``safe`` mode. Without it, approval-required operations are refused.

    Raises:
        ValueError: If the mode is unknown.
    """

    def __init__(
        self,
        mode: str = "safe",
        approver: Callable[[Any, str], bool] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(
                f"unknown policy mode {mode!r}; expected one of {sorted(MODES)}"
            )
        self.mode = mode
        self.approver = approver

    def check(self, call: Any) -> str | None:
        """Return ``None`` to allow a call or a reason string to block it.

        Args:
            call: ToolCall-like object with ``name`` and ``args``.

        Returns:
            ``None`` when permitted, otherwise a descriptive refusal.
        """
        name = _call_name(call)
        if name is None:
            return "tool request has no name"
        args = _call_args(call)

        if name in COMMAND_TOOLS or "command" in args:
            command = args.get("command")
            if isinstance(command, str):
                matched = dangerous_pattern(command)
                if matched is not None:
                    return (
                        "dangerous command blocked by policy: "
                        f"matched {matched}"
                    )

        if name in READ_TOOLS:
            return None
        if self.mode == "yolo":
            return None
        if self.mode == "read-only":
            return f"{name} is not allowed in read-only mode"

        reason = self._reason(name)
        if self.approver is None:
            return (
                f"approval required for {name}, but no approver is configured"
            )
        try:
            approved = self.approver(call, reason)
        except Exception as exc:
            return (
                "approval callback failed: "
                f"{type(exc).__name__}: {exc}"
            )
        if approved is True:
            return None
        return f"approval denied for {name}"

    @staticmethod
    def _reason(name: str) -> str:
        """Return a short explanation shown to the approver."""
        reasons = {
            "write_file": "writes a file in the workspace",
            "edit_file": "edits a file in the workspace",
            "bash": "runs a command inside the sandbox",
        }
        return reasons.get(name, "requests a non-read-only operation")
