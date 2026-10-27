"""Stage 3: sequential sub-agent delegation.

Concept: expose one ``spawn_agent`` tool that creates a fresh child harness,
runs a self-contained task, and returns the child's final report as a normal
tool observation. Children share the approved workspace and policy but never
the parent's conversation.

Design constraints:
* Sequential and synchronous only.
* No child session is persisted.
* Depth, turn, time, and total-task limits bound delegated work.
* The child inherits the parent's policy and sandbox; no new yolo policy.
* A child result is not assumed to be a string; it must be a structured result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..tools_and_guardrails.tools import Tool, tool

MAX_RESULT_CHARS = 12000


@dataclass
class DelegationBudget:
    """Shared total delegated-task counter for a parent/child tree."""

    max_tasks: int = 4
    used: int = 0

    def consume(self) -> bool:
        """Reserve one delegated task, returning false when exhausted."""
        if self.max_tasks >= 0 and self.used >= self.max_tasks:
            return False
        self.used += 1
        return True


def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Bound a child report for the parent observation."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [sub-agent report truncated]"


def subagent_tool(
    make_harness: Callable[[int], Any],
    depth: int = 0,
    max_depth: int = 2,
    *,
    budget: DelegationBudget | None = None,
    max_tasks: int = 4,
) -> Tool:
    """Create the ``spawn_agent`` tool.

    Args:
        make_harness: Factory called with ``depth + 1`` to create a child.
        depth: Current agent depth. A top-level Harness is depth 0.
        max_depth: Maximum delegated depth.
        budget: Optional shared delegated-work budget.
        max_tasks: Local task limit when no shared budget is supplied.

    Returns:
        A Tool named ``spawn_agent``.
    """
    local_budget = budget if budget is not None else DelegationBudget(max_tasks)

    @tool(
        "Delegate a self-contained task to a fresh child agent. The child has "
        "its own conversation and task state, shares the approved project "
        "workspace, and returns a final report. The parent remains responsible "
        "for final integration and verification. Children run sequentially.",
        task=(
            "Self-contained task for the child. Include expected output files "
            "and verification steps because the child cannot see this "
            "conversation."
        ),
    )
    def spawn_agent(task: str) -> str:
        if depth >= max_depth:
            return "ERROR: sub-agent depth limit reached; do this task yourself"
        if not isinstance(task, str) or not task.strip():
            return "ERROR: sub-agent task must not be empty"
        if not local_budget.consume():
            return "ERROR: sub-agent task limit reached; continue in the parent"

        try:
            child = make_harness(depth + 1)
        except Exception as exc:
            return (
                "ERROR: could not create sub-agent: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            result = child.run(task)
            if not hasattr(result, "completed") or not hasattr(result, "text"):
                return (
                    "ERROR: child harness returned an unsupported result; "
                    "expected a RunResult"
                )
            text = _clip(str(getattr(result, "text", "") or ""))
            if result.completed:
                return f"[sub-agent completed]\n{text}"
            status = getattr(getattr(result, "status", None), "value", "unknown")
            reason = str(getattr(result, "reason", ""))
            return (
                f"ERROR: sub-agent finished with status {status}: {reason}\n"
                f"{text}"
            )
        except Exception as exc:
            return (
                "ERROR: sub-agent failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            close = getattr(child, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return spawn_agent


__all__ = ["DelegationBudget", "subagent_tool"]
