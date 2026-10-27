"""Day 1: execution-loop package.

Concept: the loop separates run state, termination policy, and the runner so
each concern can be tested and extended independently.

Design constraints:
* No provider SDK imports.
* The public surface is re-exported for convenient imports.
"""

from .runner import EventType, RunEvent, run_loop
from .state import ExecutionState, RunResult, RunStatus, tool_call_signature
from .termination import TerminationPolicy

__all__ = [
    "EventType",
    "ExecutionState",
    "RunEvent",
    "RunResult",
    "RunStatus",
    "TerminationPolicy",
    "run_loop",
    "tool_call_signature",
]
