"""Day 1: configurable termination policy.

Concept: a bounded agent loop needs explicit ceilings on turns, tool calls,
time, tokens, empty replies, repeated calls, and tool failures.

Design constraints:
* Defaults are conservative but not tuned to one application.
* Stall detection is configurable so legitimate repeated operations can be
  allowed by raising ``max_repeated_calls`` or disabling detection.
* Token accounting is a soft upper bound: output tokens are unknown until a
  provider call returns, and no output reservation is made on Day 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TerminationPolicy:
    """Ceilings used by the execution loop.

    Attributes:
        max_turns: Maximum provider completion calls.
        max_tool_calls: Maximum model-requested tool calls processed.
        max_seconds: Maximum elapsed monotonic seconds.
        max_total_tokens: Soft ceiling on accumulated input plus output tokens.
        max_repeated_calls: Consecutive identical tool calls allowed before the
            run is classified as stalled.
        max_empty_responses: Consecutive empty assistant replies allowed.
        max_consecutive_tool_errors: Consecutive failed or blocked tool results
            allowed before the run is classified as a tool error.
        enable_stall_detection: Set to ``False`` to allow unbounded repetition.
    """

    max_turns: int = 40
    max_tool_calls: int = 100
    max_seconds: float = 300
    max_total_tokens: int = 200000
    max_repeated_calls: int = 3
    max_empty_responses: int = 2
    max_consecutive_tool_errors: int = 5
    enable_stall_detection: bool = True
