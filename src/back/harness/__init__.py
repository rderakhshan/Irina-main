"""Day 1: Irina agent harness.

Concept: the harness is a small execution kernel, not a framework. It owns the
bounded execution loop, explicit run state, and provider-neutral tool protocol.

Design constraints:
* Standard library only and synchronous.
* Provider-specific behaviour stays inside ``providers``.
* Later capabilities (memory, orchestration, compaction) are not implemented on
  Day 1, but the loop and provider contracts leave room for them.
"""
