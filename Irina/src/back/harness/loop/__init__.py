"""Day 1: execution-loop package.

Concept: the loop separates run state, termination policy, and the runner so
each concern can be tested and extended independently.

Design constraints:
* No provider SDK imports.
* Modules are imported directly until the loop lands in a later milestone.
"""
