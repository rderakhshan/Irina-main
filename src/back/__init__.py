"""Day 1: backend package namespace for Irina.

Concept: backend code is grouped under ``src.back`` so the agent harness can
grow alongside datasets, evaluations, and experiments.

Design constraints:
* Standard library only.
* Importing this package must not configure providers or read credentials.
"""
