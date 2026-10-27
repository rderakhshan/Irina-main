"""Day 1: Irina source root.

Concept: the harness lives under a stable package path so it can be imported
from the repository root without installing a package.

Design constraints:
* Python 3.10+ standard library only.
* The import path is anchored at ``src.back.harness``.
* No runtime side effects on import.
"""
