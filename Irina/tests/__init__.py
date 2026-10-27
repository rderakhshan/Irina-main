"""Day 1: test package for the Irina harness.

Concept: tests live beside the repository as an importable package so
``python -m unittest discover -s tests`` works from the repository root.

Design constraints:
* Standard library ``unittest`` only.
* Tests must run without network access or credentials.
"""
