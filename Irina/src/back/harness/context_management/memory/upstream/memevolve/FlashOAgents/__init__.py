"""Lazy public exports for the vendored FlashOAgents package.

Upstream imported every submodule eagerly, including web and media tools that
require optional packages (``requests``, ``crawl4ai``, ``pandas``, ``biopython``,
``python-pptx``). Irina only needs the agent core for MemEvolve analysis, so
submodules are resolved on first attribute access instead.
"""

from __future__ import annotations

import importlib

_MODULES = (
    ".agent_types",
    ".agents",
    ".memory",
    ".models",
    ".monitoring",
    ".tools",
    ".utils",
    ".search_tools",
    ".mm_tools",
)


def __getattr__(name: str):
    for module_name in _MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
