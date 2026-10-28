#!/usr/bin/env python
# coding=utf-8

"""
MemEvolve - Memory System Evolution Framework

Directory structure:
- core/: Core evolution logic (MemoryEvolver, AutoEvolver)
- phases/: Evolution phases (Analyzer, Generator, Validator, Creator)
- validators/: System validators (SWEAgentValidator)
- utils/: Utility tools (trajectory tools, run_provider)

Irina compatibility patch: public names are exported lazily. Upstream imported
the analyzer (and therefore FlashOAgents web/media tools) merely to import the
package, which made the evolution core unimportable without optional research
dependencies. Lazy exports keep the upstream class names and algorithms intact
while allowing `MemEvolve.core.memory_evolver` to load on its own.
"""

from __future__ import annotations

import importlib

_MODULES = (
    ".core.memory_evolver",
    ".core.auto_evolver",
    ".phases.memory_creator",
    ".phases.phase_analyzer",
    ".phases.phase_generator",
    ".phases.phase_validator",
    ".utils.trajectory_tools",
    ".utils.run_provider",
    ".validators.swe_agent_validator",
)

__all__ = [
    'MemoryEvolver',
    'AutoEvolver',
    'MemorySystemCreator',
    'PhaseAnalyzer',
    'PhaseGenerator',
    'PhaseValidator',
    'TrajectoryViewerTool',
    'StepViewerTool',
    'TrajectoryFeedbackAggregator',
    'run_provider',
    'SWEAgentValidator'
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _MODULES:
        module = importlib.import_module(module_name, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
