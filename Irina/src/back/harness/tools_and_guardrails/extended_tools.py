"""Unified construction of the extended tool surface.

The twelve web/document/data modules only describe provider schemas and forward
one request to the sandbox tool worker. This function keeps the single registry
decision in one place.
"""

from __future__ import annotations

from typing import Any

from .data_tools import data_tools
from .document_tools import document_tools
from .tools import Tool
from .web_tools import web_tools


def extended_tools(workdir: str, backend: Any) -> list[Tool]:
    """Return all 18 extended tools for one approved workspace."""
    tools: list[Tool] = []
    tools.extend(web_tools(workdir, backend))
    tools.extend(document_tools(workdir, backend))
    tools.extend(data_tools(workdir, backend))
    return tools


__all__ = ["extended_tools"]
