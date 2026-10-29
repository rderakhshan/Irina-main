"""Database and data tool group."""

from __future__ import annotations

from typing import Any

from ..tools import Tool
from .database import database_tools
from .datasets import dataset_tools
from .notebook import notebook_tools


def data_tools(workdir: str, backend: Any) -> list[Tool]:
    """Return the five database and data tools for one workspace."""
    del workdir
    tools: list[Tool] = []
    tools.extend(database_tools(backend))
    tools.extend(dataset_tools(backend))
    tools.extend(notebook_tools(backend))
    return tools


__all__ = ["data_tools"]
