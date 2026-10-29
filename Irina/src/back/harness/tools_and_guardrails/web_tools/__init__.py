"""Web and network tool group."""

from __future__ import annotations

from typing import Any

from ..tools import Tool
from .fetch import web_fetch_tools
from .http import http_tools
from .search import web_search_tools


def web_tools(workdir: str, backend: Any) -> list[Tool]:
    """Return the four web and network tools for one workspace."""
    del workdir
    tools: list[Tool] = []
    tools.extend(web_search_tools(backend))
    tools.extend(web_fetch_tools(backend))
    tools.extend(http_tools(backend))
    return tools


__all__ = ["web_tools"]
