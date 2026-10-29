"""Web search tool."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def web_search_tools(backend: Any) -> list[Tool]:
    """Return the ``web_search`` tool for one workspace backend."""

    @tool(
        "Search the web; returns titles, URLs, and snippets.",
        query="Search query.",
        max_results="Max results (default 5).",
    )
    def web_search(query: str, max_results: str = "5") -> str:
        try:
            result = run_operation(
                backend,
                "web_search",
                exec_timeout=45,
                query=query,
                max_results=max_results,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [web_search]


__all__ = ["web_search_tools"]
