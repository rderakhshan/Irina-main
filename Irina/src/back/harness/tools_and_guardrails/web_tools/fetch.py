"""Web page retrieval tool."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def web_fetch_tools(backend: Any) -> list[Tool]:
    """Return the ``web_fetch`` tool for one workspace backend."""

    @tool(
        "Fetch a URL and extract readable text.",
        url="HTTP(S) URL.",
        max_chars="Max characters (default 12000).",
    )
    def web_fetch(url: str, max_chars: str = "12000") -> str:
        try:
            result = run_operation(
                backend,
                "web_fetch",
                exec_timeout=45,
                url=url,
                max_chars=max_chars,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [web_fetch]


__all__ = ["web_fetch_tools"]
