"""PowerPoint generation tool."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def pptx_tools(backend: Any) -> list[Tool]:
    """Return the ``create_pptx`` tool."""

    @tool(
        "Create a PowerPoint presentation.",
        path="Output .pptx path.",
        content="Text or JSON slides.",
        title="Title.",
        overwrite="Replace existing.",
    )
    def create_pptx(
        path: str,
        content: str,
        title: str = "",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "create_pptx",
                exec_timeout=120,
                path=path,
                content=content,
                title=title,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [create_pptx]


__all__ = ["pptx_tools"]
