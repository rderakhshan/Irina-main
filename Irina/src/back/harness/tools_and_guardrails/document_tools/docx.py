"""Word document tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def docx_tools(backend: Any) -> list[Tool]:
    """Return ``read_docx`` and ``write_docx``."""

    @tool(
        "Read DOCX paragraphs and tables.",
        path="DOCX path.",
        max_paragraphs="Max paragraphs.",
        max_tables="Max tables.",
    )
    def read_docx(
        path: str,
        max_paragraphs: str = "500",
        max_tables: str = "20",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "read_docx",
                exec_timeout=120,
                path=path,
                max_paragraphs=max_paragraphs,
                max_tables=max_tables,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Create or edit a DOCX.",
        path="Output .docx path.",
        content="Text or JSON sections/tables.",
        title="Title.",
        append="Append to existing.",
        overwrite="Replace existing.",
    )
    def write_docx(
        path: str,
        content: str,
        title: str = "",
        append: str = "false",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "write_docx",
                exec_timeout=120,
                path=path,
                content=content,
                title=title,
                append=append,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [read_docx, write_docx]


__all__ = ["docx_tools"]
