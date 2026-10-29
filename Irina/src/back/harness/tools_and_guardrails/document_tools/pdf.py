"""PDF read, create, and edit tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def pdf_tools(backend: Any) -> list[Tool]:
    """Return ``read_pdf``, ``create_pdf``, and ``edit_pdf``."""

    @tool(
        "Extract PDF text, pages, and metadata.",
        path="PDF path.",
        pages="Pages, e.g. '1-3,5'.",
        max_chars="Max characters.",
    )
    def read_pdf(
        path: str, pages: str = "", max_chars: str = "20000"
    ) -> str:
        try:
            result = run_operation(
                backend,
                "read_pdf",
                exec_timeout=120,
                path=path,
                pages=pages,
                max_chars=max_chars,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Create a PDF from text or structured JSON.",
        path="Output .pdf path.",
        content="Text or JSON (title/paragraphs/tables).",
        title="Title.",
        overwrite="Replace existing file.",
    )
    def create_pdf(
        path: str,
        content: str,
        title: str = "",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "create_pdf",
                exec_timeout=120,
                path=path,
                content=content,
                title=title,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Merge, split, extract, or rotate PDFs.",
        operation="merge/split/extract/rotate.",
        output="Output path.",
        input="Input PDF.",
        inputs="JSON input list (merge).",
        pages="Pages.",
        angle="Rotation (default 90).",
        overwrite="Replace existing files.",
    )
    def edit_pdf(
        operation: str,
        output: str,
        input: str = "",
        inputs: str = "[]",
        pages: str = "",
        angle: str = "90",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "edit_pdf",
                exec_timeout=180,
                operation=operation,
                output=output,
                input=input,
                inputs=inputs,
                pages=pages,
                angle=angle,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [read_pdf, create_pdf, edit_pdf]


__all__ = ["pdf_tools"]
