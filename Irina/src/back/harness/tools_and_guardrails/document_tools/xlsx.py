"""Spreadsheet tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def xlsx_tools(backend: Any) -> list[Tool]:
    """Return ``read_xlsx`` and ``write_xlsx``."""

    @tool(
        "Read Excel worksheets and cells.",
        path="Workbook path.",
        sheet="Sheet name.",
        max_rows="Max rows.",
        max_cols="Max columns.",
    )
    def read_xlsx(
        path: str,
        sheet: str = "",
        max_rows: str = "200",
        max_cols: str = "50",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "read_xlsx",
                exec_timeout=120,
                path=path,
                sheet=sheet,
                max_rows=max_rows,
                max_cols=max_cols,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Create or modify an Excel workbook.",
        path="Output .xlsx path.",
        content="JSON rows or sheets.",
        sheet="Sheet name.",
        append="Append.",
        overwrite="Replace.",
    )
    def write_xlsx(
        path: str,
        content: str,
        sheet: str = "Sheet1",
        append: str = "false",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "write_xlsx",
                exec_timeout=120,
                path=path,
                content=content,
                sheet=sheet,
                append=append,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [read_xlsx, write_xlsx]


__all__ = ["xlsx_tools"]
