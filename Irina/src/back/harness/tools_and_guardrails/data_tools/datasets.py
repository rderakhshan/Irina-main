"""Structured dataset tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def dataset_tools(backend: Any) -> list[Tool]:
    """Return ``read_data`` and ``write_data``."""

    @tool(
        "Read CSV, JSON, Parquet, or Excel data.",
        path="Dataset path.",
        format="Format override.",
        sheet="Excel sheet.",
        max_rows="Max rows.",
    )
    def read_data(
        path: str,
        format: str = "",
        sheet: str = "",
        max_rows: str = "200",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "read_data",
                exec_timeout=120,
                path=path,
                format=format,
                sheet=sheet,
                max_rows=max_rows,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Write CSV, JSON, Parquet, or Excel data.",
        path="Output path.",
        data="JSON records or {columns, rows}.",
        format="Format override.",
        overwrite="Replace existing.",
        index="Include index.",
    )
    def write_data(
        path: str,
        data: str,
        format: str = "",
        overwrite: str = "false",
        index: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "write_data",
                exec_timeout=180,
                path=path,
                data=data,
                format=format,
                overwrite=overwrite,
                index=index,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [read_data, write_data]


__all__ = ["dataset_tools"]
