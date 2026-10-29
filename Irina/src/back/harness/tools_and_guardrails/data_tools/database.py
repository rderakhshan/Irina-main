"""SQLite and SQL database tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def database_tools(backend: Any) -> list[Tool]:
    """Return ``sql_query`` and ``database_schema``."""

    @tool(
        "Run one SQL statement on a SQLite database.",
        database="DB path or connection URL.",
        query="SQL statement.",
        params="JSON parameters.",
        read_only="Reject writes.",
        max_rows="Max rows.",
    )
    def sql_query(
        database: str,
        query: str,
        params: str = "[]",
        read_only: str = "false",
        max_rows: str = "200",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "sql_query",
                exec_timeout=120,
                database=database,
                query=query,
                params=params,
                read_only=read_only,
                max_rows=max_rows,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Inspect tables, columns, indexes, and keys.",
        database="DB path or connection URL.",
        table="Table name.",
    )
    def database_schema(database: str, table: str = "") -> str:
        try:
            result = run_operation(
                backend,
                "database_schema",
                exec_timeout=120,
                database=database,
                table=table,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [sql_query, database_schema]


__all__ = ["database_tools"]
