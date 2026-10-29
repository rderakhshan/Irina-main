"""Jupyter notebook execution tool."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def notebook_tools(backend: Any) -> list[Tool]:
    """Return the ``execute_notebook`` tool."""

    @tool(
        "Execute a Jupyter notebook in OpenShell.",
        path="Notebook path.",
        output="Output path.",
        timeout="Cell timeout.",
        allow_errors="Continue on errors.",
        overwrite="Replace output.",
    )
    def execute_notebook(
        path: str,
        output: str = "",
        timeout: str = "120",
        allow_errors: str = "false",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "execute_notebook",
                exec_timeout=min(1900, int(float(timeout or 120)) + 60),
                path=path,
                output=output,
                timeout=timeout,
                allow_errors=allow_errors,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [execute_notebook]


__all__ = ["notebook_tools"]
