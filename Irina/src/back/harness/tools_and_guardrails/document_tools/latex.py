"""LaTeX compilation tool."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def latex_tools(backend: Any) -> list[Tool]:
    """Return the ``compile_latex`` tool."""

    @tool(
        "Compile a .tex file to PDF.",
        path="Input .tex.",
        output="Output .pdf.",
        engine="LaTeX engine.",
        runs="Compilation passes.",
        overwrite="Replace existing PDF.",
    )
    def compile_latex(
        path: str,
        output: str = "",
        engine: str = "pdflatex",
        runs: str = "2",
        overwrite: str = "false",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "compile_latex",
                exec_timeout=600,
                path=path,
                output=output,
                engine=engine,
                runs=runs,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [compile_latex]


__all__ = ["latex_tools"]
