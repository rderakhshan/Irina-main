"""PDF and document tool group."""

from __future__ import annotations

from typing import Any

from ..tools import Tool
from .docx import docx_tools
from .latex import latex_tools
from .pdf import pdf_tools
from .pptx import pptx_tools
from .xlsx import xlsx_tools


def document_tools(workdir: str, backend: Any) -> list[Tool]:
    """Return the nine PDF and document tools for one workspace."""
    del workdir
    tools: list[Tool] = []
    tools.extend(pdf_tools(backend))
    tools.extend(latex_tools(backend))
    tools.extend(docx_tools(backend))
    tools.extend(xlsx_tools(backend))
    tools.extend(pptx_tools(backend))
    return tools


__all__ = ["document_tools"]
