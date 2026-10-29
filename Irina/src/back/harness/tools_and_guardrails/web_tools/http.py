"""Generic HTTP request and download tools."""

from __future__ import annotations

import json
from typing import Any

from .._runtime import format_error, run_operation
from ..tools import Tool, tool


def http_tools(backend: Any) -> list[Tool]:
    """Return ``http_request`` and ``download_url`` for one backend."""

    @tool(
        "Send an HTTP/API request.",
        method="HTTP method.",
        url="HTTP(S) URL.",
        headers="JSON headers.",
        body="Body text.",
        timeout="Timeout seconds.",
        max_bytes="Max response bytes.",
    )
    def http_request(
        method: str,
        url: str,
        headers: str = "{}",
        body: str = "",
        timeout: str = "30",
        max_bytes: str = "2000000",
    ) -> str:
        try:
            result = run_operation(
                backend,
                "http_request",
                exec_timeout=min(180, int(float(timeout or 30)) + 15),
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout=timeout,
                max_bytes=max_bytes,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    @tool(
        "Download a file into the workspace.",
        url="HTTP(S) URL.",
        destination="Workspace path.",
        overwrite="Replace existing file.",
    )
    def download_url(
        url: str, destination: str, overwrite: str = "false"
    ) -> str:
        try:
            result = run_operation(
                backend,
                "download_url",
                timeout=300,
                url=url,
                destination=destination,
                overwrite=overwrite,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return format_error(exc)

    return [http_request, download_url]


__all__ = ["http_tools"]
