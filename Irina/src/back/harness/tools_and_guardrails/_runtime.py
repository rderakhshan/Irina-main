"""Host-side bridge from tool wrappers to the sandbox tool worker."""

from __future__ import annotations

import json
from typing import Any

TOOL_WORKER_SANDBOX_PATH = "/tmp/irina_tool_worker.py"


class ToolExecutionError(RuntimeError):
    """A sandbox tool invocation failed."""


def run_operation(
    backend: Any,
    op: str,
    *,
    exec_timeout: int = 60,
    **payload: Any,
) -> dict[str, Any]:
    """Run one operation inside the approved workspace backend.

    Args:
        backend: Workspace backend exposing ``workspace`` and ``exec``.
        op: Worker operation name.
        exec_timeout: Sandbox command timeout in seconds.
        **payload: Operation arguments.

    Returns:
        The worker's successful response.

    Raises:
        ToolExecutionError: If execution fails, times out, or the worker
            reports an error.
    """
    workspace = getattr(backend, "workspace", None)
    if not isinstance(workspace, str) or not workspace:
        raise ToolExecutionError("workspace backend is not ready")
    request = {"op": op, "workdir": workspace, **payload}
    result = backend.exec(
        ["python3", TOOL_WORKER_SANDBOX_PATH],
        workdir=workspace,
        timeout=exec_timeout,
        input_text=json.dumps(request, ensure_ascii=False),
    )
    if getattr(result, "timed_out", False):
        raise ToolExecutionError(f"{op} timed out after {exec_timeout}s")
    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    if not stdout:
        detail = stderr or f"worker exited with {getattr(result, 'exit_code', '?')}"
        raise ToolExecutionError(f"{op} produced no response: {detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        raise ToolExecutionError(
            f"{op} returned an invalid response: {stdout[:500]}"
        ) from None
    if not isinstance(payload, dict):
        raise ToolExecutionError(f"{op} returned a non-object response")
    if not payload.get("ok"):
        error = payload.get("error") or {}
        error_type = str(error.get("type", "ToolError"))
        message = str(error.get("message", "unknown sandbox error"))
        raise ToolExecutionError(f"{error_type}: {message}")
    payload.pop("ok", None)
    return payload


def format_error(exc: BaseException) -> str:
    """Render one tool failure as a compact model-readable observation."""
    return f"ERROR: {type(exc).__name__}: {exc}"
