"""Day 2: six workspace tools and their JSON schemas.

Concept: expose a small, consistent tool surface for reading, writing, editing,
listing, searching, and running commands. Every tool is bound to one approved
workspace; with OpenShell, all operations go through the sandbox worker.

Design constraints:
* Schemas match the existing provider's ``{"schema": {...}}`` shape.
* Every argument is a string in the schema, as required by the harness contract.
* File operations are independently testable through a local file-only backend;
  command execution never falls back to the host.
* Path checks run where the operation runs. The sandbox worker owns the
  sandbox-side resolver; local mode only exists for tests and development.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .sandbox import (
    SandboxError,
    SandboxResult,
    SandboxUnavailable,
    parse_worker_response,
)
from .sandbox_worker import resolve_path
from .shell import DEFAULT_SHELL, ShellConfig, run_command

MAX_READ_LINES = 4000
MAX_LIST_ENTRIES = 500
MAX_GREP_MATCHES = 200
MAX_LINE_CHARS = 200
MAX_OUTPUT_CHARS = 12000
HALF_OUTPUT_CHARS = 6000


@dataclass
class Tool:
    """One callable tool with a provider-facing schema.

    Attributes:
        name: Tool name used by the model.
        spec: Neutral specification shaped as ``{"schema": {...}}``.
        run: Original Python callable.
    """

    name: str
    spec: dict[str, Any]
    run: Callable[..., Any]


def tool(
    description: str,
    **params: str,
) -> Callable[[Callable[..., Any]], Tool]:
    """Convert a plain function into a :class:`Tool`.

    Args:
        description: Tool description shown to the model.
        **params: Per-argument descriptions keyed by parameter name.

    Returns:
        A decorator that inspects the function signature and preserves the
        original function as ``Tool.run``.
    """

    def decorate(function: Callable[..., Any]) -> Tool:
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            properties[name] = {
                "type": "string",
                "description": params.get(name, ""),
            }
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        spec = {
            "schema": {
                "name": function.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        }
        return Tool(name=function.__name__, spec=spec, run=function)

    return decorate


class WorkspaceBackend(Protocol):
    """File and command operations required by :func:`core_tools`."""

    workspace: str

    def read_file(self, path: str, *, max_lines: int = ...) -> dict[str, Any]:
        """Read a file."""
        ...

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write a file."""
        ...

    def edit_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Edit a file."""
        ...

    def list_files(
        self, pattern: str = "**/*", *, limit: int = ...
    ) -> dict[str, Any]:
        """List files."""
        ...

    def grep(
        self, regex: str, pattern: str = "*", *, limit: int = ...
    ) -> dict[str, Any]:
        """Search files."""
        ...

    def exec(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: int = 120,
        input_text: str | None = None,
    ) -> SandboxResult:
        """Run a command."""
        ...


class LocalWorkspace:
    """File-only backend for unit tests and local development.

    This is not a sandbox and must not be used to claim isolation. It runs the
    same ``sandbox_worker.py`` logic on the host, but command execution is
    deliberately disabled.
    """

    def __init__(self, workdir: str | Path) -> None:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.workspace = str(root)
        self._worker = Path(__file__).with_name("sandbox_worker.py")

    def _request(self, op: str, **values: Any) -> dict[str, Any]:
        """Run one file operation with the shared worker on the host."""
        request = {"op": op, "workdir": self.workspace, **values}
        result = subprocess.run(
            [sys.executable, str(self._worker)],
            input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        stdout = result.stdout.decode("utf-8", "replace")
        stderr = result.stderr.decode("utf-8", "replace")
        if result.returncode != 0 and not stdout.strip():
            raise SandboxError(
                f"local workspace worker failed (exit {result.returncode}): "
                f"{stderr.strip()}"
            )
        return parse_worker_response(stdout)

    def read_file(
        self, path: str, *, max_lines: int = MAX_READ_LINES
    ) -> dict[str, Any]:
        """Read a file with the shared worker."""
        return self._request(
            "read_file", path=path, max_lines=max_lines
        )

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write a file with the shared worker."""
        return self._request("write_file", path=path, content=content)

    def edit_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Edit a file with the shared worker."""
        return self._request(
            "edit_file", path=path, old=old, new=new
        )

    def list_files(
        self, pattern: str = "**/*", *, limit: int = MAX_LIST_ENTRIES
    ) -> dict[str, Any]:
        """List files with the shared worker."""
        return self._request(
            "list_files", pattern=pattern, limit=limit
        )

    def grep(
        self, regex: str, pattern: str = "*", *, limit: int = MAX_GREP_MATCHES
    ) -> dict[str, Any]:
        """Search files with the shared worker."""
        return self._request(
            "grep", regex=regex, pattern=pattern, limit=limit
        )

    def exec(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: int = 120,
        input_text: str | None = None,
    ) -> SandboxResult:
        """Refuse command execution without OpenShell."""
        raise SandboxUnavailable(
            "OpenShell is required for command execution; "
            "local file-only mode cannot run commands"
        )


def _format_error(exc: BaseException) -> str:
    """Render an expected tool failure as a model-readable observation."""
    return f"ERROR: {type(exc).__name__}: {exc}"


def core_tools(
    workdir: str | Path,
    backend: WorkspaceBackend | None = None,
    shell: ShellConfig | None = None,
) -> list[Tool]:
    """Build the six workspace tools for one approved workspace.

    Args:
        workdir: Workspace path. With OpenShell this is the sandbox-side
            workspace path returned by :class:`OpenShellSandbox`.
        backend: Workspace implementation. Defaults to a local file-only
            backend for tests and development.
        shell: Shell configuration used by ``bash``. Defaults to Bash.

    Returns:
        Tools named ``read_file``, ``write_file``, ``edit_file``, ``bash``,
        ``list_files``, and ``grep``.
    """
    workspace_backend = backend or LocalWorkspace(workdir)
    workspace = getattr(workspace_backend, "workspace", str(workdir))
    shell_config = shell or DEFAULT_SHELL

    @tool(
        "Read a text file from the workspace. Each line is numbered with its "
        "line number and a tab. At most 4,000 lines are returned.",
        path="File path relative to the workspace.",
    )
    def read_file(path: str) -> str:
        try:
            data = workspace_backend.read_file(
                path, max_lines=MAX_READ_LINES
            )
            lines = str(data.get("text", "")).splitlines()
            total = int(data.get("total_lines", len(lines)))
            numbered = "\n".join(
                f"{index}\t{line}"
                for index, line in enumerate(lines, start=1)
            )
            if total > MAX_READ_LINES:
                numbered += (
                    f"\n\n... truncated: showing {MAX_READ_LINES} "
                    f"of {total} lines"
                )
            return numbered
        except (PermissionError, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)

    @tool(
        "Write a text file in the workspace. Missing parent directories are "
        "created.",
        path="File path relative to the workspace.",
        content="Complete text to write.",
    )
    def write_file(path: str, content: str) -> str:
        try:
            data = workspace_backend.write_file(path, content)
            return f"Wrote {int(data.get('chars', len(content)))} chars to {path}"
        except (PermissionError, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)

    @tool(
        "Replace a snippet in a file only when it appears exactly once. Read "
        "the file first and copy the snippet exactly.",
        path="File path relative to the workspace.",
        old="Exact snippet to replace.",
        new="Replacement text.",
    )
    def edit_file(path: str, old: str, new: str) -> str:
        if old == "":
            return "ERROR: old snippet must not be empty"
        try:
            data = workspace_backend.edit_file(path, old, new)
            matches = int(data.get("matches", 0))
            if matches == 0:
                return (
                    "ERROR: snippet not found — read the file and copy it "
                    "exactly"
                )
            if matches > 1:
                return (
                    f"ERROR: snippet appears {matches} times — include more "
                    "context to make it unique"
                )
            return f"Edited {path}"
        except (PermissionError, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)

    @tool(
        f"Run a command in the workspace inside OpenShell using "
        f"{shell_config.name}. Use {shell_config.name} syntax, not PowerShell.",
        command="Command text to run in the sandbox.",
        timeout="Timeout in seconds (default 120).",
    )
    def bash(command: str, timeout: str = "120") -> str:
        try:
            seconds = int(float(timeout))
        except (TypeError, ValueError):
            return "ERROR: timeout must be a positive number of seconds"
        if seconds <= 0:
            return "ERROR: timeout must be a positive number of seconds"
        try:
            result = run_command(
                workspace_backend,
                command,
                workspace,
                seconds,
                shell_config,
            )
        except (SandboxUnavailable, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)
        if result.timed_out:
            return f"ERROR: timed out after {seconds}s"
        output = result.stdout
        if result.stderr:
            if output and not output.endswith("\n"):
                output += "\n"
            output += result.stderr
        if not output:
            return f"(exit {result.exit_code}, no output)"
        if len(output) > MAX_OUTPUT_CHARS:
            output = (
                output[:HALF_OUTPUT_CHARS]
                + "\n... [output truncated] ...\n"
                + output[-HALF_OUTPUT_CHARS:]
            )
        return output

    @tool(
        "List files in the workspace. Skips .git, node_modules, __pycache__, "
        "and .venv. At most 500 entries are returned.",
        pattern="Glob pattern tested against relative paths and basenames.",
    )
    def list_files(pattern: str = "**/*") -> str:
        try:
            data = workspace_backend.list_files(
                pattern, limit=MAX_LIST_ENTRIES
            )
            paths = [str(path) for path in data.get("paths", [])]
            total = int(data.get("total", len(paths)))
            if not paths:
                return "(no files)"
            text = "\n".join(paths[:MAX_LIST_ENTRIES])
            if total > len(paths):
                text += f"\nand {total - len(paths)} more"
            return text
        except (PermissionError, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)

    @tool(
        "Search file contents in the workspace with a regular expression. "
        "Matches are formatted as path:line: text. At most 200 matches are "
        "returned.",
        regex="Python regular expression to find in each line.",
        pattern="Glob pattern used to choose files (default all files).",
    )
    def grep(regex: str, pattern: str = "*") -> str:
        try:
            re.compile(regex)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        try:
            data = workspace_backend.grep(
                regex, pattern, limit=MAX_GREP_MATCHES
            )
            matches = list(data.get("matches", []))[:MAX_GREP_MATCHES]
            if not matches:
                return "(no matches)"
            lines = [
                f"{item['path']}:{item['line']}: "
                f"{str(item['text'])[:MAX_LINE_CHARS]}"
                for item in matches
            ]
            return "\n".join(lines)
        except (PermissionError, SandboxError, OSError, ValueError) as exc:
            return _format_error(exc)

    return [read_file, write_file, edit_file, bash, list_files, grep]


__all__ = [
    "LocalWorkspace",
    "Tool",
    "core_tools",
    "resolve_path",
    "tool",
]
