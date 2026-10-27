"""Day 2: sandbox-side workspace worker.

Concept: file tools must resolve and touch paths in the same filesystem where
the operation happens. When OpenShell is active, this script runs inside the
sandbox; it is uploaded by the adapter and invoked once per operation over
stdin/stdout JSON.

Design constraints:
* Standard library only; Python 3.10+.
* The worker receives the approved workspace path from the trusted host tool.
* Paths are resolved with symlinks followed and rejected if they leave the
  workspace.
* The module is importable on the host for the shared ``resolve_path`` helper.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

IGNORED_DIRECTORIES = frozenset({".git", "node_modules", "__pycache__", ".venv"})


def resolve_path(workdir: str | Path, path: str) -> Path:
    """Resolve ``path`` against ``workdir`` and reject escapes.

    Symbolic links are resolved before the boundary check, so a link inside the
    workspace cannot point at a file outside it.

    Args:
        workdir: Approved workspace root.
        path: Relative or absolute requested path.

    Returns:
        The canonical path inside the workspace.

    Raises:
        PermissionError: If the resolved destination is outside the workspace.
    """
    root = Path(workdir).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"{path!r} escapes the working directory"
        ) from None
    return resolved


def _matches(path: str, pattern: str) -> bool:
    """Return whether a relative path or basename matches a glob pattern."""
    if pattern in {"**", "**/*", "*"}:
        return True
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
        os.path.basename(path), pattern
    )


def _iter_files(workdir: Path):
    """Yield canonical files inside the workspace, skipping ignored trees."""
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = sorted(
            name for name in dirnames if name not in IGNORED_DIRECTORIES
        )
        for filename in sorted(filenames):
            full = Path(dirpath) / filename
            try:
                resolved = full.resolve()
                resolved.relative_to(workdir)
            except (OSError, ValueError):
                continue
            relative = full.relative_to(workdir).as_posix()
            yield relative, resolved


def _read_file(workdir: Path, path: str, max_lines: int) -> dict[str, Any]:
    """Read a text file and return the first ``max_lines`` lines plus the total."""
    target = resolve_path(workdir, path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "text": "\n".join(lines[:max_lines]),
        "total_lines": len(lines),
    }


def _write_file(workdir: Path, path: str, content: str) -> dict[str, Any]:
    """Write text to a workspace path, creating parent directories."""
    target = resolve_path(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"chars": len(content)}


def _edit_file(
    workdir: Path, path: str, old: str, new: str
) -> dict[str, Any]:
    """Replace one unique snippet, or report the match count without editing."""
    if old == "":
        raise ValueError("old snippet must not be empty")
    target = resolve_path(workdir, path)
    text = target.read_text(encoding="utf-8", errors="replace")
    matches = text.count(old)
    if matches == 1:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return {"matches": matches, "edited": matches == 1}


def _list_files(
    workdir: Path, pattern: str, limit: int
) -> dict[str, Any]:
    """List matching workspace files in deterministic order."""
    paths = sorted(
        relative
        for relative, _ in _iter_files(workdir)
        if _matches(relative, pattern)
    )
    return {"paths": paths[:limit], "total": len(paths)}


def _grep(
    workdir: Path, regex: str, pattern: str, limit: int
) -> dict[str, Any]:
    """Search workspace files line by line and return matching lines."""
    compiled = re.compile(regex)
    matches: list[dict[str, Any]] = []
    for relative, target in _iter_files(workdir):
        if not _matches(relative, pattern):
            continue
        try:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                for number, raw_line in enumerate(handle, start=1):
                    line = raw_line.rstrip("\r\n")
                    if compiled.search(line):
                        matches.append(
                            {
                                "path": relative,
                                "line": number,
                                "text": line[:200],
                            }
                        )
        except OSError:
            continue
    matches.sort(key=lambda item: (item["path"], item["line"]))
    return {"matches": matches[:limit]}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Handle one structured file operation and return its result."""
    operation = request.get("op")
    workdir = Path(str(request["workdir"])).resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workdir}")

    if operation == "read_file":
        return _read_file(
            workdir,
            str(request["path"]),
            int(request.get("max_lines", 4000)),
        )
    if operation == "write_file":
        return _write_file(
            workdir,
            str(request["path"]),
            str(request["content"]),
        )
    if operation == "edit_file":
        return _edit_file(
            workdir,
            str(request["path"]),
            str(request["old"]),
            str(request["new"]),
        )
    if operation == "list_files":
        return _list_files(
            workdir,
            str(request.get("pattern", "**/*")),
            int(request.get("limit", 500)),
        )
    if operation == "grep":
        return _grep(
            workdir,
            str(request["regex"]),
            str(request.get("pattern", "*")),
            int(request.get("limit", 200)),
        )
    raise ValueError(f"unsupported operation: {operation!r}")


def main() -> int:
    """Read one JSON request from stdin and write one JSON response."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise ValueError("empty worker request")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("worker request must be a JSON object")
        response = {"ok": True, **handle(request)}
    except Exception as exc:
        response = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
