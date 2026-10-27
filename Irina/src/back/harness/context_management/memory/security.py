"""Static and isolated checks for generated memory candidates."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CandidateSecurityError(RuntimeError):
    """Generated candidate failed a safety check."""


@dataclass
class CandidateValidation:
    ok: bool
    static_ok: bool
    isolated_import_ok: bool | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "__import__",
    "compile",
    "input",
    "breakpoint",
}
_FORBIDDEN_MODULES = {
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "paramiko",
    "ctypes",
    "multiprocessing",
}


def static_validate(path: str | Path) -> CandidateValidation:
    candidate = Path(path)
    errors: list[str] = []
    try:
        tree = ast.parse(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        return CandidateValidation(False, False, errors=[f"{type(exc).__name__}: {exc}"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                errors.append(f"forbidden call {node.func.id} at line {node.lineno}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in _FORBIDDEN_MODULES:
                    errors.append(f"forbidden import {name} at line {node.lineno}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                if node.attr in {"path", "modules", "meta_path"}:
                    errors.append(f"sys.{node.attr} mutation is not allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                # Generated providers need local storage; require an explicit
                # candidate-relative path is enforced by isolated execution and
                # the caller's review, not by this static check.
                continue
    return CandidateValidation(not errors, not errors, errors=errors)


def isolated_import(path: str | Path, *, timeout: int = 10) -> CandidateValidation:
    """Import a candidate in a separate Python process from a temp cwd.

    This is a bounded smoke check, not a complete sandbox. Production
    promotion additionally requires an explicit review and OpenShell/host
    isolation; no candidate is imported into the trusted Irina process here.
    """
    candidate = Path(path).resolve()
    static = static_validate(candidate)
    if not static.ok:
        return static
    with tempfile.TemporaryDirectory() as temp:
        code = (
            "import importlib.util, pathlib, sys; "
            f"p=pathlib.Path({str(candidate)!r}); "
            "spec=importlib.util.spec_from_file_location('candidate', p); "
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=temp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONPATH": ""},
            )
        except Exception as exc:
            return CandidateValidation(
                False, True, False, errors=[f"{type(exc).__name__}: {exc}"]
            )
    if result.returncode != 0:
        return CandidateValidation(
            False,
            True,
            False,
            errors=[(result.stderr or result.stdout)[-2000:]],
        )
    return CandidateValidation(True, True, True)
