"""Static and isolated checks for generated memory candidates."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class CandidateSecurityError(RuntimeError):
    """Generated candidate failed a safety check."""


@dataclass
class CandidateValidation:
    ok: bool
    static_ok: bool
    isolated_import_ok: bool | None = None
    contract_ok: bool | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    return CandidateValidation(not errors, not errors, errors=errors)


def _subprocess_env() -> dict[str, str]:
    """Minimal environment: no provider credentials reach generated code."""
    keep = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC", "PATHEXT")
    return {name: os.environ[name] for name in keep if name in os.environ}


def isolated_import(
    path: str | Path,
    *,
    timeout: int = 10,
    python_paths: list[str | Path] | None = None,
) -> CandidateValidation:
    """Import a candidate in a separate Python process from a temp cwd.

    This is a bounded smoke check, not a complete sandbox. Production
    promotion additionally requires explicit review and OpenShell/host
    isolation; no candidate is imported into the trusted Irina process here.
    """
    candidate = Path(path).resolve()
    static = static_validate(candidate)
    if not static.ok:
        return static
    paths = [str(Path(item).resolve()) for item in (python_paths or [])]
    with tempfile.TemporaryDirectory() as temp:
        code = (
            "import importlib.util, pathlib, sys; "
            f"sys.path[:0]={paths!r}; "
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
                env=_subprocess_env(),
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


_CONTRACT_METHODS = ("initialize", "take_in_memory", "provide_memory")


def validate_candidate(
    path: str | Path,
    *,
    python_paths: list[str | Path] | None = None,
    timeout: int = 20,
) -> CandidateValidation:
    """Static check, isolated import, and provider-contract probe.

    The probe imports the file in a separate interpreter, finds concrete
    provider classes, and attempts construction with a temporary storage
    directory. It never executes the candidate in the Irina process and it
    never promotes it.
    """
    candidate = Path(path).resolve()
    static = static_validate(candidate)
    if not static.ok:
        return static
    if not candidate.is_file():
        return CandidateValidation(
            False,
            True,
            errors=[f"candidate file does not exist: {candidate}"],
        )
    paths = [str(Path(item).resolve()) for item in (python_paths or [])]
    with tempfile.TemporaryDirectory() as temp:
        code = f"""
import importlib.util, inspect, json, pathlib, sys, tempfile
sys.path[:0]={paths!r}
p=pathlib.Path({str(candidate)!r})
spec=importlib.util.spec_from_file_location('candidate_provider', p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
classes=[]
for name, obj in vars(m).items():
    if not inspect.isclass(obj) or obj.__module__ != m.__name__:
        continue
    methods=[x for x in {list(_CONTRACT_METHODS)!r} if callable(getattr(obj, x, None))]
    classes.append({{"name": name, "methods": methods}})
contract=[c for c in classes if len(c["methods"]) == len({list(_CONTRACT_METHODS)!r})]
instantiated=None
instantiation_error=None
if contract:
    cls=getattr(m, contract[0]["name"])
    storage=tempfile.mkdtemp()
    try:
        obj=cls({{"storage_dir": storage}})
        instantiated=True
    except Exception as exc:
        try:
            obj=cls()
            instantiated=True
        except Exception as exc2:
            instantiated=False
            instantiation_error=f"{{type(exc).__name__}}: {{exc}} / {{type(exc2).__name__}}: {{exc2}}"
print(json.dumps({{"classes": classes, "contract": bool(contract), "instantiated": instantiated, "instantiation_error": instantiation_error}}))
"""
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=temp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_subprocess_env(),
            )
        except Exception as exc:
            return CandidateValidation(
                False,
                True,
                False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
    if result.returncode != 0:
        return CandidateValidation(
            False,
            True,
            False,
            errors=[(result.stderr or result.stdout)[-2000:]][:1],
        )
    try:
        details = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return CandidateValidation(
            False,
            True,
            True,
            errors=["candidate probe did not return JSON"],
            details={"stdout": result.stdout[-2000:]},
        )
    contract_ok = bool(details.get("contract"))
    instantiated = bool(details.get("instantiated"))
    errors: list[str] = []
    if not contract_ok:
        errors.append(
            "candidate does not expose initialize/take_in_memory/provide_memory"
        )
    if contract_ok and not instantiated:
        errors.append(
            "candidate could not be instantiated: "
            + str(details.get("instantiation_error"))
        )
    return CandidateValidation(
        ok=contract_ok and instantiated,
        static_ok=True,
        isolated_import_ok=True,
        contract_ok=contract_ok,
        errors=errors,
        details=details,
    )
