"""Coding-task evaluation fixtures and metrics for memory architectures."""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class CodingTask:
    name: str
    prompt: str
    fixture_files: dict[str, str]
    validation_command: tuple[str, ...]
    held_out: bool = False
    tags: tuple[str, ...] = ()


@dataclass
class EvaluationMetrics:
    task_success_rate: float = 0.0
    tests_passed: int = 0
    verified_completions: int = 0
    tool_calls: int = 0
    model_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None
    wall_clock_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    memory_update_latency_seconds: float = 0.0
    memory_storage_bytes: int = 0
    failure_rate: float = 0.0
    recovery_behavior: str | None = None
    per_task: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_coding_tasks() -> list[CodingTask]:
    """Small deterministic coding fixtures; held-out tasks are explicit."""
    return [
        CodingTask(
            name="fix-import-error",
            prompt="Fix the failing package import in app.py and verify it.",
            fixture_files={
                "app.py": "from helpers.math import add\nprint(add(2, 3))\n",
                "helpers/__init__.py": "",
                "helpers/math.py": "def add(a, b):\n    return a + b\n",
                "test_app.py": (
                    "import subprocess, sys, unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_import(self):\n"
                    "        r=subprocess.run([sys.executable,'app.py'],capture_output=True)\n"
                    "        self.assertEqual(r.returncode,0)\n"
                ),
            },
            validation_command=(("python", "-m", "unittest", "-v")),
            tags=("imports", "tests"),
        ),
        CodingTask(
            name="repair-failing-test",
            prompt="Repair the failing test without weakening it, then verify.",
            fixture_files={
                "calc.py": "def double(value):\n    return value\n",
                "test_calc.py": (
                    "import unittest\nfrom calc import double\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_double(self):\n"
                    "        self.assertEqual(double(4),8)\n"
                ),
            },
            validation_command=(("python", "-m", "unittest", "-v")),
            tags=("tests", "debugging"),
        ),
        CodingTask(
            name="cli-feature",
            prompt="Add a --count option to counter.py and verify it.",
            fixture_files={
                "counter.py": "import argparse\nprint('todo')\n",
                "test_counter.py": (
                    "import subprocess,sys,unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_count(self):\n"
                    "        r=subprocess.run([sys.executable,'counter.py','--count','3'],capture_output=True,text=True)\n"
                    "        self.assertEqual(r.stdout.strip(),'1\\n2\\n3')\n"
                ),
            },
            validation_command=(("python", "-m", "unittest", "-v")),
            tags=("cli", "feature"),
        ),
        CodingTask(
            name="heldout-multi-file-refactor",
            prompt="Move the greeting into a helper and keep the tests green.",
            fixture_files={
                "main.py": "def greet(name):\n    return 'Hello, ' + name\n",
                "test_main.py": (
                    "import unittest\nfrom main import greet\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_greet(self):\n"
                    "        self.assertEqual(greet('Ada'),'Hello, Ada')\n"
                ),
            },
            validation_command=(("python", "-m", "unittest", "-v")),
            held_out=True,
            tags=("refactor", "multi-file"),
        ),
    ]


def run_fixture(
    task: CodingTask,
    run_task: Callable[[CodingTask, Path], dict[str, Any]],
    *,
    work_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one task in an isolated fixture directory and a validation command."""
    root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="irina-eval-"))
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in task.fixture_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    started = time.monotonic()
    result = run_task(task, root)
    elapsed = time.monotonic() - started
    validation = subprocess.run(
        list(task.validation_command),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "name": task.name,
        "held_out": task.held_out,
        "result": result,
        "duration_seconds": elapsed,
        "validation_exit": validation.returncode,
        "validation_output": (validation.stdout + validation.stderr)[-4000:],
        "verified_success": validation.returncode == 0,
    }


def evaluate(
    tasks: Iterable[CodingTask],
    run_task: Callable[[CodingTask, Path], dict[str, Any]],
    *,
    work_root: str | Path | None = None,
) -> EvaluationMetrics:
    rows = [run_fixture(task, run_task, work_root=work_root) for task in tasks]
    total = len(rows)
    success = sum(1 for row in rows if row["verified_success"])
    metrics = EvaluationMetrics(
        task_success_rate=(success / total if total else 0.0),
        tests_passed=success,
        verified_completions=success,
        wall_clock_seconds=sum(float(row["duration_seconds"]) for row in rows),
        failure_rate=((total - success) / total if total else 0.0),
        per_task=rows,
    )
    for row in rows:
        result = row.get("result") or {}
        metrics.tool_calls += int(result.get("tool_calls", 0) or 0)
        metrics.model_turns += int(result.get("turns", 0) or 0)
        usage = result.get("usage") or {}
        metrics.input_tokens += int(usage.get("input_tokens", 0) or 0)
        metrics.output_tokens += int(usage.get("output_tokens", 0) or 0)
    return metrics
