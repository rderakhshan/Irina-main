"""Stage 4 tests for parallel fleet execution."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from src.back.harness.orchestration.fleet import (
    FleetError,
    FleetJob,
    run_fleet,
)
from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.providers.base import TokenUsage


def make_result(
    text: str,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    turns: int = 1,
    tool_calls: int = 0,
    usage: tuple[int, int] = (10, 5),
) -> RunResult:
    """Build a structured result for a fake harness."""
    return RunResult(
        status=status,
        text=text,
        messages=[],
        turns=turns,
        tool_calls=tool_calls,
        usage=TokenUsage(*usage),
        reason="scripted",
    )


class FakeHarness:
    """Minimal Harness-shaped object for fleet tests."""

    def __init__(
        self,
        results: list[RunResult | BaseException],
        *,
        session: Path | None = None,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.results = list(results)
        self.session_path = session
        self.barrier = barrier
        self.build_calls: list[str] = []
        self.review_calls: list[str] = []
        self.closed = False
        self.sandbox = object()

    def _next(self, task: str) -> RunResult:
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def run(self, task: str) -> RunResult:
        self.build_calls.append(task)
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        return self._next(task)

    def continue_task(self, task: str) -> RunResult:
        self.review_calls.append(task)
        return self._next(task)

    def resume(self, path=None) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    @property
    def workspace(self) -> str:
        return "/sandbox/irina"


class FleetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.dirs = []
        for name in ("one", "two", "three"):
            path = self.root / name
            path.mkdir()
            self.dirs.append(path)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_empty_fleet_returns_empty(self) -> None:
        self.assertEqual(run_fleet([], lambda job: FakeHarness([])), [])

    def test_results_keep_input_order(self) -> None:
        harnesses = {
            "slow": FakeHarness([make_result("slow")]),
            "fast": FakeHarness([make_result("fast")]),
        }
        jobs = [
            {"name": "slow", "workdir": str(self.dirs[0]), "task": "a"},
            {"name": "fast", "workdir": str(self.dirs[1]), "task": "b"},
        ]
        results = run_fleet(jobs, lambda job: harnesses[job.name], max_workers=2)
        self.assertEqual([item["name"] for item in results], ["slow", "fast"])
        self.assertTrue(all(item["ok"] for item in results))
        self.assertTrue(harnesses["slow"].closed)
        self.assertTrue(harnesses["fast"].closed)

    def test_jobs_run_in_parallel(self) -> None:
        barrier = threading.Barrier(2)
        harnesses = {
            "a": FakeHarness([make_result("a")], barrier=barrier),
            "b": FakeHarness([make_result("b")], barrier=barrier),
        }
        jobs = [
            {"name": "a", "workdir": str(self.dirs[0]), "task": "a"},
            {"name": "b", "workdir": str(self.dirs[1]), "task": "b"},
        ]
        results = run_fleet(jobs, lambda job: harnesses[job.name], max_workers=2)
        self.assertTrue(all(item["ok"] for item in results))

    def test_review_phase_reuses_the_same_harness_and_session(self) -> None:
        session = self.root / "session.jsonl"
        harness = FakeHarness(
            [
                make_result("built", turns=2, tool_calls=3, usage=(4, 2)),
                make_result("reviewed", turns=2, tool_calls=1, usage=(1, 1)),
            ],
            session=session,
        )
        seen = {}

        def on_complete(live, final):
            seen["same"] = live is harness
            seen["text"] = final.text
            seen["closed"] = harness.closed
            return {"checks": ["files", "tests"]}

        job = {
            "name": "project",
            "workdir": str(self.dirs[0]),
            "task": "build",
            "review_task": "review",
            "on_complete": on_complete,
        }
        result = run_fleet([job], lambda item: harness)[0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["turns"], 4)
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["phases"][0]["status"], "completed")
        self.assertEqual(result["phases"][1]["status"], "completed")
        self.assertEqual(result["session"], str(session))
        self.assertEqual(result["metadata"], {"checks": ["files", "tests"]})
        self.assertEqual(harness.build_calls, ["build"])
        self.assertEqual(harness.review_calls, ["review"])
        self.assertEqual(seen, {"same": True, "text": "reviewed", "closed": False})
        self.assertTrue(harness.closed)

    def test_build_failure_skips_review_and_is_isolated(self) -> None:
        good = FakeHarness([make_result("good")])
        bad = FakeHarness([RuntimeError("boom")])
        jobs = [
            {"name": "bad", "workdir": str(self.dirs[0]), "task": "bad"},
            {"name": "good", "workdir": str(self.dirs[1]), "task": "good"},
        ]
        harnesses = {"bad": bad, "good": good}
        results = run_fleet(jobs, lambda job: harnesses[job.name])
        self.assertFalse(results[0]["ok"])
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("RuntimeError: boom", results[0]["report"])
        self.assertTrue(results[1]["ok"])
        self.assertEqual(bad.review_calls, [])
        self.assertTrue(bad.closed)
        self.assertTrue(good.closed)

    def test_on_complete_can_fail_a_job(self) -> None:
        harness = FakeHarness([make_result("built")])

        def on_complete(live, final):
            return {"ok": False, "report": "required file missing"}

        result = run_fleet(
            [
                {
                    "name": "project",
                    "workdir": str(self.dirs[0]),
                    "task": "build",
                    "on_complete": on_complete,
                }
            ],
            lambda job: harness,
        )[0]
        self.assertFalse(result["ok"])
        self.assertEqual(result["report"], "required file missing")
        self.assertEqual(result["metadata"]["ok"], False)

    def test_invalid_fleets_are_rejected_before_execution(self) -> None:
        factory = lambda job: FakeHarness([])
        with self.assertRaises(FleetError):
            run_fleet(
                [
                    {"name": "same", "workdir": str(self.dirs[0]), "task": "a"},
                    {"name": "same", "workdir": str(self.dirs[1]), "task": "b"},
                ],
                factory,
            )
        with self.assertRaises(FleetError):
            run_fleet(
                [
                    {"name": "parent", "workdir": str(self.root), "task": "a"},
                    {"name": "child", "workdir": str(self.dirs[0]), "task": "b"},
                ],
                factory,
            )
        with self.assertRaises(FleetError):
            run_fleet([{"name": "missing"}], factory)
        with self.assertRaises(FleetError):
            run_fleet(
                [{"name": "x", "workdir": str(self.dirs[0]), "task": "a", "bad": 1}],
                factory,
            )
        with self.assertRaises(FleetError):
            run_fleet([], factory, max_workers=0)
        with self.assertRaises(FleetError):
            run_fleet([], None)


if __name__ == "__main__":
    unittest.main()
