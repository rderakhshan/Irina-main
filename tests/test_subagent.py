"""Stage 3: tests for sequential sub-agent delegation."""

from __future__ import annotations

import unittest

from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.providers.base import TokenUsage
from src.back.harness.orchestration.subagent import DelegationBudget, subagent_tool


def make_result(
    text: str = "child done",
    status: RunStatus = RunStatus.COMPLETED,
    reason: str = "done",
) -> RunResult:
    """Build a structured child result."""
    return RunResult(
        status=status,
        text=text,
        messages=[],
        turns=1,
        tool_calls=0,
        usage=TokenUsage(1, 1),
        reason=reason,
    )


class FakeChild:
    """Child harness stub with a fresh-run record."""

    def __init__(
        self,
        result: RunResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result or make_result()
        self.error = error
        self.calls: list[str] = []
        self.closed = False

    def run(self, task: str) -> RunResult:
        self.calls.append(task)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.closed = True


class SubagentTests(unittest.TestCase):
    def test_depth_limit_returns_exact_error_without_factory(self) -> None:
        created: list[int] = []
        tool = subagent_tool(
            lambda depth: created.append(depth),
            depth=2,
            max_depth=2,
        )
        self.assertEqual(
            tool.run(task="do it"),
            "ERROR: sub-agent depth limit reached; do this task yourself",
        )
        self.assertEqual(created, [])

    def test_child_is_created_at_next_depth_and_gets_only_the_task(self) -> None:
        child = FakeChild(make_result("report"))
        depths: list[int] = []
        tool = subagent_tool(
            lambda depth: (depths.append(depth), child)[1],
            depth=0,
            max_depth=2,
        )
        result = tool.run(task="write utils.py")
        self.assertEqual(depths, [1])
        self.assertEqual(child.calls, ["write utils.py"])
        self.assertIn("[sub-agent completed]", result)
        self.assertIn("report", result)
        self.assertTrue(child.closed)

    def test_failed_child_is_reported_as_error_not_success(self) -> None:
        child = FakeChild(
            make_result(
                text="not finished",
                status=RunStatus.MAX_TURNS,
                reason="turn limit",
            )
        )
        tool = subagent_tool(lambda depth: child, depth=0, max_depth=2)
        result = tool.run(task="x")
        self.assertTrue(result.startswith("ERROR: sub-agent finished"))
        self.assertIn("max_turns", result)
        self.assertIn("not finished", result)
        self.assertTrue(child.closed)

    def test_task_limit_bounds_total_delegation(self) -> None:
        children: list[FakeChild] = []

        def factory(depth: int) -> FakeChild:
            child = FakeChild()
            children.append(child)
            return child

        budget = DelegationBudget(max_tasks=1)
        tool = subagent_tool(
            factory, depth=0, max_depth=2, budget=budget
        )
        first = tool.run(task="one")
        second = tool.run(task="two")
        self.assertIn("[sub-agent completed]", first)
        self.assertIn("task limit", second)
        self.assertEqual(len(children), 1)

    def test_empty_task_and_factory_failure_are_reported(self) -> None:
        child = FakeChild()
        tool = subagent_tool(lambda depth: child, depth=0, max_depth=2)
        self.assertIn("must not be empty", tool.run(task="  "))

        def broken_factory(depth: int):
            raise RuntimeError("no sandbox")

        broken = subagent_tool(broken_factory, depth=0, max_depth=2)
        result = broken.run(task="x")
        self.assertTrue(result.startswith("ERROR: could not create sub-agent"))
        self.assertIn("no sandbox", result)

    def test_child_exception_and_unsupported_result_are_reported(self) -> None:
        failing = FakeChild(error=RuntimeError("child crashed"))
        tool = subagent_tool(lambda depth: failing, depth=0, max_depth=2)
        result = tool.run(task="x")
        self.assertTrue(result.startswith("ERROR: sub-agent failed"))
        self.assertIn("child crashed", result)
        self.assertTrue(failing.closed)

        class StringChild:
            def run(self, task):
                return "a string"

            def close(self):
                pass

        unsupported = subagent_tool(
            lambda depth: StringChild(), depth=0, max_depth=2
        )
        result = unsupported.run(task="x")
        self.assertTrue(result.startswith("ERROR: child harness returned"))

    def test_tool_schema_and_description(self) -> None:
        tool = subagent_tool(lambda depth: FakeChild(), depth=0, max_depth=2)
        self.assertEqual(tool.name, "spawn_agent")
        schema = tool.spec["schema"]
        self.assertEqual(schema["name"], "spawn_agent")
        self.assertIn("fresh", schema["description"])
        self.assertIn("workspace", schema["description"])
        self.assertIn("parent", schema["description"])
        self.assertEqual(schema["parameters"]["required"], ["task"])


if __name__ == "__main__":
    unittest.main()
