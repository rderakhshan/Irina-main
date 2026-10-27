"""Stage 3: real OpenShell crash-recovery and resume integration test."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.back.harness import Harness
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.session import latest, load
from src.back.harness.tools_and_guardrails import (
    OpenShellSandbox,
    Policy,
)


class ResumeProvider:
    """Inspect the uncertain file, then report the result."""

    default_model = "resume-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="read-call",
                        name="read_file",
                        args={"path": "part1.txt"},
                    )
                ],
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            )
        return ProviderResponse(
            text="Inspected part1.txt and continued.",
            tool_calls=[],
            finish_reason="stop",
            usage=TokenUsage(1, 1),
        )


class OpenShellCrashRecoveryTests(unittest.TestCase):
    def test_crash_then_resume_reports_unknown_without_replay(self) -> None:
        if not OpenShellSandbox.is_available():
            self.skipTest("OpenShell is not available")

        worker = Path(__file__).with_name("_crash_worker.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            state = root / "state"
            workspace.mkdir()

            process = subprocess.run(
                [sys.executable, str(worker), str(workspace), str(state)],
                cwd=str(worker.parents[1]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
            )
            self.assertEqual(
                process.returncode,
                9,
                msg=f"worker output: {process.stdout}\n{process.stderr}",
            )

            session = latest(workspace, state_dir=state)
            self.assertIsNotNone(session)
            pre_resume = load(session)
            self.assertEqual(pre_resume.status, "interrupted")
            self.assertTrue(
                any(
                    call.get("id") == "crash-call"
                    for message in pre_resume.messages
                    if message.get("role") == "assistant"
                    for call in message.get("tool_calls", [])
                )
            )
            self.assertTrue(
                any(
                    "may or may not have completed" in message.get("text", "")
                    for message in pre_resume.messages
                    if message.get("role") == "tool"
                )
            )

            sandbox_name = pre_resume.metadata.get("sandbox_name")
            harness = Harness(
                workdir=str(workspace),
                model="resume-model",
                provider=ResumeProvider(),
                policy=Policy("yolo"),
                state_dir=state,
            )
            try:
                self.assertTrue(harness.resume(session))
                self.assertFalse(
                    (workspace / "part1.txt").exists(),
                    "agent writes must stay in the sandbox workspace",
                )
                result = harness.run("Continue the task.")
                self.assertTrue(result.completed)
                self.assertEqual(result.text, "Inspected part1.txt and continued.")
                data = load(session)
                self.assertEqual(data.status, "completed")
                self.assertTrue(
                    any(
                        event.get("outcome") == "unknown"
                        for event in data.recovery_events
                    )
                )
                self.assertTrue(
                    any(
                        "partial" in message.get("text", "")
                        for message in harness.messages
                        if message.get("role") == "tool"
                    )
                )
            finally:
                harness.close()
                if sandbox_name:
                    try:
                        OpenShellSandbox(name=sandbox_name).delete()
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
