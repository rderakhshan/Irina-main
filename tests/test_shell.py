"""Day 2: offline tests for the shell interface and bash formatting.

Concept: verify that the shell layer forwards the configured shell, command,
working directory, and timeout to the backend, and that observations are
formatted, truncated, and timed out correctly.

Design constraints:
* Standard library ``unittest`` only.
* No host command execution is performed.
"""

from __future__ import annotations

import unittest

from src.back.harness.tools_and_guardrails.sandbox import SandboxResult
from src.back.harness.tools_and_guardrails.shell import (
    DEFAULT_SHELL,
    ShellConfig,
    run_command,
)
from src.back.harness.tools_and_guardrails.tools import core_tools


class FakeBackend:
    """Backend that records exec requests and returns a fixed result."""

    def __init__(self, result: SandboxResult) -> None:
        self.result = result
        self.workspace = "/sandbox/workspace"
        self.calls: list[dict] = []

    def exec(
        self,
        argv,
        *,
        workdir=None,
        timeout=120,
        input_text=None,
    ) -> SandboxResult:
        self.calls.append(
            {
                "argv": list(argv),
                "workdir": workdir,
                "timeout": timeout,
                "input_text": input_text,
            }
        )
        return self.result


def bash_tool(backend: FakeBackend):
    """Build the tool set and return the bash tool."""
    return {item.name: item for item in core_tools("/sandbox/workspace", backend)}[
        "bash"
    ]


class ShellInterfaceTests(unittest.TestCase):
    def test_default_shell_is_bash_reading_from_stdin(self) -> None:
        backend = FakeBackend(SandboxResult(0, "hello\n", "", False))
        result = run_command(
            backend, "echo hello", "/sandbox/workspace", 30
        )
        self.assertEqual(backend.calls[0]["argv"], ["/bin/bash", "-s"])
        self.assertEqual(backend.calls[0]["input_text"], "echo hello")
        self.assertEqual(backend.calls[0]["workdir"], "/sandbox/workspace")
        self.assertEqual(backend.calls[0]["timeout"], 30)
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.shell, "bash")
        self.assertEqual(DEFAULT_SHELL.name, "bash")

    def test_custom_shell_configuration_is_used(self) -> None:
        backend = FakeBackend(SandboxResult(0, "ok", "", False))
        shell = ShellConfig(path="/bin/sh", args=("-c",), name="sh")
        result = run_command(
            backend, "echo ok", "/sandbox/workspace", 5, shell
        )
        self.assertEqual(backend.calls[0]["argv"], ["/bin/sh", "-c"])
        self.assertEqual(result.shell, "sh")


class BashFormattingTests(unittest.TestCase):
    def test_combines_stdout_and_stderr(self) -> None:
        backend = FakeBackend(SandboxResult(3, "out\n", "err\n", False))
        output = bash_tool(backend).run(command="do it", timeout="10")
        self.assertEqual(output, "out\nerr\n")
        self.assertEqual(backend.calls[0]["timeout"], 10)

    def test_no_output_returns_exit_status(self) -> None:
        backend = FakeBackend(SandboxResult(7, "", "", False))
        output = bash_tool(backend).run(command="false")
        self.assertEqual(output, "(exit 7, no output)")

    def test_timeout_returns_a_clear_error(self) -> None:
        backend = FakeBackend(SandboxResult(124, "", "", True))
        output = bash_tool(backend).run(command="sleep 10", timeout="5")
        self.assertEqual(output, "ERROR: timed out after 5s")

    def test_long_output_keeps_first_and_last_6000_characters(self) -> None:
        output = ("a" * 6000) + ("b" * 6000) + ("c" * 1000)
        backend = FakeBackend(SandboxResult(0, output, "", False))
        formatted = bash_tool(backend).run(command="long")
        self.assertTrue(formatted.startswith("a" * 6000))
        self.assertTrue(formatted.endswith("c" * 1000))
        self.assertIn("output truncated", formatted)

    def test_invalid_timeout_is_rejected_before_execution(self) -> None:
        backend = FakeBackend(SandboxResult(0, "", "", False))
        tool = bash_tool(backend)
        for value in ("abc", "0", "-1"):
            with self.subTest(value=value):
                output = tool.run(command="echo hi", timeout=value)
                self.assertIn("ERROR: timeout", output)
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
