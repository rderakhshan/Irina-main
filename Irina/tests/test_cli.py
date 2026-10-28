"""Stage 4 tests for the CLI front door and headless exit codes."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.loop.state import RunResult, RunStatus
from src.back.harness.orchestration import cli
from src.back.harness.providers.base import TokenUsage
from src.back.harness.tools_and_guardrails import Policy


def result(
    status: RunStatus = RunStatus.COMPLETED,
    *,
    text: str = "done",
) -> RunResult:
    return RunResult(
        status=status,
        text=text,
        messages=[],
        turns=1,
        tool_calls=0,
        usage=TokenUsage(2, 1),
        reason="scripted",
    )


class FakeHarness:
    def __init__(
        self,
        outcomes: list[RunResult | BaseException] | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [result()])
        self.session_path = Path("session.jsonl")
        self.workspace = "/sandbox/irina"
        self.model = "fake-model"
        self.policy = Policy("safe")
        self.resume_calls: list[object] = []
        self.run_calls: list[str] = []
        self.continue_calls: list[str] = []
        self.closed = False

    def _next(self) -> RunResult:
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def run(self, task: str) -> RunResult:
        self.run_calls.append(task)
        return self._next()

    def continue_task(self, task: str) -> RunResult:
        self.continue_calls.append(task)
        return self._next()

    def resume(self, path=None) -> bool:
        self.resume_calls.append(path)
        return False

    def close(self) -> None:
        self.closed = True


class ParserTests(unittest.TestCase):
    def test_defaults_are_safe_headless_ready(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertIsNone(args.prompt)
        self.assertEqual(args.workdir, ".")
        self.assertEqual(args.mode, "safe")
        self.assertEqual(args.max_turns, 120)
        self.assertFalse(args.resume)
        self.assertTrue(args.stream)
        self.assertIsNone(args.color)
        self.assertFalse(args.plain)
        self.assertIsNone(args.memory)
        self.assertIsNone(args.memory_provider)

    def test_memory_flags_are_explicit(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            ["--memory", "--memory-provider", "expel"]
        )
        self.assertTrue(args.memory)
        self.assertEqual(args.memory_provider, "expel")
        self.assertFalse(parser.parse_args(["--no-memory"]).memory)

    def test_resume_accepts_latest_or_explicit_session(self) -> None:
        parser = cli.build_parser()
        self.assertIs(parser.parse_args(["--resume"]).resume, True)
        self.assertEqual(
            parser.parse_args(["--resume", "session.jsonl"]).resume,
            "session.jsonl",
        )

    def test_invalid_mode_and_max_turns_are_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.build_parser().parse_args(["--mode", "unsafe"])
        self.assertEqual(caught.exception.code, 2)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main(["--max-turns", "0"])
        self.assertEqual(caught.exception.code, 2)


class EnvDefaultTests(unittest.TestCase):
    def test_application_env_defaults_fill_missing_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            present = Path(temp) / "examples"
            present.mkdir()
            env = {
                "IRINA_SANDBOX_IMAGE": "irina/application-tex:1",
                "IRINA_INPUTS": "cv.tex;reference.pdf",
                "IRINA_OPTIONAL_INPUTS": f"{present};missing-dir",
                "IRINA_EXPORT_PATHS": "application",
                "IRINA_ARTIFACTS_DIR": "D:\\Application",
                "IRINA_EXPORT_CONTENTS": "1",
                "IRINA_AUTO_EXPORT": "1",
                "IRINA_MODE": "yolo",
            }
            with patch.dict(os.environ, env, clear=False):
                args = cli.build_parser().parse_args([])
                cli._apply_env_defaults(args)
            self.assertEqual(args.sandbox_image, "irina/application-tex:1")
            self.assertEqual(
                args.inputs, ["cv.tex", "reference.pdf", str(present)]
            )
            self.assertEqual(args.export_paths, ["application"])
            self.assertEqual(args.artifacts_dir, "D:\\Application")
            self.assertTrue(args.export_contents)
            self.assertTrue(args.auto_export)
            self.assertEqual(args.mode, "yolo")

    def test_require_gate_skips_all_env_defaults(self) -> None:
        env = {
            "IRINA_DEFAULTS_REQUIRE": "definitely-missing-file",
            "IRINA_SANDBOX_IMAGE": "env-image",
            "IRINA_INPUTS": "env-input",
        }
        with patch.dict(os.environ, env, clear=False):
            args = cli.build_parser().parse_args([])
            cli._apply_env_defaults(args)
        self.assertIsNone(args.sandbox_image)
        self.assertEqual(args.inputs, [])

    def test_explicit_flags_are_not_replaced(self) -> None:
        env = {
            "IRINA_SANDBOX_IMAGE": "env-image",
            "IRINA_INPUTS": "env-input",
            "IRINA_EXPORT_PATHS": "env-export",
            "IRINA_ARTIFACTS_DIR": "env-artifacts",
        }
        with patch.dict(os.environ, env, clear=False):
            args = cli.build_parser().parse_args(
                [
                    "--sandbox-image", "cli-image",
                    "--input", "cli-input",
                    "--export", "cli-export",
                    "--artifacts-dir", "cli-artifacts",
                ]
            )
            cli._apply_env_defaults(args)
        self.assertEqual(args.sandbox_image, "cli-image")
        self.assertEqual(args.inputs, ["cli-input"])
        self.assertEqual(args.export_paths, ["cli-export"])
        self.assertEqual(args.artifacts_dir, "cli-artifacts")


class PolicyTests(unittest.TestCase):
    def test_safe_interactive_has_an_approver_but_headless_does_not(self) -> None:
        parser = cli.build_parser()
        interactive = cli.build_policy(
            parser.parse_args([]), interactive=True
        )
        headless = cli.build_policy(
            parser.parse_args([]), interactive=False
        )
        self.assertEqual(interactive.mode, "safe")
        self.assertIsNotNone(interactive.approver)
        self.assertEqual(headless.mode, "safe")
        self.assertIsNone(headless.approver)

    def test_yolo_and_read_only_never_gain_an_approver(self) -> None:
        parser = cli.build_parser()
        for mode in ("yolo", "read-only"):
            policy = cli.build_policy(
                parser.parse_args(["--mode", mode]), interactive=True
            )
            self.assertEqual(policy.mode, mode)
            self.assertIsNone(policy.approver)

    def test_approver_requires_explicit_yes(self) -> None:
        yes = cli.make_approver(
            input_fn=lambda prompt: "y",
            stream=io.StringIO(),
        )
        no = cli.make_approver(
            input_fn=lambda prompt: "",
            stream=io.StringIO(),
        )
        self.assertTrue(yes(None, "write"))
        self.assertFalse(no(None, "write"))

    def test_approver_stops_progress_before_prompting(self) -> None:
        called = []
        approver = cli.make_approver(
            input_fn=lambda prompt: "y",
            stream=io.StringIO(),
            on_prompt=lambda: called.append(True),
        )
        self.assertTrue(approver(None, "write"))
        self.assertEqual(called, [True])


class EventTests(unittest.TestCase):
    def test_event_printer_redacts_assistant_text(self) -> None:
        stream = io.StringIO()
        printer = cli.EventPrinter(stream)
        printer(
            RunEvent(
                EventType.ASSISTANT,
                1,
                text="key sk-abcdefghijklmnopqrstuvwxyz",
            )
        )
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", stream.getvalue())
        self.assertIn("key", stream.getvalue())

    def test_final_result_redacts_text(self) -> None:
        stream = io.StringIO()
        cli._print_result(
            result(text="key sk-abcdefghijklmnopqrstuvwxyz"),
            stream=stream,
        )
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", stream.getvalue())


class PromptInputTests(unittest.TestCase):
    def test_multiline_paste_reads_until_terminator(self) -> None:
        lines = iter(["Line one.", "Line two.", '"""', "ignored"])
        text = cli._read_multiline(lambda _: next(lines))
        self.assertEqual(text, "Line one.\nLine two.")

    def test_multiline_paste_accepts_end_marker(self) -> None:
        lines = iter(["only line", "/end"])
        self.assertEqual(
            cli._read_multiline(lambda _: next(lines)), "only line"
        )

    def test_prompt_file_is_bounded_and_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "jd.txt"
            path.write_text("Full job description.\n", encoding="utf-8")
            self.assertEqual(
                cli._task_from_file(f'"{path}"'), "Full job description."
            )

    def test_paste_markers_are_removed(self) -> None:
        self.assertEqual(
            cli._clean_paste("\x1b[200~hello\nworld\x1b[201~"),
            "hello\nworld",
        )

    def test_capture_prompt_keeps_commands_and_single_lines(self) -> None:
        self.assertEqual(cli._capture_prompt("exit"), "exit")
        self.assertEqual(cli._capture_prompt("/file jd.txt"), "/file jd.txt")
        self.assertEqual(cli._capture_prompt("hello"), "hello")

    def test_prompt_file_flag_is_exposed(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--prompt-file", "jd.txt"])
        self.assertEqual(args.prompt_file, "jd.txt")


class HeadlessTests(unittest.TestCase):
    def test_completed_headless_run_closes_and_returns_zero(self) -> None:
        harness = FakeHarness([result()])
        with patch.object(cli, "build_harness", return_value=harness):
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["-p", "hello"])
        self.assertEqual(code, 0)
        self.assertTrue(harness.closed)

    def test_incomplete_headless_run_preserves_session_and_returns_one(self) -> None:
        harness = FakeHarness([result(RunStatus.MAX_TURNS)])
        with patch.object(cli, "build_harness", return_value=harness):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main(["-p", "hello"])
        self.assertEqual(code, 1)
        self.assertFalse(harness.closed)

    def test_interrupted_headless_run_returns_130_without_cleanup(self) -> None:
        harness = FakeHarness([KeyboardInterrupt()])
        with patch.object(cli, "build_harness", return_value=harness):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main(["-p", "hello"])
        self.assertEqual(code, 130)
        self.assertFalse(harness.closed)

    def test_failed_headless_run_returns_one_without_cleanup(self) -> None:
        harness = FakeHarness([RuntimeError("provider failed")])
        with patch.object(cli, "build_harness", return_value=harness):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main(["-p", "hello"])
        self.assertEqual(code, 1)
        self.assertFalse(harness.closed)

    def test_headless_safe_policy_has_no_approver(self) -> None:
        harness = FakeHarness([result()])
        captured = {}

        def build(args, *, policy, on_event):
            captured["policy"] = policy
            return harness

        with patch.object(cli, "build_harness", side_effect=build):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["-p", "hello"]), 0)
        self.assertIsNone(captured["policy"].approver)

    def test_resume_is_forwarded(self) -> None:
        harness = FakeHarness([result()])
        with patch.object(cli, "build_harness", return_value=harness):
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(
                    ["--resume", "session.jsonl", "-p", "hello"]
                )
        self.assertEqual(code, 0)
        self.assertEqual(harness.resume_calls, ["session.jsonl"])


class InteractiveTests(unittest.TestCase):
    def test_interactive_normal_exit_closes(self) -> None:
        harness = FakeHarness([result()])
        with patch("builtins.input", side_effect=["hello", "exit"]):
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli._run_interactive(
                    harness, cli.build_parser().parse_args([])
                )
        self.assertEqual(code, 0)
        self.assertTrue(harness.closed)
        self.assertEqual(harness.run_calls, ["hello"])

    def test_interactive_follow_up_uses_continue_task(self) -> None:
        harness = FakeHarness(
            [result(RunStatus.AWAITING_INPUT), result()]
        )
        with patch("builtins.input", side_effect=["build", "continue", "exit"]):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = cli._run_interactive(
                        harness, cli.build_parser().parse_args([])
                    )
        self.assertEqual(code, 0)
        self.assertTrue(harness.closed)
        self.assertEqual(harness.run_calls, ["build"])
        self.assertEqual(harness.continue_calls, ["continue"])

    def test_interactive_ctrl_c_leaves_sandbox_for_resume(self) -> None:
        harness = FakeHarness([KeyboardInterrupt()])
        with patch("builtins.input", side_effect=["hello"]):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = cli._run_interactive(
                        harness, cli.build_parser().parse_args([])
                    )
        self.assertEqual(code, 130)
        self.assertFalse(harness.closed)


class HeadlessStreamingTests(unittest.TestCase):
    def test_no_stream_is_forwarded_to_harness(self) -> None:
        harness = FakeHarness([result()])
        captured = {}

        def build(args, *, policy, on_event):
            captured["stream"] = args.stream
            return harness

        with patch.object(cli, "build_harness", side_effect=build):
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["-p", "hello", "--no-stream"])
        self.assertEqual(code, 0)
        self.assertFalse(captured["stream"])

    def test_forced_color_emits_ansi_without_a_tty(self) -> None:
        class EventHarness(FakeHarness):
            def __init__(self, on_event):
                super().__init__([result()])
                self.on_event = on_event

            def run(self, task):
                text = "Hello"
                self.on_event(RunEvent(EventType.ASSISTANT_DELTA, 1, text=text))
                self.on_event(RunEvent(EventType.ASSISTANT, 1, text=text))
                self.on_event(
                    RunEvent(
                        EventType.RUN_END,
                        1,
                        status=RunStatus.COMPLETED,
                        reason="scripted",
                    )
                )
                return result(text=text)

        def build(args, *, policy, on_event):
            return EventHarness(on_event)

        out = io.StringIO()
        with patch.object(cli, "build_harness", side_effect=build):
            with contextlib.redirect_stdout(out):
                code = cli.main(["-p", "hello", "--color"])
        output = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("\x1b[", output)
        self.assertIn("Completed", output)

    def test_plain_headless_output_has_no_ansi_and_prints_deltas_once(self) -> None:
        class EventHarness(FakeHarness):
            def __init__(self, on_event):
                super().__init__([result()])
                self.on_event = on_event

            def run(self, task):
                self.run_calls.append(task)
                text = "Hello\x1b[31m red"
                self.on_event(
                    RunEvent(EventType.ASSISTANT_DELTA, 1, text=text)
                )
                self.on_event(RunEvent(EventType.ASSISTANT, 1, text=text))
                self.on_event(
                    RunEvent(
                        EventType.RUN_END,
                        1,
                        status=RunStatus.COMPLETED,
                        reason="scripted",
                    )
                )
                return result(text="Hello red")

        holder = {}

        def build(args, *, policy, on_event):
            holder["harness"] = EventHarness(on_event)
            return holder["harness"]

        out = io.StringIO()
        with patch.object(cli, "build_harness", side_effect=build):
            with contextlib.redirect_stdout(out):
                code = cli.main(["-p", "hello", "--plain"])
        output = out.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("?", output)
        self.assertEqual(output.count("Hello"), 1)
        self.assertIn("Completed", output)


class MainTests(unittest.TestCase):
    def test_interactive_main_prepares_workspace(self) -> None:
        harness = FakeHarness([result()])
        called = {}

        def run_interactive(live, args):
            called["workspace"] = live.workspace
            return 0

        with patch.object(cli, "build_harness", return_value=harness):
            with patch.object(
                cli, "_run_interactive", side_effect=run_interactive
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = cli.main([])
        self.assertEqual(code, 0)
        self.assertEqual(called["workspace"], "/sandbox/irina")

    def test_startup_error_returns_two(self) -> None:
        with patch.object(
            cli, "build_harness", side_effect=RuntimeError("no key")
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(["-p", "hello"]), 2)


if __name__ == "__main__":
    unittest.main()
