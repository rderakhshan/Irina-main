"""Tests for memory log visibility and coloring."""

from __future__ import annotations

import io
import logging
import tempfile
import unittest
from pathlib import Path

from src.back.harness.context_management.memory import (
    ExperienceRecord,
    MemoryConfig,
    MemoryManager,
)
from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.orchestration.cli import build_parser
from src.back.harness.utils.memory_logging import (
    MemoryLogHandler,
    _memory_loggers,
    capture_memory_output,
    configure_memory_logging,
    reset_memory_logging,
)
from src.back.harness.utils.theme import Theme


class FakeMemoryModel:
    def __call__(self, messages):
        class Response:
            pass

        response = Response()
        response.content = (
            '{"strategic":["Inspect the assertion."],'
            '"operational":["Rerun the tests."]}'
        )
        return response


def _snapshot():
    return {
        logger: (list(logger.handlers), logger.level, logger.propagate)
        for logger in _memory_loggers()
    }


def _restore(snapshot) -> None:
    for logger in _memory_loggers():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    for logger, (handlers, level, propagate) in snapshot.items():
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
    reset_memory_logging()


class MemoryLoggingTests(unittest.TestCase):
    def test_default_hides_routine_logs_and_keeps_errors(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=False, theme=Theme(color=True), stream=stream
            )
            logger.info("initializing")
            logger.debug("details")
            logger.warning("routine warning")
            self.assertEqual(stream.getvalue(), "")
            logger.error("provider failed")
            output = stream.getvalue()
            self.assertIn("provider failed", output)
            self.assertIn("ERROR", output)
        finally:
            _restore(snapshot)

    def test_verbose_uses_level_colors(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=True), stream=stream
            )
            logger.debug("details")
            logger.info("retrieving")
            logger.warning("careful")
            logger.error("broken")
            output = stream.getvalue()
            self.assertIn("\033[2m", output)   # dim DEBUG
            self.assertIn("\033[36m", output)  # cyan INFO
            self.assertIn("\033[33m", output)  # yellow WARNING
            self.assertIn("\033[31m", output)  # red ERROR
            self.assertIn("Lightweight", output)
        finally:
            _restore(snapshot)

    def test_success_messages_are_green(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=True), stream=stream
            )
            logger.info("Ingestion complete: 2 items")
            self.assertIn("\033[32m", stream.getvalue())
        finally:
            _restore(snapshot)

    def test_no_color_and_no_ansi(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=False), stream=stream
            )
            logger.info("retrieving")
            self.assertNotIn("\033[", stream.getvalue())
            self.assertIn("INFO", stream.getvalue())
        finally:
            _restore(snapshot)

    def test_reconfigure_does_not_duplicate_logs(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=False), stream=stream
            )
            configure_memory_logging(
                verbose=True, theme=Theme(color=False), stream=stream
            )
            ours = [
                handler
                for handler in logger.handlers
                if isinstance(handler, MemoryLogHandler)
            ]
            self.assertEqual(len(ours), 1)
            logger.info("once")
            self.assertEqual(stream.getvalue().count("once"), 1)
        finally:
            _restore(snapshot)

    def test_provider_handler_is_replaced(self) -> None:
        logger = logging.getLogger("EvolveLab.providers.test_provider.Lightweight")
        provider_stream = io.StringIO()
        logger.addHandler(logging.StreamHandler(provider_stream))
        snapshot = _snapshot()
        stream = io.StringIO()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=False), stream=stream
            )
            logger.info("retrieving")
            self.assertEqual(provider_stream.getvalue(), "")
            self.assertIn("retrieving", stream.getvalue())
        finally:
            _restore(snapshot)

    def test_captured_prints_are_hidden_but_errors_remain(self) -> None:
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=False, theme=Theme(color=False), stream=stream
            )
            with capture_memory_output(provider="expel"):
                print("routine ingestion detail")
                print("Error: provider failed")
            output = stream.getvalue()
            self.assertNotIn("routine ingestion detail", output)
            self.assertIn("provider failed", output)
            self.assertIn("ERROR", output)
        finally:
            _restore(snapshot)

    def test_captured_prints_are_colored_in_verbose_mode(self) -> None:
        stream = io.StringIO()
        snapshot = _snapshot()
        try:
            configure_memory_logging(
                verbose=True, theme=Theme(color=True), stream=stream
            )
            with capture_memory_output(provider="expel"):
                print("Loaded 5 memories")
                print("Warning: partial index")
                print("Error: broken")
            output = stream.getvalue()
            self.assertIn("Expel", output)
            self.assertIn("\033[32m", output)  # success/loaded
            self.assertIn("\033[33m", output)  # warning
            self.assertIn("\033[31m", output)  # error
        finally:
            _restore(snapshot)

    def test_renderer_memory_events_follow_verbose(self) -> None:
        from src.back.harness.utils.renderer import TerminalRenderer

        stream = io.StringIO()
        renderer = TerminalRenderer(Theme(color=False), stream)
        event = RunEvent(
            type=EventType.MEMORY,
            turn=0,
            text="Retrieved 1 relevant experience(s)",
            memory_provider="lightweight_memory",
            memory_phase="retrieve",
            memory_ok=True,
            memory_count=1,
        )
        renderer(event)
        self.assertEqual(stream.getvalue(), "")
        renderer.memory_verbose = True
        renderer(event)
        self.assertIn("Retrieved 1 relevant experience(s)", stream.getvalue())

        failed = RunEvent(
            type=EventType.MEMORY,
            turn=0,
            text="Memory update failed",
            memory_phase="update",
            memory_ok=False,
        )
        renderer.memory_verbose = False
        renderer(failed)
        self.assertIn("Memory update failed", stream.getvalue())

    def test_parser_exposes_verbose_flag(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["--memory"])
        verbose = parser.parse_args(["--memory", "--verbose"])
        self.assertFalse(default.verbose)
        self.assertTrue(verbose.verbose)

    def test_real_manager_logs_are_hidden_then_shown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            config = MemoryConfig(
                enabled=True, state_dir=str(Path(temp) / "memory")
            )
            snapshot = _snapshot()
            silent = io.StringIO()
            try:
                configure_memory_logging(
                    verbose=False, theme=Theme(color=False), stream=silent
                )
                manager = MemoryManager(
                    config, workspace=workspace, model=FakeMemoryModel()
                )
                self.assertEqual(silent.getvalue(), "")
                manager.after_task(
                    ExperienceRecord(
                        experience_id="log-1",
                        task_description="fix a failing test",
                        verified_success=True,
                        run_status="completed",
                        execution_outcome="completed",
                        tool_observations=[
                            {
                                "tool_call_id": "c1",
                                "name": "bash",
                                "text": "OK",
                            }
                        ],
                    )
                )
                self.assertEqual(silent.getvalue(), "")

                verbose = io.StringIO()
                configure_memory_logging(
                    verbose=True, theme=Theme(color=False), stream=verbose
                )
                manager.before_task("fix a failing test")
                output = verbose.getvalue()
                self.assertIn("RETRIEVAL", output.upper())
                self.assertIn("INFO", output)
            finally:
                manager.close()
                _restore(snapshot)


if __name__ == "__main__":
    unittest.main()
