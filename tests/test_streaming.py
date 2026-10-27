"""Stage 5 streaming, renderer, Markdown, and fleet presentation tests."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

from src.back.harness.loop.runner import (
    EventType,
    _consume_stream,
    run_loop,
)
from src.back.harness.loop.state import RunStatus
from src.back.harness.orchestration.harness import Harness
from src.back.harness.orchestration.session import load
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy
from src.back.harness.providers import deepseek
from src.back.harness.providers.base import (
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderStreamError,
    ProviderStreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from src.back.harness.providers.deepseek import DeepSeekProvider
from src.back.harness.tools_and_guardrails.tools import Tool
from src.back.harness.utils import (
    FleetRenderer,
    StreamingRedactor,
    TerminalRenderer,
    Theme,
)
from src.back.harness.utils.markdown import MarkdownStream


class FakeJSON:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeSSE:
    def __init__(self, chunks: list[dict], *, close_flag: list[bool] | None = None):
        self.lines = [
            ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")
            for chunk in chunks
        ] + [b"data: [DONE]\n\n"]
        self.close_flag = close_flag
        self.closed = False

    def __iter__(self):
        yield from self.lines

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        if self.close_flag is not None:
            self.close_flag.append(True)
        return False


def stream_chunks(
    text: str = "Hello",
    *,
    tool_calls: bool = False,
    usage: bool = True,
) -> list[dict]:
    chunks: list[dict] = []
    for piece in (text[:2], text[2:]) if text else ():
        chunks.append({"choices": [{"delta": {"content": piece}}]})
    if tool_calls:
        chunks.extend(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": '{"path": "x", ',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": '"content": "hi"}'
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            ]
        )
    final = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ]
    }
    if usage:
        final["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    chunks.append(final)
    return chunks


def make_provider(**kwargs) -> DeepSeekProvider:
    return DeepSeekProvider(api_key="test", retries=0, **kwargs)


class ProviderStreamTests(unittest.TestCase):
    def test_text_deltas_usage_and_finish_are_preserved(self) -> None:
        response = FakeSSE(stream_chunks("Hello"))
        with mock.patch.object(deepseek, "urlopen", return_value=response):
            events = list(make_provider().stream("m", "", [], []))
        self.assertEqual(
            [event.text for event in events if event.type is StreamEventType.TEXT_DELTA],
            ["He", "llo"],
        )
        finish = events[-1]
        self.assertEqual(finish.finish_reason, "stop")
        self.assertEqual(finish.usage, TokenUsage(10, 5))
        self.assertTrue(response.closed)

    def test_partial_tool_arguments_and_ids_assemble(self) -> None:
        response = FakeSSE(stream_chunks("", tool_calls=True))
        with mock.patch.object(deepseek, "urlopen", return_value=response):
            assembled = _consume_stream(
                make_provider(), "m", "", [], [], 1, None
            )
        self.assertEqual(len(assembled.tool_calls), 1)
        call = assembled.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args, {"path": "x", "content": "hi"})
        self.assertEqual(assembled.finish_reason, "tool_calls")

    def test_stream_and_complete_are_semantically_equal(self) -> None:
        non_stream = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "x", "content": "hi"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        stream = FakeSSE(stream_chunks("Hello", tool_calls=True))

        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            return stream if body.get("stream") else FakeJSON(non_stream)

        provider = make_provider()
        with mock.patch.object(deepseek, "urlopen", side_effect=fake_urlopen):
            completed = provider.complete("m", "", [], [])
            assembled = _consume_stream(provider, "m", "", [], [], 1, None)
        self.assertEqual(assembled.text, completed.text)
        self.assertEqual(
            [(c.id, c.name, c.args) for c in assembled.tool_calls],
            [(c.id, c.name, c.args) for c in completed.tool_calls],
        )
        self.assertEqual(assembled.finish_reason, completed.finish_reason)
        self.assertEqual(assembled.usage, completed.usage)

    def test_conflicting_tool_identity_is_accumulated_without_loss(self) -> None:
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {"name": "a"},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "function": {"name": "b"},
                                },
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"x":1}'}},
                                {"index": 1, "function": {"arguments": '{"y":2}'}},
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        with mock.patch.object(deepseek, "urlopen", return_value=FakeSSE(chunks)):
            assembled = _consume_stream(
                make_provider(), "m", "", [], [], 1, None
            )
        self.assertEqual([call.id for call in assembled.tool_calls], ["call_a", "call_b"])
        self.assertEqual([call.args for call in assembled.tool_calls], [{"x": 1}, {"y": 2}])

    def test_truncated_stream_is_invalid(self) -> None:
        chunks = [{"choices": [{"delta": {"content": "partial"}}]}]
        with mock.patch.object(deepseek, "urlopen", return_value=FakeSSE(chunks)):
            with self.assertRaises(ProviderInvalidResponse):
                list(make_provider().stream("m", "", [], []))

    def test_connection_failure_is_a_stream_error(self) -> None:
        class Broken(FakeSSE):
            def __iter__(self):
                raise URLError("network down")

        with mock.patch.object(deepseek, "urlopen", return_value=Broken([])):
            with self.assertRaises(ProviderStreamError):
                list(make_provider().stream("m", "", [], []))


class FakeStreamingProvider:
    def __init__(self, responses: list[list[ProviderStreamEvent]]) -> None:
        self.responses = list(responses)
        self.closed = False

    def complete(self, *args):
        raise AssertionError("streaming path must not call complete")

    def stream(self, model, system, messages, tools):
        try:
            for event in self.responses.pop(0):
                yield event
        finally:
            self.closed = True


class InterruptingProvider:
    def complete(self, *args):
        raise AssertionError("streaming path required")

    def stream(self, *args):
        yield ProviderStreamEvent(
            StreamEventType.TEXT_DELTA, text="partial"
        )
        raise KeyboardInterrupt()


class LoopStreamingTests(unittest.TestCase):
    def make_tool(self, calls: list[dict]) -> Tool:
        return Tool(
            name="write_file",
            spec={
                "schema": {
                    "name": "write_file",
                    "description": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            },
            run=lambda **kwargs: calls.append(kwargs) or "wrote",
        )

    def test_loop_emits_deltas_then_final_assistant(self) -> None:
        provider = FakeStreamingProvider(
            [
                [
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="Hel"),
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="lo"),
                    ProviderStreamEvent(
                        StreamEventType.TOOL_CALL_START,
                        tool_index=0,
                        tool_id="call_1",
                        tool_name="write_file",
                        arguments_delta='{"path": "x"}',
                    ),
                    ProviderStreamEvent(
                        StreamEventType.FINISH,
                        finish_reason="tool_calls",
                        usage=TokenUsage(2, 3),
                    ),
                ],
                [
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="Done"),
                    ProviderStreamEvent(
                        StreamEventType.FINISH,
                        finish_reason="stop",
                        usage=TokenUsage(1, 1),
                    ),
                ],
            ]
        )
        calls: list[dict] = []
        events = []
        result = run_loop(
            provider,
            "m",
            "",
            [{"role": "user", "text": "go"}],
            {"write_file": self.make_tool(calls)},
            on_event=events.append,
            stream=True,
        )
        self.assertTrue(result.completed)
        self.assertEqual(
            [e.text for e in events if e.type is EventType.ASSISTANT_DELTA],
            ["Hel", "lo", "Done"],
        )
        self.assertEqual(calls, [{"path": "x"}])
        assistant = [
            message for message in result.messages if message.get("role") == "assistant"
        ]
        self.assertEqual(assistant[0]["tool_calls"][0]["id"], "call_1")
        self.assertIn("Done", result.text)

    def test_interruption_during_text_never_executes_tools(self) -> None:
        class Interrupting(FakeStreamingProvider):
            def stream(self, model, system, messages, tools):
                try:
                    yield ProviderStreamEvent(
                        StreamEventType.TEXT_DELTA, text="partial"
                    )
                    raise KeyboardInterrupt()
                finally:
                    self.closed = True

        provider = Interrupting([])
        calls: list[dict] = []
        with self.assertRaises(KeyboardInterrupt):
            run_loop(
                provider,
                "m",
                "",
                [{"role": "user", "text": "go"}],
                {"write_file": self.make_tool(calls)},
                stream=True,
            )
        self.assertTrue(provider.closed)
        self.assertEqual(calls, [])

    def test_partial_tool_arguments_never_execute(self) -> None:
        provider = FakeStreamingProvider(
            [
                [
                    ProviderStreamEvent(
                        StreamEventType.TOOL_CALL_START,
                        tool_index=0,
                        tool_id="call_1",
                        tool_name="write_file",
                        arguments_delta='{"path":',
                    ),
                    ProviderStreamEvent(
                        StreamEventType.FINISH, finish_reason="tool_calls"
                    ),
                ],
                [
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="done"),
                    ProviderStreamEvent(
                        StreamEventType.FINISH, finish_reason="stop"
                    ),
                ],
            ]
        )
        calls: list[dict] = []
        result = run_loop(
            provider,
            "m",
            "",
            [{"role": "user", "text": "go"}],
            {"write_file": self.make_tool(calls)},
            stream=True,
        )
        self.assertTrue(result.completed)
        self.assertEqual(calls, [])

    def test_interruption_during_argument_assembly_executes_nothing(self) -> None:
        class Interrupting(FakeStreamingProvider):
            def stream(self, model, system, messages, tools):
                try:
                    yield ProviderStreamEvent(
                        StreamEventType.TOOL_CALL_START,
                        tool_index=0,
                        tool_id="call_1",
                        tool_name="write_file",
                        arguments_delta='{"path":',
                    )
                    raise KeyboardInterrupt()
                finally:
                    self.closed = True

        provider = Interrupting([])
        calls: list[dict] = []
        with self.assertRaises(KeyboardInterrupt):
            run_loop(
                provider,
                "m",
                "",
                [{"role": "user", "text": "go"}],
                {"write_file": self.make_tool(calls)},
                stream=True,
            )
        self.assertTrue(provider.closed)
        self.assertEqual(calls, [])

    def test_interrupted_harness_session_is_persisted(self) -> None:
        provider = InterruptingProvider()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            harness = Harness(
                workdir=workspace,
                model="m",
                provider=provider,
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace),
                state_dir=root / "state",
                stream=True,
            )
            with self.assertRaises(KeyboardInterrupt):
                harness.run("go")
            self.assertEqual(harness.session_status, "interrupted")
            self.assertIsNotNone(harness.session_path)
            session = load(harness.session_path)
            self.assertEqual(session.status, "interrupted")
            harness.close()

    def test_stream_error_before_tool_completion_blocks_execution(self) -> None:
        provider = FakeStreamingProvider(
            [
                [
                    ProviderStreamEvent(
                        StreamEventType.TOOL_CALL_START,
                        tool_index=0,
                        tool_id="call_1",
                        tool_name="write_file",
                        arguments_delta='{"path":',
                    ),
                    ProviderStreamEvent(
                        StreamEventType.ERROR, error="connection reset"
                    ),
                ]
            ]
        )
        calls: list[dict] = []
        result = run_loop(
            provider,
            "m",
            "",
            [{"role": "user", "text": "go"}],
            {"write_file": self.make_tool(calls)},
            stream=True,
        )
        self.assertEqual(result.status, RunStatus.PROVIDER_ERROR)
        self.assertEqual(calls, [])

    def test_missing_stream_usage_is_not_invented(self) -> None:
        provider = FakeStreamingProvider(
            [
                [
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="ok"),
                    ProviderStreamEvent(
                        StreamEventType.FINISH, finish_reason="stop", usage=None
                    ),
                ]
            ]
        )
        events = []
        result = run_loop(
            provider, "m", "", [{"role": "user", "text": "go"}], {},
            on_event=events.append, stream=True,
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.usage, TokenUsage(0, 0))

    def test_deltas_are_not_persisted_as_session_records(self) -> None:
        provider = FakeStreamingProvider(
            [
                [
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="Hel"),
                    ProviderStreamEvent(StreamEventType.TEXT_DELTA, text="lo"),
                    ProviderStreamEvent(
                        StreamEventType.FINISH,
                        finish_reason="stop",
                        usage=TokenUsage(1, 1),
                    ),
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            harness = Harness(
                workdir=workspace,
                model="m",
                provider=provider,
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace),
                state_dir=root / "state",
                stream=True,
            )
            result = harness.run("go")
            self.assertTrue(result.completed)
            self.assertIsNotNone(harness.session_path)
            session = load(harness.session_path)
            types = [record.get("type") for record in session.records]
            self.assertNotIn("assistant_delta", types)
            assistant = [
                message
                for message in session.messages
                if message.get("role") == "assistant"
            ]
            self.assertEqual(assistant[0]["text"], "Hello")
            harness.close()

    def test_streaming_is_not_used_when_flag_is_false(self) -> None:
        class CompleteOnly:
            def __init__(self):
                self.called = False
                self.stream_called = False

            def complete(self, *args):
                self.called = True
                return ProviderResponse(
                    "ok", [], "stop", TokenUsage(1, 1)
                )

            def stream(self, *args):
                self.stream_called = True
                yield

        provider = CompleteOnly()
        result = run_loop(
            provider, "m", "", [{"role": "user", "text": "go"}], {},
            stream=False,
        )
        self.assertTrue(result.completed)
        self.assertTrue(provider.called)
        self.assertFalse(provider.stream_called)


class RendererTests(unittest.TestCase):
    def make_renderer(self) -> tuple[TerminalRenderer, io.StringIO]:
        stream = io.StringIO()
        theme = Theme.detect(stream, plain=True)
        return TerminalRenderer(theme, stream), stream

    def test_streamed_text_is_not_duplicated(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        renderer(RunEvent(EventType.ASSISTANT_DELTA, 1, text="Hello"))
        renderer(RunEvent(EventType.ASSISTANT, 1, text="Hello"))
        self.assertEqual(stream.getvalue().count("Hello"), 1)

    def test_tool_success_failure_blocked_and_interrupted(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        for index, result in enumerate(
            ("wrote 4 lines", "ERROR: boom", "BLOCKED: policy", "cancelled")
        ):
            call = ToolCall(f"c{index}", "write_file", {"path": "x"})
            renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
            renderer(
                RunEvent(
                    EventType.TOOL_END, 1, tool_call=call, result=result
                )
            )
        output = stream.getvalue()
        self.assertIn("o write_file", output)
        self.assertIn("+ wrote 4 lines", output)
        self.assertIn("x boom", output)
        self.assertIn("! policy", output)

    def test_blocked_end_without_a_start_is_visible(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        call = ToolCall("blocked", "bash", {"command": "sudo rm -rf /"})
        renderer(
            RunEvent(
                EventType.TOOL_END,
                1,
                tool_call=call,
                result="BLOCKED: dangerous command",
            )
        )
        output = stream.getvalue()
        self.assertIn("! bash", output)
        self.assertIn("dangerous command", output)

    def test_cancelled_run_has_a_clear_summary(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        renderer(
            RunEvent(
                EventType.RUN_END,
                1,
                status=RunStatus.CANCELLED,
                reason="stream interrupted",
            )
        )
        output = stream.getvalue()
        self.assertIn("Cancelled", output)
        self.assertIn("stream interrupted", output)

    def test_markdown_structure_and_code_fence(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        text = "# Heading\n- item\n1. first\n```py\nprint(1)\n```\n"
        renderer(RunEvent(EventType.ASSISTANT_DELTA, 1, text=text))
        renderer(RunEvent(EventType.ASSISTANT, 1, text=text))
        output = stream.getvalue()
        self.assertIn("Heading", output)
        self.assertIn("- item", output)
        self.assertIn("1. first", output)
        self.assertIn("print(1)", output)
        self.assertNotIn("# Heading", output)

    def test_markdown_partial_blocks_and_malformed_fences(self) -> None:
        outputs: list[str] = []
        stream = MarkdownStream(
            Theme.detect(io.StringIO(), plain=True), outputs.append
        )
        stream.feed("Plain ")
        self.assertEqual("".join(outputs), "Plain ")
        stream.feed("text with `code`")
        stream.feed("\n# Partial heading")
        self.assertIn("code", "".join(outputs))
        self.assertNotIn("Partial heading", "".join(outputs))
        stream.finish()
        self.assertIn("Partial heading", "".join(outputs))
        self.assertNotIn("# Partial", "".join(outputs))

        outputs.clear()
        malformed = MarkdownStream(
            Theme.detect(io.StringIO(), plain=True), outputs.append
        )
        malformed.feed("```\nopen fence without close")
        malformed.finish()
        self.assertIn("open fence without close", "".join(outputs))

    def test_markdown_inline_code_is_styled_without_backticks(self) -> None:
        outputs: list[str] = []
        stream = MarkdownStream(
            Theme(color=True, unicode=True), outputs.append
        )
        stream.feed("Use `irina` now\n")
        joined = "".join(outputs)
        self.assertIn("irina", joined)
        self.assertNotIn("`irina`", joined)
        self.assertIn("\x1b[36m", joined)

    def test_progress_clear_does_not_erase_streamed_text(self) -> None:
        import time

        class TTYStream(io.StringIO):
            encoding = "utf-8"

            def isatty(self) -> bool:
                return True

        from src.back.harness.loop.runner import RunEvent

        stream = TTYStream()
        theme = Theme(color=True, unicode=True)
        renderer = TerminalRenderer(theme, stream, show_progress=True)
        renderer.begin_run()
        time.sleep(0.4)
        tool = ToolCall("t1", "bash", {"command": "true"})
        renderer(
            RunEvent(
                EventType.ASSISTANT,
                1,
                text="",
                tool_calls=[tool],
            )
        )
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=tool))
        renderer(
            RunEvent(EventType.TOOL_END, 1, tool_call=tool, result="ok")
        )
        renderer(RunEvent(EventType.ASSISTANT_DELTA, 2, text="Visible answer"))
        renderer(RunEvent(EventType.ASSISTANT, 2, text="Visible answer"))
        renderer(
            RunEvent(
                EventType.RUN_END,
                2,
                status=RunStatus.COMPLETED,
                reason="ok",
            )
        )
        renderer.stop_progress()
        self.assertIn("Visible answer", stream.getvalue())

    def test_plain_output_has_no_ansi_and_ascii_symbols(self) -> None:
        renderer, stream = self.make_renderer()
        renderer.begin_run()
        renderer.banner(
            model="m", workspace="w", mode="safe", session="s"
        )
        renderer.info("done")
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertIn("* IRINA", stream.getvalue())

    def test_control_characters_and_secrets_are_not_displayed(self) -> None:
        from src.back.harness.loop.runner import RunEvent

        renderer, stream = self.make_renderer()
        renderer.begin_run()
        secret = "sk-" + "a" * 32
        renderer(RunEvent(EventType.ASSISTANT_DELTA, 1, text="\x1b[31m" + secret))
        renderer(RunEvent(EventType.ASSISTANT, 1, text="\x1b[31m" + secret))
        output = stream.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertNotIn(secret, output)

    def test_cross_chunk_secret_redaction(self) -> None:
        redactor = StreamingRedactor()
        secret = "sk-" + "z" * 40
        output = ""
        for char in secret:
            output += redactor.feed(char)
        output += redactor.flush()
        self.assertNotIn(secret, output)
        self.assertIn("REDACTED", output)

    def test_fleet_output_is_serialized(self) -> None:
        out = io.StringIO()
        theme = Theme.detect(out, plain=True)
        renderer = FleetRenderer(["a", "b", "c"], theme, out)
        renderer.header()
        errors: list[BaseException] = []

        def run(name: str) -> None:
            try:
                callback = renderer.callback(name)
                from src.back.harness.loop.runner import RunEvent

                callback(
                    RunEvent(
                        EventType.TOOL_START,
                        1,
                        tool_call=ToolCall(f"{name}1", "bash", {"command": "x"}),
                    )
                )
                callback(
                    RunEvent(
                        EventType.TOOL_END,
                        1,
                        tool_call=ToolCall(f"{name}1", "bash", {"command": "x"}),
                        result="ok",
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(name,)) for name in "abc"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        self.assertTrue(any("a" in line for line in lines))
        self.assertTrue(any("b" in line for line in lines))
        self.assertTrue(any("c" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
