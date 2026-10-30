"""Rich Markdown rendering, streaming, and tool presentation tests."""

from __future__ import annotations

import io
import unittest

from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.providers.base import ToolCall
from src.back.harness.utils.renderer import TerminalRenderer
from src.back.harness.utils.rich_markdown import (
    RichMarkdownStream,
    build_console,
    build_markdown,
    stable_boundary,
)
from src.back.harness.utils.theme import Theme


def make_renderer(
    *, width: int = 100, color: bool = False
) -> tuple[TerminalRenderer, io.StringIO]:
    stream = io.StringIO()
    renderer = TerminalRenderer(
        Theme(color=color, unicode=True),
        stream,
        show_progress=False,
        rich_markdown=True,
        width=width,
    )
    renderer.begin_run()
    return renderer, stream


def render_assistant(text: str, *, width: int = 100) -> str:
    renderer, stream = make_renderer(width=width)
    renderer(RunEvent(EventType.ASSISTANT, 1, text=text))
    return stream.getvalue()


def assistant_turn(renderer: TerminalRenderer, text: str) -> None:
    renderer(RunEvent(EventType.ASSISTANT, 1, text=text))


TABLE = (
    "| Tool | Purpose | Status |\n"
    "|------|---------|--------|\n"
    "| bash | Run commands | Ready |\n"
    "| grep | Search files | Ready |\n"
)


class MarkdownRenderingTests(unittest.TestCase):
    def test_markdown_table_becomes_a_table(self) -> None:
        output = render_assistant(TABLE)
        self.assertIn("Tool", output)
        self.assertIn("Run commands", output)
        self.assertIn("\u2500", output)  # horizontal rule of the table box
        self.assertNotIn("|------|", output)
        self.assertNotIn("| Tool |", output)

    def test_table_respects_terminal_width(self) -> None:
        wide = "| Column | Details |\n|---|---|\n| row | " + ("x" * 300) + " |\n"
        output = render_assistant(wide, width=60)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 60, line)

    def test_table_survives_split_deltas(self) -> None:
        renderer, stream = make_renderer()
        chunks = [
            "| Tool | Purpose |\n",
            "|---|---|\n| bash | Run |\n",
            "| grep | Search |\n",
        ]
        for chunk in chunks:
            renderer(RunEvent(EventType.ASSISTANT_DELTA, 1, text=chunk))
        renderer(RunEvent(EventType.ASSISTANT, 1, text="".join(chunks)))
        output = stream.getvalue()
        self.assertIn("Purpose", output)
        self.assertIn("Search", output)
        self.assertNotIn("|---|", output)
        self.assertEqual(output.count("Search"), 1)

    def test_headings_lists_and_blockquotes(self) -> None:
        output = render_assistant(
            "# Title\n\n### Sub\n\n- one\n- two\n\n1. first\n\n> quoted text\n"
        )
        self.assertIn("Title", output)
        self.assertIn("Sub", output)
        self.assertNotIn("# Title", output)
        self.assertIn("\u2022 one", output)
        self.assertIn("quoted text", output)
        self.assertNotIn("> quoted", output)

    def test_inline_code_and_emphasis(self) -> None:
        output = render_assistant(
            "Use `irina` and **bold** plus *italic* now.\n"
        )
        self.assertNotIn("`irina`", output)
        self.assertIn("irina", output)
        self.assertNotIn("**bold**", output)
        self.assertIn("bold", output)
        self.assertIn("italic", output)

    def test_code_block_is_syntax_highlighted(self) -> None:
        renderer, stream = make_renderer(color=True)
        assistant_turn(
            renderer,
            "```python\ndef hello():\n    print('hi')\n```\n",
        )
        output = stream.getvalue()
        self.assertIn("def hello():", output)
        self.assertIn("print('hi')", output)
        self.assertNotIn("```", output)
        self.assertIn("\x1b[", output)

    def test_code_block_without_language_still_renders(self) -> None:
        output = render_assistant("```\nplain text inside fence\n```\n")
        self.assertIn("plain text inside fence", output)
        self.assertNotIn("```", output)

    def test_malformed_markdown_does_not_crash(self) -> None:
        output = render_assistant(
            "**unclosed emphasis and | broken table\n```\nunclosed fence\n"
        )
        self.assertIn("unclosed emphasis", output)
        self.assertIn("unclosed fence", output)

    def test_plain_text_remains_readable(self) -> None:
        output = render_assistant("Hello there, nothing special.\n")
        self.assertIn("Hello there, nothing special.", output)

    def test_response_is_not_duplicated(self) -> None:
        output = render_assistant("UNIQUEWORD response.\n")
        self.assertEqual(output.count("UNIQUEWORD"), 1)

    def test_original_text_is_preserved_in_the_stream(self) -> None:
        console = build_console(
            io.StringIO(), terminal=False, color=False, width=80
        )
        stream = RichMarkdownStream(console)
        original = TABLE + "\n```py\nx = 1\n```\n"
        for piece in (original[:20], original[20:60], original[60:]):
            stream.feed(piece)
        self.assertEqual(stream.text, original)
        stream.finish()

    def test_links_are_clickable_when_supported(self) -> None:
        buffer = io.StringIO()
        console = build_console(buffer, terminal=True, color=False, width=80)
        console.print(build_markdown("See [docs](https://example.com).\n"))
        raw = buffer.getvalue()
        self.assertIn("\x1b]8;", raw)  # OSC-8 hyperlink
        self.assertIn("docs", raw)
        self.assertNotIn("](", raw)

    def test_build_markdown_renders_without_error(self) -> None:
        self.assertIsNotNone(build_markdown(TABLE))


class LiveStreamingTests(unittest.TestCase):
    def test_live_stream_clears_and_prints_the_final_document_once(self) -> None:
        from src.back.harness.utils.formatting import strip_ansi

        buffer = io.StringIO()
        console = build_console(buffer, terminal=True, color=False, width=80)
        stream = RichMarkdownStream(
            console, live=True, redraw_interval=0.0, min_growth=0
        )
        stream.feed("| A | B |\n")
        stream.feed("|---|---|\n| 1 | 2 |\n")
        stream.feed("Done.\n")
        self.assertTrue(stream.is_live)
        # A transient region is erased on stop, so the repaint frames cannot
        # remain on screen as duplicates.
        self.assertTrue(stream._live.transient)  # type: ignore[union-attr]
        stream.finish()
        self.assertFalse(stream.is_live)
        self.assertEqual(stream.text, "| A | B |\n|---|---|\n| 1 | 2 |\nDone.\n")
        visible = strip_ansi(buffer.getvalue())
        self.assertIn("Done.", visible)
        self.assertTrue(visible.rstrip().endswith("Done."))

    def test_large_response_switches_to_append_blocks(self) -> None:
        from src.back.harness.utils.formatting import strip_ansi

        buffer = io.StringIO()
        console = build_console(buffer, terminal=True, color=False, width=80)
        stream = RichMarkdownStream(
            console,
            live=True,
            redraw_interval=0.0,
            min_growth=0,
            max_live_chars=200,
        )
        stream.feed("# Title\n\n" + ("paragraph text " * 20) + "\n\n")
        self.assertTrue(stream.append_mode)
        stream.feed("| A | B |\n|---|---|\n| 1 | 2 |\n\n")
        stream.finish()
        visible = strip_ansi(buffer.getvalue())
        # Append mode never repaints the prefix, so it appears once.
        self.assertEqual(visible.count("Title"), 1)
        self.assertIn("paragraph text", visible)
        self.assertNotIn("| A |", visible)

    def test_stable_boundary_respects_fences_and_blank_lines(self) -> None:
        self.assertEqual(stable_boundary("a\n\nb"), 3)
        self.assertEqual(stable_boundary("no blank line"), 0)
        fenced = "```\ncode\n\nmore\n```\n\ntail"
        self.assertEqual(stable_boundary(fenced), fenced.index("tail"))
        self.assertEqual(stable_boundary("| a |\n|---|\n| 1 |"), 0)

    def test_live_region_is_not_started_for_blank_text(self) -> None:
        console = build_console(
            io.StringIO(), terminal=True, color=False, width=80
        )
        stream = RichMarkdownStream(console, live=True)
        stream.feed("   \n")
        stream.finish()


class ToolPresentationTests(unittest.TestCase):
    def test_tool_start_and_completion_are_structured(self) -> None:
        renderer, stream = make_renderer()
        call = ToolCall("t1", "read_file", {"path": "src/app.py"})
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
        renderer(
            RunEvent(EventType.TOOL_END, 1, tool_call=call, result="wrote 4 lines")
        )
        output = stream.getvalue()
        self.assertIn("\u25cf read_file", output)  # running marker
        self.assertIn("src/app.py", output)
        self.assertIn("\u2713 read_file completed", output)
        self.assertIn("wrote 4 lines", output)

    def test_tool_failure_and_blocked_are_distinguishable(self) -> None:
        renderer, stream = make_renderer()
        failed = ToolCall("t2", "bash", {"command": "pytest"})
        blocked = ToolCall("t3", "bash", {"command": "sudo rm -rf /"})
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=failed))
        renderer(
            RunEvent(EventType.TOOL_END, 1, tool_call=failed, result="ERROR: boom")
        )
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=blocked))
        renderer(
            RunEvent(
                EventType.TOOL_END,
                1,
                tool_call=blocked,
                result="BLOCKED: dangerous command",
            )
        )
        output = stream.getvalue()
        self.assertIn("\u2717 bash failed", output)
        self.assertIn("boom", output)
        self.assertIn("! bash blocked", output)

    def test_verbose_mode_shows_diagnostics_without_flooding(self) -> None:
        renderer, stream = make_renderer()
        renderer.verbose = True
        call = ToolCall("t4", "bash", {"command": "seq 1 60"})
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
        body = "\n".join(f"line {index}" for index in range(1, 61))
        renderer(RunEvent(EventType.TOOL_END, 1, tool_call=call, result=body))
        output = stream.getvalue()
        self.assertIn("line 1", output)
        self.assertIn("more lines", output)
        self.assertNotIn("line 60", output)

    def test_non_verbose_mode_hides_large_output(self) -> None:
        renderer, stream = make_renderer()
        call = ToolCall("t5", "bash", {"command": "seq 1 60"})
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
        body = "\n".join(f"line {index}" for index in range(1, 61))
        renderer(RunEvent(EventType.TOOL_END, 1, tool_call=call, result=body))
        output = stream.getvalue()
        self.assertNotIn("line 60", output)

    def test_tool_events_flush_assistant_markdown_first(self) -> None:
        renderer, stream = make_renderer()
        renderer(
            RunEvent(
                EventType.ASSISTANT_DELTA,
                1,
                text="| Step | State |\n|---|---|\n| verify | Ready |\n",
            )
        )
        call = ToolCall("t6", "bash", {"command": "pytest"})
        renderer(RunEvent(EventType.TOOL_START, 1, tool_call=call))
        output = stream.getvalue()
        self.assertIn("verify", output)
        self.assertLess(output.index("verify"), output.index("\u25cf bash"))
        self.assertNotIn("|---|", output)


class InterruptTests(unittest.TestCase):
    def test_interrupt_finalises_the_partial_response(self) -> None:
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self) -> bool:
                return True

        stream = TTY()
        renderer = TerminalRenderer(
            Theme(color=True, unicode=True),
            stream,
            show_progress=False,
            width=80,
        )
        renderer.begin_run()
        renderer(
            RunEvent(
                EventType.ASSISTANT_DELTA,
                1,
                text="## Partial\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
            )
        )
        self.assertIsNotNone(renderer._markdown)
        renderer.interrupt()
        self.assertIsNone(renderer._markdown)
        self.assertIn("Partial", stream.getvalue())


class FallbackTests(unittest.TestCase):
    def test_plain_mode_keeps_line_oriented_output(self) -> None:
        stream = io.StringIO()
        renderer = TerminalRenderer(
            Theme.detect(stream, plain=True), stream, show_progress=False
        )
        renderer.begin_run()
        assistant_turn(renderer, "# Title\n- item\n")
        output = stream.getvalue()
        self.assertNotIn("\x1b", output)
        self.assertIn("- item", output)

    def test_rich_module_falls_back_when_unavailable(self) -> None:
        from src.back.harness.utils import renderer as renderer_module

        stream = io.StringIO()
        original = renderer_module._RICH_AVAILABLE
        renderer_module._RICH_AVAILABLE = False
        try:
            renderer = TerminalRenderer(
                Theme(color=False, unicode=True),
                stream,
                show_progress=False,
                rich_markdown=True,
            )
            renderer.begin_run()
            assistant_turn(renderer, "# Title\n")
        finally:
            renderer_module._RICH_AVAILABLE = original
        self.assertFalse(renderer.rich_markdown)
        self.assertIn("Title", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
