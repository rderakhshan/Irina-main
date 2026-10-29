"""Tests for collapsible multiline paste presentation."""

from __future__ import annotations

import unittest

from src.back.harness.utils.paste import (
    COLLAPSE_CHARS,
    PASTE_END,
    PASTE_START,
    PromptComposer,
    compose_stream,
    is_large_paste,
    paste_placeholder,
    strip_paste_markers,
)
from src.back.harness.orchestration.cli import (
    _capture_prompt,
    _clean_paste,
    _flush_burst,
)


class PasteComposerTests(unittest.TestCase):
    def test_small_paste_stays_visible(self) -> None:
        composer = PromptComposer()
        visible = composer.add_paste("ValueError: invalid input")
        self.assertEqual(visible, "ValueError: invalid input")
        self.assertEqual(composer.render(), visible)
        self.assertEqual(composer.submit(), "ValueError: invalid input")

    def test_three_lines_remain_visible(self) -> None:
        text = "one\ntwo\nthree"
        composer = PromptComposer()
        self.assertFalse(is_large_paste(text))
        self.assertIn("one", composer.add_paste(text))
        self.assertEqual(composer.submit(), text)

    def test_four_lines_collapse(self) -> None:
        text = "one\ntwo\nthree\nfour"
        composer = PromptComposer()
        visible = composer.add_paste(text)
        self.assertEqual(visible, "[Pasted text · 4 lines]")
        self.assertEqual(composer.render(), visible)
        self.assertEqual(composer.submit(), text)

    def test_long_single_line_collapses(self) -> None:
        text = "x" * (COLLAPSE_CHARS + 1)
        composer = PromptComposer()
        self.assertEqual(composer.add_paste(text), "[Pasted text · 1 line]")
        self.assertEqual(composer.submit(), text)
        self.assertEqual(paste_placeholder(text), "[Pasted text · 1 line]")

    def test_text_before_and_after_multiple_blocks_is_preserved(self) -> None:
        first = "a\nb\nc\nd"
        second = "def f():\n    return 1\n\nprint(f())\n"
        composer = PromptComposer()
        composer.add_typed("Explain this error: ")
        composer.add_paste(first)
        composer.add_typed(" and this code: ")
        composer.add_paste(second)
        composer.add_typed(" please")
        self.assertEqual(
            composer.render(),
            "Explain this error: [Pasted text · 4 lines]"
            " and this code: [Pasted text · 4 lines] please",
        )
        self.assertEqual(
            composer.submit(),
            "Explain this error: " + first + " and this code: " + second + " please",
        )

    def test_indentation_newlines_and_special_characters_survive(self) -> None:
        text = 'def f():\n\treturn "a\\nb"  # ünïcode\n\n\nprint(f())\n'
        composer = PromptComposer()
        composer.add_paste(text)
        self.assertEqual(composer.submit(), text)

    def test_backspace_removes_a_collapsed_block_as_one_unit(self) -> None:
        composer = PromptComposer()
        composer.add_typed("before ")
        placeholder = composer.add_paste("1\n2\n3\n4")
        removed = composer.backspace()
        self.assertEqual(removed, len(placeholder))
        self.assertEqual(composer.submit(), "before ")

    def test_expand_last_paste_reveals_original_text(self) -> None:
        composer = PromptComposer()
        text = "1\n2\n3\n4"
        composer.add_paste(text)
        self.assertEqual(composer.expand_last_paste(), text)
        self.assertEqual(composer.render(), text)

    def test_markers_are_removed_and_blocks_composed(self) -> None:
        raw = (
            "prefix "
            + PASTE_START
            + "1\n2\n3\n4"
            + PASTE_END
            + " middle "
            + PASTE_START
            + "small"
            + PASTE_END
            + " suffix"
        )
        composer = compose_stream(raw)
        self.assertEqual(
            composer.submit(),
            "prefix 1\n2\n3\n4 middle small suffix",
        )
        self.assertEqual(
            composer.render(),
            "prefix [Pasted text · 4 lines] middle small suffix",
        )
        self.assertEqual(strip_paste_markers(raw), composer.submit())

    def test_cli_capture_submits_the_original_text(self) -> None:
        raw = (
            "Explain this error: "
            + PASTE_START
            + "ValueError: bad input\n\tat line 2\nline 3\nline 4\n"
            + PASTE_END
        )
        self.assertEqual(
            _capture_prompt(raw),
            "Explain this error: ValueError: bad input\n\tat line 2\nline 3\nline 4",
        )
        self.assertEqual(
            _clean_paste(f"{PASTE_START}abc{PASTE_END}"), "abc"
        )

    def test_burst_fallback_collapses_multiline_paste(self) -> None:
        composer = PromptComposer()
        visible: list[str] = []
        ended = _flush_burst(
            composer, list("a\nb\nc\nd"), visible.append
        )
        self.assertFalse(ended)
        self.assertEqual(visible, ["[Pasted text · 4 lines]"])
        self.assertEqual(composer.submit(), "a\nb\nc\nd")


if __name__ == "__main__":
    unittest.main()
