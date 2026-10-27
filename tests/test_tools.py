"""Day 2: offline tests for the six workspace tools.

Concept: verify schemas, file operations, path containment, listing, searching,
truncation, and the refusal to execute commands without OpenShell.

Design constraints:
* Standard library ``unittest`` only.
* File tools use the shared sandbox worker through the local file-only backend.
* No network, credentials, or OpenShell sandbox is required for these tests.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.back.harness.tools_and_guardrails import Tool, core_tools, resolve_path, tool


def make_tools(root: Path) -> dict:
    """Return the six tools keyed by name for a temporary workspace."""
    return {item.name: item for item in core_tools(root)}


class ToolDecoratorTests(unittest.TestCase):
    def test_decorator_builds_string_schema_and_required_list(self) -> None:
        @tool("demo tool", alpha="first arg", beta="second arg")
        def sample(alpha: str, beta: str = "default") -> str:
            return alpha + beta

        self.assertIsInstance(sample, Tool)
        self.assertEqual(sample.name, "sample")
        self.assertEqual(sample.run.__name__, "sample")
        schema = sample.spec["schema"]
        self.assertEqual(schema["name"], "sample")
        self.assertEqual(schema["description"], "demo tool")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertEqual(
            schema["parameters"]["properties"]["alpha"],
            {"type": "string", "description": "first arg"},
        )
        self.assertEqual(
            schema["parameters"]["properties"]["beta"],
            {"type": "string", "description": "second arg"},
        )
        self.assertEqual(schema["parameters"]["required"], ["alpha"])
        self.assertEqual(sample.run("a"), "adefault")
        self.assertEqual(sample.run("a", "b"), "ab")

    def test_core_tools_exposes_the_six_expected_names_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            names = [item.name for item in core_tools(directory)]
        self.assertEqual(
            names,
            [
                "read_file",
                "write_file",
                "edit_file",
                "bash",
                "list_files",
                "grep",
            ],
        )


class FileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.tools = make_tools(self.root)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_write_creates_parents_and_reports_character_count(self) -> None:
        result = self.tools["write_file"].run(
            path="nested/notes.txt", content="hello\n"
        )
        self.assertEqual(result, "Wrote 6 chars to nested/notes.txt")
        self.assertEqual(
            (self.root / "nested" / "notes.txt").read_text(encoding="utf-8"),
            "hello\n",
        )

    def test_read_numbers_each_line_with_a_tab(self) -> None:
        (self.root / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        result = self.tools["read_file"].run(path="notes.txt")
        self.assertEqual(result, "1\talpha\n2\tbeta")

    def test_read_truncates_after_4000_lines_and_reports_total(self) -> None:
        content = "\n".join(f"line {number}" for number in range(1, 4006))
        (self.root / "long.txt").write_text(content, encoding="utf-8")
        result = self.tools["read_file"].run(path="long.txt")
        lines = result.splitlines()
        self.assertEqual(lines[0], "1\tline 1")
        self.assertEqual(lines[3999], "4000\tline 4000")
        self.assertNotIn("4001\tline 4001", result)
        self.assertIn("4005", result)
        self.assertIn("truncated", result)

    def test_edit_replaces_only_a_unique_snippet(self) -> None:
        (self.root / "sample.txt").write_text(
            "alpha beta alpha", encoding="utf-8"
        )
        ambiguous = self.tools["edit_file"].run(
            path="sample.txt", old="alpha", new="gamma"
        )
        self.assertEqual(
            ambiguous,
            "ERROR: snippet appears 2 times — include more context to make it "
            "unique",
        )
        self.assertEqual(
            (self.root / "sample.txt").read_text(encoding="utf-8"),
            "alpha beta alpha",
        )

        edited = self.tools["edit_file"].run(
            path="sample.txt", old="beta", new="delta"
        )
        self.assertEqual(edited, "Edited sample.txt")
        self.assertEqual(
            (self.root / "sample.txt").read_text(encoding="utf-8"),
            "alpha delta alpha",
        )

    def test_edit_reports_a_missing_snippet_without_modifying_the_file(self) -> None:
        (self.root / "sample.txt").write_text("alpha", encoding="utf-8")
        result = self.tools["edit_file"].run(
            path="sample.txt", old="missing", new="new"
        )
        self.assertEqual(
            result,
            "ERROR: snippet not found — read the file and copy it exactly",
        )
        self.assertEqual(
            (self.root / "sample.txt").read_text(encoding="utf-8"), "alpha"
        )

    def test_read_rejects_a_path_that_escapes_the_workspace(self) -> None:
        result = self.tools["read_file"].run(path="../../etc/passwd")
        self.assertIn("ERROR: PermissionError", result)
        self.assertIn("escapes the working directory", result)
        self.assertNotIn("root:", result)

    def test_resolve_path_raises_permission_error_on_escape(self) -> None:
        with self.assertRaises(PermissionError) as caught:
            resolve_path(self.root, "../outside.txt")
        self.assertIn("escapes the working directory", str(caught.exception))

    def test_symlink_escape_is_rejected_when_symlinks_are_available(self) -> None:
        outside = self.root.parent / "outside-secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this host")
        result = self.tools["read_file"].run(path="link.txt")
        self.assertIn("escapes the working directory", result)
        self.assertNotIn("secret", result)

    def test_write_rejects_escape_and_does_not_create_the_file(self) -> None:
        result = self.tools["write_file"].run(
            path="../escape.txt", content="nope"
        )
        self.assertIn("ERROR: PermissionError", result)
        self.assertFalse((self.root.parent / "escape.txt").exists())

    def test_list_files_skips_ignored_directories_and_is_sorted(self) -> None:
        (self.root / "zeta.txt").write_text("z", encoding="utf-8")
        (self.root / "alpha.txt").write_text("a", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src" / "keep.py").write_text("print(1)", encoding="utf-8")
        for ignored in (".git", "node_modules", "__pycache__", ".venv"):
            directory = self.root / ignored
            directory.mkdir()
            (directory / "hidden.txt").write_text("x", encoding="utf-8")

        result = self.tools["list_files"].run(pattern="**/*")
        self.assertEqual(
            result.splitlines(),
            ["alpha.txt", "src/keep.py", "zeta.txt"],
        )
        self.assertNotIn("hidden", result)

        python_files = self.tools["list_files"].run(pattern="*.py")
        self.assertEqual(python_files, "src/keep.py")

    def test_list_files_caps_at_500_and_reports_remaining(self) -> None:
        for number in range(505):
            (self.root / f"file-{number:03d}.txt").write_text(
                "x", encoding="utf-8"
            )
        result = self.tools["list_files"].run(pattern="**/*")
        lines = result.splitlines()
        self.assertEqual(len(lines), 501)
        self.assertEqual(lines[-1], "and 5 more")

    def test_grep_formats_matches_and_ignores_directories(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text(
            "first\nneedle here\n", encoding="utf-8"
        )
        (self.root / ".git").mkdir()
        (self.root / ".git" / "hidden.py").write_text(
            "needle hidden\n", encoding="utf-8"
        )
        result = self.tools["grep"].run(regex="needle", pattern="*.py")
        self.assertEqual(result, "src/a.py:2: needle here")
        self.assertNotIn("hidden", result)

    def test_grep_clips_long_lines_and_reports_invalid_regex(self) -> None:
        long_line = "needle" + ("x" * 300)
        (self.root / "long.txt").write_text(long_line, encoding="utf-8")
        result = self.tools["grep"].run(regex="needle", pattern="*")
        line = result.splitlines()[0]
        text = line.split(": ", 1)[1]
        self.assertLessEqual(len(text), 200)

        invalid = self.tools["grep"].run(regex="[", pattern="*")
        self.assertTrue(invalid.startswith("ERROR: invalid regex:"))

    def test_bash_without_openshell_fails_clearly(self) -> None:
        result = self.tools["bash"].run(command="echo hello")
        self.assertIn("ERROR: SandboxUnavailable", result)
        self.assertIn("OpenShell is required", result)

    def test_all_tool_specs_match_the_provider_shape(self) -> None:
        for name, item in self.tools.items():
            schema = item.spec["schema"]
            self.assertEqual(schema["name"], name)
            self.assertEqual(schema["parameters"]["type"], "object")
            for property_schema in schema["parameters"]["properties"].values():
                self.assertEqual(property_schema["type"], "string")


if __name__ == "__main__":
    unittest.main()
