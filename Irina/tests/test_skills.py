"""Stage 2: tests for skill discovery, loading, and policy behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.back.harness.providers.base import ToolCall
from src.back.harness.context_management.skills import (
    SKILLS_DIR,
    catalog,
    catalog_prompt,
    read_skill,
    skill_tools,
)
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy


class SkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.workspace = LocalWorkspace(self.root)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _write_skill(
        self,
        name: str,
        description: str = "A test skill.",
        body: str = "Follow these instructions.",
    ) -> Path:
        directory = self.root / SKILLS_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "SKILL.md"
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
            encoding="utf-8",
        )
        return path

    def test_catalog_discovers_front_matter_and_sorts_names(self) -> None:
        self._write_skill("zeta", "Last skill.")
        self._write_skill("alpha", "First skill.")
        entries = catalog(str(self.root), workspace=self.workspace)
        self.assertEqual(list(entries), ["alpha", "zeta"])
        self.assertEqual(entries["alpha"]["description"], "First skill.")
        self.assertEqual(
            entries["alpha"]["path"], "skills/alpha/SKILL.md"
        )

    def test_catalog_ignores_malformed_and_invalid_names(self) -> None:
        self._write_skill("good", "Good skill.")
        bad = self.root / SKILLS_DIR / "bad name"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("---\ndescription: bad\n---\n", encoding="utf-8")
        hidden = self.root / SKILLS_DIR / ".hidden"
        hidden.mkdir(parents=True)
        (hidden / "SKILL.md").write_text("hidden", encoding="utf-8")
        irregular = self.root / SKILLS_DIR / "nested" / "deeper"
        irregular.mkdir(parents=True)
        (irregular / "SKILL.md").write_text("nested", encoding="utf-8")
        entries = catalog(str(self.root), workspace=self.workspace)
        self.assertEqual(list(entries), ["good"])

    def test_catalog_prompt_has_only_names_and_short_descriptions(self) -> None:
        self._write_skill("brand-voice", "Friendly pirate voice.")
        prompt = catalog_prompt(str(self.root), workspace=self.workspace)
        self.assertEqual(
            prompt,
            "Skills available (load one with the use_skill tool when relevant):\n"
            "- brand-voice: Friendly pirate voice.",
        )
        self.assertNotIn("Follow these instructions", prompt)

    def test_catalog_prompt_is_empty_without_skills(self) -> None:
        self.assertEqual(catalog_prompt(str(self.root), workspace=self.workspace), "")

    def test_read_skill_returns_complete_contents(self) -> None:
        path = self._write_skill("brand-voice", "Pirate.", "Use pirate words.")
        result = read_skill(
            str(self.root), "brand-voice", workspace=self.workspace
        )
        self.assertEqual(result, path.read_text(encoding="utf-8"))

    def test_missing_skill_lists_available_names(self) -> None:
        self._write_skill("brand-voice", "Pirate.")
        result = read_skill(
            str(self.root), "missing", workspace=self.workspace
        )
        self.assertEqual(
            result,
            "ERROR: no skill named missing. Available: brand-voice",
        )

    def test_invalid_names_and_traversal_are_rejected(self) -> None:
        for name in ("../secret", "..", ".", "a/b", "bad\\name", "x" * 100):
            with self.subTest(name=name):
                result = read_skill(
                    str(self.root), name, workspace=self.workspace
                )
                self.assertTrue(result.startswith("ERROR: invalid skill name:"))

    def test_oversized_and_empty_skills_are_reported(self) -> None:
        huge = self.root / SKILLS_DIR / "huge"
        huge.mkdir(parents=True)
        (huge / "SKILL.md").write_text(
            "\n".join(f"line {index}" for index in range(4001)),
            encoding="utf-8",
        )
        result = read_skill(str(self.root), "huge", workspace=self.workspace)
        self.assertIn("too large", result)

        empty = self.root / SKILLS_DIR / "empty"
        empty.mkdir(parents=True)
        (empty / "SKILL.md").write_text("", encoding="utf-8")
        result = read_skill(str(self.root), "empty", workspace=self.workspace)
        self.assertIn("is empty", result)

    def test_secrets_are_redacted_in_catalog_and_skill_output(self) -> None:
        self._write_skill(
            "secretive",
            "Uses sk-abcdefghijklmnopqrstuvwx",
            "Token: sk-abcdefghijklmnopqrstuvwx",
        )
        entries = catalog(str(self.root), workspace=self.workspace)
        self.assertIn("[REDACTED]", entries["secretive"]["description"])
        loaded = read_skill(
            str(self.root), "secretive", workspace=self.workspace
        )
        self.assertIn("[REDACTED]", loaded)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", loaded)

    def test_symlink_escape_is_not_catalogued_or_read(self) -> None:
        outside = self.root.parent / "outside-skill.md"
        outside.write_text("outside secret", encoding="utf-8")
        directory = self.root / SKILLS_DIR / "escape"
        directory.mkdir(parents=True)
        try:
            os.symlink(outside, directory / "SKILL.md")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this host")

        entries = catalog(str(self.root), workspace=self.workspace)
        self.assertNotIn("escape", entries)
        result = read_skill(
            str(self.root), "escape", workspace=self.workspace
        )
        self.assertTrue(result.startswith("ERROR:"))
        self.assertNotIn("outside secret", result)

    def test_use_skill_tool_and_read_only_policy(self) -> None:
        self._write_skill("brand-voice", "Pirate.", "Use pirate words.")
        tool = skill_tools(str(self.root), workspace=self.workspace)[0]
        self.assertEqual(tool.name, "use_skill")
        self.assertEqual(tool.spec["schema"]["name"], "use_skill")
        self.assertIn("Use pirate words.", tool.run(name="brand-voice"))
        self.assertIn(
            "ERROR: no skill named missing",
            tool.run(name="missing"),
        )
        call = ToolCall(id="s1", name="use_skill", args={"name": "brand-voice"})
        self.assertIsNone(Policy("read-only").check(call))
        self.assertIsNone(Policy("safe").check(call))

    def test_front_matter_is_optional(self) -> None:
        directory = self.root / SKILLS_DIR / "plain"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "# Plain skill\n\nA plain body.\n", encoding="utf-8"
        )
        entries = catalog(str(self.root), workspace=self.workspace)
        self.assertIn("plain", entries)
        self.assertEqual(entries["plain"]["description"], "A plain body.")


if __name__ == "__main__":
    unittest.main()
