"""Stage 2: tests for durable project memory and prompt construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.back.harness.context_management.memory import (
    BASE_SYSTEM_PROMPT,
    MEMORY_FILE,
    build_system_prompt,
    memory_tools,
    read_memory,
    remember,
)
from src.back.harness.providers.base import ToolCall
from src.back.harness.context_management.skills import SKILLS_DIR
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.workspace = LocalWorkspace(self.root)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_missing_memory_produces_a_prompt_without_memory_section(self) -> None:
        prompt = build_system_prompt(
            str(self.root), workspace=self.workspace
        )
        self.assertIn(BASE_SYSTEM_PROMPT, prompt)
        self.assertIn(f"Approved workspace: {self.root}", prompt)
        self.assertNotIn("Project memory (IRINA.md):", prompt)
        self.assertIsNone(read_memory(str(self.root), workspace=self.workspace))

    def test_remember_adds_a_readable_fact(self) -> None:
        result = remember(
            str(self.root),
            "build command: python -m unittest",
            workspace=self.workspace,
        )
        self.assertEqual(result, "Remembered in IRINA.md")
        content = (self.root / MEMORY_FILE).read_text(encoding="utf-8")
        self.assertIn("# Project memory (IRINA.md)", content)
        self.assertIn("- build command: python -m unittest", content)

    def test_exact_duplicates_are_not_appended(self) -> None:
        remember(
            str(self.root),
            "build command: python -m unittest",
            workspace=self.workspace,
        )
        remember(
            str(self.root),
            "build command: python -m unittest",
            workspace=self.workspace,
        )
        content = (self.root / MEMORY_FILE).read_text(encoding="utf-8")
        self.assertEqual(content.count("build command:"), 1)

    def test_keyed_fact_can_be_superseded(self) -> None:
        remember(
            str(self.root),
            "test command: python old.py",
            workspace=self.workspace,
        )
        remember(
            str(self.root),
            "test command: python -m unittest",
            workspace=self.workspace,
        )
        content = (self.root / MEMORY_FILE).read_text(encoding="utf-8")
        self.assertNotIn("python old.py", content)
        self.assertIn("test command: python -m unittest", content)

    def test_empty_notes_are_rejected(self) -> None:
        for note in ("", "   ", "\n\t"):
            with self.subTest(note=note):
                with self.assertRaises(ValueError):
                    remember(str(self.root), note, workspace=self.workspace)

    def test_likely_secrets_are_rejected_and_not_written(self) -> None:
        notes = (
            "api key: sk-abcdefghijklmnopqrstuvwx",
            "DEEPSEEK_API_KEY=super-secret-value",
            "password is hunter2",
            "Authorization: Bearer abcdefghijklmnopqrstuvwx",
        )
        for note in notes:
            with self.subTest(note=note):
                with self.assertRaises(ValueError):
                    remember(str(self.root), note, workspace=self.workspace)
        self.assertFalse((self.root / MEMORY_FILE).exists())

    def test_manual_secret_in_memory_is_redacted_in_the_prompt(self) -> None:
        (self.root / MEMORY_FILE).write_text(
            "- api key: sk-abcdefghijklmnopqrstuvwx\n",
            encoding="utf-8",
        )
        prompt = build_system_prompt(
            str(self.root), workspace=self.workspace
        )
        self.assertIn("[REDACTED]", prompt)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", prompt)

    def test_oversized_memory_is_not_overwritten(self) -> None:
        content = "# Project memory (IRINA.md)\n\n- " + ("x" * 20_100) + "\n"
        path = self.root / MEMORY_FILE
        path.write_text(content, encoding="utf-8")
        with self.assertRaises(ValueError):
            remember(str(self.root), "another fact", workspace=self.workspace)
        self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_unreadable_memory_is_skipped_gracefully(self) -> None:
        (self.root / MEMORY_FILE).mkdir()
        prompt = build_system_prompt(
            str(self.root), workspace=self.workspace
        )
        self.assertNotIn("Project memory (IRINA.md):", prompt)

    def test_prompt_contains_platform_memory_skills_and_extra(self) -> None:
        remember(
            str(self.root),
            "build: python -m unittest",
            workspace=self.workspace,
        )
        skill_dir = self.root / SKILLS_DIR / "brand-voice"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Friendly pirate voice.\n---\nUse pirate words.\n",
            encoding="utf-8",
        )
        prompt = build_system_prompt(
            str(self.root),
            extra="Be brief.",
            workspace=self.workspace,
            platform_name="Linux",
        )
        self.assertIn("Platform: Linux", prompt)
        self.assertIn("Project memory (IRINA.md):", prompt)
        self.assertIn("build: python -m unittest", prompt)
        self.assertIn("Skills available (load one with the use_skill tool", prompt)
        self.assertIn("brand-voice: Friendly pirate voice.", prompt)
        self.assertIn("Additional instructions:\nBe brief.", prompt)
        self.assertLess(
            prompt.index("You are Irina"), prompt.index("Project memory")
        )

    def test_memory_persists_across_fresh_workspace_instances(self) -> None:
        remember(
            str(self.root),
            "project codename: lighthouse",
            workspace=self.workspace,
        )
        fresh_workspace = LocalWorkspace(self.root)
        prompt = build_system_prompt(
            str(self.root), workspace=fresh_workspace
        )
        self.assertIn("project codename: lighthouse", prompt)

    def test_remember_tool_and_policy_modes(self) -> None:
        tool = memory_tools(str(self.root), workspace=self.workspace)[0]
        self.assertEqual(tool.name, "remember")
        self.assertEqual(tool.spec["schema"]["name"], "remember")
        self.assertEqual(
            tool.run(note="build: python -m unittest"),
            "Remembered in IRINA.md",
        )
        self.assertTrue(
            tool.run(note="password is hunter2").startswith("ERROR: ValueError")
        )

        call = ToolCall(id="m1", name="remember", args={"note": "x"})
        self.assertIsNotNone(Policy("read-only").check(call))
        self.assertIn(
            "approval required", Policy("safe").check(call) or ""
        )
        self.assertIsNone(Policy("yolo").check(call))


if __name__ == "__main__":
    unittest.main()
