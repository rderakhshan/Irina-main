"""Stage 2: tests for skill discovery, loading, and policy behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.back.harness import Harness
from src.back.harness.orchestration.cli import build_parser
from src.back.harness.providers.base import ToolCall
from src.back.harness.context_management.skills import (
    SKILLS_DIR,
    catalog,
    catalog_prompt,
    discard_skill_draft,
    draft_root,
    ensure_host_skills_dir,
    list_skill_drafts,
    promote_skill_draft,
    read_skill,
    seed_skills,
    skill_tools,
    sync_skill_drafts,
)
from src.back.harness.tools_and_guardrails import LocalWorkspace, Policy
from src.back.harness.tools_and_guardrails.transfers import export_artifacts


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


class SeedSkillsTests(unittest.TestCase):
    def test_seed_uploads_only_valid_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "host-skills"
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            valid = host / "brand-voice"
            valid.mkdir(parents=True)
            (valid / "SKILL.md").write_text(
                "---\ndescription: Friendly voice.\n---\n\nUse short sentences.\n",
                encoding="utf-8",
            )
            invalid = host / "bad name"
            invalid.mkdir()
            (invalid / "SKILL.md").write_text("ignored\n", encoding="utf-8")
            empty = host / "empty"
            empty.mkdir()
            (empty / "SKILL.md").write_text("", encoding="utf-8")
            oversized = host / "oversized"
            oversized.mkdir()
            (oversized / "SKILL.md").write_text(
                "x" * 200_001, encoding="utf-8"
            )
            no_file = host / "no-file"
            no_file.mkdir()

            workspace = LocalWorkspace(workspace_root)
            installed = seed_skills(str(host), str(workspace_root), workspace=workspace)
            self.assertEqual(installed, ["brand-voice"])
            entries = catalog(str(workspace_root), workspace=workspace)
            self.assertIn("brand-voice", entries)
            self.assertNotIn("bad name", entries)
            self.assertIn(
                "Use short sentences.",
                read_skill(str(workspace_root), "brand-voice", workspace=workspace),
            )

    def test_seed_uploads_scripts_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "skills"
            skill = host / "with-assets"
            (skill / "assets").mkdir(parents=True)
            (skill / "scripts").mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\n"
                "description: Asset skill.\n"
                "---\n"
                "\n"
                "Use assets.\n",
                encoding="utf-8",
            )
            (skill / "assets" / "template.tex").write_text(
                "TEMPLATE\n", encoding="utf-8"
            )
            (skill / "scripts" / "run.py").write_text(
                "print('run')\n", encoding="utf-8"
            )
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            workspace = LocalWorkspace(workspace_root)
            installed = seed_skills(
                str(host), str(workspace_root), workspace=workspace
            )
            self.assertEqual(installed, ["with-assets"])
            self.assertIn(
                "TEMPLATE",
                workspace.read_file("skills/with-assets/assets/template.tex")[
                    "text"
                ],
            )
            self.assertIn(
                "print",
                workspace.read_file("skills/with-assets/scripts/run.py")["text"],
            )

    def test_seed_is_repeatable_and_missing_host_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = LocalWorkspace(root)
            self.assertEqual(
                seed_skills(str(root / "missing"), str(root), workspace=workspace),
                [],
            )
            host = root / "host"
            skill = host / "one"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("first\n", encoding="utf-8")
            self.assertEqual(
                seed_skills(str(host), str(root), workspace=workspace), ["one"]
            )
            self.assertEqual(
                seed_skills(str(host), str(root), workspace=workspace), ["one"]
            )
            self.assertEqual(len(catalog(str(root), workspace=workspace)), 1)

    def test_harness_seeds_before_building_the_prompt(self) -> None:
        class Provider:
            default_model = "fake-model"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "host-skills"
            skill = host / "release-notes"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\ndescription: Write release notes.\n---\n\nGroup by change type.\n",
                encoding="utf-8",
            )
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            harness = Harness(
                workdir=workspace_root,
                provider=Provider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace_root),
                persist=False,
                skills_dir=host,
            )
            prompt = harness.system_prompt
            self.assertIn("release-notes", prompt)
            self.assertEqual(harness.seeded_skills, ["release-notes"])
            harness.close()

    def test_cli_exposes_skills_dir(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--memory"]).skills_dir, "skills")
        self.assertEqual(
            parser.parse_args(["--skills-dir", "custom"]).skills_dir, "custom"
        )
        self.assertEqual(parser.parse_args(["--skills-dir", ""]).skills_dir, "")
        transfer = parser.parse_args(
            [
                "--input", "cv",
                "--export", "application",
                "--artifacts-dir", "out",
                "--sandbox-image", "irina/application-tex:1",
            ]
        )
        self.assertEqual(transfer.inputs, ["cv"])
        self.assertEqual(transfer.export_paths, ["application"])
        self.assertEqual(transfer.artifacts_dir, "out")
        self.assertEqual(transfer.sandbox_image, "irina/application-tex:1")


class SkillDraftTests(unittest.TestCase):
    def test_host_skills_directory_is_created_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            host = Path(temp) / "skills"
            created = ensure_host_skills_dir(host)
            self.assertIsNotNone(created)
            self.assertTrue((host / "README.md").is_file())
            self.assertTrue((host / ".gitkeep").is_file())
            first = (host / "README.md").read_text(encoding="utf-8")
            ensure_host_skills_dir(host)
            self.assertEqual(
                (host / "README.md").read_text(encoding="utf-8"), first
            )

    def test_sync_captures_new_and_modified_but_not_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "skills"
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            state = root / "state"
            for name, body in (
                ("stable", "stable body\n"),
                ("edited", "trusted body\n"),
            ):
                skill = host / name
                skill.mkdir(parents=True)
                (skill / "SKILL.md").write_text(body, encoding="utf-8")
            workspace = LocalWorkspace(workspace_root)
            seed_skills(str(host), str(workspace_root), workspace=workspace)
            workspace.write_file(
                "skills/edited/SKILL.md", "agent changed this\n"
            )
            workspace.write_file("skills/created/SKILL.md", "new skill\n")

            captured = sync_skill_drafts(
                host, str(workspace_root), workspace=workspace, state_dir=state
            )
            self.assertEqual(captured, ["created", "edited"])
            drafts = list_skill_drafts(str(workspace_root), state_dir=state)
            self.assertEqual(sorted(drafts), ["created", "edited"])
            self.assertNotIn("stable", drafts)
            self.assertFalse((host / "created" / "SKILL.md").exists())
            self.assertEqual(
                sync_skill_drafts(
                    host, str(workspace_root), workspace=workspace, state_dir=state
                ),
                [],
            )

    def test_promote_and_discard_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "skills"
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            state = root / "state"
            workspace = LocalWorkspace(workspace_root)
            workspace.write_file("skills/promoted/SKILL.md", "promote me\n")
            sync_skill_drafts(
                host, str(workspace_root), workspace=workspace, state_dir=state
            )
            target = promote_skill_draft(
                "promoted",
                host_dir=host,
                workdir=str(workspace_root),
                state_dir=state,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "promote me\n")
            self.assertFalse(
                (draft_root(str(workspace_root), state_dir=state) / "promoted").exists()
            )
            workspace.write_file("skills/promoted/SKILL.md", "changed\n")
            sync_skill_drafts(
                host, str(workspace_root), workspace=workspace, state_dir=state
            )
            with self.assertRaises(FileExistsError):
                promote_skill_draft(
                    "promoted",
                    host_dir=host,
                    workdir=str(workspace_root),
                    state_dir=state,
                )
            promote_skill_draft(
                "promoted",
                host_dir=host,
                workdir=str(workspace_root),
                state_dir=state,
                force=True,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "changed\n")

            workspace.write_file("skills/discard-me/SKILL.md", "delete\n")
            sync_skill_drafts(
                host, str(workspace_root), workspace=workspace, state_dir=state
            )
            self.assertTrue(
                discard_skill_draft(
                    "discard-me", workdir=str(workspace_root), state_dir=state
                )
            )
            self.assertFalse(
                discard_skill_draft(
                    "discard-me", workdir=str(workspace_root), state_dir=state
                )
            )

    def test_harness_creates_and_syncs_skills_on_close(self) -> None:
        class Provider:
            default_model = "fake-model"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            host = root / "skills"
            host_skill = host / "host-skill"
            host_skill.mkdir(parents=True)
            (host_skill / "SKILL.md").write_text(
                "---\ndescription: A host skill.\n---\n\nHost instructions.\n",
                encoding="utf-8",
            )
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            harness = Harness(
                workdir=workspace_root,
                provider=Provider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace_root),
                persist=False,
                skills_dir=host,
                state_dir=root / "state",
            )
            self.assertIn("Skills", harness.system_prompt)
            self.assertTrue((host / "README.md").is_file())
            harness._backend.write_file(
                "skills/agent-made/SKILL.md", "created during a run\n"
            )
            harness.close()
            drafts = list_skill_drafts(
                str(workspace_root), state_dir=root / "state"
            )
            self.assertIn("agent-made", drafts)
            self.assertFalse((host / "agent-made" / "SKILL.md").exists())


    def test_harness_uploads_inputs_and_exports_artifacts(self) -> None:
        class Provider:
            default_model = "fake-model"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            source = root / "host-input"
            source.mkdir()
            (source / "resume.tex").write_text("CV\n", encoding="utf-8")
            single = root / "single.tex"
            single.write_text("SINGLE\n", encoding="utf-8")
            artifacts = root / "artifacts"
            harness = Harness(
                workdir=workspace_root,
                provider=Provider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace_root),
                persist=False,
                inputs=[source, single],
                export_paths=["out"],
                artifacts_dir=artifacts,
            )
            harness.system_prompt
            self.assertIn(
                "CV", harness._backend.read_file("input/host-input/resume.tex")["text"]
            )
            self.assertEqual(
                harness._backend.read_file("input/single.tex")["text"].strip(),
                "SINGLE",
            )
            harness._backend.write_file("out/report.txt", "result\n")
            harness.close()
            self.assertEqual(
                (artifacts / "out" / "report.txt").read_text(encoding="utf-8"),
                "result\n",
            )
            self.assertEqual(harness.export_errors, [])


    def test_export_contents_lands_company_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            workspace = LocalWorkspace(workspace_root)
            workspace.write_file("application/Acme/application.pdf", "PDF\n")
            host = root / "Application"
            exported = export_artifacts(
                ["application"], workspace, host, contents=True
            )
            self.assertEqual(
                (host / "Acme" / "application.pdf").read_text(encoding="utf-8"),
                "PDF\n",
            )
            self.assertEqual(len(exported), 1)

            # A second export merges into the existing folder, replacing the
            # file the new run regenerated. It never raises "already exists".
            workspace.write_file("application/Acme/application.pdf", "NEW\n")
            third = export_artifacts(
                ["application"], workspace, host, contents=True
            )
            self.assertEqual([path.name for path in third], ["Acme"])
            self.assertEqual(
                (host / "Acme" / "application.pdf").read_text(encoding="utf-8"),
                "NEW\n",
            )

            # A new company lands next to the existing one.
            workspace.write_file("application/Globex/application.pdf", "B\n")
            export_artifacts(["application"], workspace, host, contents=True)
            self.assertEqual(
                (host / "Globex" / "application.pdf").read_text(encoding="utf-8"),
                "B\n",
            )
            self.assertEqual(
                (host / "Acme" / "application.pdf").read_text(encoding="utf-8"),
                "NEW\n",
            )


    def test_export_now_supports_force_and_skips_close_reexport(self) -> None:
        class Provider:
            default_model = "fake-model"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            artifacts = root / "Application"
            harness = Harness(
                workdir=workspace_root,
                provider=Provider(),
                policy=Policy("yolo"),
                workspace_backend=LocalWorkspace(workspace_root),
                persist=False,
                export_paths=["application"],
                export_contents=True,
                artifacts_dir=artifacts,
            )
            harness.system_prompt
            self.assertEqual(harness.export_now(), [])
            harness._backend.write_file("application/Acme/out.txt", "one\n")
            exported = harness.export_now()
            self.assertEqual(
                (artifacts / "Acme" / "out.txt").read_text(encoding="utf-8"),
                "one\n",
            )
            self.assertEqual(len(exported), 1)
            # Re-export merges in place; it must not fail on the existing
            # company folder.
            self.assertEqual(harness.export_now(), [artifacts / "Acme"])
            harness._backend.write_file("application/Acme/out.txt", "two\n")
            harness.export_now()
            self.assertEqual(
                (artifacts / "Acme" / "out.txt").read_text(encoding="utf-8"),
                "two\n",
            )
            harness.close()


if __name__ == "__main__":
    unittest.main()
