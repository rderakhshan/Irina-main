"""Stage 4 tests for artifact download from an OpenShell sandbox."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.back.harness.tools_and_guardrails.sandbox import (
    OpenShellSandbox,
    SandboxError,
)


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.sandbox = OpenShellSandbox(workspace="irina")
        self.sandbox._started = True
        self.sandbox.workspace = "/sandbox/irina"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_downloads_workspace_file_and_creates_parent(self) -> None:
        target = self.root / "project" / "index.html"
        with patch.object(
            self.sandbox,
            "_cli",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as cli:
            returned = self.sandbox.download("index.html", target)
        self.assertEqual(returned, target.resolve())
        self.assertTrue(target.parent.is_dir())
        command = cli.call_args.args[0]
        self.assertEqual(command[:3], ["sandbox", "download", self.sandbox.name])
        self.assertEqual(command[3], "/sandbox/irina/index.html")

    def test_download_rejects_workspace_escape(self) -> None:
        with self.assertRaises(SandboxError):
            self.sandbox.download("../secret.txt", self.root / "out.txt")
        with self.assertRaises(SandboxError):
            self.sandbox.download("/etc/passwd", self.root / "out.txt")

    def test_download_requires_a_started_sandbox(self) -> None:
        self.sandbox._started = False
        with self.assertRaises(SandboxError):
            self.sandbox.download("index.html", self.root / "out.txt")

    def test_download_reports_cli_failure(self) -> None:
        with patch.object(
            self.sandbox,
            "_cli",
            return_value=subprocess.CompletedProcess([], 7, b"", b"denied"),
        ):
            with self.assertRaises(SandboxError):
                self.sandbox.download("index.html", self.root / "out.txt")


if __name__ == "__main__":
    unittest.main()
