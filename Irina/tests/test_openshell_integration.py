"""Day 2: OpenShell integration tests for filesystem, network, process, and credentials.

Concept: verify the protections the policy claims to provide, using a real
OpenShell 0.1.5 sandbox. Application-level tests cannot prove runtime isolation.

Design constraints:
* Tests skip when OpenShell is not installed or its gateway is unavailable.
* A fresh sandbox is started once and deleted after the class.
* The sandbox contains no provider credentials.
* These tests do not print environment contents or credentials.
"""

from __future__ import annotations

import os
import unittest

from src.back.harness.tools_and_guardrails import (
    OpenShellSandbox,
    SandboxUnavailable,
    core_tools,
)


class OpenShellIntegrationTests(unittest.TestCase):
    """Integration checks against a real OpenShell sandbox."""

    sandbox: OpenShellSandbox

    @classmethod
    def setUpClass(cls) -> None:
        cls.sandbox = OpenShellSandbox(workspace="integration")
        try:
            cls.sandbox.start()
        except SandboxUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sandbox.stop()

    def test_tools_operate_inside_the_sandbox(self) -> None:
        tools = {item.name: item for item in core_tools(self.sandbox.workspace, self.sandbox)}
        self.assertEqual(
            tools["write_file"].run(path="src/example.txt", content="one\ntwo\n"),
            "Wrote 8 chars to src/example.txt",
        )
        self.assertEqual(
            tools["read_file"].run(path="src/example.txt"),
            "1\tone\n2\ttwo",
        )
        self.assertEqual(
            tools["edit_file"].run(
                path="src/example.txt", old="two", new="three"
            ),
            "Edited src/example.txt",
        )
        self.assertEqual(
            tools["grep"].run(regex="three", pattern="*.txt"),
            "src/example.txt:2: three",
        )
        listing = tools["list_files"].run(pattern="**/*")
        self.assertIn("src/example.txt", listing)

        command = tools["bash"].run(
            command="pwd; python3 -c 'print(6 * 7)'", timeout="20"
        )
        self.assertIn(self.sandbox.workspace, command)
        self.assertIn("42", command)

    def test_d_filesystem_isolation_denies_outside_paths(self) -> None:
        denied = self.sandbox.exec(["ls", "/var"], timeout=10)
        self.assertNotEqual(denied.exit_code, 0)
        self.assertIn("Permission denied", denied.stdout + denied.stderr)

        protected = self.sandbox.exec(["ls", "/.openshell"], timeout=10)
        self.assertNotEqual(protected.exit_code, 0)
        self.assertIn(
            "Permission denied", protected.stdout + protected.stderr
        )

        outside_write = self.sandbox.exec(
            ["touch", "/dev/shm/irina-outside"], timeout=10
        )
        self.assertNotEqual(outside_write.exit_code, 0)
        self.assertIn(
            "Permission denied", outside_write.stdout + outside_write.stderr
        )

        passwd = self.sandbox.exec(["head", "-1", "/etc/passwd"], timeout=10)
        self.assertNotEqual(passwd.exit_code, 0)
        self.assertIn(
            "Permission denied", passwd.stdout + passwd.stderr
        )

        allowed = self.sandbox.exec(["cat", "/etc/hosts"], timeout=10)
        self.assertEqual(allowed.exit_code, 0)
        self.assertIn("localhost", allowed.stdout)

    def test_e_network_isolation_denies_unlisted_destinations(self) -> None:
        probe = (
            "import sys\n"
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=3)\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__, exc)\n"
            "    sys.exit(1)\n"
            "print('CONNECTED')\n"
        )
        self.sandbox.write_file("network_probe.py", probe)
        result = self.sandbox.exec(
            ["python3", "network_probe.py"],
            workdir=self.sandbox.workspace,
            timeout=15,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn("CONNECTED", combined)
        self.assertIn("Permission", combined)

    def test_f_process_restrictions_use_a_non_root_user(self) -> None:
        identity = self.sandbox.exec(["id", "-u"], timeout=10)
        self.assertEqual(identity.exit_code, 0)
        self.assertNotEqual(identity.stdout.strip(), "0")

        probe = (
            "import os\n"
            "import sys\n"
            "try:\n"
            "    os.setuid(0)\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__, exc)\n"
            "    sys.exit(1)\n"
            "print('ESCALATED')\n"
        )
        self.sandbox.write_file("privilege_probe.py", probe)
        result = self.sandbox.exec(
            ["python3", "privilege_probe.py"],
            workdir=self.sandbox.workspace,
            timeout=15,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn("ESCALATED", combined)
        self.assertIn("Operation not permitted", combined)

    def test_g_provider_credentials_are_not_in_the_sandbox(self) -> None:
        environment = self.sandbox.exec(["printenv"], timeout=10)
        self.assertEqual(environment.exit_code, 0)
        self.assertNotIn("DEEPSEEK_API_KEY", environment.stdout)

        direct = self.sandbox.exec(["printenv", "DEEPSEEK_API_KEY"], timeout=10)
        self.assertNotEqual(direct.exit_code, 0)
        self.assertEqual(direct.stdout.strip(), "")

        listing = self.sandbox.exec(["ls", "-a"], timeout=10)
        self.assertNotIn(".env", listing.stdout)

    def test_h_windows_powershell_launch_uses_the_sandbox(self) -> None:
        if os.name != "nt":
            self.skipTest("PowerShell launch check only applies on Windows")
        self.assertIsNotNone(self.sandbox.version)
        result = self.sandbox.exec(
            [
                "python3",
                "-c",
                "import os; print(os.getcwd()); print(os.getuid())",
            ],
            workdir=self.sandbox.workspace,
            timeout=15,
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[0], self.sandbox.workspace)
        self.assertEqual(lines[1], "1000")


if __name__ == "__main__":
    unittest.main()
