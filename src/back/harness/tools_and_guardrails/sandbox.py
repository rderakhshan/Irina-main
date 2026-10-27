"""Day 2: NVIDIA OpenShell sandbox adapter.

Concept: this module is the only place that talks to OpenShell. It starts or
attaches to a sandbox, uploads the workspace worker, runs commands and file
operations inside the sandbox, and removes the sandbox when the task ends.

Design constraints:
* Only documented OpenShell 0.1.5 CLI commands are used.
* Commands and worker requests are passed as argument lists and bytes over
  stdin. No host shell is used.
* Windows launches the WSL-installed ``openshell`` CLI through ``wsl.exe``.
* The host model key is never passed into the sandbox. No provider is attached.
* File paths are resolved by the sandbox worker, not by the host.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_POLICY_PATH = Path(__file__).with_name("openshell_policy.yaml")
DEFAULT_WORKER_PATH = Path(__file__).with_name("sandbox_worker.py")
DEFAULT_IMAGE = "python:3.12-slim"
WORKER_SANDBOX_PATH = "/tmp/irina_worker.py"
DEFAULT_CREATE_TIMEOUT = 300
DEFAULT_FILE_TIMEOUT = 60


class SandboxError(RuntimeError):
    """A sandbox operation failed in a way the caller should report."""


class SandboxUnavailable(SandboxError):
    """OpenShell or its gateway is not available on this machine."""


@dataclass
class SandboxResult:
    """Result of one command executed inside the sandbox.

    Attributes:
        exit_code: Remote command exit code, or 124 when the command timed out.
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: Whether the command hit its time limit.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _decode(data: bytes) -> str:
    """Decode CLI output as UTF-8 without failing on unusual bytes."""
    return data.decode("utf-8", "replace")


def parse_worker_response(stdout: str) -> dict[str, Any]:
    """Parse one worker response and raise a useful error on failure.

    Args:
        stdout: Raw worker standard output.

    Returns:
        The successful response object.

    Raises:
        PermissionError, FileNotFoundError, IsADirectoryError,
        NotADirectoryError, ValueError: Reconstructed from the worker error.
        SandboxError: If the response is missing or malformed.
    """
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        raise SandboxError(f"invalid worker response: {exc}") from None
    if not isinstance(payload, dict):
        raise SandboxError("worker response must be a JSON object")
    if payload.get("ok"):
        return payload

    error = payload.get("error")
    if not isinstance(error, dict):
        raise SandboxError("worker returned an error without details")
    error_type = str(error.get("type", "SandboxError"))
    message = str(error.get("message", ""))
    if error_type == "PermissionError":
        raise PermissionError(message)
    if error_type == "FileNotFoundError":
        raise FileNotFoundError(message)
    if error_type == "IsADirectoryError":
        raise IsADirectoryError(message)
    if error_type == "NotADirectoryError":
        raise NotADirectoryError(message)
    if error_type == "ValueError":
        raise ValueError(message)
    raise SandboxError(f"{error_type}: {message}")


class OpenShellSandbox:
    """Lifecycle and execution adapter for one OpenShell sandbox.

    Args:
        name: Sandbox name. A unique name is generated when omitted.
        policy_path: OpenShell policy YAML. Defaults to the Day 2 policy file.
        image: OCI image used for the sandbox. It must contain Python 3.
        workspace: Workspace under the sandbox working directory. A relative
            name is created inside the sandbox working directory.
        timeout: Host-side timeout for sandbox creation in seconds.
        wsl_distro: Optional WSL distribution name on Windows.
        gateway: Optional OpenShell gateway name.
        worker_path: Host path to the sandbox worker script.
        keep: Stop instead of delete when :meth:`stop` is called.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        policy_path: str | Path | None = None,
        image: str = DEFAULT_IMAGE,
        workspace: str | None = None,
        timeout: int = DEFAULT_CREATE_TIMEOUT,
        wsl_distro: str | None = None,
        gateway: str | None = None,
        worker_path: str | Path | None = None,
        keep: bool = False,
    ) -> None:
        self.name = name or f"irina-{uuid.uuid4().hex[:8]}"
        self.policy_path = Path(
            policy_path if policy_path is not None else DEFAULT_POLICY_PATH
        ).resolve()
        self.worker_path = Path(
            worker_path if worker_path is not None else DEFAULT_WORKER_PATH
        ).resolve()
        self.image = image
        self.workspace_name = workspace
        self.timeout = timeout
        self.wsl_distro = wsl_distro
        self.gateway = gateway
        self.keep = keep
        self.workspace: str | None = None
        self.version: str | None = None
        self.sandbox_info: dict[str, Any] | None = None
        self._started = False

    @property
    def started(self) -> bool:
        """Return whether the adapter currently holds a running sandbox."""
        return self._started

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the CLI and gateway are usable on this machine."""
        try:
            cls().check_available()
        except SandboxUnavailable:
            return False
        return True

    def check_available(self) -> str:
        """Verify the CLI and gateway, returning the CLI version string.

        Raises:
            SandboxUnavailable: If OpenShell, the gateway, or WSL is missing.
        """
        try:
            version = self._cli(["--version"], timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(
                "OpenShell CLI is not available. Install OpenShell 0.1.5 in "
                "WSL 2 and ensure `wsl.exe -e openshell --version` works. "
                f"Details: {exc}"
            ) from None
        if version.returncode != 0:
            raise SandboxUnavailable(
                "OpenShell CLI failed: "
                f"{_decode(version.stderr).strip() or version.returncode}"
            )
        version_line = _decode(version.stdout).strip().splitlines()
        self.version = version_line[0] if version_line else "openshell unknown"

        try:
            status = self._cli(["status"], timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(
                f"OpenShell gateway status failed: {exc}"
            ) from None
        status_text = _decode(status.stdout)
        if status.returncode != 0 or "Connected" not in status_text:
            detail = _decode(status.stderr).strip() or status_text.strip()
            raise SandboxUnavailable(
                "OpenShell gateway is not connected. Start it with the "
                f"OpenShell CLI. Details: {detail[:500]}"
            )
        return self.version

    def start(self) -> "OpenShellSandbox":
        """Start a new sandbox or attach to the named one when it is ready.

        Returns:
            This adapter, ready for file and command operations.

        Raises:
            SandboxUnavailable: If OpenShell is not configured.
            SandboxError: If the sandbox cannot be started or prepared.
        """
        if self._started:
            return self
        self.check_available()
        if not self.policy_path.is_file():
            raise SandboxError(
                f"OpenShell policy file not found: {self.policy_path}"
            )
        if not self.worker_path.is_file():
            raise SandboxError(
                f"sandbox worker not found: {self.worker_path}"
            )

        existing = self._get_sandbox()
        created = existing is None
        if existing is not None:
            phase = str(existing.get("phase", ""))
            if phase != "Ready":
                started = self._cli(
                    ["sandbox", "start", self.name], timeout=self.timeout
                )
                if started.returncode != 0:
                    raise SandboxError(
                        f"sandbox {self.name!r} is {phase or 'unknown'} and "
                        "could not be started: "
                        f"{_decode(started.stderr).strip()}"
                    )
                existing = self._wait_ready()
            self.sandbox_info = existing
        else:
            self.sandbox_info = self._create()

        try:
            self._setup_workspace()
        except Exception:
            if created:
                try:
                    self._cli(
                        ["sandbox", "delete", self.name], timeout=120
                    )
                except Exception:
                    pass
            raise
        self._started = True
        return self

    @classmethod
    def attach(cls, name: str, **kwargs: Any) -> "OpenShellSandbox":
        """Attach to an existing ready sandbox by name.

        Args:
            name: Existing sandbox name.
            **kwargs: Same options as the constructor.

        Returns:
            A ready adapter.

        Raises:
            SandboxError: If the sandbox is missing or not ready.
        """
        sandbox = cls(name=name, **kwargs)
        sandbox.check_available()
        info = sandbox._get_sandbox()
        if info is None:
            raise SandboxError(f"sandbox {name!r} was not found")
        if str(info.get("phase", "")) != "Ready":
            raise SandboxError(
                f"sandbox {name!r} is not ready: {info.get('phase')}"
            )
        sandbox.sandbox_info = info
        sandbox._setup_workspace()
        sandbox._started = True
        return sandbox

    def exec(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: int = 120,
        input_text: str | None = None,
    ) -> SandboxResult:
        """Execute one command inside the sandbox.

        Args:
            argv: Command and arguments. No host shell is involved.
            workdir: Working directory inside the sandbox. Defaults to the
                approved workspace.
            timeout: Remote timeout in seconds. ``0`` disables it.
            input_text: Optional text written to the command's stdin.

        Returns:
            Captured output and exit status.
        """
        if not self._started:
            raise SandboxError("sandbox is not started")
        if isinstance(argv, str):
            raise TypeError("argv must be a list of strings")
        command = [str(part) for part in argv]
        directory = workdir if workdir is not None else self.workspace
        return self._exec_cli(
            command,
            workdir=directory,
            timeout=timeout,
            input_text=input_text,
        )

    def read_file(self, path: str, *, max_lines: int = 4000) -> dict[str, Any]:
        """Read a workspace file through the sandbox worker."""
        return self._file_request(
            "read_file", path=path, max_lines=max_lines
        )

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write a workspace file through the sandbox worker."""
        return self._file_request(
            "write_file", path=path, content=content
        )

    def edit_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Edit a workspace file through the sandbox worker."""
        return self._file_request(
            "edit_file", path=path, old=old, new=new
        )

    def list_files(
        self, pattern: str = "**/*", *, limit: int = 500
    ) -> dict[str, Any]:
        """List workspace files through the sandbox worker."""
        return self._file_request(
            "list_files", pattern=pattern, limit=limit
        )

    def grep(
        self, regex: str, pattern: str = "*", *, limit: int = 200
    ) -> dict[str, Any]:
        """Search workspace files through the sandbox worker."""
        return self._file_request(
            "grep", regex=regex, pattern=pattern, limit=limit
        )

    def stop(self) -> None:
        """Delete the sandbox, or stop it when ``keep`` is set.

        Raises:
            SandboxError: If OpenShell reports a cleanup failure.
        """
        if not self._started:
            return
        action = "stop" if self.keep else "delete"
        result = self._cli(
            ["sandbox", action, self.name], timeout=120
        )
        self._started = False
        self.workspace = None
        if result.returncode != 0:
            raise SandboxError(
                f"failed to {action} sandbox {self.name!r}: "
                f"{_decode(result.stderr).strip()}"
            )

    close = stop

    def __enter__(self) -> "OpenShellSandbox":
        return self.start()

    def __exit__(self, *exc_info: object) -> bool:
        self.stop()
        return False

    def _base_command(self) -> list[str]:
        """Return the host command prefix for the OpenShell CLI."""
        if os.name == "nt":
            command = ["wsl.exe"]
            if self.wsl_distro:
                command.extend(["-d", self.wsl_distro])
            command.append("-e")
        else:
            command = []
        command.append("openshell")
        if self.gateway:
            command.extend(["--gateway", self.gateway])
        command.extend(["--color", "never"])
        return command

    def _wsl_path(self, path: Path) -> str:
        """Return a WSL path for a local Windows file."""
        raw = str(path)
        if os.name != "nt" or raw.startswith("/"):
            return raw
        drive = path.drive
        if not drive or not drive.endswith(":"):
            raise SandboxUnavailable(
                "OpenShell on Windows requires the policy, worker, and files "
                "on a local drive visible to WSL. "
                f"Unsupported path: {raw}"
            )
        remainder = raw[len(drive) :].replace("\\", "/")
        return f"/mnt/{drive[0].lower()}{remainder}"

    def _cli(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: float = 120,
    ) -> subprocess.CompletedProcess:
        """Run one OpenShell CLI command without a host shell."""
        command = self._base_command() + args
        options: dict[str, Any] = {
            "capture_output": True,
            "timeout": timeout,
        }
        if input_text is None:
            options["stdin"] = subprocess.DEVNULL
        else:
            options["input"] = input_text.encode("utf-8")
        return subprocess.run(command, **options)

    def _get_sandbox(self) -> dict[str, Any] | None:
        """Return sandbox metadata, or ``None`` when it does not exist."""
        try:
            result = self._cli(
                ["sandbox", "get", self.name, "--output", "json"],
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(_decode(result.stdout).strip())
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _wait_ready(self, timeout: int = 120) -> dict[str, Any]:
        """Poll until the sandbox is ready or the wait budget expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self._get_sandbox()
            if info is not None:
                phase = str(info.get("phase", ""))
                if phase == "Ready":
                    return info
                if phase in {"Error", "Failed"}:
                    raise SandboxError(
                        f"sandbox {self.name!r} entered phase {phase}"
                    )
            time.sleep(1)
        raise SandboxError(
            f"timed out waiting for sandbox {self.name!r} to become ready"
        )

    def _create(self) -> dict[str, Any]:
        """Create a detached sandbox with the Day 2 policy."""
        policy = self._wsl_path(self.policy_path)
        keepalive = "while :; do sleep 3600; done"
        args = [
            "sandbox",
            "create",
            "--name",
            self.name,
            "--from",
            self.image,
            "--policy",
            policy,
            "--detach",
            "--no-auto-providers",
            "--label",
            "irina=day2",
            "--output",
            "json",
            "--",
            "/bin/sh",
            "-c",
            keepalive,
        ]
        try:
            result = self._cli(args, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(
                f"timed out creating sandbox {self.name!r}: {exc}"
            ) from None
        if result.returncode != 0:
            raise SandboxError(
                f"failed to create sandbox {self.name!r}: "
                f"{_decode(result.stderr).strip()}"
            )
        text = _decode(result.stdout).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SandboxError(
                f"unexpected create output: {text[:500]!r} ({exc})"
            ) from None
        if not isinstance(payload, dict):
            raise SandboxError("unexpected create output: expected an object")
        if str(payload.get("phase", "")) != "Ready":
            return self._wait_ready()
        return payload

    def _setup_workspace(self) -> None:
        """Create the approved workspace and upload the worker script."""
        current = self._exec_cli(
            ["/bin/pwd"], workdir=None, timeout=60
        )
        if current.exit_code != 0:
            raise SandboxError(
                "could not read the sandbox working directory: "
                f"{current.stderr.strip()}"
            )
        canonical = current.stdout.strip().splitlines()
        if not canonical or not canonical[-1].startswith("/"):
            raise SandboxError(
                f"unexpected sandbox working directory: {current.stdout!r}"
            )
        root = canonical[-1]

        requested = self.workspace_name or root
        workspace = (
            requested
            if requested.startswith("/")
            else posixpath.join(root, requested)
        )
        workspace = posixpath.normpath(workspace)
        if workspace != root and not workspace.startswith(root.rstrip("/") + "/"):
            raise SandboxError(
                "workspace must be inside the sandbox working directory: "
                f"{workspace!r} is outside {root!r}"
            )

        made = self._exec_cli(
            ["/bin/mkdir", "-p", workspace], workdir=None, timeout=60
        )
        if made.exit_code != 0:
            raise SandboxError(
                f"could not create sandbox workspace {workspace!r}: "
                f"{made.stderr.strip()}"
            )

        uploaded = self._cli(
            [
                "sandbox",
                "upload",
                self.name,
                self._wsl_path(self.worker_path),
                WORKER_SANDBOX_PATH,
                "--no-git-ignore",
            ],
            timeout=120,
        )
        if uploaded.returncode != 0:
            raise SandboxError(
                "could not upload the sandbox worker: "
                f"{_decode(uploaded.stderr).strip()}"
            )
        self.workspace = workspace

    def _exec_cli(
        self,
        argv: list[str],
        *,
        workdir: str | None,
        timeout: int,
        input_text: str | None = None,
    ) -> SandboxResult:
        """Build and run one ``openshell sandbox exec`` request."""
        args = [
            "sandbox",
            "exec",
            "-n",
            self.name,
            "--no-tty",
            "--no-login-shell",
        ]
        if workdir is not None:
            args.extend(["--workdir", workdir])
        args.extend(["--timeout", str(int(timeout))])
        args.extend(["--", *argv])
        host_timeout = timeout + 15 if timeout > 0 else 3600
        try:
            result = self._cli(
                args, input_text=input_text, timeout=host_timeout
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(124, "", "", True)
        return SandboxResult(
            exit_code=result.returncode,
            stdout=_decode(result.stdout),
            stderr=_decode(result.stderr),
            timed_out=result.returncode == 124,
        )

    def _file_request(self, op: str, **values: Any) -> dict[str, Any]:
        """Send one operation to the uploaded sandbox worker."""
        if not self.workspace:
            raise SandboxError("sandbox workspace is not initialized")
        request = {"op": op, "workdir": self.workspace, **values}
        result = self.exec(
            ["python3", WORKER_SANDBOX_PATH],
            workdir=self.workspace,
            timeout=DEFAULT_FILE_TIMEOUT,
            input_text=json.dumps(request, ensure_ascii=False),
        )
        if result.timed_out:
            raise SandboxError("sandbox file operation timed out")
        if result.exit_code != 0 and not result.stdout.strip():
            raise SandboxError(
                f"sandbox worker failed (exit {result.exit_code}): "
                f"{result.stderr.strip()}"
            )
        return parse_worker_response(result.stdout)
