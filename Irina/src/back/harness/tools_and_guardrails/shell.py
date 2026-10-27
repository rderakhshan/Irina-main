"""Day 2: shell selection and command result formatting input.

Concept: the shell tool accepts a command string, selects the shell configured
for the sandbox, and forwards the request to the workspace backend. The real
execution happens in ``sandbox.py``.

Design constraints:
* No host-side command execution.
* The shell is explicit, so the model can be told whether to write Bash or
  another shell's syntax.
* The command is written to the shell's stdin, which avoids fragile quoting
  across Windows, WSL, and OpenShell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .sandbox import SandboxResult


@dataclass(frozen=True)
class ShellConfig:
    """Description of the shell used inside the sandbox.

    Attributes:
        path: Absolute path to the shell executable.
        args: Arguments that make the shell read its script from stdin.
        name: Human-readable shell name shown to the model.
    """

    path: str = "/bin/bash"
    args: tuple[str, ...] = ("-s",)
    name: str = "bash"


DEFAULT_SHELL = ShellConfig()


@dataclass
class ShellResult:
    """Consistent result returned by the shell interface.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Remote command exit code.
        timed_out: Whether the command hit its time limit.
        shell: Shell name used for the command.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    shell: str


class CommandBackend(Protocol):
    """Minimal execution interface required by :func:`run_command`."""

    def exec(
        self,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: int = 120,
        input_text: str | None = None,
    ) -> SandboxResult:
        """Execute a command and return its captured result."""
        ...


def run_command(
    backend: CommandBackend,
    command: str,
    workdir: str,
    timeout: int,
    shell: ShellConfig = DEFAULT_SHELL,
) -> ShellResult:
    """Run ``command`` in the configured shell through the backend.

    Args:
        backend: Sandbox or test backend implementing ``exec``.
        command: Shell script text.
        workdir: Working directory inside the sandbox.
        timeout: Timeout in seconds.
        shell: Shell configuration.

    Returns:
        A normalised result with output, errors, exit status, and timeout flag.
    """
    argv = [shell.path, *shell.args]
    result = backend.exec(
        argv,
        workdir=workdir,
        timeout=timeout,
        input_text=command,
    )
    return ShellResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        shell=shell.name,
    )
