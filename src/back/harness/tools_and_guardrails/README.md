# Tools and guardrails

This folder gives Irina six workspace tools, an application permission policy, a
shell interface, and the OpenShell sandbox adapter. The policy decides whether a
request is allowed. OpenShell decides what the sandbox can actually reach.

## Files

| File | Purpose |
| --- | --- |
| `tools.py` | `Tool`, `tool`, the six tools, and the local file-only test backend. |
| `security.py` | `Policy`, modes, approval callback, and dangerous-command rules. |
| `shell.py` | Explicit shell configuration and the command result shape. |
| `sandbox.py` | OpenShell lifecycle, command execution, and worker calls. |
| `sandbox_worker.py` | Sandbox-side file operations and the shared path resolver. |
| `openshell_policy.yaml` | Filesystem, network, and process policy used at creation. |

## The six tools

| Tool | What it does |
| --- | --- |
| `read_file` | Reads a text file, numbers lines, returns at most 4,000 lines. |
| `write_file` | Writes text, creates parent directories, reports characters written. |
| `edit_file` | Replaces a snippet only when it appears exactly once. |
| `bash` | Runs a command in the sandbox workspace with Bash. |
| `list_files` | Lists files with glob matching and ignored build directories. |
| `grep` | Searches file contents with a regex and returns `path:line: text`. |

All six take string arguments, which is what the existing provider expects.

## Where paths are checked

`sandbox_worker.py` is uploaded into the sandbox. It resolves the workspace and
the requested path, follows symlinks, and rejects anything outside the workspace
with `PermissionError("... escapes the working directory")`. File operations run
inside the sandbox, so the check happens in the same filesystem as the write.

`core_tools(workdir)` without a backend uses the same worker locally for unit
tests. That mode is file-only and cannot run commands; it is not a sandbox. The
demo and integration tests pass an `OpenShellSandbox`, which makes all six tools
use the sandbox.

## Permissions

`Policy` modes:

- `read-only`: allows `read_file`, `list_files`, and `grep`.
- `safe`: also allows writes and commands after `approver(call, reason)` returns
  exactly `True`. Without an approver, those requests are refused.
- `yolo`: allows writes and commands without asking, but still checks dangerous
  commands first.

Dangerous-command rules cover destructive `rm -rf` against root, home, and
`/home/<user>`; `sudo`; `mkfs`; `dd if=`; `curl`/`wget` piped into a shell;
forced `git push`; writes to `/dev/sd*`; and destructive PowerShell/Windows
commands such as `Remove-Item -Recurse -Force`. These are application
safeguards, not a shell parser.

## Shell

The sandbox shell is explicit: `/bin/bash -s`. The command is sent on stdin,
which avoids quoting problems between PowerShell, WSL, and OpenShell. The model
is told to write Bash, not PowerShell. The Windows host only launches
`wsl.exe -e openshell`; it never runs the command itself.

## OpenShell

The adapter uses the documented OpenShell 0.1.5 CLI:

1. check the CLI and gateway;
2. create or attach to a sandbox;
3. upload the worker;
4. create the workspace;
5. execute commands and file operations;
6. delete the sandbox.

On Windows it calls `wsl.exe -e openshell ...`. The default image is
`python:3.12-slim` because the stock OpenShell Ubuntu image does not include
Python. Docker must be able to see that image.

`openshell_policy.yaml` sets:

- **Filesystem:** the workspace is read-write; system libraries are read-only;
  `/var`, `/home`, and `/etc/passwd` are unavailable; `/tmp` is writable.
  TeX configuration and font metadata (`/etc/texmf`, `/etc/fonts`,
  `/var/lib/texmf`) are read-only when a document image is used.
- **Network:** no rules, so all outbound connections are denied.
- **Process:** UID/GID 1000, non-root, with Landlock set to
  `hard_requirement`.
- **Credentials:** no provider is attached and `DEEPSEEK_API_KEY` is never passed
  into the sandbox.

Approved images can add language or document tools without changing the policy.
Use `--sandbox-image` and keep the image pinned and reviewed; the image is part
of the trusted supply chain. See
[images/application](../../../../images/application/README.md) for TeX, Poppler,
and pypdf.

The integration tests run commands inside a real sandbox to check these layers.
They skip if OpenShell is unavailable.

## Run it

```powershell
python -m unittest tests.test_tools tests.test_security tests.test_shell -v
python -m unittest tests.test_openshell_integration -v
python demos/day2_openshell.py
```

The Day 2 demo needs `DEEPSEEK_API_KEY`, OpenShell 0.1.5 in WSL 2, Docker, and
the `python:3.12-slim` image. It prints model and tool events, verifies the
Fibonacci program independently, exercises the destructive-command and
path-escape guardrails, and deletes the sandbox.

## Verified in this environment

OpenShell **0.1.5** on WSL 2 Ubuntu, with Docker Desktop and the
`python:3.12-slim` image. Windows launches the CLI with
`wsl.exe -e openshell ...`.

| Check | Observed result |
| --- | --- |
| Filesystem | `/var`, `/.openshell`, `/dev/shm` writes, and `/etc/passwd` were denied; `/etc/hosts` was readable. |
| Network | `https://example.com` failed with `PermissionError` under the deny-all policy. |
| Process | `id -u` returned `1000`; `os.setuid(0)` failed with `Operation not permitted`. |
| Credentials | `DEEPSEEK_API_KEY` was absent from the sandbox environment, and `.env` was not present. |
| Six tools | read, write, edit, list, grep, and bash all worked inside the sandbox. |
| Blocked command | a scripted agent received `BLOCKED` for `rm -rf ~`; a spy confirmed `bash` was not executed. |
| Path escape | a scripted agent received a `PermissionError` observation for `../../etc/passwd`; no `root:` content leaked. |

Current checks:

- `python -m unittest discover -s tests` → **100 tests, OK**, with 1 skip for
  Windows symlink creation.
- `tests.test_openshell_integration` → **8 tests, OK**.
- `demos/day2_openshell.py` → **A, B, and C passed**, and the sandbox was
  deleted. The live model refused the destructive and escape requests without
  calling a tool; the direct policy/tool checks and the scripted-agent checks
  cover the blocked observations.

## Milestones

| Commit | Scope |
| --- | --- |
| `3ef6f56` | Six tools, policy, shell, worker, sandbox adapter, policy YAML, unit tests. |
| `33149b4` | Real OpenShell filesystem, network, process, credential, and PowerShell tests. |
| `3eb3c0b` | Dangerous-command matching also catches commands inside shell wrappers. |
| `6ec49e3` | OpenShell demo and README updates. |
| `a490115` | Scripted-agent checks for `BLOCKED` and path-error observations. |
| `2010156` | Sandbox cleanup when workspace setup fails. |

## Limits

No memory, multi-agent work, saved sessions, or context compaction. The
application policy is intentionally heuristic. OpenShell is the isolation
boundary, and the demo does not claim protections that the integration tests did
not exercise.

See the [demos README](../../../../demos/README.md),
[tests README](../../../../tests/README.md), and
[project README](../../../../README.md).
