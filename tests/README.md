# Tests

Most tests run offline. They use a scripted provider and mocked HTTP, so they
need no API key and no network. The OpenShell integration tests need a local
gateway and create a real sandbox; they skip when OpenShell is unavailable.

```powershell
python -m unittest discover -s tests -v
```

Current run: **147 tests, OK** (2 skips for Windows symlink creation). Eight of
those tests use a real OpenShell sandbox; that class creates one sandbox and
deletes it afterwards.

## What is covered

`test_deepseek.py`:

- missing API key
- message conversion and tool call IDs
- bad tool argument JSON
- usage and finish reasons
- temporary retries and permanent failures
- retry limits and `Retry-After`
- no API key in error messages

`test_loop.py`:

- normal text completion
- tool calls, order, and results
- unknown tools, exceptions, blocked calls, bad arguments
- completion validation and clarification
- turn, tool, time, and token limits
- repeated calls, empty replies, and cancellation
- provider errors and total usage
- `before_turn` replacing the transcript in place
- one tool result for every tool call

`test_tools.py`:

- tool decorator schemas and required arguments
- all six workspace tools
- line numbering, 4,000-line reads, and search clipping
- ignored directories and deterministic listing
- path escape rejection
- command execution refusal without OpenShell

`test_security.py`:

- read-only, safe, and yolo modes
- explicit approval handling
- destructive `rm -rf`, sudo, mkfs, dd, curl/pipe, git force, and `/dev/sd`
- destructive PowerShell commands

`test_shell.py`:

- configured shell and stdin command forwarding
- combined output, truncation, and timeout formatting
- invalid timeout rejection before execution

`test_openshell_integration.py`:

- all six tools inside a real sandbox
- filesystem isolation and `/etc/passwd` denial
- outbound network denial
- non-root process identity and setuid denial
- no provider credentials in the sandbox
- launch from Windows PowerShell through WSL
- a scripted agent receiving `BLOCKED` for a destructive command
- a scripted agent receiving a `PermissionError` observation for a path escape

`test_context.py`:

- token estimation and provider-usage calibration
- compaction that preserves complete tool-call groups
- original-task, recent-group, and task-state preservation
- preservation-boundary moves under tight budgets
- fixed system/tool reserve accounting
- pending, orphaned, and duplicate tool-call failures
- provider failure, empty summary, and oversized-summary handling
- `context_compacted` event fields
- task-state updates from actual tool results

`test_memory.py`:

- base prompt and memory section ordering
- add, dedupe, update, and bounded `IRINA.md`
- secret rejection and redaction
- missing, unreadable, and oversized memory
- cross-session persistence
- `remember` policy behavior

`test_skills.py`:

- catalog discovery, front matter, sorting, and malformed names
- catalog prompt and full skill loading
- missing, invalid, oversized, and empty skills
- secret redaction and symlink escape handling
- `use_skill` and read-only policy behavior

`test_stage2_security.py`:

- memory and skill operations fail without a sandbox
- prompt construction omits unavailable storage
- memory and skill text cannot change policy or sandbox rules
- traversal names are rejected

The integration class starts one sandbox and deletes it after the class. It does
not print credentials.

Stage 2 checks can be run separately:

```powershell
python -m unittest tests.test_context tests.test_memory tests.test_skills tests.test_stage2_security -v
```

See the [tools and guardrails README](../src/back/harness/tools_and_guardrails/README.md)
for the sandbox policy and the
[context, memory, and skills README](../src/back/harness/context_management/README.md)
for compaction and memory/skill behavior. The live checks are separate:
[demos/README.md](../demos/README.md). Back to the
[harness overview](../src/back/harness/README.md).
