# Tests

Most tests run offline. They use a scripted provider and mocked HTTP, so they
need no API key and no network. The OpenShell integration tests need a local
gateway and create a real sandbox; they skip when OpenShell is unavailable.

```powershell
python -m unittest discover -s tests -v
```

Current run: **188 tests, OK** (2 skips for Windows symlink creation). Nine of
those tests use real OpenShell sandboxes: the Stage 1 integration class and the
Stage 3 crash/resume test.

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

`test_session.py`:

- exclusive session creation, slugs, and workspace isolation
- JSONL flush/fsync, locks, and secret redaction
- torn-tail truncation and middle-corruption rejection
- exact tool-call ID repair and unknown-outcome observations
- confirmed tool_end restoration without duplicate results
- recovery sidecar separation
- latest/resumable session selection
- recorder lifecycle and compaction-history separation

`test_subagent.py`:

- exact depth-limit error
- fresh child task delivery
- successful and failed child reports
- total task budget
- factory/child failures and unsupported results
- `spawn_agent` schema and description

`test_harness.py`:

- tool registration, duplicate rejection, and safe default policy
- plain runs and tool runs with session records
- interrupted-session resume and unknown-outcome inspection
- completed-session refusal and workspace isolation
- sub-agent delegation, fresh conversations, and one parent session
- child policy inheritance and blocked child writes
- child turn/time/total-task limits
- compaction on resume with durable history preserved
- unavailable sandbox errors instead of host fallback

`test_stage3_integration.py`:

- a real worker process writes through OpenShell, exits before tool completion,
  and a fresh `Harness` resumes the original sandbox
- the unknown outcome is reported, the file is inspected, and the write is not
  replayed
- the session remains provider-valid and completes

`_crash_worker.py` is the helper process used by that integration test.

The integration class starts one sandbox and deletes it after the class. The
Stage 3 crash test starts and deletes its own worker sandbox. Neither prints
credentials.

Stage 2 checks can be run separately:

```powershell
python -m unittest tests.test_context tests.test_memory tests.test_skills tests.test_stage2_security -v
```

Stage 3 checks:

```powershell
python -m unittest tests.test_session tests.test_subagent tests.test_harness -v
python -m unittest tests.test_stage3_integration -v
```

See the [tools and guardrails README](../src/back/harness/tools_and_guardrails/README.md)
for the sandbox policy, the
[context, memory, and skills README](../src/back/harness/context_management/README.md)
for compaction and memory/skill behavior, and the
[Stage 3 harness section](../src/back/harness/README.md#stage-3-sessions-crash-recovery-and-sub-agents)
for sessions, repair, and sub-agents. The live checks are separate:
[demos/README.md](../demos/README.md). Back to the
[harness overview](../src/back/harness/README.md).
