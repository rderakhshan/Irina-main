# Tests

Most tests run offline. They use a scripted provider and mocked HTTP, so they
need no API key and no network. The OpenShell integration tests need a local
gateway and create a real sandbox; they skip when OpenShell is unavailable.

```powershell
python -m unittest discover -s tests -v
```

Current full run: **295 tests, OK** (2 skips for Windows symlink creation).
OpenShell integration tests run when the local gateway is available and skip
when it is not. Stage 4/5/6 live checks are separate; see the demo and memory
READMEs.

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
- `continue_task` appending to the same completed session
- sub-agent delegation, fresh conversations, and one parent session
- child policy inheritance and blocked child writes
- child turn/time/total-task limits
- compaction on resume with durable history preserved
- unavailable sandbox errors instead of host fallback

`test_cli.py`:

- parser defaults, help, invalid modes, and resume forms
- interactive safe approvals and headless safe refusal behavior
- completed/incomplete/interrupted/failed exit codes
- Ctrl-C session preservation without sandbox cleanup
- interactive prompt loop and `continue_task` follow-ups
- event/output redaction
- startup errors and workspace preparation

`test_fleet.py`:

- empty fleets and ordered results
- parallel execution with a barrier
- review in the same Harness/session
- failure isolation and per-job cleanup
- verification callbacks that fail a job
- duplicate names and duplicate/overlapping workspace rejection
- invalid worker counts and job shapes

`test_sandbox_download.py`:

- downloading from inside the approved workspace
- parent-directory creation
- workspace escape rejection
- started-sandbox and CLI-failure handling

`test_memory_integration.py`:

- canonical store idempotency, torn-tail recovery, isolation, redaction, corruption
- evidence-based trajectory conversion and stable experience IDs
- bounded lower-trust retrieval context
- all 13 upstream provider status probes and actual Lightweight/ExpeL integration
- MemoryManager canonical-before-provider update lifecycle
- Harness memory retrieval/update hooks
- architecture activation/rollback and canonical replay without source loss
- candidate static/isolated checks, coding evaluation fixtures, and honest evolution status

`test_memory_finalization.py`:

- provider storage is workspace-scoped (no cross-project retrieval)
- registry mutations are visible across independent instances and survive restart
- an approved activation is preserved when a manager restarts
- evolution fixture: generate → create → validate → register, never activate
- a generated provider can be activated, store/retrieve, and roll back
- evaluation rejects candidates that fail the provider contract

`test_memory_logging.py`:

- routine memory logs are hidden by default; errors remain visible
- `--verbose` applies INFO/DEBUG/SUCCESS/WARNING/ERROR colors
- `--no-color` produces plain output
- provider handlers are replaced and reconfiguration never duplicates lines
- provider `print` output is captured and filtered per memory operation
- renderer memory notices follow `--verbose`; failed updates stay visible
- the CLI parser exposes `--verbose`

`test_streaming.py`:

- DeepSeek SSE text deltas, finish reasons, and usage
- multiple/partial tool-call arguments with exact IDs
- complete-vs-stream semantic equivalence
- truncated streams and connection failures
- loop delta events, no partial tool execution, and cancellation
- streaming deltas are not persisted as session messages
- interrupted stream session status
- renderer duplicate suppression and tool success/failure/blocked/cancelled
- Markdown headings, lists, fences, inline code, partial, and malformed input
- plain/ASCII output and cross-chunk secret/control-sequence protection
- thread-safe fleet rendering

`test_stage4_projects.py`:

- artisan-coffee section/word/SVG/interaction checks
- viper canvas/rAF/localStorage/game-state checks
- taskman files, test count, test result, and help result
- project configuration and default CLI demo settings

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

Stage 4 checks:

```powershell
python -m unittest tests.test_cli tests.test_fleet tests.test_sandbox_download tests.test_stage4_projects -v
```

Stage 5/6 checks:

```powershell
python -m unittest tests.test_streaming -v
python -m unittest tests.test_memory_integration -v
python -m unittest tests.test_loop tests.test_harness tests.test_cli -v
```

Use `uv run` for the optional ExpeL/evolution dependency set:

```powershell
uv run python -m unittest tests.test_memory_integration -v
```

The Stage 4 product proof is a live run, not an offline test:

```powershell
python demos/stage4_projects.py --only taskman --workers 1
```

`taskman` completed; the two web projects passed static checks but ended at
`token_budget` before a completed review. Browser interaction is not automated.
See [demos/README.md](../demos/README.md) and
[stage4_results.json](../stage4_results.json) for the exact results.

See the [tools and guardrails README](../src/back/harness/tools_and_guardrails/README.md)
for the sandbox policy, the
[context, memory, and skills README](../src/back/harness/context_management/README.md)
for compaction and memory/skill behavior, and the
[Stage 3 harness section](../src/back/harness/README.md#stage-3-sessions-crash-recovery-and-sub-agents)
for sessions, repair, and sub-agents. The live checks are separate:
[demos/README.md](../demos/README.md). Back to the
[harness overview](../src/back/harness/README.md).
