# Harness

The harness runs one conversation from start to finish. It asks the model for a
reply, runs any tools the model asks for, and stops for a clear reason.

The loop is deliberately small. Day 1 added one model adapter and a dice demo;
Day 2 added six workspace tools and an OpenShell sandbox. Stage 2 added context
compaction, durable memory, and skills. Stage 3 added durable sessions, truthful
crash repair, sequential sub-agents, and the central `Harness`. Stage 4 adds the
interactive/headless CLI, parallel fleet execution, and artifact download.

## Files

| File | Purpose |
| --- | --- |
| `orchestration/harness.py` | Central composition of provider, loop, tools, policy, context, sessions, and OpenShell. |
| `orchestration/session.py` | Append-only JSONL sessions, recorder, load/repair, latest-session lookup. |
| `orchestration/subagent.py` | `spawn_agent`, delegation budget, sequential child execution. |
| `orchestration/cli.py` | Interactive/headless front door, approvals, exit codes, resume. |
| `orchestration/fleet.py` | Parallel isolated jobs, validation, ordering, optional review/repair. |
| `__main__.py` | `python -m src.back.harness` entry point. |
| `utils/theme.py` | Semantic terminal colors, symbols, and capability detection. |
| `utils/formatting.py` | Redaction, safe display, previews, durations, and token formatting. |
| `utils/markdown.py` | Incremental Markdown presentation. |
| `utils/renderer.py` | Streaming event renderer and progress indicator. |
| `utils/fleet.py` | Serialized, job-prefixed concurrent fleet output. |
| `context_management/memory/` | Canonical experience history, EvolveLab adapter, registry, replay, evolution/evaluation bridges. |
| `memory/upstream/memevolve/` | Vendored original MemEvolve/EvolveLab source at the recorded commit. |
| `loop/runner.py` | Runs turns, handles tool calls, emits events. |
| `loop/state.py` | Defines run statuses, the result object, and counters. |
| `loop/termination.py` | Sets limits for turns, tools, time, tokens, repeats, and failures. |
| `providers/base.py` | Defines what a provider must implement and the shared data types. |
| `providers/deepseek.py` | Talks to the DeepSeek Chat Completions API. |
| `tools_and_guardrails/` | Six workspace tools, permission policy, shell, and OpenShell adapter. |
| `context_management/` | Compaction, task state, `IRINA.md` memory, and skills. |

## How a run works

1. Check whether the caller cancelled and whether time or a budget is used up.
2. Give an optional `before_turn` hook a chance to change the transcript.
   Stage 2 uses this for context compaction.
3. Ask the provider for a reply.
4. Add the reply and its token use to the run.
5. If the reply asks for tools, run them one at a time.
6. If the reply looks final, ask the completion policy whether to accept it.
7. Continue until the run completes or stops.

A reply with no tool calls is only a candidate answer. For work that needs proof
(a dice roll, a changed file, a passing test), pass a `completion_validator`. It
can reject the answer and ask the model to keep working. If no validator is set,
a non-empty text answer is accepted.

## Results

`run_loop` returns a `RunResult`. `status` tells you what happened. Only
`completed` means success.

| Status | Meaning |
| --- | --- |
| `completed` | The answer was accepted. |
| `awaiting_input` | The task needs information from the user. |
| `max_turns` | The model used all its turns. |
| `max_tool_calls` | The tool-call limit was reached. |
| `timeout` | The time limit was reached. |
| `token_budget` | The token limit was reached. |
| `stalled` | The model repeated calls or sent empty replies. |
| `provider_error` | The model API failed. |
| `invalid_response` | The reply could not be used. |
| `tool_error` | Too many tool calls failed in a row. |
| `cancelled` | The caller stopped the run. |
| `blocked` | A policy or the provider blocked the work. |

The transcript and the last assistant text are kept for every result.

## Tools

A tool is any object with a `spec` and a `run` method:

```python
tool.spec = {"schema": {"name": "roll_dice", "description": "...", "parameters": {...}}}
tool.run(count=3) -> "rolls=[4, 2, 6], total=12"
```

`before_tool(call)` is the permission check. Return `None` to allow the call, or
a reason string to block it. A blocked call still gets a tool result.

Unknown tools, bad arguments, and tool exceptions become error results. They do
not crash the run. Every tool call gets exactly one matching tool result before
the next model request.

Day 2 uses `tools_and_guardrails/` for its six tools. The caller passes
`Policy.check` as the `before_tool` hook, and the demo runs the tools inside
OpenShell. The loop itself is unchanged.

## Stage 3: sessions, crash recovery, and sub-agents

`Harness` is the composition layer. It resolves the host workspace, starts or
reattaches OpenShell lazily, registers core tools plus `remember`, `use_skill`,
`spawn_agent`, and caller extras, builds the system prompt, configures context
compaction and task state, and calls the existing `run_loop`. It defaults to
`Policy("safe")`, not `yolo`.

### Sessions

- Session files live under `$IRINA_STATE_DIR/sessions` or `~/.irina/sessions`,
  keyed by a hash of the canonical host workspace. They are outside the
  agent-writable sandbox workspace.
- `new_session` creates an exclusive file with a timestamp, bounded slug, and
  collision-resistant suffix.
- `append` writes one JSON object per line, flushes, calls `fsync`, and uses a
  lock file to prevent concurrent writers. Likely credentials are redacted.
- Records cover `session_start`, messages, `tool_start`, `tool_end`,
  `context_compacted`, task-state checkpoints, and `run_end`.
- `load` preserves every complete record. It ignores and truncates a torn final
  line, but raises on corruption in the middle. It matches tool calls and results
  by exact ID.
- A missing result becomes an explicit unknown-outcome observation with the call
  ID and tool name. Confirmed `tool_end` records are restored without duplicating
  messages. No uncertain operation is replayed automatically.
- Recovery and resume events go to a separate `.recovery.jsonl` sidecar, so the
  original execution history remains append-only.
- `latest` selects the newest session for one workspace, ignores child sessions,
  and can restrict selection to resumable sessions.
- Durable history and working context are separate. Compaction events are
  recorded, but the original messages are not rewritten. Resume loads the
  durable history, restores task state, and lets `before_turn` compact again.
- Resume reattaches the session's OpenShell sandbox by name and verifies that its
  workspace still exists. It does not create a replacement sandbox and pretend
  old files survived.

### Sub-agents

- `spawn_agent` is registered through the existing `Tool` abstraction.
- Each child gets a fresh conversation, task state, and context budget.
- Children share the parent policy, backend, sandbox, workspace, and project
  memory/skills. They do not create top-level sessions.
- Execution is sequential. Depth, child turns, child time, and total delegated
  tasks are bounded. The parent remains responsible for integration and tests.
- Child reports come from structured `RunResult` values and return as normal tool
  observations.

### Harness API

`Harness(workdir, model, policy, ..., provider, workspace_backend, state_dir)`
exposes `run(task)`, `continue_task(task)`, `resume(path=None)`, `close()`,
`tools`, `workspace`, `sandbox`, `system_prompt`, `session_path`, and
`session_status`. The deterministic Stage 3 demo uses a scripted provider; the
real OpenShell crash integration test uses a forced process exit and a fresh
Harness to resume.

## Stage 4: CLI, fleet, and artifacts

### CLI

`orchestration/cli.py` is a thin front door over `Harness`:

```powershell
python main.py                         # interactive
python main.py -p "Fix the failing test" --mode safe
python main.py -p "Build the release" --mode yolo --workdir .
python main.py --resume
python main.py --resume <session.jsonl>
python -m src.back.harness --help
```

- `safe` is the default. Interactive `safe` asks for explicit approval; headless
  `safe` refuses approval-required operations instead of auto-approving them.
- `--mode yolo` skips application approvals; OpenShell and dangerous-command
  checks remain.
- Headless exit `0` means `RunStatus.COMPLETED`. Any other status returns
  nonzero, and the session is preserved.
- Ctrl-C during a run returns `130`, preserves the session, and leaves the owned
  sandbox alive for `--resume`. A normal exit closes/deletes the owned sandbox.
- Assistant text, tool arguments, results, status reasons, and session labels are
  redacted before display or persistence.

### Fleet

`orchestration/fleet.py` runs independent jobs concurrently:

```python
from src.back.harness import run_fleet

results = run_fleet(jobs, make_harness, max_workers=3)
```

- Jobs require `name`, `workdir`, and `task`; optional `review_task` and
  `on_complete` are supported.
- Names must be unique. Duplicate or overlapping workspaces are rejected before
  any job starts.
- Each job gets its own `Harness`, session, transcript, and OpenShell sandbox.
  One job's exception is reported without stopping the others.
- Results preserve input order and include `ok`, `report`, status, turns, tool
  calls, session, usage, and phase metadata.
- A completed build may run a review in the same session via `continue_task`.
  `on_complete` receives the live Harness, can verify or download artifacts, and
  can mark the job failed by returning `{"ok": False}`.
- The demo uses one focused repair pass when mechanical verification fails. It
  never reuses a workspace across jobs.

Each job's `Harness.close()` deletes its owned sandbox. `Harness.sandbox` exposes
the live adapter to the completion callback. `OpenShellSandbox.download(path,
destination)` copies a file or directory only from inside the approved workspace,
so artifacts can be saved before cleanup.

### Stage 4 live results

The three product builds used real DeepSeek + OpenShell. Results and hashes are
in [`stage4_results.json`](../../../stage4_results.json).

| Project | Mechanical result | Status | Caveat |
| --- | --- | --- | --- |
| `taskman` | PASS | `completed` | Build + review completed; 19 tests pass. |
| `artisan-coffee` | PASS | `token_budget` | 9 sections, 2,171 words, 16 SVGs; review incomplete. |
| `viper` | PASS | `token_budget` | Canvas, rAF, localStorage, 9 listeners; review incomplete. |

Web verification is static; browser interaction was not automated. The web
projects met their mechanical bars but did not finish the review phase within
the live token budget, so the Stage 4 process is not fully complete for them.

## Stage 5: streaming and terminal UI

### Provider stream

`ProviderStreamEvent` is a neutral event type for text deltas, tool-call
identity/argument deltas, finish metadata, usage, and explicit stream errors.
`DeepSeekProvider.stream()` parses server-sent events from the real
`/chat/completions` streaming endpoint. It uses `stream_options` for usage when
the API supplies it, preserves chunk order, preserves exact tool IDs, supports
multiple tool calls, and never executes a tool.

`complete()` remains unchanged. A provider without `stream()` continues to use
the completion path.

### Loop integration

`run_loop(..., stream=True)` uses `provider.stream()` when available and falls
back to `complete()`. It emits `ASSISTANT_DELTA` immediately, then assembles the
complete assistant message and validated tool calls, then emits the existing
`ASSISTANT` event. Tools still pass through the existing policy and OpenShell
path. Deltas are UI events only; they are not written as separate conversation
messages or session records.

Cancellation or a truncated stream closes the generator before any partial tool
call can execute. `Harness` records an interrupted stream as `interrupted`, not
`completed`.

### Terminal UI

`utils/` owns presentation only:

- `theme.py`: semantic colors, Unicode/ASCII symbols, TTY/capability detection.
- `formatting.py`: redaction, terminal-control stripping, path/args/clip/duration/
  token formatting, result previews, and the bounded cross-chunk redactor.
- `markdown.py`: incremental headings, lists, code fences, tables, and inline
  code without full-document buffering.
- `renderer.py`: streaming text, tool start/success/failure/blocked, compaction,
  cancellation, errors, progress indicators, and completion summary.
- `fleet.py`: thread-safe, job-prefixed fleet output.

CLI flags:

```text
--stream / --no-stream
--color / --no-color
--plain
```

Streaming is on by default. Redirected output disables spinners and ANSI by
default; `--plain` also forces ASCII fallbacks. The interactive prompt is `>`
(plain/ASCII mode).

Security limits: known credential markers are held across chunks and redacted;
this is bounded protection, not a guarantee for arbitrary unknown secrets.
ANSI, C0/C1 controls, bidi controls, and terminal escape sequences are stripped
from display text. Original text remains in the durable transcript.

### Stage 5 verification

```powershell
python demos/stage5_streaming.py --text
python demos/stage5_streaming.py --live --plain
```

- Real DeepSeek text-only stream: 92 deltas, first at 11.392s, last at 12.066s,
  12.067s total, finish `stop`, 110 input / 961 output tokens. Provider latency
  varies; the 0.67s between first and last delta is the observed streaming window.
- Real DeepSeek + OpenShell CLI task: 4 turns, 3 tools, 10.8s, session preserved.
- Safe write: visibly blocked; `false`: visibly failed; `--plain`: no ANSI.
- Controlled cancellation: session status `interrupted`, zero tools executed.
- Fleet display: serialized prefixed output in the synthetic demo.
- Offline suite: **316 tests, OK (2 skipped)**, including memory finalization and
  logging coverage.

## Stage 6: EvolveLab and canonical memory

The original MemEvolve repository is vendored at commit
`6035d5659d7a092dbfa6a87b1a32a3cee652ba54` under
`context_management/memory/upstream/memevolve/`, with Apache-2.0 licenses,
manifest, and compatibility notes. The provider adapter uses the real
`take_in_memory`, `provide_memory`, and `initialize` methods.

Irina adds:

- a stable `MemoryBackend` adapter that hides provider paths and schemas;
- a canonical, append-only experience store with stable IDs, schema versioning,
  redaction, torn-line recovery, idempotency, and workspace isolation;
- a trajectory bridge that preserves tool IDs and labels outcomes from observed
  evidence rather than assistant prose;
- bounded, lower-trust retrieval injection into the system prompt;
- post-task updates with an audit ledger and nonfatal failure handling;
- workspace-scoped provider storage, so shared memory roots do not leak between
  projects;
- an architecture registry with isolated candidate storage, one atomic active
  pointer, replay, explicit promotion, and rollback;
- the original MemEvolve analyze/generate/create/validate workflow, writing into
  an isolated build tree and never modifying the vendored source;
- a candidate gate (static AST, separate-process import, provider contract,
  instantiation) and in-process loading only after `--approve`;
- quiet-by-default memory logging; `--verbose` adds theme-colored diagnostics
  while errors stay visible.

Available/tested here:

- `lightweight_memory`: dependency-free upstream provider; integrated and the
  default.
- `expel`: original upstream provider, imported with the `memory-research` extra
  and exercised with a real DeepSeek memory model.
- Canonical + ExpeL live memory check: update succeeded, canonical count 1,
  retrieval returned labeled historical context.
- Live DeepSeek evolution reached analysis, generation, and creation. The
  generated provider was correctly rejected for a missing `initialize()`; a
  controlled fixture candidate passed the same gate and was registered
  `validated` without activation.

The remaining providers are discovered with separate source/import/init/
integration statuses. Missing optional dependencies are reported as blocked.
Memory is opt-in for the harness/CLI. Evolution needs the `memory-evolution`
extra; upstream dataset validation is skipped unless a dataset runner is
provided. Canonical replay, activation, rollback, and evolved-provider loading
are implemented. See [the memory README](context_management/memory/README.md).

## Limits

The defaults are in `loop/termination.py`: 40 turns, 100 tool calls, 300 seconds,
and 200,000 tokens. Token use is checked before and after each model call. It is
a soft limit because the size of the reply is unknown until it arrives.

Repeated identical calls and empty replies are warnings that can be tuned or
turned off. A run that hits a limit stops with that status; it is not marked
complete.

Stage 3 limits: child runs are sequential, child sessions are not persisted, and
resume requires the original OpenShell sandbox/workspace. The deterministic
sub-agent demo uses a scripted provider; the live DeepSeek model variant of the
Stage 3 acceptance task was not run.

Stage 4 limits: fleet jobs are independent parallel jobs, not child agents
inside one conversation. There is no queue, distributed scheduler, database, or
multi-tenant isolation beyond OpenShell. Web browser interaction is not
automated. `artisan-coffee` and `viper` ended at `token_budget` before a completed
review, although their static checks passed.

Stage 5 limits: streaming is synchronous and single-stream per Harness; there is
no provider-side cancellation token beyond closing the connection. Markdown is
streamed with line/structural buffering, not a full terminal Markdown renderer.
The cross-chunk redactor is bounded and documented; it does not claim absolute
secret detection. Streaming usage is zero when the provider omits usage rather
than being invented.

Stage 6 limits: only tested providers are claimed integrated; optional research
dependencies are isolated and reported as blocked when absent. Upstream
benchmark-dataset validation is skipped unless a dataset runner is supplied.
Generated code is only as contract-compliant as the model output; Irina rejects
non-conforming candidates instead of repairing them. Candidate validation is a
bounded static + separate-process probe, not an OpenShell execution sandbox;
activation imports generated code into the Irina process only after explicit
`--approve`. Legacy shared provider storage is not auto-attributed to a
workspace; rebuild it with `memory replay`. Live evolution uses the engine's
`OPENAI_API_KEY`/`OPENAI_BASE_URL` contract and its trajectory summary tool
still defaults to `gpt-4o-mini`.

## More detail

- [Loop](loop/README.md) - turn order, completion rules, events.
- [Providers](providers/README.md) - model adapter and retries.
- [Tools and guardrails](tools_and_guardrails/README.md) - six tools, policy, OpenShell.
- [Context, memory, skills](context_management/README.md) - compaction, task state, `IRINA.md`, `SKILL.md`.
- [Memory and evolution](context_management/memory/README.md) - lifecycle, providers, replay, evolution, safety, logging.
- [Stage 3 sub-agent demo](../../../demos/stage3_subagents.py) - deterministic sequential delegation.
- [Stage 4 fleet demo](../../../demos/stage4_projects.py) - parallel product builds and verification.
- [Stage 4 results](../../../stage4_results.json) - checks, hashes, and sessions.
- [Stage 5 streaming demo](../../../demos/stage5_streaming.py) - real and synthetic terminal UI checks.
- [Demos](../../../demos/README.md) - dice, OpenShell, Stage 2, Stage 3, Stage 4, Stage 5, Stage 6.
- [Tests](../../../tests/README.md) - unit and OpenShell integration coverage.
- [Project overview](../../../README.md) - the short version.
