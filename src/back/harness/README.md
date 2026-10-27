# Harness

The harness runs one conversation from start to finish. It asks the model for a
reply, runs any tools the model asks for, and stops for a clear reason.

The loop is deliberately small. Day 1 added one model adapter and a dice demo;
Day 2 added six workspace tools and an OpenShell sandbox. Stage 2 added context
compaction, durable memory, and skills. Stage 3 adds durable sessions, truthful
crash repair, sequential sub-agents, and the central `Harness`.

## Files

| File | Purpose |
| --- | --- |
| `harness.py` | Central composition of provider, loop, tools, policy, context, sessions, and OpenShell. |
| `session.py` | Append-only JSONL sessions, recorder, load/repair, latest-session lookup. |
| `subagent.py` | `spawn_agent`, delegation budget, sequential child execution. |
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
exposes `run(task)`, `resume(path=None)`, `close()`, `tools`, `workspace`,
`system_prompt`, `session_path`, and `session_status`. The deterministic Stage 3
demo uses a scripted provider; the real OpenShell crash integration test uses a
forced process exit and a fresh Harness to resume.

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
Stage 3 acceptance task was not run. No parallel scheduling, queue, database, or
production CLI is included.

## More detail

- [Loop](loop/README.md) — turn order, completion rules, events.
- [Providers](providers/README.md) — model adapter and retries.
- [Tools and guardrails](tools_and_guardrails/README.md) — six tools, policy, OpenShell.
- [Context, memory, skills](context_management/README.md) — compaction, task state, `IRINA.md`, `SKILL.md`.
- [Stage 3 sub-agent demo](../../../demos/stage3_subagents.py) — deterministic sequential delegation.
- [Demos](../../../demos/README.md) — dice, OpenShell, Stage 2, Stage 3.
- [Tests](../../../tests/README.md) — unit and OpenShell integration coverage.
- [Project overview](../../../README.md) — the short version.
