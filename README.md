# Irina

Irina is a Python project for building and evaluating AI agents. Day 1 built a
small agent loop that talks to DeepSeek and returns a clear result for every run.
Day 2 added six workspace tools, an application permission policy, and execution
inside an NVIDIA OpenShell sandbox. Stage 2 added context compaction, durable
project memory, and loadable skills. Stage 3 adds durable sessions, truthful
crash recovery, sequential sub-agents, and one central `Harness`.

The harness uses only the Python standard library. It does not use an agent
framework or a vendor SDK. The repo targets Python 3.13; the harness code uses
features available from Python 3.10.

## What works today

- A bounded loop with explicit stop reasons.
- A DeepSeek Chat Completions adapter.
- Six workspace tools: read, write, edit, bash, list, and grep.
- `read-only`, `safe`, and `yolo` permission modes with dangerous-command checks.
- OpenShell 0.1.5 execution with a non-root user, confined filesystem, denied
  outbound network, and no provider credentials in the sandbox.
- Context compaction that preserves complete tool-call groups and emits a
  `context_compacted` event.
- Durable `IRINA.md` project memory across separate conversations.
- `SKILL.md` discovery and a `use_skill` tool, with skills kept lower-trust.
- Structured session task state updated from actual tool events.
- Append-only JSONL sessions with tool-call ID repair and unknown-outcome markers.
- Sequential `spawn_agent` delegation with depth, turn, time, and total-task limits.
- A central `Harness` that composes the provider, loop, tools, policy, context,
  memory, skills, sessions, and OpenShell.
- Clear run results, so "stopped" is not confused with "finished".
- Offline tests with a scripted model.
- Live checks for the Day 1, Day 2, and Stage 2 tasks, plus deterministic Stage 3
  checks over real OpenShell.

The Day 1 live check passed against DeepSeek for both tasks. The Day 2 acceptance
demo passed Fibonacci, destructive-command, and workspace-escape checks. The
Stage 2 demo passed long-task compaction, cross-session memory, and runtime skill
loading. The Stage 3 demonstration passed two sequential sub-agent tasks, one
parent-run test, and one top-level session. The real OpenShell crash/resume test
also passed. The full local suite is 188 tests; 9 of them use real OpenShell
sandboxes and verify isolation plus crash recovery.

## Start here

| Area | What it covers | Details |
| --- | --- | --- |
| Harness | How a run is put together and how to use it | [src/back/harness/README.md](src/back/harness/README.md) |
| Loop | Turn order, stop conditions, completion, events | [src/back/harness/loop/README.md](src/back/harness/loop/README.md) |
| Providers | DeepSeek settings, message mapping, retries | [src/back/harness/providers/README.md](src/back/harness/providers/README.md) |
| Tools and guardrails | Six tools, permission policy, shell, OpenShell | [src/back/harness/tools_and_guardrails/README.md](src/back/harness/tools_and_guardrails/README.md) |
| Context, memory, skills | Compaction, `IRINA.md`, `SKILL.md`, `use_skill` | [src/back/harness/context_management/README.md](src/back/harness/context_management/README.md) |
| Sessions, sub-agents | Append-only history, crash repair, `Harness`, `spawn_agent` | [Stage 3 in the harness README](src/back/harness/README.md#stage-3-sessions-crash-recovery-and-sub-agents) |
| Demos | Dice, OpenShell, Stage 2, Stage 3 sub-agents | [demos/README.md](demos/README.md) |
| Tests | What is tested and how to run it | [tests/README.md](tests/README.md) |

## Quick start

```powershell
uv sync
python -m unittest discover -s tests -v
```

For the live check, add your key to the ignored `.env` file:

```dotenv
DEEPSEEK_API_KEY=<your key>
```

```powershell
python demos/verify_live.py
```

The key is never printed. Without a key, the command exits with code `2` and
says the live run was skipped.

For the Day 2 OpenShell demo, install OpenShell 0.1.5 in WSL 2, make sure Docker
can see the `python:3.12-slim` image, then run:

```powershell
python demos/day2_openshell.py
```

The Day 2 demo creates a sandbox, runs the Fibonacci task, checks the destructive
command and path-escape guardrails, and deletes the sandbox. The
[tools and guardrails README](src/back/harness/tools_and_guardrails/README.md)
has the policy details.

The Stage 2 demo uses the same OpenShell setup:

```powershell
python demos/stage2_context_memory_skills.py
```

It runs the long five-file task under a 1,500-token context budget, memory across
a fresh session, and a runtime-created skill. See the
[context, memory, and skills README](src/back/harness/context_management/README.md)
for the details.

The Stage 3 demonstration uses the same OpenShell setup and a scripted provider
for deterministic verification:

```powershell
python main.py
```

It delegates `utils.py` and `test_utils.py` to two sequential child agents, runs
the generated test in the parent, verifies one top-level session, and deletes the
sandbox. The same demo is available as
`python demos/stage3_subagents.py`. Crash recovery is exercised by
`tests/test_stage3_integration.py` with a real forced process exit and sandbox
reattachment.

## Layout

```
src/back/harness/                   Agent loop, providers, tools, OpenShell, context
src/back/harness/harness.py         Central Harness composition
src/back/harness/session.py         JSONL sessions and truthful crash repair
src/back/harness/subagent.py        Sequential spawn_agent delegation
src/back/harness/context_management/
                                    Compaction, memory, skills, task state
demos/                              Runnable examples and live checks
tests/                              Unit tests and OpenShell integration tests
src/back/datasets/                  Dataset code (later)
src/back/evaluations/               Evaluation code (later)
src/back/experiments/               Experiment tracking (later)
src/back/opik/                      Opik source for local tracing and evaluation
src/front/                          Frontend application
```

`harness/loop`, `harness/providers`, `harness/tools_and_guardrails`, and
`harness/context_management` are implemented, along with `harness.py`,
`session.py`, and `subagent.py`. The other folders under `src/back/harness/` are
placeholders for later work.

## Opik local UI

See [docs/opik-local.md](docs/opik-local.md) for the Windows Docker Compose
setup and common commands.

## Limits

No parallel sub-agents, task queue, distributed workers, database, semantic
memory, or full production CLI yet. Child agents run sequentially, share the
parent workspace, and do not create top-level sessions. Resuming requires the
original OpenShell sandbox and workspace; if they are gone, the harness reports
that limitation instead of creating a replacement and pretending files survived.
The Stage 3 demonstration uses a scripted provider for determinism; the live
DeepSeek model variant of the Stage 3 acceptance task was not run. Project memory
is a bounded Markdown file, not a vector store. Day 2's sandboxing is a local
OpenShell 0.1.5 setup launched through WSL 2, not a production multi-tenant
deployment. The local file-only tool mode exists for tests and development; it
is not an isolation boundary. A run that hits a limit stops; it is not turned
into a success. The [harness README](src/back/harness/README.md#limits) has the
details.
