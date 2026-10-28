# Irina

Irina is a Python project for building and evaluating AI agents. Day 1 built a
small agent loop that talks to DeepSeek and returns a clear result for every run.
Day 2 added six workspace tools, an application permission policy, and execution
inside an NVIDIA OpenShell sandbox. Stage 2 added context compaction, durable
project memory, and loadable skills. Stage 3 added durable sessions, truthful
crash recovery, sequential sub-agents, and one central `Harness`. Stage 4 adds
the interactive/headless CLI, parallel fleet execution, artifact download, and
three real product builds. Stage 5 adds genuine provider streaming and a
polished event-driven terminal UI. Stage 6 vendors EvolveLab/MemEvolve and adds
canonical experience storage, retrieval/update hooks, architecture replay, and
safe memory-evolution bridges.

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
- An interactive and headless CLI with `safe`, `yolo`, and `read-only` modes,
  session resume, truthful exit codes, and Windows-safe output.
- Genuine DeepSeek streaming with provider-neutral delta events, safe tool-call
  assembly, and a streaming/non-streaming compatibility path.
- A semantic terminal theme, Markdown-aware streaming renderer, progress
  indicators, tool lifecycle display, plain/redirected mode, and serialized
  fleet presentation.
- Canonical, architecture-independent experience storage with redaction,
  idempotent ingestion, recovery, workspace isolation, and versioned schemas.
- Vendored original EvolveLab/MemEvolve source, a provider registry, memory
  retrieval/update hooks, architecture replay/registry, coding evaluation
  fixtures, and a safe candidate-validation bridge.
- Original Lightweight Memory and ExpeL provider integration; optional
  research dependencies remain isolated from ordinary Irina execution.
- A parallel fleet runner with isolated workspaces, ordered results, per-job
  failure isolation, optional review/repair phases, and metadata.
- Artifact download from an owned OpenShell sandbox before cleanup.
- Clear run results, so "stopped" is not confused with "finished".
- Offline tests with a scripted model.
- Live checks for the Day 1, Day 2, and Stage 2 tasks, plus deterministic Stage 3
  checks over real OpenShell.

The Day 1 live check passed against DeepSeek for both tasks. The Day 2 acceptance
demo passed Fibonacci, destructive-command, and workspace-escape checks. The
Stage 2 demo passed long-task compaction, cross-session memory, and runtime skill
loading. The Stage 3 demonstration passed two sequential sub-agent tasks, one
parent-run test, and one top-level session. The real OpenShell crash/resume test
also passed. Stage 4 produced `taskman/`, `artisan-coffee/`, and `viper/` through
real DeepSeek + OpenShell runs. `taskman` completed both build and review with 19
tests passing. The two web projects met their mechanical checks, but their runs
ended at `token_budget` before a completed review; browser interaction was not
automated. The offline suite is 270 tests, OK (2 skipped). The repository
contains 279 tests total; 9 use real OpenShell sandboxes and could not be rerun
during final memory verification because the local Docker service was
unavailable.

Stage 5 live streaming checks:
- A real DeepSeek text-only stream delivered 92 deltas: first delta at 11.392s,
  last at 12.066s, total 12.067s, finish `stop`, 110 input / 961 output tokens.
  Provider latency varies; the point is the 0.67s between first and last delta.
- A real DeepSeek + OpenShell CLI task used 4 turns and 3 tools and completed in
  10.8s. Its final assistant text arrived after the tools, as the model chose.
- A safe headless write was visibly blocked; a `false` command was displayed as
  failed. Plain output contained no ANSI. The controlled cancellation demo left
  the session `interrupted` and executed no tool.

### Stage 4 product results

| Project | Mechanical result | Turns | Evidence | Caveat |
| --- | --- | --- | --- | --- |
| `taskman` | PASS | 27 | 19 tests pass; `--help` works; JSON/atomic persistence checks | Build and review completed. |
| `artisan-coffee` | PASS | 27 | 9 sections, 2,171 visible words, 16 inline SVGs, 8 listeners | Static checks only; review ended at `token_budget`. |
| `viper` | PASS | 25 | Canvas, `requestAnimationFrame`, `localStorage`, 9 listeners | Static checks only; review ended at `token_budget`. |

See [stage4_results.json](stage4_results.json) for checks, hashes, sessions, and
the exact caveats.

## Start here

| Area | What it covers | Details |
| --- | --- | --- |
| Harness | How a run is put together and how to use it | [src/back/harness/README.md](src/back/harness/README.md) |
| CLI and fleet | Interactive/headless CLI, parallel jobs, artifact download | [Stage 4 in the harness README](src/back/harness/README.md#stage-4-cli-fleet-and-artifacts) |
| Streaming UI | Real streaming, theme, Markdown, progress, fleet display | [Stage 5 in the harness README](src/back/harness/README.md#stage-5-streaming-and-terminal-ui) |
| Memory / EvolveLab | Vendored providers, canonical experience history, replay, evolution bridge | [Memory README](src/back/harness/context_management/memory/README.md) |
| Loop | Turn order, stop conditions, completion, events | [src/back/harness/loop/README.md](src/back/harness/loop/README.md) |
| Providers | DeepSeek settings, message mapping, retries | [src/back/harness/providers/README.md](src/back/harness/providers/README.md) |
| Tools and guardrails | Six tools, permission policy, shell, OpenShell | [src/back/harness/tools_and_guardrails/README.md](src/back/harness/tools_and_guardrails/README.md) |
| Context, memory, skills | Compaction, `IRINA.md`, `SKILL.md`, `use_skill` | [src/back/harness/context_management/README.md](src/back/harness/context_management/README.md) |
| Sessions, sub-agents | Append-only history, crash repair, `Harness`, `spawn_agent` | [Stage 3 in the harness README](src/back/harness/README.md#stage-3-sessions-crash-recovery-and-sub-agents) |
| Demos | Dice, OpenShell, Stage 2, Stage 3, Stage 4 fleet | [demos/README.md](demos/README.md) |
| Tests | What is tested and how to run it | [tests/README.md](tests/README.md) |

## Quick start

```powershell
uv sync
python -m unittest discover -s tests -v
```

## Command line

The CLI is the default entry point:

```powershell
python main.py                         # interactive
python main.py -p "Fix the failing test" --mode safe
python main.py -p "Build the release" --mode yolo --workdir .
python main.py --resume                # latest resumable session
python main.py --resume <session.jsonl>
python main.py -p "Explain the test failure" --no-stream
python main.py -p "Explain the test failure" --plain
```

Streaming is on by default. `--no-stream` uses the original completion path;
`--plain` disables color, animation, and Unicode fallbacks. `safe` is the
default. Headless `safe` refuses approval-required operations
instead of silently approving them; use `--mode yolo` explicitly. A run exits
`0` only when the result is `completed`. Ctrl-C preserves the session and leaves
an owned sandbox alive for `--resume`; a normal exit cleans it up. The same
module entry point works as `python -m src.back.harness`.

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
python demos/stage3_subagents.py
```

It delegates `utils.py` and `test_utils.py` to two sequential child agents, runs
the generated test in the parent, verifies one top-level session, and deletes the
sandbox. Crash recovery is exercised by
`tests/test_stage3_integration.py` with a real forced process exit and sandbox
reattachment.

Stage 4 runs the three product jobs through the parallel fleet:

```powershell
python demos/stage4_projects.py --workers 3
```

Each job gets its own OpenShell sandbox and session. The demo seeds
`skills/design-engineering/SKILL.md` for the web projects, runs build + review,
verifies artifacts, downloads them to `artisan-coffee/`, `taskman/`, and
`viper/`, and writes [stage4_results.json](stage4_results.json). Web verification
is static; no browser automation is included.

Stage 5's terminal UI can be exercised without credentials:

```powershell
python demos/stage5_streaming.py --plain
```

It demonstrates streaming-style Markdown, success/failure/blocked tools,
context compaction, controlled cancellation, and concurrent fleet presentation.
The controlled cancellation is synthetic; it is labeled as such. A real run is:

```powershell
python demos/stage5_streaming.py --text
python demos/stage5_streaming.py --live --plain
```

`--text` measures provider deltas without tools; `--live` runs the full
OpenShell tool task.

Memory is opt-in in ordinary CLI runs. The original source is vendored, and
the active provider can be inspected or selected explicitly:

```powershell
uv run python main.py memory status
uv run python main.py memory providers
uv run python main.py memory --workdir . inspect
uv run python main.py memory select lightweight_memory
uv run python main.py memory replay <target-arch> --target-provider lightweight_memory
uv run python main.py memory candidates
uv run python main.py memory evaluate <candidate>
uv run python main.py memory promote <candidate> --approve
uv run python main.py memory rollback
uv run python main.py memory evolve --from-canonical --work-dir .irina/evolution
uv run python main.py -p "Fix the failing test" --mode yolo --memory
uv run python main.py -p "Fix the failing test" --mode yolo --memory --memory-provider expel
uv run python main.py -p "Fix the failing test" --mode yolo --memory --verbose
```

Routine memory/provider logs stay hidden by default. `--verbose` shows them
with Irina's level colors (cyan INFO, dim DEBUG, green success, yellow
warning, red error) and respects `--no-color`/`--plain`; errors remain visible
in both modes.

Canonical experiences live under `~/.irina/memory` by default, and provider
storage is scoped per workspace. Install the optional sets with
`uv sync --extra memory-research` (providers), `--extra memory-embeddings`
(sentence-transformer providers), and `--extra memory-evolution` (original
MemEvolve engine). EvolveLab providers with missing optional research
dependencies are reported as blocked; source presence is not treated as
availability. The original evolution engine is callable, writes generated code
into an isolated build tree, validates candidates with a static + isolated
provider-contract probe, registers them without activating them, and skips the
upstream benchmark-dataset validation stage unless `--validation-dataset` is
provided.

## Layout

```
src/back/harness/                   Agent loop, providers, tools, OpenShell, context
src/back/harness/orchestration/     Harness, sessions, sub-agents, CLI, fleet
src/back/harness/orchestration/harness.py
                                    Central Harness composition
src/back/harness/orchestration/session.py
                                    JSONL sessions and truthful crash repair
src/back/harness/orchestration/subagent.py
                                    Sequential spawn_agent delegation
src/back/harness/orchestration/cli.py
                                    Interactive and headless front door
src/back/harness/orchestration/fleet.py
                                    Parallel isolated jobs and result ordering
src/back/harness/utils/             Terminal theme, formatting, Markdown, renderer, fleet UI
src/back/harness/context_management/memory/
                                    Canonical experience store, EvolveLab adapters,
                                    provider registry, architecture/replay,
                                    evaluation, and vendored MemEvolve source
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
`harness/context_management` are implemented. `harness/orchestration` contains
the composition, durable sessions, sub-agents, CLI, and fleet. The other folders
under `src/back/harness/` are placeholders for later work.

## Opik local UI

See [docs/opik-local.md](docs/opik-local.md) for the Windows Docker Compose
setup and common commands.

## Limits

Stage 5 streaming redaction uses a bounded cross-chunk buffer for known
credential markers. It is not a guarantee for arbitrary unknown secrets in
model-generated prose. Terminal control sequences are stripped only for display;
session history remains the source of truth. Stage 5 has no full-screen TUI and
no artificial typing delay. Memory retrieval is lower-trust prompt context, not
authorization. The canonical store is host-side. EvolveLab providers may need
optional research dependencies; only tested providers are claimed available.
MemEvolve's original engine is vendored but unavailable here until its
FlashOAgents dependency set is installed. Stage 4 has a real CLI and parallel
fleet, but no queue, distributed workers, database, or production multi-tenant
scheduler. Fleet jobs are independent
process-level jobs; child agents inside one Harness still run sequentially.
Resuming requires the original OpenShell sandbox and workspace; if they are gone,
the harness reports that limitation instead of creating a replacement and
pretending files survived. The Stage 3 demonstration uses a scripted provider for
determinism; the Stage 4 live runs used DeepSeek and are recorded in
`stage4_results.json`. Web verification is static, not browser interaction.
`artisan-coffee` and `viper` met mechanical checks but ended at `token_budget`
before a completed review. Project memory is a bounded Markdown file, not a
vector store. Day 2's sandboxing is a local OpenShell 0.1.5 setup launched through
WSL 2, not a production multi-tenant deployment. The local file-only tool mode
exists for tests and development; it is not an isolation boundary. A run that
hits a limit stops; it is not turned into a success. The
[harness README](src/back/harness/README.md#limits) has the details.
