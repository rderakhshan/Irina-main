# Irina

**A memory-first coding agent that learns from verified work — and can evolve its own memory system.**

Irina is a Python agent harness for real terminal work. It runs a bounded model
loop, executes tools inside an NVIDIA OpenShell sandbox, records truthful session
history, and turns completed work into memory that is recalled on later tasks.
The harness core uses only the Python standard library; optional memory research
extras use their upstream dependencies.

The project grew in visible stages: a DeepSeek loop, workspace tools and
OpenShell, compaction and skills, durable sessions and sub-agents, an
interactive CLI and fleet, genuine streaming and a semantic UI, and finally the
vendored EvolveLab/MemEvolve memory system with canonical experience history.

## Why Irina is different

- **Truth over optimism.** A run has an explicit status. A stopped run is not
  reported as finished. A memory is not "verified" because the model said so.
- **Real isolation.** Tools execute in OpenShell with a confined filesystem and
  no provider credentials, not in the host process.
- **Durable evidence.** Sessions are append-only JSONL with crash repair and
  unknown-outcome markers.
- **Memory with a constitution.** Canonical experiences are host-side,
  append-only, redacted, workspace-isolated, and independent of any provider's
  storage format.
- **Evolution with a gate.** The original MemEvolve engine can generate a new
  memory provider; Irina validates it, registers it, and refuses to activate it
  without explicit approval.
- **Readable automation.** Streaming, Markdown, tool lifecycle, progress, and
  colored diagnostics are designed for a terminal — not a wall of logs.

## Contents

- [Why Irina is different](#why-irina-is-different)
- [Capability map](#capability-map)
- [Memory that learns, not just notes](#memory-that-learns-not-just-notes)
- [What works today](#what-works-today)
- [Start here](#start-here)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Layout](#layout)
- [Limits](#limits)

## Capability map

| Area | What Irina gives you | Entry point |
| --- | --- | --- |
| Agent loop | Bounded turns, tokens, time, and tool calls with explicit stop reasons | `harness/loop/` |
| Sandbox | OpenShell execution with non-root user, confined filesystem, denied egress | `harness/tools_and_guardrails/` |
| Sessions | Append-only history, resume, truthful crash recovery | `harness/orchestration/session.py` |
| Memory | 13-provider EvolveLab registry, canonical ledger, retrieval/learning hooks | `harness/context_management/memory/` |
| Evolution | Original MemEvolve analyze/generate/create/validate + safe activation | `memory/evolution_bridge.py` |
| CLI | Interactive and headless modes, sessions, memory administration | `harness/orchestration/cli.py` |
| Fleet | Parallel isolated jobs with ordered results and failure isolation | `harness/orchestration/fleet.py` |
| Terminal UI | Streaming Markdown, semantic colors, quiet-by-default memory logs | `harness/utils/` |

## Memory that learns, not just notes

Most agents "remember" by writing a note. Irina records **what happened**, decides
whether it actually worked, and recalls the lesson later.

```text
before task ──▶ retrieve relevant experience ──▶ bounded lower-trust prompt block
                                                             │
task runs ──▶ tool/test evidence ──▶ canonical experience ───┘
                                                             │
                                     provider learns ──▶ next task recalls it
```

What that changes in practice:

- A completed, verified coding run becomes a canonical experience automatically.
- The active provider (for example Lightweight or ExpeL) extracts reusable
  guidance from it.
- A later task receives the most relevant guidance, capped by token budget and
  labeled as historical evidence — never as authorization.
- Provider failures are logged for retry; they never erase the experience.
- A new architecture can be rebuilt from history, activated, and rolled back
  without losing the timeline.

```powershell
# learn silently (routine memory logs hidden)
uv run python main.py --mode yolo --memory -p "Fix the failing test"

# see initialization, retrieval, ingestion, and persistence
uv run python main.py --mode yolo --memory --verbose -p "Fix the failing test"

# audit what was stored
uv run python main.py memory --workdir . inspect
```

See the [Memory README](src/back/harness/context_management/memory/README.md) for
the full lifecycle, storage layout, provider matrix, evolution workflow, safety
model, and troubleshooting.

## What works today

- A bounded loop with explicit stop reasons.
- A DeepSeek Chat Completions adapter.
- Six core workspace tools: read, write, edit, bash, list, and grep.
- Eighteen web, document, and data tools (search, fetch, HTTP, download, PDF,
  LaTeX, DOCX, XLSX, PPTX, SQL, datasets, and notebooks), all registered in the
  same tool registry and executed inside OpenShell.
- Collapsible multiline paste: large paste blocks show a compact placeholder
  while the complete original text is sent to DeepSeek.
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
  idempotent ingestion, torn-line recovery, workspace isolation, and versioned
  schemas.
- Vendored original EvolveLab/MemEvolve source with provenance, license,
  manifest, and documented compatibility patches.
- A provider registry with 13 upstream identifiers, layered availability
  reporting, and a stable adapter that the agent loop never bypasses.
- Automatic retrieval before a task and one learning update after it, with a
  retry-safe `updates.jsonl` audit ledger.
- Workspace-scoped provider storage, so two projects sharing a memory root
  cannot read each other's provider memories.
- Versioned architectures with isolated candidate storage, one active pointer,
  atomic cross-process registry updates, explicit promotion, and rollback.
- Replay from canonical history into a new architecture or storage
  representation without modifying the source.
- The original MemEvolve analyze/generate/create/validate workflow, writing into
  an isolated build tree, with static + isolated + contract validation and no
  automatic activation.
- Dynamic loading of an approved generated provider from its recorded build tree.
- Quiet memory logging by default; `--verbose` adds theme-colored diagnostics
  while keeping errors visible.
- Complete skill-directory seeding (`SKILL.md`, assets, scripts), explicit
  `--input` upload and `--export` artifact download, and an approved
  TeX/Poppler/pypdf image for document skills.
- Original Lightweight Memory and ExpeL provider integration; optional research
  dependencies remain isolated from ordinary Irina execution.
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
automated. The current offline suite is **352 tests, OK (3 skipped)**, including
the memory integration, finalization, extended-tools, and paste suites.

Memory finalization checks:
- Canonical store, provider update, and restart retrieval passed across separate
  processes.
- A full harness run produced one canonical experience and a restarted manager
  retrieved it into bounded prompt context.
- A live DeepSeek MemEvolve run completed analysis, generation, and creation; the
  generated provider was correctly **rejected** because it omitted
  `initialize()`. A controlled fixture candidate passed the same gate and was
  registered `validated` without activation.
- Replay, promotion, rollback, workspace isolation, and failure rejection were
  exercised; canonical history survived activation and rollback.
- Default memory logging produced zero routine lines; `--verbose` produced
  colored diagnostics while memory operations continued normally.

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
| Memory / EvolveLab | Learning lifecycle, 13 providers, canonical ledger, replay, evolution, safety, logging | [Memory README](src/back/harness/context_management/memory/README.md) |
| Loop | Turn order, stop conditions, completion, events | [src/back/harness/loop/README.md](src/back/harness/loop/README.md) |
| Providers | DeepSeek settings, message mapping, retries | [src/back/harness/providers/README.md](src/back/harness/providers/README.md) |
| Tools and guardrails | Six tools, permission policy, shell, OpenShell | [src/back/harness/tools_and_guardrails/README.md](src/back/harness/tools_and_guardrails/README.md) |
| Context, memory, skills | Compaction, `IRINA.md`, `SKILL.md`, `use_skill` | [src/back/harness/context_management/README.md](src/back/harness/context_management/README.md) |
| Sessions, sub-agents | Append-only history, crash repair, `Harness`, `spawn_agent` | [Stage 3 in the harness README](src/back/harness/README.md#stage-3-sessions-crash-recovery-and-sub-agents) |
| Demos | Dice, OpenShell, Stage 2, Stage 3, Stage 4 fleet | [demos/README.md](demos/README.md) |
| Tests | What is tested and how to run it | [tests/README.md](tests/README.md) |

## Quick start

```powershell
uv sync                                                    # core agent
uv sync --extra memory-research --extra memory-evolution   # optional memory sets
uv run python -m unittest discover -s tests -v              # full suite
uv run python main.py --memory                              # interactive with memory
```

Requires Python 3.13 and NVIDIA OpenShell 0.1.5 for sandboxed tool execution.
Add `DEEPSEEK_API_KEY` to the ignored `.env` file for live model calls.

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
python main.py --memory                # enable learned experience
python main.py --memory --verbose      # show colored memory diagnostics
python main.py --skills-dir .\skills   # upload host skills before the first prompt
python main.py --prompt-file job.txt   # headless task from a host file
```

For long or multi-line text, paste it directly. Irina uses bracketed paste when
the terminal provides it and falls back to a short inter-key burst detector.
Small pastes stay visible. A paste larger than 800 characters or three lines is
shown as a compact placeholder such as `[Pasted text · 145 lines]`; the
complete original text, including indentation and newlines, is still submitted
unchanged. Text before and after a paste is preserved, multiple paste blocks
work in one prompt, and Backspace removes a collapsed block as one unit. If a
terminal splits the paste, use the explicit `/paste` mode:

```text
/paste
<paste the job description>
"""
```

or load a host file:

```text
/file D:\Application\job_description.txt
```

`/paste` reads until a line containing only `"""` or `/end`. `/file` reads a
UTF-8 host file (up to 200,000 characters) as the task. `--prompt-file` does the
same for a headless run.

When the first line is `Use the application skill` and the pasted text contains
a complete JD, the skill runs its automatic mode: it drafts, finalizes, and
reports without a review loop. `IRINA_AUTO_EXPORT=1` saves the folder to the
host after the run.

Use `/export` to save configured artifacts immediately after they are created,
without waiting for exit. `/export force` replaces an existing host folder.
`--auto-export` does this automatically after each completed run.

Streaming is on by default. `--no-stream` uses the original completion path;
`--plain` disables color, animation, and Unicode fallbacks. `safe` is the
default. Headless `safe` refuses approval-required operations
instead of silently approving them; use `--mode yolo` explicitly. A run exits
`0` only when the result is `completed`. Ctrl-C preserves the session and leaves
an owned sandbox alive for `--resume`; a normal exit cleans it up. The same
module entry point works as `python -m src.back.harness`.

Memory flags:

| Flag | Effect |
| --- | --- |
| `--memory` | Enable canonical experience + the configured EvolveLab provider |
| `--no-memory` | Disable memory even when configuration enables it |
| `--memory-provider <id>` | Select an upstream provider explicitly |
| `--skills-dir <path>` | Upload complete skill directories (`SKILL.md`, assets, scripts) at startup (default `./skills`; empty string disables) |
| `--input <host path>` | Upload a file or directory into `input/` before the run |
| `--export <sandbox path>` | Download a workspace path on clean shutdown |
| `--artifacts-dir <host path>` | Host destination for `--export` (required with `--export`) |
| `--export-contents` | Merge an exported directory's contents into `--artifacts-dir` (e.g. `D:\Application\<Company>`) |
| `--auto-export` | Export configured artifacts after each completed run |
| `--force` | Replace an existing exported artifact |
| `--sandbox-image <image>` | Use an approved OpenShell image, e.g. `irina/irina-tools:1` |
| `--sandbox-policy <path>` | Use an approved OpenShell policy; `openshell_policy_network.yaml` enables the listed web destinations |
| `--verbose` | Show routine memory logs with level colors |
| `--no-color` / `--plain` | Keep verbose logs readable without ANSI |

Default memory logging is intentionally silent. `--verbose` reports
initialization, retrieval, ingestion, and persistence; errors stay visible in
both modes.

Skills are host directories at `skills/<name>/` with a `SKILL.md` plus optional
assets and scripts. On first run Irina creates `./skills` with a README and
uploads complete valid skill directories into the sandbox before building the
system prompt (`--skills-dir`, default `./skills`). The agent sees a compact
catalog and calls `use_skill(name)` only when relevant. Invalid names, symlinks,
empty or oversized files are skipped; skills are instructions only and cannot
grant permissions.

Files explicitly selected by the user can be uploaded into `input/` (a file
lands at `input/<filename>`, a directory under `input/<dirname>/`); declared
workspace paths can be exported before cleanup:

```powershell
uv run python main.py `
  --skills-dir .\skills `
  --input .\input-cv\resume.tex `
  --export application `
  --export-contents `
  --artifacts-dir "D:\Application" `
  --mode yolo -p "Prepare an application for this job description"
```

With `--export-contents`, the sandbox folder `application/<Company>/` lands
directly as `D:\Application\<Company>\`. Without it, the export creates
`D:\Application\application\<Company>\`. Existing company folders are
refused unless `--force` is passed.

The `irina-tools` image includes TeX Live, Poppler, and the Python document,
data, and notebook libraries used by the web/document/data tool groups:

```powershell
wsl docker build -t irina/irina-tools:1 .\images\application
uv run python main.py --sandbox-image irina/irina-tools:1 ...
```

Web tools are off by default because the default OpenShell policy denies
network access. Select the shipped network policy explicitly to allow only its
listed destinations:

```powershell
uv run python main.py `
  --sandbox-image irina/irina-tools:1 `
  --sandbox-policy src/back/harness/tools_and_guardrails/openshell_policy_network.yaml `
  -p "Fetch https://example.com and summarize it"
```

Edit `openshell_policy_network.yaml` to add destinations. Enabling network
policies makes OpenShell add its L7 proxy baseline paths, so the strict default
policy remains the right choice when web access is not needed.

See [images/application](images/application/README.md) and the
[application skill](skills/application/SKILL.md).

Skills created or edited by Irina during a run are captured as drafts on a
clean shutdown, not promoted automatically:

```powershell
uv run python main.py skills list
uv run python main.py skills promote <name>
uv run python main.py skills discard <name>
```

Drafts live under `<state>/skill-drafts/<workspace-key>/`. Review before
promoting; one bad run should not be able to rewrite the instructions for every
later run.

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

| Command | What to look for |
| --- | --- |
| `memory status` | Active architecture, config root, upstream provenance |
| `memory providers` | Per-provider source/import/init/integration status |
| `memory inspect` | Canonical count and latest experiences for the workspace |
| `memory replay` | `validation_status: "validated"` and zero failed experiences |
| `memory candidates` | State, storage namespace, evaluation evidence, active/previous |
| `memory evaluate` | Contract + instantiation probe for one candidate |
| `memory promote --approve` | `Activated <id>`; no automatic promotion exists |
| `memory rollback` | `Rolled back to <id>`; canonical history remains |

Storage at a glance:

```text
~/.irina/memory/
├── workspaces/<workspace-id>/experiences.jsonl   # canonical, shared history
├── providers/<provider>/<workspace-id>/          # per-workspace provider data
├── providers/<provider>/<arch-id>/<workspace-id>/ # isolated candidate data
├── architectures.json                            # active/previous versions
├── selection.json                                # explicit provider choice
├── updates.jsonl                                 # update audit ledger
└── migrations/<target>.json                      # replay evidence
```

## Flag-free application profile

Put machine-specific defaults in the ignored `.env` file once:

```dotenv
IRINA_SANDBOX_IMAGE=irina/irina-tools:1
IRINA_SANDBOX_POLICY=src/back/harness/tools_and_guardrails/openshell_policy_network.yaml
IRINA_SKILLS_DIR=skills
IRINA_INPUTS=D:\Application\input\resume.tex
IRINA_OPTIONAL_INPUTS=D:\Application\reference.pdf;D:\Application\examples
IRINA_EXPORT_PATHS=application
IRINA_EXPORT_CONTENTS=1
IRINA_ARTIFACTS_DIR=D:\Application
IRINA_AUTO_EXPORT=1
IRINA_MODE=safe
IRINA_DEFAULTS_REQUIRE=D:\Application\input\resume.tex
```

Then start with no application flags:

```powershell
uv run python main.py
```

At the prompt use `/paste` for the JD. Configured artifacts are exported after
each completed run (`IRINA_AUTO_EXPORT=1`) and again on clean exit. Explicit CLI
flags override these defaults. Missing optional inputs are skipped; a missing
required input fails before the run with its exact path.
`IRINA_DEFAULTS_REQUIRE` disables the whole block when the CV is absent, so a
normal Irina session starts instead.

## Add your own skill

A skill is a directory under `skills/` with a `SKILL.md` and optional assets or
scripts:

```text
skills/
??? release-notes/
    ??? SKILL.md
    ??? assets/
        ??? template.md
```

Create it once:

```markdown
---
name: release-notes
description: Write concise release notes from git history.
---

# Release Notes

When asked for release notes:
1. Read commits since the last tag.
2. Group by feature, fix, and breaking change.
3. Keep each bullet under 120 characters.
```

Rules:

- the directory name is the skill id;
- `description` is what appears in the system prompt catalog;
- the full `SKILL.md` loads only when the agent calls `use_skill(name)`;
- assets and scripts in the same directory are uploaded too;
- skills are lower-trust instructions and cannot grant permissions.

On the next start:

```powershell
uv run python main.py
```

Irina uploads the complete directory into OpenShell before building the system
prompt. The agent sees the skill in its catalog and can load it with
`use_skill`. A skill created by Irina during a run goes to the drafts directory
instead; review it with:

```powershell
uv run python main.py skills list
uv run python main.py skills promote <name>
```

Use `--skills-dir <path>` for a different host directory, or
`--skills-dir ""` to disable loading.

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
src/back/harness/utils/             Terminal theme, renderer, Markdown, memory logging, fleet UI
src/back/harness/context_management/memory/
                                    Canonical experience store, EvolveLab adapters,
                                    provider registry, architecture/replay,
                                    evolution/evaluation/security, memory logging,
                                    and vendored MemEvolve source
src/back/harness/context_management/
                                    Compaction, memory, skills, task state
skills/                             Host skill packs uploaded into the sandbox
images/application/                 Approved TeX/Poppler/pypdf sandbox image
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

Streaming redaction uses a bounded cross-chunk buffer for known credential
markers; it is not a guarantee for arbitrary unknown secrets. Terminal control
sequences are stripped only for display; session history remains the source of
truth. There is no full-screen TUI and no artificial typing delay.

Memory limits:

- Retrieval is lower-trust prompt context, not authorization.
- Only tested providers are claimed available; optional research dependencies
  are reported as blocked, not simulated.
- The canonical store is host-side and is not readable or writable from
  OpenShell tools.
- Upstream benchmark validation is skipped unless `--validation-dataset` is
  supplied, because those dataset runners are not vendored.
- Generated provider code is only as good as the model output. Irina validates
  and rejects non-conforming candidates; it does not silently repair them.
- Candidate isolation is a bounded validation gate, not a complete OpenShell
  sandbox. Activation imports generated code into the Irina process, so
  `--approve` is mandatory.
- Legacy shared provider storage is not auto-attributed to a workspace; rebuild
  it with `memory replay`.
- `IRINA.md` remains a bounded human-readable project-memory file; the provider
  memory system is separate from it.

Other limits: there is no queue, distributed worker pool, database, or
multi-tenant scheduler. Fleet jobs are independent process-level jobs; child
agents inside one Harness still run sequentially. Resuming requires the original
OpenShell sandbox and workspace; if they are gone, the harness reports that
limitation instead of pretending files survived. The Stage 3 demonstration uses
a scripted provider for determinism; Stage 4 live runs used DeepSeek and are
recorded in `stage4_results.json`. Web verification is static, not browser
interaction. `artisan-coffee` and `viper` met mechanical checks but ended at
`token_budget` before a completed review. Day 2 sandboxing is a local OpenShell
0.1.5 setup through WSL 2, not a production deployment. The local file-only tool
mode exists for tests and development; it is not an isolation boundary. A run
that hits a limit stops; it is not turned into a success. The
[harness README](src/back/harness/README.md#limits) has more detail.
