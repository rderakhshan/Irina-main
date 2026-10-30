# Memory: EvolveLab, MemEvolve, and Canonical Experience Evolution

Irina does not just remember notes. It records what actually happened in a run,
turns verified work into reusable memory, recalls relevant experience before the
next task, and can evolve its own memory architecture without losing history.

This package is the boundary between **Irina's stable memory contract** and the
**original EvolveLab/MemEvolve source**. Providers may change their schema,
storage format, or retrieval algorithm; Irina's canonical history does not.

> **Design rule:** the agent loop never touches a provider's files or internal
> format. It talks to one stable adapter, and the canonical ledger remains the
> source of truth.

## Contents

- [Why this exists](#why-this-exists)
- [At a glance](#at-a-glance)
- [The lifecycle](#the-lifecycle)
- [What a canonical experience contains](#what-a-canonical-experience-contains)
- [Storage layout](#storage-layout)
- [Providers](#providers)
- [Everyday use](#everyday-use)
- [Architecture evolution](#architecture-evolution)
- [Migration and replay](#migration-and-replay)
- [Activation and rollback guarantees](#activation-and-rollback-guarantees)
- [Safety model](#safety-model)
- [Upstream provenance](#upstream-provenance)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Where to look](#where-to-look)

## Why this exists

| Before | With this package |
| --- | --- |
| `IRINA.md` stored facts the model chose to write | Every completed run can become a structured experience |
| Memory stopped at a Markdown file | A registry of 13 original EvolveLab identifiers, with tested Lightweight and ExpeL integrations |
| Nothing was learned from tool results | Verified test/tool evidence decides what counts as success |
| A provider change could strand history | Canonical experiences survive provider and format changes |
| One memory implementation | Versioned architectures with isolated storage, activation, and rollback |
| No path from experience to improvement | Original MemEvolve can analyze Irina's own trajectories and generate a new provider |

## At a glance

| Capability | What it gives Irina | Main code |
| --- | --- | --- |
| Canonical experience ledger | Append-only, redacted, idempotent history of completed work | `experience_store.py` |
| Trajectory bridge | Turns a run into a record using observed tool evidence, never prose | `trajectory_bridge.py` |
| Provider adapter | Maps Irina to upstream `initialize` / `take_in_memory` / `provide_memory` | `adapter.py` |
| Retrieval hook | Bounded, lower-trust experience block injected before a task | `manager.py`, `retrieval_bridge.py` |
| Learning hook | One provider update per experience, with retry-safe failure ledger | `manager.py` |
| Architecture registry | Baseline/candidates, one active pointer, atomic cross-process updates | `architecture_registry.py` |
| Replay/migration | Rebuild a new architecture from canonical history without touching the source | `migration.py` |
| Evolution bridge | Original MemEvolve analyze → generate → create → validate | `evolution_bridge.py` |
| Candidate safety | Static AST checks, isolated import, contract + instantiation probe | `security.py` |
| Terminal presentation | Silent by default; colored memory diagnostics with `--verbose` | `../../utils/memory_logging.py` |

## The lifecycle

```text
                    ┌──────────────────────────────────────────────┐
                    │              before each task                │
                    │  query ──▶ provider.retrieve ──▶ dedupe      │
                    │        ──▶ token bound ──▶ lower-trust block │
                    └──────────────────────┬───────────────────────┘
                                           │ injected into system prompt
                                           ▼
   Irina coding task ──▶ session + tool events ──▶ observed evidence
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │               after the run                  │
                    │  trajectory bridge ──▶ canonical ledger      │
                    │        ──▶ active provider update            │
                    │        ──▶ updates.jsonl audit entry         │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                              next task recalls the lesson
```

Two properties matter:

1. **Canonical first.** The experience is written to host-side JSONL before the
   provider is updated. A provider failure never corrupts or erases the record.
2. **Evidence, not claims.** `verified_success` is derived from observed bash or
   test output (for example `OK`, `PASSED`, `Ran N tests`, `exit 0`), not from
   the assistant saying it succeeded.

## What a canonical experience contains

Schema version 1 stores:

- task, session, workspace, and experience identity;
- timestamps, run status, execution outcome, and failure reason when present;
- bounded task context and the exact tool actions;
- tool observations keyed by the original tool-call IDs;
- test evidence and the verification decision;
- source-event references (session path, turns, tool count).

Unavailable fields stay empty. The bridge never invents a test result.

The store is:

- **append-only** and **idempotent** by experience ID;
- flushed and `fsync`ed per record;
- bounded per record and per list;
- secret-redacted before becoming durable;
- workspace-isolated;
- recoverable from a torn final line;
- strict about corruption in the middle of the file (raises instead of losing
  records silently).

## Storage layout

Default root: `~/.irina/memory` (override with `IRINA_MEMORY_STATE_DIR`).

```text
<state>/
├── workspaces/
│   └── <workspace-id>/
│       └── experiences.jsonl          # canonical, architecture-independent
├── providers/
│   └── <provider>/
│       ├── <workspace-id>/...         # active baseline storage per workspace
│       └── <architecture-id>/
│           └── <workspace-id>/...     # isolated candidate/replay storage
├── architectures.json                 # active/previous pointer + versions
├── selection.json                     # explicit provider selection
├── updates.jsonl                      # provider update audit (ok/failed)
└── migrations/
    └── <target-architecture>.json     # replay counts, timings, validation
```

- Canonical history and the architecture registry are shared state.
- Provider storage is **workspace-scoped**. Two projects sharing one memory root
  cannot read each other's provider memories.
- Older Irina versions used a shared provider directory. That data is not
  silently attributed to a workspace; rebuild it with `memory replay`.

## Providers

The registry discovers the 13 identifiers in the upstream `PROVIDER_MAPPING`.
Source presence, importability, initialization, integration, and end-to-end
testing are separate statuses — a file on disk is not an available provider.

| Provider | Current observed status |
| --- | --- |
| `lightweight_memory` | Dependency-free upstream provider; integrated and used by default |
| `expel` | Integrated with the `memory-research` extra and a real DeepSeek model |
| `cerebra_fusion_memory` | Imports with `numpy` + `scikit-learn`; initialization not claimed |
| `skillweaver` | Imports after vendoring `storage/tools/`; initialization not claimed |
| `agent_kb`, `voyager`, `dilu`, `generative`, `memp`, `dynamic_cheatsheet`, `agent_workflow_memory`, `evolver` | Source present; blocked by `sentence-transformers` |
| `mobilee` | Source present; blocked by `jieba` |

```powershell
uv run python main.py memory providers
```

Optional sets:

```powershell
uv sync --extra memory-research      # expel, cerebra-style providers
uv sync --extra memory-embeddings    # sentence-transformer providers
uv sync --extra memory-evolution     # original MemEvolve engine
```

The default is `lightweight_memory`: zero optional dependencies, real upstream
code, and a working retrieval/update loop. `expel` is the stronger research
baseline when its extra is installed.

## Everyday use

### Enable memory in a normal run

```powershell
uv run python main.py --mode yolo --memory -p "Fix the failing test"
uv run python main.py --memory                                  # interactive
uv run python main.py --memory --memory-provider expel          # explicit provider
```

Memory is opt-in. The canonical record and provider update happen automatically;
you never call internal Python modules.

### Configuration

Point `IRINA_MEMORY_CONFIG` at a TOML file or pass `--config` to the memory
commands:

```toml
[memory]
enabled = true
backend = "evolvelab"
active_provider = "lightweight_memory"
learn_after_task = true
retrieve_before_task = true
max_context_tokens = 3000

[memory.provider_config.lightweight_memory]
enable_longterm_provision = true
top_k_longterm = 3

[memory.evolution]
enabled = false
auto_promote = false
require_approval = true
```

Useful environment variables: `IRINA_MEMORY_STATE_DIR`,
`IRINA_MEMORY_ENABLED`, `IRINA_MEMORY_PROVIDER`,
`IRINA_MEMORY_MAX_CONTEXT_TOKENS`, `IRINA_MEMORY_CONFIG`.

### Logging: quiet by default, informative on demand

Routine initialization, retrieval, ingestion, and persistence logs are hidden:

```powershell
uv run python main.py --memory
```

`--verbose` shows them with Irina's terminal theme:

```powershell
uv run python main.py --memory --verbose
```

| Level | Color |
| --- | --- |
| INFO | cyan |
| DEBUG | dim gray |
| SUCCESS | green |
| WARNING | yellow |
| ERROR | red |

Provider names are highlighted; counts and ingestion/retrieval results are kept.
`--no-color` and `--plain` keep the same information without ANSI. Errors remain
visible in both modes. Provider `print` output is captured per memory operation,
so it cannot corrupt streaming text or the interactive prompt.

### Inspect and administer

```powershell
uv run python main.py memory status
uv run python main.py memory --workdir . inspect
uv run python main.py memory candidates
```

`inspect` reports the canonical count and latest experiences. `candidates`
shows states, storage namespaces, evaluation evidence, and the active/previous
pointer.

## Architecture evolution

```text
analyze ──▶ generate ──▶ create ──▶ validate ──▶ evaluate ──▶ register
                                                              │
                                             explicit promote │
                                                              ▼
                                                        activate (one)
                                                              │
                                                        rollback (safe)
```

- **analyze / generate / create / validate** are the original MemEvolve stages,
  invoked through `EvolutionBridge`.
- `create` writes into an **isolated copy of `EvolveLab/`** under the evolution
  work directory. The vendored source is never modified.
- Upstream benchmark validation needs dataset runners that Irina does not
  vendor. Without `--validation-dataset`, that stage is marked `skipped`.
- Irina then runs its own gate: static AST validation, a separate-process
  import, the three-method provider contract, and an instantiation attempt.
- Candidates are registered as `validated` or `rejected`. **Nothing is activated
  automatically.**
- `memory promote --approve` performs the explicit activation; the evolved
  module is registered in-process and loaded by file path from its recorded
  build tree.
- `memory rollback` restores the previous architecture. Canonical history and
  both isolated stores remain intact.

Start an evolution run from Irina's own canonical history:

```powershell
# The original engine expects OpenAI-compatible credentials.
$env:OPENAI_API_KEY = "<key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"   # or another compatible endpoint

uv run python main.py memory evolve --from-canonical `
  --work-dir .irina/evolution `
  --max-steps 1 `
  --analysis-model deepseek-v4-pro `
  --generation-model deepseek-v4-pro
```

Use `--no-register` to inspect generated code without recording a candidate, and
`--validation-dataset <name>` only when the upstream dataset runners are
available.

## Migration and replay

Replay rebuilds a target architecture from canonical experiences in
deterministic order:

```powershell
uv run python main.py memory replay challenge-v2 --target-provider lightweight_memory
```

Properties:

- the source store and active bucket are never modified;
- the target uses isolated, workspace-scoped storage;
- a health check decides `validated` vs failure;
- failed replays mark the target `rejected`;
- replay metadata is written to `migrations/<target>.json`;
- a clean replay becomes promotable.

This is how a new schema, storage file, or provider implementation is rebuilt
without losing history.

## Activation and rollback guarantees

- Exactly **one active architecture** by default (`baseline-<provider>`).
- The registry is authoritative on restart; a new process does not silently
  replace an approved candidate with the default provider.
- Every registry mutation takes a cross-process lock, reloads from disk, and
  replaces the file atomically.
- Candidate storage is isolated per architecture and workspace.
- Failed or `rejected` candidates cannot be activated.
- `rollback` swaps to the recorded previous architecture; canonical experiences
  accumulated after activation are preserved.
- Provider selection is explicit and persisted.

## Safety model

Generated memory code is untrusted.

- Static validation rejects obvious process/network escapes (`subprocess`,
  `socket`, `ctypes`, `eval`, `exec`, and similar).
- Isolated import runs in a **separate Python process**, a temporary working
  directory, a minimal environment without provider credentials, and an
  explicit read-only `PYTHONPATH` for the candidate's `EvolveLab` tree.
- The probe checks `initialize`, `take_in_memory`, and `provide_memory`, then
  attempts construction with a temporary storage directory.
- Only an explicit `--approve` imports the candidate into the Irina process.
- Canonical experience storage is host-side and not writable by OpenShell tools.
- Provider credentials are not stored in experience records; failed updates
  retain only a bounded error description.
- Promotion is never automatic.

This is a bounded validation gate, **not a complete sandbox**. Candidate
isolation does not execute generated code inside OpenShell. Treat approval as a
review decision.

## Upstream provenance

- Repository: `https://github.com/bingreeky/MemEvolve`
- Embedded commit: `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`
- Commit date: `2026-05-05T17:29:28+08:00`
- License: Apache-2.0 (`upstream/memevolve/LICENSE`,
  `LICENSE-FLASH-SEARCHER`)
- Manifest: `upstream/memevolve/UPSTREAM_MANIFEST.json`
- Compatibility notes: `upstream/memevolve/PATCHES.md`

Vendored: `EvolveLab/`, `MemEvolve/`, `FlashOAgents/`, `storage/tools/`,
`base_agent.py`, `evolve_cli.py`, `utils.py`, and `requirements.txt`. Research
datasets, generated outputs, model caches, assets, and unrelated runners are not
copied.

Compatibility patches are limited to lazy exports
(`EvolveLab/providers/__init__.py`, `FlashOAgents/__init__.py`,
`MemEvolve/__init__.py`), lazy web/media tool imports in `base_agent.py`, and an
optional `base_dir` for `MemoryEvolver.create()`. **No upstream algorithm or
prompt was changed.**

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| No memory logs | Expected. Add `--verbose`. |
| Provider reported `blocked` | Install the matching extra; run `memory providers`. |
| `memory evolve` stops on credentials | Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. |
| Validation says `skipped` | No upstream dataset runner; this is expected unless `--validation-dataset` is used. |
| Candidate is `rejected` | Its code failed the contract gate (often a missing `initialize` or wrong `super().__init__`). Inspect `memory candidates`. |
| Memory looks empty in another project | Provider storage is workspace-scoped; use the same `--workdir` or replay from canonical history. |
| Old memory disappeared after an upgrade | Legacy shared provider storage is not auto-attributed; rebuild with `memory replay`. |
| Promotion refused | Candidate must be `validated`/`evaluated` and you must pass `--approve`. |
| Evolution summary warns about `gpt-4o-mini` | Upstream tool default; analysis continues on a compatible endpoint. |

## Tests

```powershell
uv run python -m unittest tests.test_memory_integration -v   # provider/store/replay/security
uv run python -m unittest tests.test_memory_finalization -v  # isolation, restart, evolution, activation
uv run python -m unittest tests.test_memory_logging -v       # default silence, verbose colors
uv run python -m unittest discover -s tests                  # full suite
```

Full suite at the time of writing: **378 tests, OK (4 skipped)**.

## Where to look

| File | Responsibility |
| --- | --- |
| `adapter.py` | Stable `MemoryBackend` + `EvolveLabBackend` |
| `manager.py` | Lifecycle orchestration, retrieval, updates, activation |
| `experience_store.py` | Canonical ledger and schema |
| `trajectory_bridge.py` | Run result → experience record |
| `retrieval_bridge.py` | Query, dedupe, token bound, prompt rendering |
| `provider_registry.py` | Provider discovery/probing + generated-provider loading |
| `architecture_registry.py` | Versioned states, atomic active pointer |
| `migration.py` | Deterministic replay and validation |
| `evolution_bridge.py` | Original MemEvolve stages + candidate registration |
| `security.py` | Static, isolated, and contract checks |
| `evaluation_bridge.py` | Deterministic coding fixtures and metrics |
| `cli.py` | Administration commands |
| `../../utils/memory_logging.py` | Quiet/verbose terminal presentation |
