# Memory: EvolveLab, MemEvolve, and canonical experiences

This package embeds the original upstream source and keeps Irina's memory
contract independent of the selected architecture.

## Upstream source

- Repository: `https://github.com/bingreeky/MemEvolve`
- Embedded commit: `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`
- Commit date: `2026-05-05T17:29:28+08:00`
- License: Apache-2.0 (`upstream/memevolve/LICENSE` and
  `LICENSE-FLASH-SEARCHER`)
- Manifest: `upstream/memevolve/UPSTREAM_MANIFEST.json`
- Compatibility notes: `upstream/memevolve/PATCHES.md`

Vendored upstream directories: `EvolveLab/`, `MemEvolve/`, `FlashOAgents/`,
`storage/tools/`, `base_agent.py`, `evolve_cli.py`, `utils.py`, and
`requirements.txt`. Research
datasets, generated outputs, model caches, assets, and unrelated runners are
not copied. Upstream compatibility patches are limited to lazy exports in
`EvolveLab/providers/__init__.py`, `FlashOAgents/__init__.py`, and
`MemEvolve/__init__.py`, lazy web/media tool imports in `base_agent.py`, and an
optional `base_dir` for `MemoryEvolver.create()`. They are documented in
`upstream/memevolve/PATCHES.md`; no algorithms or prompts were changed.

## Stable Irina interface

`MemoryBackend` in `adapter.py` exposes:

- `add_experience(ExperienceRecord) -> MemoryUpdateResult`
- `retrieve(MemoryQuery) -> list[RetrievedMemory]`
- `health_check() -> dict`
- `close()`

`EvolveLabBackend` maps those methods onto the real upstream
`take_in_memory`, `provide_memory`, and `initialize` APIs. The agent loop never
reads a provider's internal file paths or schema.

## Canonical experience history

`ExperienceStore` lives under the trusted host state root (default
`~/.irina/memory/workspaces/<workspace-id>/experiences.jsonl`). It is
append-only, idempotent by experience ID, flushed/fsynced, bounded, secret
redacted, workspace-isolated, and repairs only a torn final line. Complete
corrupt records raise instead of disappearing.

Schema version 1 records task/session/workspace identity, timestamps, task
description, bounded context, tool actions and observations with exact call
IDs, execution outcome, test evidence, verification, failure reason, run
status, and source references. Unavailable fields remain null/empty; the
bridge does not invent test outcomes.

`MemoryManager` stores canonical history before updating the provider. A
provider update failure is recorded in `updates.jsonl` and never corrupts the
completed task. The canonical history remains readable when the provider or
its storage format changes.

## Providers

The registry discovers the 13 identifiers in the upstream `PROVIDER_MAPPING`.
Status is intentionally layered: source present, importable, initialized,
storage/retrieval tested, and end-to-end tested are separate fields. Current
observed status in this checkout:

| Provider | Status |
| --- | --- |
| `lightweight_memory` | upstream source imports and initializes without optional deps; integrated with Irina |
| `expel` | imports with the `memory-research` extra; integrated with Irina using a real DeepSeek model |
| `cerebra_fusion_memory` | imports with `numpy` + `scikit-learn`; initialization not tested |
| `agent_kb`, `voyager`, `dilu`, `generative`, `memp`, `dynamic_cheatsheet`, `agent_workflow_memory`, `evolver` | source present; blocked by `sentence-transformers` |
| `skillweaver` | imports after vendoring upstream `storage/tools/`; initialization/integration not tested |
| `mobilee` | source present; blocked by `jieba` |

Do not interpret source presence as provider availability. `python main.py
memory providers` probes and reports the actual state.

The default active provider is `lightweight_memory` because it is dependency-
free and can run in the ordinary environment. `expel` is the recommended
baseline when the optional `memory-research` extra is installed. The original
evolution engine additionally needs the `memory-evolution` extra:

```powershell
uv sync --extra memory-research
uv sync --extra memory-evolution
# sentence-transformer providers additionally need:
uv sync --extra memory-embeddings
```

Provider storage is workspace-scoped:
`<state>/providers/<provider>/<workspace-id>/...`. Two projects that share one
memory state root cannot retrieve each other's provider memories; only the
canonical store and the architecture registry are shared state. Provider data
created by older Irina versions used a shared directory; it is not attributed
to a workspace automatically. Rebuild it with `memory replay` from canonical
history.

## Retrieval and lifecycle

Before a task, `MemoryManager.before_task` asks the active provider for
memories, deduplicates them, applies `max_context_tokens`, labels them as
historical lower-trust evidence, and appends that bounded block to the system
prompt. Existing `IRINA.md`, task state, skills, and compaction remain intact.
Retrieval failure is nonfatal.

After a terminal run, the trajectory bridge creates a canonical record and
`MemoryManager.after_task` updates the active provider. Memory updates happen
once per experience; failed updates remain retryable. Nothing is learned per
streamed token.

`Harness(memory_manager=...)` integrates these hooks. Memory is disabled by
default unless configured. With `--memory`, the CLI creates the manager and
uses the active provider; `--memory-provider` selects an upstream identifier.
`memory.example.toml` shows the configuration shape; point
`IRINA_MEMORY_CONFIG` at it or pass `--config` to the memory commands.

## Architecture evolution and migration

`ArchitectureRegistry` stores architecture versions/states in the trusted
state root. Every mutation takes a cross-process lock, reloads the on-disk
registry, and replaces the file atomically, so the running agent and CLI
commands cannot clobber each other. The registry is authoritative on startup:
a restart returns the approved active architecture instead of silently
creating a new baseline for the default provider. `ReplayManager` replays
canonical experiences into a separate, workspace-scoped target backend,
records counts/timings/validation, never deletes the source, and rejects failed
targets. Promotion through the CLI is explicit (`--approve`) and never
automatic.

`EvolutionBridge` invokes the original `MemEvolve.core.memory_evolver` stages:
`analyze` → `generate` → `create` → `validate`. The engine is importable with
the `memory-evolution` extra (lazy upstream exports keep web/media tools out of
the import path). `create` writes into an isolated copy of `EvolveLab/` under
the evolution work directory; the vendored source is never modified.

The upstream `validate()` stage runs benchmark dataset runners that are not
vendored with Irina. Without `--validation-dataset` the bridge marks that stage
`skipped` and runs Irina's own static + isolated-import + provider-contract
probe for each generated provider. Generated candidates are registered with
`state="validated"` when the probe passes and `state="rejected"` otherwise.
They are **never activated automatically**. On explicit
`memory promote --approve`, Irina registers the provider module in-process and
loads it by file path from the recorded implementation root, so a generated
provider with a different schema or storage file can be activated and rolled
back like any other architecture.

`evaluation_bridge.py` provides deterministic coding fixtures and metrics.

## CLI

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
```

The active selection is stored in the memory state root and is only changed by
an explicit command. Ordinary Irina execution:

```powershell
uv run python main.py -p "Fix the failing test" --mode yolo --memory
uv run python main.py -p "Fix the failing test" --mode yolo --memory --memory-provider expel
```

## Security boundaries

Upstream generated candidates are untrusted. The vendored source is never
modified by candidates; static validation rejects obvious process/network
escapes, and isolated import uses a separate Python process, a temporary cwd,
a minimal environment without provider credentials, and an explicit read-only
`PYTHONPATH` for the candidate's `EvolveLab` tree. The probe additionally
checks the `initialize` / `take_in_memory` / `provide_memory` contract and
tries to construct the provider with a temporary storage directory.
This smoke test is not a complete sandbox. Production promotion requires
explicit approval, and the original OpenShell boundary remains in force for
agent tools. After that explicit approval, an evolved provider module is
imported into the Irina process (from its recorded build tree), which is why
approval is mandatory and never automatic. Canonical experience storage is
host-side, not writable by OpenShell tools. Provider credentials are not
persisted in memory records; provider update failures retain only a bounded
error description.

Known limitation: candidate isolation does not yet execute generated code
inside OpenShell. Do not describe the smoke test as a complete security sandbox.

Live evolution uses the original engine's environment contract:
`OPENAI_API_KEY` and optional `OPENAI_BASE_URL` (an OpenAI-compatible endpoint
such as DeepSeek). The engine's trajectory summary tool hardcodes
`gpt-4o-mini`; on non-OpenAI endpoints that summary step fails while analysis
continues. Generated code is only as contract-compliant as the model output:
Irina validates and rejects non-conforming candidates instead of repairing them
silently.

## Tests

```powershell
uv run python -m unittest tests.test_memory_integration -v
uv run python -m unittest tests.test_memory_finalization -v
uv run python -m unittest discover -s tests
```
