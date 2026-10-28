# Context, memory, and skills

Stage 2 adds three capabilities to the existing harness:

- context compaction that keeps long tasks inside a token budget;
- durable project memory in `IRINA.md`;
- skill discovery and loading from `skills/<name>/SKILL.md`.

The code lives in this package rather than at the harness root so the existing
`context_management` area stays the single home for context-related work.
The `memory/` subpackage now also contains canonical experience storage and the
vendored EvolveLab/MemEvolve integration; see [memory/README.md](memory/README.md).

## Files

| File | Purpose |
| --- | --- |
| `context.py` | Token estimate, one-call compaction, task state, events. |
| `memory/legacy.py` | Base system prompt, `IRINA.md` loading, `remember`, memory tool. |
| `memory/` | Canonical experiences, EvolveLab adapters, replay/evolution bridges, vendored upstream source. |
| `skills.py` | Catalog, front-matter parsing, `read_skill`, `use_skill`, host seeding, and draft sync. |
| `skills_cli.py` | `skills list`, `skills promote`, and `skills discard`. |
| `__init__.py` | Public exports for the rest of the harness. |

## Context compaction

`estimate_tokens(messages)` serializes the neutral messages deterministically
and divides the character count by `CHARS_PER_TOKEN = 4`. The result is rounded
up and scaled by the most conservative provider-observed ratio. It is an
estimate, not the model tokenizer.

`compact(model, messages, budget_tokens, provider=..., task_state=..., on_event=...)`
does this:

1. Estimate the current context, including an optional fixed-context reserve.
2. Return the messages unchanged when they fit.
3. Split the transcript into atomic groups. An assistant tool call and its
   results are one group.
4. Keep the original task group. Select recent groups targeting
   `KEEP_RECENT = 6`.
5. If the recent groups leave nothing older to summarize, or the fixed context
   leaves no room, move the preservation boundary older. The summary and task
   state keep the verified work.
6. Call the provider once with the compaction instruction. The summary is
   inserted as a user message with the prefix:
   `[Conversation so far, compacted]`.
7. Insert a separate task-state message, then the preserved recent groups.
8. Recheck the estimate. If the summary is too large, clip it deterministically
   and try again. The provider is not called again.

If the result still cannot fit, `CompactionError` is raised. Pending tool calls,
orphaned tool results, and duplicate results stop compaction instead of being
repaired by inventing data.

A successful attempt emits `context_compacted` through the existing event
callback with the before/after estimates, summarized and preserved message
counts, success flag, and reason. The event never contains transcript text.

`reserve_tokens` accounts for fixed system/tool context that is not in the
message list. The live 1,500-token demonstration uses this reserve and moves the
preservation boundary to the original task when needed; the summary and task
state carry the verified file work. Test A registers only `read_file`,
`write_file`, and `bash` so the fixed tool-schema context fits the intentionally
small budget; the full six-tool set and memory/skill tools are exercised in
Stage 1 and the Stage 2 tests.

## Task state

`TaskState` tracks the goal, completed steps, pending steps, modified files, and
unresolved errors. `TaskStateObserver` updates completed steps, modified files,
and errors from actual `tool_end` events. It does not treat model claims as
proof. `pending_steps` is available for the caller to set.

The state is kept outside the summary and reinserted as a separate user message
after compaction, so the agent does not have to recover it from prose.

## Durable memory

`build_system_prompt(workdir, extra="", workspace=..., platform_name=...)`
joins:

1. the base coding-agent rules;
2. the platform and approved workspace path;
3. `Project memory (IRINA.md):` when the file exists;
4. the compact skills catalog;
5. additional task instructions.

Memory is lower-trust project context. The prompt says it cannot override the
base rules, application permissions, or OpenShell restrictions.

`remember(workdir, note, workspace=...)`:

- rejects empty notes and likely credentials;
- ignores exact duplicates;
- replaces a fact when the same `key: value` key is reused;
- keeps the file human-readable;
- caps the file at 20,000 characters and 200 facts;
- writes through the workspace backend, so production writes stay in OpenShell.

Agent-initiated writes go through the `remember` tool. In `safe` mode it needs
approval; `read-only` refuses it; `yolo` allows it but still checks dangerous
commands and OpenShell policy.

`read_memory` silently skips missing or unreadable files and truncates oversized
memory in the prompt with a visible notice. It never modifies the file.

## Skills

Skills live at `skills/<name>/SKILL.md`. `catalog` reads simple front matter:

```markdown
---
name: brand-voice
description: Friendly pirate voice for project updates.
---
```

Discovery validates names, ignores nested or malformed paths, skips unreadable
or escaping symlinks, and sorts results. `catalog_prompt` advertises only names
and short descriptions in the system prompt.

`read_skill(workdir, name)` returns the complete file when it is at most 4,000
lines and 200,000 characters. Missing, invalid, empty, oversized, or unreadable
skills return clear `ERROR:` text. The `use_skill` tool wraps this through the
existing `Tool` abstraction and is allowed in read-only mode because it only
reads instructions.

Skills are lower-trust instructions. They cannot grant permissions or change
sandbox rules.

## Installing host skills

Put skills in the host project as `skills/<name>/SKILL.md`. On the first run
Irina creates the directory with a `README.md` and `.gitkeep`, then uploads
valid files into the sandbox before the system prompt is built:

```powershell
uv run python main.py                              # uses ./skills
uv run python main.py --skills-dir .\skills
uv run python main.py --skills-dir ""     # disable upload
```

`--skills-dir` defaults to `./skills` (relative to the current directory; a
missing relative path is also looked up under `--workdir`). The whole skill
directory is uploaded, including `assets/` and `scripts/`; binary files use the
sandbox upload path. Only one level of `<name>/SKILL.md` is accepted. Invalid
names, symlinks, empty files, oversized `SKILL.md` files, and excessive asset
totals are skipped. Existing sandbox copies are replaced, so an edited host
skill takes effect on the next run or resume.

Skills that need input files or produce files use explicit transfer flags. No
host mount exists:

```powershell
uv run python main.py `
  --skills-dir .\skills `
  --input .\input-cv\resume.tex `
  --export application `
  --export-contents `
  --artifacts-dir "D:\Application" `
  --sandbox-image irina/application-tex:1 `
  --mode yolo -p "Prepare the application"
```

`--input` uploads a file to `input/<filename>` or a directory tree to
`input/<dirname>/...`. `--export` names a workspace path downloaded on clean
shutdown into `--artifacts-dir`. With `--export-contents`, a directory's
children merge directly into the host directory, so
`application/<Company>/` becomes `D:\Application\<Company>\`. Existing
targets are refused unless `--force` is passed. Use `/export` in an interactive
session to save immediately, or `--auto-export` to save after each completed
run. The image
flag selects an approved OpenShell image with extra tools. For the included
`application` skill, build [images/application](../../../../images/application/README.md)
with TeX Live, Poppler, and pypdf.

Skills created or modified by Irina during a run are not written into the
trusted folder. They are captured on clean shutdown under:

```text
<state>/skill-drafts/<workspace-key>/<name>/SKILL.md
```

Review and promote them explicitly:

```powershell
uv run python main.py skills --workdir . list
uv run python main.py skills --workdir . promote <name>
uv run python main.py skills --workdir . discard <name>
```

`promote` refuses to overwrite an existing trusted skill unless `--force` is
passed. This review gate exists because skills are instructions for future
runs: auto-persisting agent-written skills would let one bad run poison later
runs.

## Integration

- `before_turn` calls `compact`; the agent loop is unchanged.
- The system prompt is built once from base rules, location, memory, and skills.
- `memory_tools` and `skill_tools` use the existing `Tool` registry.
- `TaskStateObserver` is composed with the existing `on_event` callback.
- All file, memory, and skill operations use the workspace backend; the demo
  passes the OpenShell sandbox.
- The context engine itself runs in the trusted harness because it only handles
  conversation data.

## Security boundary

- With a sandbox backend, memory and skill reads/writes use the sandbox worker.
- The demo does not fall back to host execution when OpenShell is unavailable.
- `memory.py` and `skills.py` have a local file-only backend for tests; it is not
  an isolation boundary and cannot run commands.
- Secrets are rejected in memory and redacted from prompts, summaries, task
  state, skill catalogs, and skill output.

## Tests and demo

```powershell
python -m unittest tests.test_context tests.test_memory tests.test_skills tests.test_stage2_security -v
python demos/stage2_context_memory_skills.py
```

Current full offline suite: **316 tests, OK** with 2 Windows symlink skips. The live Stage 2
demo passed:

- **A:** five files, correct manifest, independent `wc -l` check, three
  successful compaction events, and valid tool pairs;
- **B:** memory persisted across a stopped/started sandbox and a fresh
  conversation answered from `IRINA.md`;
- **C:** a runtime-created `brand-voice` skill was discovered, loaded with
  `use_skill`, and changed the writing without Python changes.

The demo deletes its sandboxes. It uses OpenShell 0.1.5 through WSL 2, Docker,
and `python:3.12-slim`.

## Limits

No vector database, embeddings, semantic search, or second agent loop. Memory is
one bounded Markdown file. Skill discovery is lightweight and supports only
simple front matter. `pending_steps` is not inferred from model text. Compaction
failure is reported as `CompactionError`; the existing loop has no separate
status for it. The provider call used for summarization is calibrated from
usage, but its tokens are not added to the loop's `RunResult.usage`.
