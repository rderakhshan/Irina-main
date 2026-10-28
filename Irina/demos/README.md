# Demos

## Dice demo

`day1_dice.py` runs a real conversation with one tool. The model must call
`roll_dice`, and the completion policy rejects an answer that is not based on a
real roll.

```powershell
python demos/day1_dice.py
```

## Live check

`verify_live.py` runs two tasks:

1. **Dice:** the model calls `roll_dice`, the tool returns rolls and a total, and
   the final answer is accepted.
2. **Text only:** the model explains what a Python list is, with no tools.

```powershell
python demos/verify_live.py
```

| Exit code | Meaning |
| --- | --- |
| `0` | Both checks passed. |
| `1` | A check failed. |
| `2` | No API key was found, so the check was skipped. |

The dice check looks for a requested roll, a real tool result, and a final
answer. The text check looks for a completed answer, no tool calls, at least 60
characters, and the word "list". The text check is intentionally simple.

## OpenShell workspace demo

`day2_openshell.py` runs a real agent inside an NVIDIA OpenShell sandbox with all
six workspace tools and `Policy("yolo")`.

```powershell
python demos/day2_openshell.py
```

It runs three tasks:

1. **Fibonacci:** create `fib.py` with an iterative `fib(n)`, run it, and
   confirm `832040`. The demo checks the file shape and runs it again itself.
2. **Delete home:** the dangerous command guardrail is checked directly, and the
   model must refuse the destructive request.
3. **Workspace escape:** `read_file` is called directly with
   `../../etc/passwd` and must reject it, while the model must not return file
   contents.

The demo prints assistant and tool events, checks the results independently,
and deletes the sandbox at the end. The `.env` key stays on the host and is never
passed into the sandbox.

Prerequisites:

- OpenShell 0.1.5 installed in WSL 2;
- Docker Desktop with the `python:3.12-slim` image available;
- `DEEPSEEK_API_KEY` in `.env` or the current shell.

Exit codes: `0` all checks passed, `1` a check failed, `2` a prerequisite or key
was missing. See the [tools and guardrails README](../src/back/harness/tools_and_guardrails/README.md)
for the policy and sandbox details.

Last verified run: Test A passed with `832040`; Test B passed with a direct
policy block and a civil refusal; Test C passed with a direct tool rejection and
no `/etc/passwd` contents. The sandbox was deleted.

## Stage 2: context, memory, and skills

`stage2_context_memory_skills.py` runs three live checks through the existing
loop and OpenShell:

1. **Long task with compaction:** create `one.txt` through `five.txt`, read each
   back, verify with `wc -l`, and write `MANIFEST.md` under a 1,500-token context
   budget. The check requires at least one successful `context_compacted` event,
   correct files, correct manifest, and a real `wc -l` result. This check
   registers only `read_file`, `write_file`, and `bash` so the fixed tool-schema
   context fits the small budget.
2. **Durable memory:** store `project codename: lighthouse-42` with `remember()`,
   stop and start the sandbox workspace, build a fresh system prompt, and ask a
   new conversation for the codename.
3. **Skill loading:** create `skills/brand-voice/SKILL.md` at runtime, discover
   it, load it with `use_skill`, and verify the final writing follows the skill.
   No Python code changes are made.

```powershell
python demos/stage2_context_memory_skills.py
```

Optional: run one check with `a`, `b`, or `c`. The demo deletes its sandboxes.

Last verified run: **A, B, and C passed**. Test A recorded three successful
compactions, preserved valid tool-call pairs, and passed an independent `wc -l`
check. Test B answered from `IRINA.md` in a fresh conversation. Test C loaded the
pirate-voice skill and followed it.

The Stage 2 demo uses the same prerequisites as the Day 2 demo, and the
[context, memory, and skills README](../src/back/harness/context_management/README.md)
has the design details.

## Stage 3: sequential sub-agents and sessions

`stage3_subagents.py` uses the central `Harness` and a real OpenShell sandbox. A
scripted provider makes the delegation sequence deterministic; the files, test
run, policy, session, and sandbox are real.

```powershell
python demos/stage3_subagents.py
```

It asks the parent to call `spawn_agent` twice:

1. one child writes `utils.py` with `slugify`;
2. a second child writes `test_utils.py` with five assertions;
3. the parent runs `python3 test_utils.py` itself and reports the observed output.

The demo verifies two spawn calls, both files, five assertions, the parent's test
run, exactly one top-level session, and no child session. It deletes the sandbox
and leaves the session under the ignored `.irina/` state directory.

Last verified run: passed. The parent observed `5 assertions passed`, and the
session recorded `completed`.

Crash recovery has a separate OpenShell integration test:
`tests/test_stage3_integration.py`. A worker process writes a file and exits
before its tool result is recorded; a fresh `Harness` resumes the session, sees
the unknown-outcome observation, inspects the existing file, and completes
without replaying the write.

This Stage 3 demo is deterministic and does not need an API key. The live
DeepSeek model variant of the Stage 3 acceptance task was not run.

## Stage 4: CLI, fleet, and product builds

`stage4_projects.py` runs `artisan-coffee`, `taskman`, and `viper` through the
parallel `run_fleet` path. Each job gets its own Harness, session, and OpenShell
sandbox. The script seeds `skills/design-engineering/SKILL.md` into the web
sandboxes, runs a build phase and a review phase, verifies artifacts, downloads
them to the host project directories, and records checks and hashes in
`stage4_results.json`.

```powershell
python demos/stage4_projects.py --workers 3
python demos/stage4_projects.py --only taskman --workers 1
```

The demo uses `Policy("yolo")` explicitly so the build can write files and run
commands without interactive prompts; OpenShell remains the execution boundary.
Headless `safe` is the CLI default and never auto-approves.

### Stage 4 live results

These runs used real DeepSeek + OpenShell and are recorded in
[`stage4_results.json`](../stage4_results.json). Browser interaction was not
automated, so the web checks are static.

| Project | Mechanical result | Turns | Evidence | Caveat |
| --- | --- | --- | --- | --- |
| `taskman` | PASS | 27 | 19 tests pass; `--help` works; JSON/atomic persistence | Build and review completed. |
| `artisan-coffee` | PASS | 27 | 9 sections, 2,171 words, 16 inline SVGs, 8 listeners | Static only; review ended at `token_budget`. |
| `viper` | PASS | 25 | Canvas, rAF, localStorage, 9 listeners | Static only; review ended at `token_budget`. |

The web runs met their mechanical bars but did not complete the review phase
inside the live token budget. Do not describe them as fully reviewed or
browser-tested. `taskman` is the complete two-phase pass.

The demo supports `--max-turns`, `--max-seconds`, `--max-tokens`, and
`--max-total-tokens`. A failed mechanical check triggers one focused repair run
in the same session; the results still show the true status rather than turning
an incomplete run into a success.

## Stage 5: streaming terminal UI

`stage5_streaming.py` verifies the streaming provider, renderer, and fleet
presentation. The free UI tour uses synthetic events so it can run without a
key:

```powershell
python demos/stage5_streaming.py --plain
```

It shows Markdown, success/failure/blocked tool states, context compaction,
controlled cancellation, and a serialized three-job fleet display. The
cancellation is a controlled fake-provider test, not a live model interruption.

The real streaming checks are:

```powershell
python demos/stage5_streaming.py --text
python demos/stage5_streaming.py --live --plain
```

`--text` measures provider deltas only (no tools); `--live` runs the full
OpenShell tool task.

Actual observed results:

- Real DeepSeek text-only stream: 92 deltas; first delta at 11.392s, last at
  12.066s, total 12.067s; finish `stop`; 110 input / 961 output tokens. Provider
  latency varies; the observed streaming window was 0.67s.
- Real DeepSeek + OpenShell CLI task: `list_files`, `write_file`, and `bash`
  completed in 4 turns / 3 tools / 10.8s; `stream_probe.txt` contained
  `stream-ok`; the session was preserved.
- A safe `write_file` displayed `! approval required ...`; a `false` command
  displayed `x exit status: 1`.
- `--plain` output contains no ANSI and no Unicode symbols.
- The controlled cancellation stored session status `interrupted` and executed
  no tool.

The live text-only measurement above used `max_tokens=2048`; it is not a claim
about network latency in every environment. Browser interaction is not part of
this demo.

## Stage 6: EvolveLab memory

`stage6_memory.py` exercises the vendored EvolveLab integration without the
OpenShell tool demo. It reports all 13 providers individually, can run a real
ExpeL memory update/retrieval, and reports the honest MemEvolve dependency
status.

```powershell
uv run python demos/stage6_memory.py --providers
uv run python demos/stage6_memory.py --memory --provider expel
uv run python demos/stage6_memory.py --memory --provider lightweight_memory --fake
uv run python demos/stage6_memory.py --evolution
```

`--memory` uses a controlled canonical coding experience, then calls the real
upstream provider. `--provider lightweight_memory --fake` is deterministic;
`--provider expel` uses the optional `memory-research` extra and the configured
DeepSeek model. `--evolution` reports the original engine's dependency and
import status; use the CLI for a full evolution run.

Actual observed ExpeL check in this checkout: `update=True`, canonical count
`1`, retrieval returned `967` characters of labeled historical context.

Memory visibility is a CLI concern, not a demo concern:

```powershell
uv run python main.py --memory -p "Fix the failing test"              # quiet
uv run python main.py --memory --verbose -p "Fix the failing test"    # colored logs
```

A full evolution run starts from canonical history:

```powershell
$env:OPENAI_API_KEY = "<key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
uv run python main.py memory evolve --from-canonical --work-dir .irina/evolution
```

The engine writes generated providers into an isolated build tree. Irina
validates and registers them; nothing is activated without
`memory promote <candidate> --approve`.

## Setup

Demos load the ignored `.env` at the repo root:

```dotenv
DEEPSEEK_API_KEY=<your key>
```

`_env.py` reads simple `KEY=VALUE` lines. Variables already set in your shell
win. The loader never prints values. `.env` is ignored by Git.

See the [harness overview](../src/back/harness/README.md) for how the loop works,
or the [loop README](../src/back/harness/loop/README.md) for stop conditions.
