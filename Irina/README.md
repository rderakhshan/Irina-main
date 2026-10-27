# Irina

Irina is a Python project for building and evaluating AI agents. Day 1 adds the
first working piece: a small agent loop that talks to DeepSeek, runs tools, and
returns a clear result for every run.

The Day 1 harness uses only the Python standard library. It does not use an agent
framework or a vendor SDK. The repo targets Python 3.13; the harness code uses
features available from Python 3.10.

## What works today

- A bounded loop with explicit stop reasons.
- A DeepSeek Chat Completions adapter.
- A simple tool interface with a permission hook.
- Clear run results, so "stopped" is not confused with "finished".
- Offline tests with a scripted model.
- A live check with a dice task and a text-only task.

The live check passed against DeepSeek for both tasks.

## Start here

| Area | What it covers | Details |
| --- | --- | --- |
| Harness | How a run is put together and how to use it | [src/back/harness/README.md](src/back/harness/README.md) |
| Loop | Turn order, stop conditions, completion, events | [src/back/harness/loop/README.md](src/back/harness/loop/README.md) |
| Providers | DeepSeek settings, message mapping, retries | [src/back/harness/providers/README.md](src/back/harness/providers/README.md) |
| Demos | Dice demo and live check | [demos/README.md](demos/README.md) |
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

## Layout

```
src/back/harness/     Agent loop, run state, and providers (Day 1)
demos/                Runnable examples and the live check
tests/                Offline unit tests
src/back/datasets/    Dataset code (later)
src/back/evaluations/ Evaluation code (later)
src/back/experiments/ Experiment tracking (later)
src/back/opik/        Opik source for local tracing and evaluation
src/front/            Frontend application
```

Only `harness/loop` and `harness/providers` are implemented. The other folders
under `src/back/harness/` are placeholders for later work.

## Opik local UI

See [docs/opik-local.md](docs/opik-local.md) for the Windows Docker Compose
setup and common commands.

## Limits

No memory, multi-agent work, skills, saved sessions, context compaction, async
runs, sandboxing, or detailed permissions yet. A run that hits a limit stops; it
is not turned into a success. The
[harness README](src/back/harness/README.md#limits) has the details.
