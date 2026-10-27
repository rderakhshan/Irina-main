# Irina

Irina is a Python project for building and evaluating AI agents. Day 1 built a
small agent loop that talks to DeepSeek and returns a clear result for every run.
Day 2 adds six workspace tools, an application permission policy, and execution
inside an NVIDIA OpenShell sandbox.

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
- Clear run results, so "stopped" is not confused with "finished".
- Offline tests with a scripted model.
- Live checks for the Day 1 tasks and the Day 2 OpenShell tasks.

The Day 1 live check passed against DeepSeek for both tasks. The Day 2 acceptance
demo passed Fibonacci, destructive-command, and workspace-escape checks. The full
local suite is 100 tests; 8 of them use a real OpenShell sandbox and verify
filesystem, network, process, and credential isolation.

## Start here

| Area | What it covers | Details |
| --- | --- | --- |
| Harness | How a run is put together and how to use it | [src/back/harness/README.md](src/back/harness/README.md) |
| Loop | Turn order, stop conditions, completion, events | [src/back/harness/loop/README.md](src/back/harness/loop/README.md) |
| Providers | DeepSeek settings, message mapping, retries | [src/back/harness/providers/README.md](src/back/harness/providers/README.md) |
| Tools and guardrails | Six tools, permission policy, shell, OpenShell | [src/back/harness/tools_and_guardrails/README.md](src/back/harness/tools_and_guardrails/README.md) |
| Demos | Dice, live check, OpenShell workspace demo | [demos/README.md](demos/README.md) |
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

## Layout

```
src/back/harness/                   Agent loop, providers, tools, OpenShell
demos/                              Runnable examples and live checks
tests/                              Offline unit tests and sandbox integration tests
src/back/datasets/                  Dataset code (later)
src/back/evaluations/               Evaluation code (later)
src/back/experiments/               Experiment tracking (later)
src/back/opik/                      Opik source for local tracing and evaluation
src/front/                          Frontend application
```

`harness/loop`, `harness/providers`, and `harness/tools_and_guardrails` are
implemented. The other folders under `src/back/harness/` are placeholders for
later work.

## Opik local UI

See [docs/opik-local.md](docs/opik-local.md) for the Windows Docker Compose
setup and common commands.

## Limits

No memory, multi-agent work, skills, saved sessions, context compaction, async
runs, or finer-grained permissions yet. Day 2's sandboxing is a local OpenShell
0.1.5 setup launched through WSL 2, not a production multi-tenant deployment.
The local file-only tool mode exists for tests and development; it is not an
isolation boundary. A run that hits a limit stops; it is not turned into a
success. The [harness README](src/back/harness/README.md#limits) has the details.
