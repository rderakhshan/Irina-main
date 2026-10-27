# Irina

AI evaluation harness project. Python 3.13, managed with [`uv`](https://docs.astral.sh/uv/).

## Layout

```
src/
  back/
    datasets/      Dataset definitions and loaders
    evaluations/   Evaluation logic and metrics
    experiments/   Experiment runs and tracking
    harness/       Core harness wiring
    opik/          Vendored Opik platform clone (self-hosted tracing/evals UI)
  front/           Frontend application
```

## Getting started

```powershell
uv sync
uv run main.py
```

## Run Opik locally on Windows

Opik provides the local tracing/evaluation UI used by the harness. It runs via Docker Compose.

```powershell
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate into the repository
cd opik

# Start the Opik platform
powershell -ExecutionPolicy Bypass -File .\opik.ps1
```

### Windows PowerShell 5.1 note

The upstream `opik.ps1` ships as UTF-8 **without a BOM**. Windows PowerShell 5.1 reads BOM-less
scripts using the system ANSI code page, which mangles the box-drawing/emoji characters and produces
parser errors such as:

```
Array index expression is missing or not valid.
The string is missing the terminator: ".
```

Fix it in one of two ways:

1. Use PowerShell 7+ (reads UTF-8 by default):

   ```powershell
   pwsh -ExecutionPolicy Bypass -File .\opik.ps1
   ```

2. Re-save `opik.ps1` as UTF-8 **with BOM** before running it:

   ```powershell
   $p = ".\opik.ps1"
   $b = [IO.File]::ReadAllBytes($p)
   [IO.File]::WriteAllBytes($p, [byte[]](0xEF, 0xBB, 0xBF) + $b)
   ```

   Then run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\opik.ps1
   ```

## Common Opik commands

```powershell
powershell -ExecutionPolicy Bypass -File .\opik.ps1            # start full stack
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --infra    # infrastructure only
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --verify   # health check
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --stop     # stop services
```

## Day 1: Agent execution loop

Day 1 adds a small, provider-neutral execution kernel. It is intentionally not
a framework: one bounded loop, one DeepSeek adapter, one handwritten tool, and
explicit terminal states.

```
src/back/harness/
  loop/
    runner.py       Turn cycle, tool dispatch, events, completion decisions
    state.py        RunStatus, RunResult, explicit execution counters
    termination.py  Turn/tool/time/token/stall/failure ceilings
  providers/
    base.py         Provider protocol, ToolCall, TokenUsage, typed errors
    deepseek.py     Chat Completions wire format, parsing, HTTP retries
demos/day1_dice.py  Live dice demo with a task completion policy
tests/              Offline unittest suite with a scripted provider
```

### Design highlights

- The loop depends on `Provider.complete(...)`, never on DeepSeek. Replacing the
  provider requires no loop changes.
- `run_loop` returns `RunResult`, not a string. Only `RunStatus.COMPLETED` means
  verified success.
- A response with no tool calls is a *candidate* final answer. A caller-supplied
  `completion_validator` can reject it and request more evidence.
- Tool-call IDs round-trip exactly, and every accepted call receives exactly one
  tool result, including unknown, blocked, malformed, and failed calls.
- Budgets cover turns, tool calls, elapsed monotonic time, and accumulated
  tokens. The token budget is a soft upper bound because output tokens are
  unknown until a call returns.
- HTTP 429/500/502/503/504 and temporary network failures are retried with
  bounded exponential backoff and jitter. Authentication and malformed requests
  are permanent failures. Retry exhaustion has its own exception type.
- API keys are read from the environment and never logged or placed in
  exception messages.

### Run the dice demo

The demo requires `DEEPSEEK_API_KEY`. Optional overrides are `IRINA_MODEL` and
`DEEPSEEK_BASE_URL`. Demo entry points load the repository's ignored `.env` file
without overwriting variables already exported in the shell.

Fill in the local file (never commit it):

```dotenv
DEEPSEEK_API_KEY=<your key>
```

Then run:

```powershell
python demos/day1_dice.py
```

Or export the key for the current shell instead:

```powershell
$env:DEEPSEEK_API_KEY = "<your key>"
python demos/day1_dice.py
```

Without a key the demo exits cleanly with a "live demo skipped" message. It does
not claim a live run succeeded.

### Run the offline tests

```powershell
python -m unittest discover -s tests -v
```

The tests use a scripted provider and mocked HTTP/sleep; no network access or
credentials are required.

### Live verification

After filling in `.env`, one command runs both required live demonstrations:

```powershell
python demos/verify_live.py
```

Test A is the dice task. Test B is a text-only landing-page request with no
tools:

```
Build a landing page for a coffee shop.
Provide the HTML and CSS directly in your answer.
Do not write files.
```

The verifier checks that Test A completed with real dice evidence, and that Test
B completed without tool calls while returning substantial HTML and CSS
(`<html`/`<!doctype html` plus `<style`/`css`, at least 300 characters). It exits
non-zero if either demonstration fails or if no key is configured. With no tools
configured, a useful non-empty answer is accepted as `COMPLETED`.

### Day 1 limitations

Memory, multi-agent orchestration, skill loading, persistent sessions, context
compaction, async execution, sandboxing, detailed permissions, and a second
LLM-based evaluator are intentionally out of scope. A final-summary call is also
deferred; hitting `max_turns` never turns an incomplete run into a completed one.
