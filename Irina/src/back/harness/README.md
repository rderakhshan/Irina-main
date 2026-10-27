# Harness

The harness runs one conversation from start to finish. It asks the model for a
reply, runs any tools the model asks for, and stops for a clear reason.

The loop is deliberately small. Day 1 added one model adapter and a dice demo;
Day 2 adds six workspace tools and an OpenShell sandbox without changing the
loop.

## Files

| File | Purpose |
| --- | --- |
| `loop/runner.py` | Runs turns, handles tool calls, emits events. |
| `loop/state.py` | Defines run statuses, the result object, and counters. |
| `loop/termination.py` | Sets limits for turns, tools, time, tokens, repeats, and failures. |
| `providers/base.py` | Defines what a provider must implement and the shared data types. |
| `providers/deepseek.py` | Talks to the DeepSeek Chat Completions API. |
| `tools_and_guardrails/` | Six workspace tools, permission policy, shell, and OpenShell adapter. |

## How a run works

1. Check whether the caller cancelled and whether time or a budget is used up.
2. Give an optional `before_turn` hook a chance to change the transcript.
3. Ask the provider for a reply.
4. Add the reply and its token use to the run.
5. If the reply asks for tools, run them one at a time.
6. If the reply looks final, ask the completion policy whether to accept it.
7. Continue until the run completes or stops.

A reply with no tool calls is only a candidate answer. For work that needs proof
(a dice roll, a changed file, a passing test), pass a `completion_validator`. It
can reject the answer and ask the model to keep working. If no validator is set,
a non-empty text answer is accepted.

## Results

`run_loop` returns a `RunResult`. `status` tells you what happened. Only
`completed` means success.

| Status | Meaning |
| --- | --- |
| `completed` | The answer was accepted. |
| `awaiting_input` | The task needs information from the user. |
| `max_turns` | The model used all its turns. |
| `max_tool_calls` | The tool-call limit was reached. |
| `timeout` | The time limit was reached. |
| `token_budget` | The token limit was reached. |
| `stalled` | The model repeated calls or sent empty replies. |
| `provider_error` | The model API failed. |
| `invalid_response` | The reply could not be used. |
| `tool_error` | Too many tool calls failed in a row. |
| `cancelled` | The caller stopped the run. |
| `blocked` | A policy or the provider blocked the work. |

The transcript and the last assistant text are kept for every result.

## Tools

A tool is any object with a `spec` and a `run` method:

```python
tool.spec = {"schema": {"name": "roll_dice", "description": "...", "parameters": {...}}}
tool.run(count=3) -> "rolls=[4, 2, 6], total=12"
```

`before_tool(call)` is the permission check. Return `None` to allow the call, or
a reason string to block it. A blocked call still gets a tool result.

Unknown tools, bad arguments, and tool exceptions become error results. They do
not crash the run. Every tool call gets exactly one matching tool result before
the next model request.

Day 2 uses `tools_and_guardrails/` for its six tools. The caller passes
`Policy.check` as the `before_tool` hook, and the demo runs the tools inside
OpenShell. The loop itself is unchanged.

## Limits

The defaults are in `loop/termination.py`: 40 turns, 100 tool calls, 300 seconds,
and 200,000 tokens. Token use is checked before and after each model call. It is
a soft limit because the size of the reply is unknown until it arrives.

Repeated identical calls and empty replies are warnings that can be tuned or
turned off. A run that hits a limit stops with that status; it is not marked
complete.

## More detail

- [Loop](loop/README.md) — turn order, completion rules, events.
- [Providers](providers/README.md) — model adapter and retries.
- [Tools and guardrails](tools_and_guardrails/README.md) — six tools, policy, OpenShell.
- [Demos](../../../demos/README.md) — dice demo, live check, OpenShell workspace demo.
- [Tests](../../../tests/README.md) — offline test coverage.
- [Project overview](../../../README.md) — the short version.
