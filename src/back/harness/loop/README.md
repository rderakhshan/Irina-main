# Loop

`run_loop` runs the conversation. It uses the message list you pass in and
returns a `RunResult`. Your list is changed in place, so any reference you hold
stays valid.

## One turn

1. Stop if the caller cancelled, time is up, or a limit is reached.
2. Run `before_turn` if provided. If it returns a new list, the loop copies it
   into your list.
3. Ask the provider for a reply.
4. Add the reply and token use to the run.
5. Send an `assistant` event.
6. Read the provider's finish reason.
7. Run any tool calls, or judge the answer.
8. Continue or stop.

## Finish reasons

| Reason | What the loop does |
| --- | --- |
| `stop` | Treats the text as a candidate answer. |
| `tool_calls` | Runs the calls. No calls means an invalid response. |
| `length` | Stops as `invalid_response`; the text may be cut off. |
| `content_filter` | Stops as `blocked`. |
| `insufficient_system_resource` | Stops as `provider_error`. |
| Anything else | Stops as `invalid_response`. |

## Completion

Without a validator, a non-empty `stop` answer is accepted. With a validator,
the validator decides:

- `True`: accept and finish as `completed`.
- `False`: add a short message asking for the missing evidence and continue.
- `RunStatus.AWAITING_INPUT`: stop and ask the user for information.

The loop does not use a special "finish" tool and does not run a second model to
judge the first one.

## Tool calls

Tool calls run in order. Before each call, the loop checks cancellation, time,
the tool-call budget, repeated calls, and the failure count.

If the run stops in the middle of a batch, the remaining calls get
`BLOCKED: ...` results. That way every call has a result and the transcript is
still valid. No tool runs after a hard stop.

| Case | Result text |
| --- | --- |
| Unknown tool | `ERROR: unknown tool <name>` |
| Blocked by `before_tool` | `BLOCKED: <reason>` |
| Tool raised an exception | `ERROR: <Type>: <message>` |
| Bad JSON arguments | `ERROR: invalid tool arguments: ...` |

`before_tool` is the permission check. The loop never skips it.

## Events

`on_event` receives `RunEvent` objects:

- `assistant`: text and any tool calls.
- `tool_start`: the call that is about to run.
- `tool_end`: the call and its result.
- `run_end`: the final status and reason.

## Stopping

- `should_cancel` is checked before each turn and each tool call.
- Elapsed time is measured with a clock that only moves forward.
- `max_turns` stops without asking for a final summary. The run stays
  `max_turns`.
- Token use adds up across every provider call.

Read [state.py](state.py) and [termination.py](termination.py) for the exact
statuses and defaults. Back to the [harness overview](../README.md).
