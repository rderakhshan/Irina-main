# Tests

The tests run offline. They use a scripted provider and mocked HTTP, so they need
no API key and no network.

```powershell
python -m unittest discover -s tests -v
```

## What is covered

`test_deepseek.py`:

- missing API key
- message conversion and tool call IDs
- bad tool argument JSON
- usage and finish reasons
- temporary retries and permanent failures
- retry limits and `Retry-After`
- no API key in error messages

`test_loop.py`:

- normal text completion
- tool calls, order, and results
- unknown tools, exceptions, blocked calls, bad arguments
- completion validation and clarification
- turn, tool, time, and token limits
- repeated calls, empty replies, and cancellation
- provider errors and total usage
- `before_turn` replacing the transcript in place
- one tool result for every tool call

The live checks are separate. See [demos/README.md](../demos/README.md).
Back to the [harness overview](../src/back/harness/README.md).
