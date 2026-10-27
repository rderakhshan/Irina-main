# Providers

A provider connects the loop to a model API. The loop only calls one method:

```python
complete(model, system, messages, tools) -> ProviderResponse
```

## Shared types

- `ProviderResponse`: `text`, `tool_calls`, `finish_reason`, `usage`.
- `ToolCall`: `id`, `name`, `args`. It also has `args_error` and
  `raw_arguments` when the model sent bad JSON.
- `TokenUsage`: `input_tokens`, `output_tokens`.

Bad tool arguments are kept as data. The loop turns them into an error result and
does not run the tool.

## DeepSeek adapter

`DeepSeekProvider` calls `https://api.deepseek.com/chat/completions` by default.
It reads:

| Variable | Required | Use |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Yes | API key. |
| `IRINA_MODEL` | No | Default model. |
| `DEEPSEEK_BASE_URL` | No | Different endpoint. |

Defaults: temperature `0.4`, max tokens `8192`, no streaming. The constructor
can change these.

The adapter converts the loop's messages to DeepSeek messages and back. Tool call
IDs are kept exactly. Tool results use the matching ID. Reasoning fields and
Gemini-only fields are not passed through.

## Retries

By default, temporary failures are tried up to five more times:

- HTTP 429, 500, 502, 503, 504
- connection errors and timeouts

The wait grows after each failure, has some randomness, and has a cap. A valid
`Retry-After` header is used when present. These are not retried:

- authentication errors such as 401
- malformed requests such as 400

Permanent HTTP errors raise `ProviderHTTPError`. Running out of retries raises
`ProviderRetryExhausted`. Error messages include the status and a short response
body, but never the API key.

## Adding another provider

Implement `complete(...)` and return the shared types. Keep vendor-specific
fields inside your adapter. The loop does not change.

See [base.py](base.py), [deepseek.py](deepseek.py), and the
[loop README](../loop/README.md). Back to the [harness overview](../README.md).
