# Providers

A provider connects the loop to a model API. The loop only calls one method:

```python
complete(model, system, messages, tools) -> ProviderResponse
```

Streaming providers also implement:

```python
stream(model, system, messages, tools) -> Iterator[ProviderStreamEvent]
```

## Shared types

- `ProviderResponse`: `text`, `tool_calls`, `finish_reason`, `usage`, and an
  optional opaque `continuation`.
- `ToolCall`: `id`, `name`, `args`. It also has `args_error` and
  `raw_arguments` when the model sent bad JSON.
- `TokenUsage`: `input_tokens`, `output_tokens`.
- `ProviderStreamEvent`: neutral text, tool-call, finish, and error events.

Bad tool arguments are kept as data. The loop turns them into an error result and
does not run the tool. `continuation` is echoed through the neutral assistant
message as `provider_state`; it carries vendor continuation data such as Gemini
thought signatures or OpenAI reasoning items without leaking vendor fields into
tool-call contracts.

## Adapters

| Adapter | Endpoint | Credentials | Default model |
| --- | --- | --- | --- |
| `DeepSeekProvider` | Chat Completions | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` |
| `OpenAIProvider` | Responses API | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| `GeminiProvider` | `generateContent` | `GEMINI_API_KEY` | `gemini-3.5-flash` |

Optional endpoints are `DEEPSEEK_BASE_URL`, `OPENAI_BASE_URL`, and
`GEMINI_BASE_URL`. Optional default models are `DEEPSEEK_MODEL`,
`OPENAI_MODEL`, and `GEMINI_MODEL`. `IRINA_MODEL` remains a DeepSeek default.

Each adapter translates neutral messages into its own wire format:

- DeepSeek/OpenAI: assistant text, function calls, and tool results.
- Gemini: role/parts contents, `functionCall`, grouped `functionResponse`
  parts, and `thoughtSignature` continuation.

The OpenAI adapter uses the Responses API (`/responses`) with flat function
tools and Responses input items. It does not reuse DeepSeek's Chat Completions
payload.

## Catalog and factory

`catalog.py` builds the configured model list. By default it exposes one model
per provider that has credentials. `IRINA_MODEL_CATALOG` can override it with a
JSON list (or `{"models": [...]}`) of entries:

```json
{
  "id": "openai/gpt-4.1-mini",
  "provider": "openai",
  "api_model": "gpt-4.1-mini",
  "enabled": true,
  "supports_tools": true,
  "supports_streaming": true,
  "description": "General reasoning and writing."
}
```

`factory.py` creates only the selected adapter. Missing credentials for an
unused provider never prevent startup; selecting an unconfigured provider gives
a clear error.

## Routing

Laya runs on the trusted host on CPU and chooses one catalog model identifier.
It is loaded lazily and reused. See
[../routing/laya_router.py](../routing/laya_router.py). Laya never receives
credentials, never generates tool arguments, and never changes policy.

Modes:

- `auto`: Laya selects from enabled, credentialed models with the required
  capabilities.
- `default`: use `IRINA_DEFAULT_PROVIDER` / `IRINA_DEFAULT_MODEL`.
- `manual`: use `--provider` / `--model` or `/model`.

Selected models stay fixed for the whole agent loop. Follow-ups to an unfinished
task keep the current model; an independent task may route again. Manual
selection always overrides automatic routing.

## Retries

By default, temporary failures are tried up to five more times:

- HTTP 429, 500, 502, 503, 504
- connection errors and timeouts

The wait grows after each failure, has some randomness, and has a cap. A valid
`Retry-After` header is used when present. Authentication and malformed-request
errors are not retried. Permanent HTTP errors raise `ProviderHTTPError`; running
out of retries raises `ProviderRetryExhausted`. API keys are never included in
messages or logs.

See [base.py](base.py), [deepseek.py](deepseek.py), [openai.py](openai.py),
[gemini.py](gemini.py), and the [loop README](../loop/README.md). Back to the
[harness overview](../README.md).
