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

## Setup

Demos load the ignored `.env` at the repo root:

```dotenv
DEEPSEEK_API_KEY=<your key>
```

`_env.py` reads simple `KEY=VALUE` lines. Variables already set in your shell
win. The loader never prints values. `.env` is ignored by Git.

See the [harness overview](../src/back/harness/README.md) for how the loop works,
or the [loop README](../src/back/harness/loop/README.md) for stop conditions.
