"""Day 1: live dice demonstration for the Irina harness.

Concept: show the full turn cycle with one handwritten tool, a printing event
callback, an always-allow authorization boundary, and a task completion policy
that requires real tool evidence before accepting an answer.

Design constraints:
* Standard library only; the DeepSeek provider is the only model adapter.
* The tool re-validates its own arguments because model schema adherence is not
  a security boundary.
* Running without ``DEEPSEEK_API_KEY`` exits cleanly instead of printing a
  traceback or claiming a live run happened.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ._env import load_dotenv
except ImportError:
    from _env import load_dotenv

from src.back.harness.loop.runner import EventType, RunEvent, run_loop
from src.back.harness.loop.termination import TerminationPolicy
from src.back.harness.providers.base import ProviderResponse, ToolCall
from src.back.harness.providers.deepseek import DeepSeekProvider

TASK = "Roll 3 dice and tell me whether the total beats 10."
SYSTEM_PROMPT = (
    "You are Irina, a concise tool-using assistant. When asked to roll dice, "
    "call the roll_dice tool instead of inventing results. Report the rolls "
    "and the total, then answer the user's question."
)


class RollDiceTool:
    """Handwritten six-sided dice roller.

    The JSON schema is advisory for the model; this method validates its own
    inputs and rejects counts outside the supported range.
    """

    spec = {
        "schema": {
            "name": "roll_dice",
            "description": "Roll count six-sided dice",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of dice to roll",
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
        }
    }

    def run(self, count: int) -> str:
        """Roll ``count`` six-sided dice and return rolls plus total.

        Args:
            count: Number of dice, from 1 to 100. Numeric strings are accepted
                because JSON models sometimes emit integers as strings.

        Returns:
            A string such as ``rolls=[4, 2, 6], total=12``.

        Raises:
            ValueError: If ``count`` is not an integer in the allowed range.
        """
        if isinstance(count, str):
            try:
                count = int(count.strip())
            except ValueError:
                raise ValueError(
                    "count must be an integer between 1 and 100"
                ) from None
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("count must be an integer between 1 and 100")
        if count < 1 or count > 100:
            raise ValueError("count must be between 1 and 100")

        rolls = [random.randint(1, 6) for _ in range(count)]
        return f"rolls={rolls}, total={sum(rolls)}"


def _format_args(args: dict[str, Any]) -> str:
    """Render tool arguments in a compact ``name=value`` form."""
    return ", ".join(f"{key}={value}" for key, value in args.items())


def print_event(event: RunEvent) -> None:
    """Print a readable, structured view of each harness event."""
    if event.type is EventType.ASSISTANT:
        printed_header = False
        if event.text.strip():
            print("ASSISTANT:")
            print(event.text)
            print()
            printed_header = True
        if event.tool_calls:
            if not printed_header:
                print("ASSISTANT:")
            for call in event.tool_calls:
                print(f"tool_call: {call.name}({_format_args(call.args)})")
            print()
    elif event.type is EventType.TOOL_END:
        call = event.tool_call
        print("TOOL:")
        if call is not None:
            print(f"{call.name} -> {event.result}")
        else:
            print(event.result)
        print()
    elif event.type is EventType.RUN_END:
        print("RUN:")
        print(f"status={event.status.value if event.status else 'unknown'}")
        print()


def always_allow(call: ToolCall) -> None:
    """Authorization boundary that permits every call in this demo."""
    return None


def dice_completion_policy(
    response: ProviderResponse,
    messages: list[dict[str, Any]],
) -> bool:
    """Accept completion only after a real dice result and a final answer."""
    rolled = any(
        message.get("role") == "tool" and message.get("name") == "roll_dice"
        for message in messages
    )
    return rolled and bool(response.text.strip())


def main() -> int:
    """Run the live dice demo and return a process exit code."""
    load_dotenv()
    try:
        provider = DeepSeekProvider()
    except RuntimeError as exc:
        print(f"Live demo skipped: {exc}")
        return 2

    messages: list[dict[str, Any]] = [{"role": "user", "text": TASK}]
    print("USER:")
    print(TASK)
    print()

    result = run_loop(
        provider=provider,
        model=provider.default_model,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools={"roll_dice": RollDiceTool()},
        on_event=print_event,
        before_tool=always_allow,
        policy=TerminationPolicy(),
        completion_validator=dice_completion_policy,
    )

    print("RUN RESULT:")
    print(f"status={result.status.value}")
    print(f"reason={result.reason}")
    print(
        f"turns={result.turns} tool_calls={result.tool_calls} "
        f"tokens={result.usage.input_tokens}+{result.usage.output_tokens}"
    )
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
