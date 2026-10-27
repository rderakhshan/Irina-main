"""Day 1: live verification of both required demonstrations.

Concept: run the dice task and the text-only landing-page task against the real
DeepSeek provider, then report whether each acceptance check passed.

Design constraints:
* Standard library only.
* Loads local ``.env`` values without ever printing the API key.
* Skips cleanly with a non-zero exit code when no key is configured.
* Reports evidence; it never claims success for a failed or skipped run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ._env import load_dotenv
    from .day1_dice import (
        SYSTEM_PROMPT as DICE_SYSTEM_PROMPT,
        TASK as DICE_TASK,
        RollDiceTool,
        always_allow,
        dice_completion_policy,
        print_event,
    )
except ImportError:
    from _env import load_dotenv
    from day1_dice import (
        SYSTEM_PROMPT as DICE_SYSTEM_PROMPT,
        TASK as DICE_TASK,
        RollDiceTool,
        always_allow,
        dice_completion_policy,
        print_event,
    )

from src.back.harness.loop.runner import run_loop
from src.back.harness.loop.termination import TerminationPolicy
from src.back.harness.providers.deepseek import DeepSeekProvider

LANDING_TASK = (
    "Build a landing page for a coffee shop.\n"
    "Provide the HTML and CSS directly in your answer.\n"
    "Do not write files."
)
LANDING_SYSTEM_PROMPT = (
    "You are Irina, a concise web developer. Return complete, self-contained "
    "HTML with embedded CSS in your answer. Do not use tools."
)


def _has_dice_evidence(messages: list[dict[str, Any]], result) -> bool:
    """Check for a requested dice call, its result, and a final answer."""
    requested = any(
        message.get("role") == "assistant"
        and any(
            call.get("name") == "roll_dice"
            for call in message.get("tool_calls", [])
        )
        for message in messages
    )
    rolled = any(
        message.get("role") == "tool"
        and message.get("name") == "roll_dice"
        and "rolls=" in message.get("text", "")
        for message in messages
    )
    return result.completed and requested and rolled and bool(result.text.strip())


def _has_landing_page_code(text: str) -> bool:
    """Apply a lightweight heuristic for substantial HTML and CSS output."""
    lowered = text.lower()
    has_html = "<html" in lowered or "<!doctype html" in lowered
    has_css = "<style" in lowered or "css" in lowered
    return has_html and has_css and len(text) >= 300


def main() -> int:
    """Run both live demonstrations and return a process exit code."""
    load_dotenv()
    try:
        provider = DeepSeekProvider()
    except RuntimeError as exc:
        print(f"Live verification skipped: {exc}")
        return 2

    print("=== TEST A: DICE TOOL ===")
    dice_messages: list[dict[str, Any]] = [{"role": "user", "text": DICE_TASK}]
    print("USER:")
    print(DICE_TASK)
    print()
    dice_result = run_loop(
        provider=provider,
        model=provider.default_model,
        system=DICE_SYSTEM_PROMPT,
        messages=dice_messages,
        tools={"roll_dice": RollDiceTool()},
        on_event=print_event,
        before_tool=always_allow,
        policy=TerminationPolicy(),
        completion_validator=dice_completion_policy,
    )
    test_a = _has_dice_evidence(dice_messages, dice_result)
    print(
        f"TEST A RESULT: status={dice_result.status.value} "
        f"evidence={'passed' if test_a else 'failed'}"
    )
    print()

    print("=== TEST B: TEXT-ONLY LANDING PAGE ===")
    landing_messages: list[dict[str, Any]] = [
        {"role": "user", "text": LANDING_TASK}
    ]
    print("USER:")
    print(LANDING_TASK)
    print()
    landing_result = run_loop(
        provider=provider,
        model=provider.default_model,
        system=LANDING_SYSTEM_PROMPT,
        messages=landing_messages,
        tools={},
        on_event=print_event,
        policy=TerminationPolicy(),
    )
    has_tool_calls = any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in landing_messages
    )
    has_code = _has_landing_page_code(landing_result.text)
    test_b = landing_result.completed and not has_tool_calls and has_code

    print("VERIFICATION SUMMARY:")
    print(f"  test_a_dice={'passed' if test_a else 'failed'}")
    print(f"  test_b_status={landing_result.status.value}")
    print(f"  test_b_tool_calls={'yes' if has_tool_calls else 'no'}")
    print(f"  test_b_landing_page_code={'yes' if has_code else 'no'}")
    print(f"  test_b_result={'passed' if test_b else 'failed'}")
    print(f"  overall={'passed' if test_a and test_b else 'failed'}")
    return 0 if test_a and test_b else 1


if __name__ == "__main__":
    raise SystemExit(main())
