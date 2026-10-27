"""Stage 3 crash worker: write a file, then terminate before tool completion.

This helper is launched by an integration test. It intentionally calls
``os._exit`` after a sandboxed write so the durable session contains a tool_start
but no tool_end. The parent test then resumes the session and checks that the
unknown outcome is reported rather than replayed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.back.harness import Harness
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.tools_and_guardrails import (
    OpenShellSandbox,
    Policy,
    Tool,
)


class CrashProvider:
    """Request one crashing tool call."""

    default_model = "crash-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("crash provider should only be called once")
        return ProviderResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="crash-call",
                    name="crash_write",
                    args={"path": "part1.txt", "content": "partial"},
                )
            ],
            finish_reason="tool_calls",
            usage=TokenUsage(1, 1),
        )


def make_crash_tool(sandbox: OpenShellSandbox) -> Tool:
    """Create a tool that writes then kills the process."""

    def crash_write(path: str, content: str) -> str:
        sandbox.write_file(path, content)
        os._exit(9)

    return Tool(
        name="crash_write",
        spec={
            "schema": {
                "name": "crash_write",
                "description": "Write a file and terminate the process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path."},
                        "content": {"type": "string", "description": "Content."},
                    },
                    "required": ["path", "content"],
                },
            }
        },
        run=crash_write,
    )


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _crash_worker.py <workdir> <state_dir>")
        return 2
    workdir = Path(sys.argv[1]).resolve()
    state_dir = Path(sys.argv[2]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    sandbox = OpenShellSandbox(workspace="stage3-crash")
    try:
        sandbox.start()
    except Exception as exc:
        print(f"OpenShell unavailable: {type(exc).__name__}: {exc}")
        return 2

    harness = Harness(
        workdir=str(workdir),
        model="crash-model",
        provider=CrashProvider(),
        policy=Policy("yolo"),
        workspace_backend=sandbox,
        state_dir=state_dir,
        extra_tools=[make_crash_tool(sandbox)],
        persist=True,
    )
    harness.run("Create part1.txt, then continue with the remaining files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
