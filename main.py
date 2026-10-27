"""Minimal Stage 3 entry point.

Runs the deterministic sequential sub-agent demonstration through the central
Harness. The full command-line interface is intentionally left for a later stage.
"""

from __future__ import annotations

from demos.stage3_subagents import main as demo_main


def main() -> int:
    """Run the Stage 3 Harness demonstration."""
    return demo_main()


if __name__ == "__main__":
    raise SystemExit(main())
