"""Irina command-line entry point.

The CLI lives in ``src.back.harness.orchestration.cli`` so the package root stays
the stable import and module-execution surface.
"""

from __future__ import annotations

from typing import Sequence

from src.back.harness.orchestration.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Irina CLI."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
