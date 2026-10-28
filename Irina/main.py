"""Irina command-line entry point.

The CLI lives in ``src.back.harness.orchestration.cli`` so the package root stays
the stable import and module-execution surface.
"""

from __future__ import annotations

import sys
from typing import Sequence

from src.back.harness.orchestration.cli import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Irina CLI or an administration command."""
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] == "memory":
        from src.back.harness.context_management.memory.cli import (
            main as memory_main,
        )

        return memory_main(values[1:])
    if values and values[0] == "skills":
        from src.back.harness.context_management.skills_cli import (
            main as skills_main,
        )

        return skills_main(values[1:])
    return cli_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
