"""Module entry point for ``python -m src.back.harness``."""

from __future__ import annotations

import sys

from .orchestration.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "memory":
        from .context_management.memory.cli import main as memory_main

        raise SystemExit(memory_main(argv[1:]))
    raise SystemExit(main())
