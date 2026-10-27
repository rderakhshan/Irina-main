"""Module entry point for ``python -m src.back.harness``."""

from __future__ import annotations

from .orchestration.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
