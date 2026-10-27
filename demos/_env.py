"""Day 1: minimal ``.env`` loader for local demos.

Concept: demos should be runnable after a developer fills in a local ``.env``
file, without asking them to export variables in every new shell.

Design constraints:
* Standard library only.
* Existing process environment variables take precedence.
* Secrets are never printed, logged, or returned.
* This helper is for demo entry points; the provider itself still reads the
  process environment and knows nothing about ``.env`` files.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> bool:
    """Load simple ``KEY=VALUE`` pairs into ``os.environ``.

    Blank lines and lines beginning with ``#`` are ignored. An optional
    ``export`` prefix and matching single or double quotes around the value are
    accepted. Variables already present in the process environment are never
    overwritten.

    Args:
        path: Optional explicit file path. Defaults to ``<repo>/.env``.

    Returns:
        ``True`` when a file was read, ``False`` when it does not exist.
    """
    env_path = path if path is not None else REPO_ROOT / ".env"
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True
