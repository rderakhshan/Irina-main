"""Skill administration: list, promote, and discard agent-created drafts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .skills import (
    discard_skill_draft,
    draft_root,
    list_skill_drafts,
    promote_skill_draft,
    resolve_host_skills_dir,
)


def _host_dir(args: argparse.Namespace) -> Path | None:
    return resolve_host_skills_dir(args.skills_dir, args.workdir)


def _host_skills(host: Path | None) -> list[str]:
    if host is None or not host.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in host.glob("*/SKILL.md")
        if (path.parent.name not in {".", ".."})
    )


def _list(args: argparse.Namespace) -> int:
    host = _host_dir(args)
    drafts = list_skill_drafts(args.workdir, state_dir=args.state_dir)
    print(
        json.dumps(
            {
                "workdir": str(Path(args.workdir).resolve()),
                "host_dir": str(host) if host else None,
                "host_skills": _host_skills(host),
                "draft_dir": str(
                    draft_root(args.workdir, state_dir=args.state_dir)
                ),
                "drafts": [
                    {"name": name, "path": str(path)}
                    for name, path in drafts.items()
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _promote(args: argparse.Namespace) -> int:
    host = _host_dir(args)
    if host is None:
        print("A host skills directory is required.", file=sys.stderr)
        return 2
    try:
        target = promote_skill_draft(
            args.name,
            host_dir=host,
            workdir=args.workdir,
            state_dir=args.state_dir,
            force=args.force,
            keep=args.keep,
        )
    except Exception as exc:
        print(f"Promotion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Promoted {args.name} -> {target}")
    return 0


def _discard(args: argparse.Namespace) -> int:
    if not discard_skill_draft(
        args.name, workdir=args.workdir, state_dir=args.state_dir
    ):
        print(f"No draft named {args.name}.", file=sys.stderr)
        return 1
    print(f"Discarded draft {args.name}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irina skills")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument(
        "--skills-dir",
        default="skills",
        help="Host skills directory (default: ./skills).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.set_defaults(func=_list)
    promote = sub.add_parser("promote")
    promote.add_argument("name")
    promote.add_argument("--force", action="store_true")
    promote.add_argument("--keep", action="store_true")
    promote.set_defaults(func=_promote)
    discard = sub.add_parser("discard")
    discard.add_argument("name")
    discard.set_defaults(func=_discard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
