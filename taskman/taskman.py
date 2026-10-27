#!/usr/bin/env python3
"""A small, production-quality command-line task manager.

Tasks are persisted as JSON.  Writes are atomic (temporary file plus
``os.replace``) so a crash or full disk never leaves a partially-written
database behind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_DATA_PATH = "tasks.json"
VALID_PRIORITIES = ("low", "normal", "high")
DATE_FORMAT = "%Y-%m-%d"


class TaskError(Exception):
    """An expected, user-facing error (message is safe to print directly)."""


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Load the task list from *path*.

    A missing file is treated as an empty list.  Malformed JSON, or JSON with
    the wrong top-level shape, raises :class:`TaskError` and the file is left
    untouched.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise TaskError(f"cannot read {path}: {exc}") from exc

    if not raw.strip():
        raise TaskError(
            f"malformed JSON in {path}: file is empty (file left unchanged)"
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskError(
            f"malformed JSON in {path}: {exc} (file left unchanged)"
        ) from exc

    if not isinstance(data, list):
        raise TaskError(f"malformed JSON in {path}: expected a JSON list (file left unchanged)")

    for item in data:
        if not isinstance(item, dict):
            raise TaskError(
                f"malformed JSON in {path}: every task must be an object (file left unchanged)"
            )
        if isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int):
            raise TaskError(
                f"malformed JSON in {path}: task object is missing an integer 'id' (file left unchanged)"
            )
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise TaskError(
                f"malformed JSON in {path}: task object has an invalid 'title' (file left unchanged)"
            )
        if item.get("priority") not in VALID_PRIORITIES:
            raise TaskError(
                f"malformed JSON in {path}: task object has an invalid 'priority' (file left unchanged)"
            )
        if not isinstance(item.get("status"), str) or item["status"] not in ("pending", "done"):
            raise TaskError(
                f"malformed JSON in {path}: task object has an invalid 'status' (file left unchanged)"
            )
        due = item.get("due")
        if due is not None:
            if not isinstance(due, str) or not _is_valid_date(due):
                raise TaskError(
                    f"malformed JSON in {path}: task object has an invalid 'due' date (file left unchanged)"
                )

    return data


def _default_file_mode() -> int:
    """Return the mode a normal ``open(..., 'w')`` would use (umask applied)."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _fsync_directory(directory: str) -> None:
    """Best-effort directory fsync so a rename is durable on POSIX."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_tasks(path: str, tasks: List[Dict[str, Any]]) -> None:
    """Atomically persist *tasks* to *path*.

    The payload is written to a temporary file in the same directory and then
    moved into place with ``os.replace``, which is atomic on POSIX and Windows
    for same-volume replacements.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise TaskError(f"cannot create directory for {path}: {exc}") from exc

    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".taskman-", suffix=".tmp", dir=directory, text=True
        )
    except OSError as exc:
        raise TaskError(f"cannot create temporary file for {path}: {exc}") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # ``mkstemp`` creates files with mode 0600; restore normal umask
        # semantics so the final data file is not unexpectedly restrictive.
        os.chmod(tmp_path, _default_file_mode())
        os.replace(tmp_path, path)
        _fsync_directory(directory)
    except OSError as exc:
        raise TaskError(f"cannot write {path}: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def next_id(tasks: List[Dict[str, Any]]) -> int:
    """Return the next stable integer id for a task list."""
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1


def _is_valid_date(value: str) -> bool:
    """Return True when *value* is a strict ``YYYY-MM-DD`` calendar date."""
    try:
        parsed = datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        return False
    # ``strptime`` accepts non-zero-padded values such as ``2025-1-1``;
    # round-tripping enforces the exact required format.
    return value == parsed.strftime(DATE_FORMAT)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _due_type(value: str) -> str:
    """argparse type callable that validates ``YYYY-MM-DD`` dates."""
    if not _is_valid_date(value):
        raise argparse.ArgumentTypeError(f"invalid date {value!r} (expected YYYY-MM-DD)")
    return value


def _get_known_ids(tasks: List[Dict[str, Any]]) -> set[int]:
    return {task["id"] for task in tasks}


def cmd_add(args: argparse.Namespace, data_path: str) -> int:
    title = args.title.strip()
    if not title:
        raise TaskError("title must not be empty")

    tasks = load_tasks(data_path)
    task = {
        "id": next_id(tasks),
        "title": title,
        "priority": args.priority,
        "due": args.due,
        "status": "pending",
    }
    tasks.append(task)
    save_tasks(data_path, tasks)

    due_text = f" due {task['due']}" if task["due"] else ""
    print(
        f"Added task #{task['id']}: {task['title']} "
        f"[{task['priority']}{due_text}]"
    )
    return 0


def cmd_list(args: argparse.Namespace, data_path: str) -> int:
    tasks = load_tasks(data_path)
    if args.pending:
        tasks = [task for task in tasks if task["status"] == "pending"]

    if not tasks:
        state = "pending " if args.pending else ""
        print(f"No {state}tasks.")
        return 0

    # Headers are sized from the widest value in each column so the table is
    # always aligned regardless of task content.
    id_width = max(len(str(task["id"])) for task in tasks)
    id_width = max(id_width, len("ID"))
    status_width = max(len(task["status"]) for task in tasks)
    status_width = max(status_width, len("STATUS"))
    priority_width = max(len(task["priority"]) for task in tasks)
    priority_width = max(priority_width, len("PRIORITY"))
    due_width = max(len(task["due"] or "-") for task in tasks)
    due_width = max(due_width, len("DUE"))

    header = (
        f"{'ID':>{id_width}}  {'STATUS':<{status_width}}  "
        f"{'PRIORITY':<{priority_width}}  {'DUE':<{due_width}}  TITLE"
    )
    print(header)
    print(
        f"{'-' * id_width}  {'-' * status_width}  "
        f"{'-' * priority_width}  {'-' * due_width}  -----"
    )
    for task in tasks:
        due = task["due"] or "-"
        print(
            f"{task['id']:>{id_width}}  {task['status']:<{status_width}}  "
            f"{task['priority']:<{priority_width}}  {due:<{due_width}}  {task['title']}"
        )
    return 0


def cmd_done(args: argparse.Namespace, data_path: str) -> int:
    tasks = load_tasks(data_path)
    known = _get_known_ids(tasks)
    ids = list(dict.fromkeys(args.ids))
    id_set = set(ids)
    unknown = sorted(task_id for task_id in id_set if task_id not in known)
    if unknown:
        formatted = ", ".join(str(task_id) for task_id in unknown)
        raise TaskError(f"unknown task id(s): {formatted}")

    for task in tasks:
        if task["id"] in id_set:
            task["status"] = "done"
    save_tasks(data_path, tasks)

    formatted = ", ".join(str(task_id) for task_id in ids)
    print(f"Marked task(s) #{formatted} as done.")
    return 0


def cmd_rm(args: argparse.Namespace, data_path: str) -> int:
    tasks = load_tasks(data_path)
    known = _get_known_ids(tasks)
    ids = list(dict.fromkeys(args.ids))
    id_set = set(ids)
    unknown = sorted(task_id for task_id in id_set if task_id not in known)
    if unknown:
        formatted = ", ".join(str(task_id) for task_id in unknown)
        raise TaskError(f"unknown task id(s): {formatted}")

    remaining = [task for task in tasks if task["id"] not in id_set]
    save_tasks(data_path, remaining)

    formatted = ", ".join(str(task_id) for task_id in ids)
    print(f"Removed task(s) #{formatted}.")
    return 0


def cmd_stats(args: argparse.Namespace, data_path: str) -> int:
    tasks = load_tasks(data_path)
    total = len(tasks)
    pending = sum(1 for task in tasks if task["status"] == "pending")
    completed = total - pending

    print(f"Total:      {total}")
    print(f"Pending:    {pending}")
    print(f"Completed:  {completed}")
    print("By priority:")
    for priority in VALID_PRIORITIES:
        count = sum(1 for task in tasks if task["priority"] == priority)
        print(f"  {priority:<8} {count}")
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="taskman",
        description="A small command-line task manager.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="path to the JSON data file (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="add a task")
    add_parser.add_argument("title", help="task title")
    add_parser.add_argument(
        "--priority",
        choices=VALID_PRIORITIES,
        default="normal",
        help="task priority (default: %(default)s)",
    )
    add_parser.add_argument(
        "--due",
        type=_due_type,
        default=None,
        help="due date as YYYY-MM-DD",
    )
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="list tasks")
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--all",
        action="store_true",
        help="show all tasks (default)",
    )
    list_group.add_argument(
        "--pending",
        action="store_true",
        help="show only pending tasks",
    )
    list_parser.set_defaults(func=cmd_list)

    done_parser = subparsers.add_parser("done", help="mark tasks complete")
    done_parser.add_argument(
        "ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="one or more task ids",
    )
    done_parser.set_defaults(func=cmd_done)

    rm_parser = subparsers.add_parser("rm", help="remove tasks")
    rm_parser.add_argument(
        "ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="one or more task ids",
    )
    rm_parser.set_defaults(func=cmd_rm)

    stats_parser = subparsers.add_parser("stats", help="show task statistics")
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args, args.data)
    except TaskError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
