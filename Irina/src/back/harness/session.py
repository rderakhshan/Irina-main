"""Stage 3: durable, append-only sessions and truthful crash recovery.

Concept: session files are trusted host-side JSONL records of what actually
happened. They are separate from the compacted working conversation. Recovery
uses exact tool-call IDs, preserves confirmed results, and marks unconfirmed
operations as unknown. It never invents success or automatically replays work.

Design constraints:
* Standard library only; one JSON object per line; flush plus fsync per record.
* Session state lives outside the agent-writable OpenShell workspace.
* A lock file prevents concurrent writers from silently interleaving records.
* A torn final record is ignored and truncated; corruption in the middle raises.
* Recovery observations are written to a separate sidecar, not as original
  execution history.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_management.context import TaskState, TaskStateObserver
from .loop.runner import EventType, RunEvent
from .loop.state import RunResult
from .providers.base import TokenUsage
from .tools_and_guardrails.security import redact_secrets

SESSION_DIR = "sessions"
SESSION_VERSION = 1
MAX_LABEL = 40
UNKNOWN_OUTCOME_TEXT = (
    "Execution was interrupted before its result was recorded.\n"
    "The operation may or may not have completed.\n"
    "Inspect the current workspace state before retrying."
)


class SessionError(RuntimeError):
    """A session is invalid, corrupt, inaccessible, or not resumable."""


class SessionLocked(SessionError):
    """Another writer currently holds the session lock."""


@dataclass
class SessionData:
    """Loaded session metadata, repaired conversation, and restored state.

    Attributes:
        path: Session file path.
        metadata: Validated ``session_start`` metadata.
        messages: Provider-valid repaired conversation.
        task_state: Restored session task state.
        status: Last run status, or ``interrupted``.
        reason: Last run reason.
        text: Last recorded assistant text.
        turns: Cumulative assistant messages in the durable history.
        tool_calls: Cumulative assistant tool calls.
        usage: Sum of recorded run usage.
        recovery_events: Separate recovery-sidecar records.
        records: All complete original session records.
    """

    path: Path
    metadata: dict[str, Any]
    messages: list[dict[str, Any]]
    task_state: TaskState
    status: str
    reason: str
    text: str
    turns: int
    tool_calls: int
    usage: TokenUsage
    recovery_events: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(repr=False)


def _canonical_workdir(workdir: str | Path) -> str:
    """Return the canonical workspace identity used for session mapping."""
    return str(Path(workdir).resolve())


def _workspace_key(workdir: str | Path) -> str:
    """Return a stable directory key for one workspace."""
    return hashlib.sha256(
        _canonical_workdir(workdir).encode("utf-8")
    ).hexdigest()[:16]


def _state_root(state_dir: str | Path | None = None) -> Path:
    """Return the trusted host-side state root."""
    if state_dir is not None:
        return Path(state_dir).resolve()
    configured = os.environ.get("IRINA_STATE_DIR")
    if configured:
        return Path(configured).resolve()
    return (Path.home() / ".irina").resolve()


def workspace_session_dir(
    workdir: str | Path,
    *,
    state_dir: str | Path | None = None,
) -> Path:
    """Return the trusted session directory for one workspace."""
    return _state_root(state_dir) / SESSION_DIR / _workspace_key(workdir)


def _slug(label: str) -> str:
    """Create a bounded lowercase alphanumeric/dash slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(label).lower()).strip("-")
    cleaned = cleaned[:MAX_LABEL].strip("-")
    return cleaned or "session"


def _now_iso() -> str:
    """Return a UTC timestamp suitable for JSON records."""
    return datetime.now(timezone.utc).isoformat()


def _redact_record(value: Any) -> Any:
    """Recursively redact likely credentials from record values."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _redact_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_record(item) for item in value]
    return value


def _lock_path(path: Path) -> Path:
    """Return the writer lock path for a session or sidecar file."""
    return path.with_name(path.name + ".lock")


class _writer_lock:
    """Exclusive lock file used for one append operation."""

    def __init__(self, path: Path) -> None:
        self.path = _lock_path(path)
        self._fd: int | None = None

    def __enter__(self) -> "_writer_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            raise SessionLocked(
                f"session is locked by another writer: {self.path}"
            ) from None
        os.write(self._fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def _append_unlocked(path: Path, record: dict[str, Any]) -> None:
    """Append one redacted JSON record with flush and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _redact_record(record), ensure_ascii=False
    ).encode("utf-8")
    last_byte = b""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last_byte = handle.read(1)
    with path.open("ab") as handle:
        if last_byte and last_byte != b"\n":
            handle.write(b"\n")
        handle.write(payload + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def append(path: str | Path, message: dict[str, Any]) -> None:
    """Append one JSON record to a session file.

    Each record is flushed and ``fsync``-ed. A sidecar lock file prevents two
    writers from appending at the same time. Likely credentials are redacted
    before serialization.

    Args:
        path: Session or recovery-sidecar path.
        message: JSON-serialisable record.

    Raises:
        SessionLocked: If another writer holds the lock.
        TypeError: If the record is not a dictionary.
    """
    if not isinstance(message, dict):
        raise TypeError("session record must be a dictionary")
    session_path = Path(path)
    with _writer_lock(session_path):
        _append_unlocked(session_path, message)


def new_session(
    workdir: str | Path,
    label: str = "session",
    *,
    state_dir: str | Path | None = None,
    sandbox_name: str | None = None,
    sandbox_workspace: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    """Create a new exclusive session file and write its metadata record.

    Args:
        workdir: Approved host workspace used for the stable session mapping.
        label: Human-readable label; slugged and limited to 40 characters.
        state_dir: Optional trusted state root.
        sandbox_name: OpenShell sandbox name used for later reattachment.
        sandbox_workspace: Approved workspace inside the sandbox.
        metadata: Extra trusted metadata to merge into the start record.
        path: Optional explicit session path inside this workspace's session
            directory. The file must not already exist.

    Returns:
        The new session path.

    Raises:
        SessionError: If the path is outside the trusted store or exists.
    """
    workspace = _canonical_workdir(workdir)
    directory = workspace_session_dir(workspace, state_dir=state_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if path is None:
        stamp = int(time.time())
        suffix = uuid.uuid4().hex[:8]
        session_path = directory / f"{stamp}-{_slug(label)}-{suffix}.jsonl"
    else:
        session_path = Path(path).resolve()
        if not session_path.is_relative_to(directory.resolve()):
            raise SessionError(
                "explicit session path must be inside the trusted "
                f"workspace session directory: {directory}"
            )
        if session_path.exists():
            raise SessionError(f"session already exists: {session_path}")

    record: dict[str, Any] = {
        "type": "session_start",
        "version": SESSION_VERSION,
        "session_id": uuid.uuid4().hex,
        "workspace": workspace,
        "label": str(label),
        "created_at": _now_iso(),
        "sandbox_name": sandbox_name,
        "sandbox_workspace": sandbox_workspace,
        "depth": 0,
        "parent": None,
    }
    if metadata:
        record.update(metadata)

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(session_path, flags)
    except FileExistsError:
        raise SessionError(f"session already exists: {session_path}") from None
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(
                json.dumps(_redact_record(record), ensure_ascii=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            session_path.unlink()
        except OSError:
            pass
        raise
    return session_path


def _validate_message(message: Any, index: int) -> dict[str, Any]:
    """Validate one neutral conversation message."""
    if not isinstance(message, dict):
        raise SessionError(f"message {index} must be an object")
    role = message.get("role")
    if role not in {"user", "assistant", "tool"}:
        raise SessionError(f"message {index} has invalid role {role!r}")
    if not isinstance(message.get("text", ""), str):
        raise SessionError(f"message {index} text must be a string")
    if role == "tool":
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise SessionError(f"tool message {index} is missing tool_call_id")
    if role == "assistant" and "tool_calls" in message:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            raise SessionError(f"assistant message {index} tool_calls must be a list")
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not call.get("id")
                or not isinstance(call.get("name"), str)
                or not call.get("name")
            ):
                raise SessionError(
                    f"assistant message {index} has an invalid tool call"
                )
            if not isinstance(call.get("args", {}), dict):
                raise SessionError(
                    f"assistant message {index} tool call args must be an object"
                )
    return message


def _validate_record(record: Any, line: int) -> dict[str, Any]:
    """Validate one session record type and its required fields."""
    if not isinstance(record, dict):
        raise SessionError(f"record {line} must be an object")
    record_type = record.get("type")
    if not isinstance(record_type, str):
        raise SessionError(f"record {line} is missing a type")

    if record_type == "session_start":
        if record.get("version") != SESSION_VERSION:
            raise SessionError(f"record {line} has unsupported version")
        if not isinstance(record.get("workspace"), str):
            raise SessionError(f"record {line} is missing workspace")
        if not isinstance(record.get("session_id"), str):
            raise SessionError(f"record {line} is missing session_id")
    elif record_type == "message":
        _validate_message(record.get("message"), line)
    elif record_type == "tool_start":
        if not isinstance(record.get("call_id"), str) or not record["call_id"]:
            raise SessionError(f"record {line} tool_start has no call_id")
        if not isinstance(record.get("name"), str):
            raise SessionError(f"record {line} tool_start has no name")
        if not isinstance(record.get("args", {}), dict):
            raise SessionError(f"record {line} tool_start args must be an object")
    elif record_type == "tool_end":
        if not isinstance(record.get("call_id"), str) or not record["call_id"]:
            raise SessionError(f"record {line} tool_end has no call_id")
        if not isinstance(record.get("name"), str):
            raise SessionError(f"record {line} tool_end has no name")
        if not isinstance(record.get("result"), str):
            raise SessionError(f"record {line} tool_end has no result")
    elif record_type == "context_compacted":
        if not isinstance(record.get("success"), bool):
            raise SessionError(f"record {line} compaction must have success")
    elif record_type == "task_state":
        if not isinstance(record.get("state"), dict):
            raise SessionError(f"record {line} task_state must be an object")
    elif record_type == "run_end":
        if not isinstance(record.get("status"), str):
            raise SessionError(f"record {line} run_end has no status")
    else:
        raise SessionError(
            f"record {line} has unsupported type {record_type!r}"
        )
    return record


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read complete session records, truncating one torn final tail."""
    raw = path.read_bytes()
    if not raw.strip():
        raise SessionError(f"session is empty: {path}")

    if not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        tail = raw[last_newline + 1 :]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if last_newline < 0:
                raise SessionError(
                    f"session has no complete records: {path}"
                ) from None
            with path.open("r+b") as handle:
                handle.truncate(last_newline + 1)
            raw = raw[: last_newline + 1]

    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionError(
                f"corrupt session record at line {index}: {exc}"
            ) from None
        records.append(_validate_record(record, index))

    if not records or records[0].get("type") != "session_start":
        raise SessionError("session must begin with a session_start record")
    return records


def _read_recovery(path: Path) -> list[dict[str, Any]]:
    """Read the recovery sidecar, ignoring only a torn final tail."""
    sidecar = recovery_path(path)
    if not sidecar.exists():
        return []
    raw = sidecar.read_bytes()
    if not raw.strip():
        return []
    if not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        tail = raw[last_newline + 1 :]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            keep = last_newline + 1 if last_newline >= 0 else 0
            with sidecar.open("r+b") as handle:
                handle.truncate(keep)
            raw = raw[:keep]
    if not raw.strip():
        return []
    events: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionError(
                f"corrupt recovery record at line {index}: {exc}"
            ) from None
        if not isinstance(event, dict) or event.get("type") != "recovery":
            raise SessionError(
                f"recovery record {index} is not a recovery event"
            )
        events.append(event)
    return events


def recovery_path(path: str | Path) -> Path:
    """Return the separate recovery sidecar path for a session."""
    session_path = Path(path)
    return session_path.with_name(session_path.name + ".recovery.jsonl")


def _task_state_from_dict(state: dict[str, Any]) -> TaskState:
    """Build a TaskState from a validated checkpoint dictionary."""
    def strings(key: str) -> list[str]:
        value = state.get(key, [])
        return [str(item) for item in value] if isinstance(value, list) else []

    return TaskState(
        goal=str(state.get("goal", "")),
        completed_steps=strings("completed_steps"),
        pending_steps=strings("pending_steps"),
        modified_files=strings("modified_files"),
        unresolved_errors=strings("unresolved_errors"),
    )


def _call_map(records: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index tool starts and ends by call ID, rejecting duplicates."""
    starts: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for record in records:
        if record["type"] == "tool_start":
            call_id = record["call_id"]
            if call_id in starts:
                raise SessionError(f"duplicate tool_start for {call_id}")
            starts[call_id] = record
        elif record["type"] == "tool_end":
            call_id = record["call_id"]
            if call_id in ends:
                raise SessionError(f"duplicate tool_end for {call_id}")
            ends[call_id] = record
    return starts, ends


def _repair_messages(
    messages: list[dict[str, Any]],
    starts: dict[str, dict],
    ends: dict[str, dict],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a provider-valid conversation and unknown-outcome events."""
    call_names: dict[str, str] = {}
    assistant_count = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        assistant_count += 1
        for call in message.get("tool_calls", []) or []:
            call_id = call["id"]
            if call_id in call_names:
                raise SessionError(f"duplicate assistant tool call id {call_id}")
            call_names[call_id] = call["name"]

    for call_id in starts:
        if call_id not in call_names:
            raise SessionError(f"tool_start without assistant call: {call_id}")
    for call_id in ends:
        if call_id not in call_names:
            raise SessionError(f"tool_end without assistant call: {call_id}")

    rendered: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    seen_tool_results: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            raise SessionError(
                f"orphaned tool result {message.get('tool_call_id')!r}"
            )
        rendered.append(dict(message))
        if role != "assistant" or not message.get("tool_calls"):
            index += 1
            continue

        call_ids = [call["id"] for call in message["tool_calls"]]
        existing: dict[str, dict] = {}
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            result = messages[cursor]
            call_id = result.get("tool_call_id")
            if call_id not in call_ids:
                raise SessionError(f"tool result for unknown call {call_id!r}")
            if call_id in existing or call_id in seen_tool_results:
                raise SessionError(f"duplicate tool result for {call_id}")
            existing[call_id] = result
            seen_tool_results.add(call_id)
            rendered.append(dict(result))
            cursor += 1

        for call in message["tool_calls"]:
            call_id = call["id"]
            if call_id in existing:
                continue
            recorded_end = ends.get(call_id)
            if recorded_end is not None:
                rendered.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": recorded_end.get("name", call["name"]),
                        "text": recorded_end["result"],
                    }
                )
                seen_tool_results.add(call_id)
                continue

            started = starts.get(call_id)
            text = (
                f"{UNKNOWN_OUTCOME_TEXT}\n"
                f"tool_call_id: {call_id}\n"
                f"tool: {call['name']}\n"
                f"start_recorded: {'yes' if started else 'no'}"
            )
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call["name"],
                    "text": text,
                }
            )
            unknown.append(
                {
                    "type": "recovery",
                    "call_id": call_id,
                    "name": call["name"],
                    "outcome": "unknown",
                    "start_recorded": started is not None,
                    "message": text,
                }
            )

        index = cursor

    for call_id in ends:
        if call_id not in seen_tool_results:
            raise SessionError(f"tool_end result was not rendered: {call_id}")
    return rendered, unknown


def load(path: str | Path, *, record_recovery: bool = True) -> SessionData:
    """Load, validate, and truthfully repair a session.

    A torn final JSONL line is truncated; a malformed middle record raises.
    Missing tool results become explicit unknown-outcome observations unless a
    confirmed tool_end or tool message exists. Recovery events go to a separate
    sidecar.

    Args:
        path: Session path.
        record_recovery: Append new unknown-outcome events to the sidecar.

    Returns:
        A :class:`SessionData` ready for a Harness to resume.
    """
    session_path = Path(path).resolve()
    records = _read_records(session_path)
    metadata = records[0]
    message_records = [
        record["message"] for record in records if record["type"] == "message"
    ]
    messages = [dict(message) for message in message_records]
    starts, ends = _call_map(records)
    repaired, unknown = _repair_messages(messages, starts, ends)

    existing_recovery = _read_recovery(session_path)
    existing_ids = {
        event.get("call_id")
        for event in existing_recovery
        if event.get("call_id")
    }
    if record_recovery:
        sidecar = recovery_path(session_path)
        for event in unknown:
            if event["call_id"] in existing_ids:
                continue
            append(sidecar, event)
            existing_recovery.append(event)
            existing_ids.add(event["call_id"])

    task_state: TaskState | None = None
    for record in records:
        if record["type"] == "task_state":
            task_state = _task_state_from_dict(record["state"])
    if task_state is None:
        goal = next(
            (
                str(message.get("text", ""))
                for message in messages
                if message.get("role") == "user"
            ),
            "",
        )
        task_state = TaskState(goal=goal)
    for event in unknown:
        task_state.add_error(
            f"{event['name']} ({event['call_id']}): interrupted; outcome unknown"
        )

    run_ends = [record for record in records if record["type"] == "run_end"]
    usage = TokenUsage(0, 0)
    for record in run_ends:
        raw_usage = record.get("usage")
        if isinstance(raw_usage, dict):
            usage = TokenUsage(
                usage.input_tokens + int(raw_usage.get("input_tokens", 0)),
                usage.output_tokens + int(raw_usage.get("output_tokens", 0)),
            )

    last_run_end_index = next(
        (
            index
            for index in range(len(records) - 1, -1, -1)
            if records[index]["type"] == "run_end"
        ),
        None,
    )
    active_after_end = False
    if last_run_end_index is not None:
        active_after_end = any(
            record["type"] in {"message", "tool_start", "tool_end"}
            for record in records[last_run_end_index + 1 :]
        )

    if last_run_end_index is not None and not active_after_end:
        last_run_end = records[last_run_end_index]
        status = str(last_run_end["status"])
        reason = str(last_run_end.get("reason", ""))
        text = str(last_run_end.get("text", ""))
    else:
        status = "interrupted"
        reason = (
            "session has activity after the last run_end record"
            if last_run_end_index is not None
            else "session ended before a run_end record"
        )
        text = next(
            (
                str(message.get("text", ""))
                for message in reversed(messages)
                if message.get("role") == "assistant"
                and str(message.get("text", "")).strip()
            ),
            "",
        )

    turns = sum(
        1 for message in repaired if message.get("role") == "assistant"
    )
    tool_calls = sum(
        len(message.get("tool_calls", []) or [])
        for message in repaired
        if message.get("role") == "assistant"
    )

    return SessionData(
        path=session_path,
        metadata=metadata,
        messages=repaired,
        task_state=task_state,
        status=status,
        reason=reason,
        text=text,
        turns=turns,
        tool_calls=tool_calls,
        usage=usage,
        recovery_events=existing_recovery,
        records=records,
    )


def latest(
    workdir: str | Path,
    *,
    state_dir: str | Path | None = None,
    resumable_only: bool = False,
) -> Path | None:
    """Return the newest valid top-level session for a workspace.

    Args:
        workdir: Approved host workspace.
        state_dir: Optional trusted state root.
        resumable_only: Skip sessions whose last run_end is completed. Corrupt
            sessions raise instead of being silently skipped.
    """
    directory = workspace_session_dir(workdir, state_dir=state_dir)
    if not directory.is_dir():
        return None
    workspace = _canonical_workdir(workdir)
    candidates = sorted(
        (
            path
            for path in directory.glob("*.jsonl")
            if not path.name.endswith(".recovery.jsonl")
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            records = _read_records(path)
        except SessionError:
            raise
        metadata = records[0]
        if metadata.get("workspace") != workspace:
            continue
        if metadata.get("depth", 0) != 0 or metadata.get("parent") is not None:
            continue
        if not resumable_only:
            return path
        data = load(path, record_recovery=False)
        if data.status != "completed":
            return path
    return None


@dataclass
class SessionRecorder:
    """Persist execution events and checkpoints for one top-level session.

    Args:
        path: Session file.
        task_state: Live task state updated by the Harness observer.
    """

    path: Path
    task_state: TaskState
    turns: int = 0
    tool_calls: int = 0
    status: str = "running"
    _user_counts: Counter[str] = field(default_factory=Counter)

    def record_user(self, text: str) -> None:
        """Persist one user message."""
        append(
            self.path,
            {
                "type": "message",
                "message": {"role": "user", "text": str(text)},
            },
        )
        self._user_counts[str(text)] += 1

    def seed_user_messages(self, messages: list[dict[str, Any]]) -> None:
        """Mark already-recorded user messages without writing them again."""
        self._user_counts.clear()
        for message in messages:
            if message.get("role") == "user":
                self._user_counts[str(message.get("text", ""))] += 1

    def sync_user_messages(self, messages: list[dict[str, Any]]) -> None:
        """Persist loop-added user messages that are not recorded yet.

        This intentionally does not use a positional index. Compaction may
        remove messages from the working context; recorded durable messages are
        never deleted.
        """
        seen: Counter[str] = Counter()
        for message in messages:
            if message.get("role") != "user":
                continue
            text = str(message.get("text", ""))
            if text.startswith(
                ("[Conversation so far, compacted]", "[Task state]")
            ):
                # Compaction summaries are working context, not original
                # execution history.
                continue
            seen[text] += 1
            if seen[text] > self._user_counts.get(text, 0):
                self.record_user(text)

    def _record_task_state(self) -> None:
        append(
            self.path,
            {
                "type": "task_state",
                "state": self.task_state.as_dict(),
                "turns": self.turns,
                "tool_calls": self.tool_calls,
            },
        )

    def record_event(self, event: RunEvent) -> None:
        """Persist assistant, tool, and compaction lifecycle records."""
        if event.type is EventType.ASSISTANT:
            message: dict[str, Any] = {
                "role": "assistant",
                "text": event.text,
            }
            if event.tool_calls:
                entries = []
                for call in event.tool_calls:
                    entry: dict[str, Any] = {
                        "id": call.id,
                        "name": call.name,
                        "args": dict(call.args),
                    }
                    if call.args_error is not None:
                        entry["args_error"] = call.args_error
                        if call.raw_arguments is not None:
                            entry["raw_arguments"] = call.raw_arguments
                    entries.append(entry)
                message["tool_calls"] = entries
            append(
                self.path, {"type": "message", "message": message}
            )
            self.turns += 1
        elif event.type is EventType.TOOL_START:
            call = event.tool_call
            if call is None:
                return
            append(
                self.path,
                {
                    "type": "tool_start",
                    "call_id": call.id,
                    "name": call.name,
                    "args": dict(call.args),
                    "turn": event.turn,
                },
            )
        elif event.type is EventType.TOOL_END:
            call = event.tool_call
            if call is None:
                return
            result = str(event.result or "")
            failed = result.startswith(("ERROR:", "BLOCKED:"))
            append(
                self.path,
                {
                    "type": "tool_end",
                    "call_id": call.id,
                    "name": call.name,
                    "result": result,
                    "failed": failed,
                    "turn": event.turn,
                },
            )
            append(
                self.path,
                {
                    "type": "message",
                    "message": {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "text": result,
                    },
                },
            )
            self.tool_calls += 1
            self._record_task_state()
        elif event.type is EventType.CONTEXT_COMPACTED:
            append(
                self.path,
                {
                    "type": "context_compacted",
                    "before_tokens": event.before_tokens,
                    "after_tokens": event.after_tokens,
                    "summarized_messages": event.summarized_messages,
                    "preserved_messages": event.preserved_messages,
                    "success": bool(event.compaction_succeeded),
                    "reason": event.reason,
                },
            )

    def record_run_end(self, result: RunResult) -> None:
        """Persist the authoritative terminal result for the run."""
        self.status = result.status.value
        append(
            self.path,
            {
                "type": "run_end",
                "status": result.status.value,
                "reason": result.reason,
                "text": result.text,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                },
            },
        )
        self._record_task_state()

    def record_error(self, exc: BaseException) -> None:
        """Persist an interrupted run without claiming success."""
        self.status = "interrupted"
        append(
            self.path,
            {
                "type": "run_end",
                "status": "interrupted",
                "reason": f"{type(exc).__name__}: {exc}",
                "text": "",
                "turns": self.turns,
                "tool_calls": self.tool_calls,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        )
        self.task_state.add_error(
            f"run interrupted: {type(exc).__name__}: {exc}"
        )
        self._record_task_state()


def record_resume(path: str | Path) -> None:
    """Record a resume event in the separate recovery sidecar."""
    append(
        recovery_path(path),
        {
            "type": "recovery",
            "outcome": "resumed",
            "message": "Session resumed after interruption.",
            "at": _now_iso(),
        },
    )


__all__ = [
    "SESSION_DIR",
    "SESSION_VERSION",
    "SessionData",
    "SessionError",
    "SessionLocked",
    "SessionRecorder",
    "append",
    "latest",
    "load",
    "new_session",
    "record_resume",
    "recovery_path",
    "workspace_session_dir",
]
