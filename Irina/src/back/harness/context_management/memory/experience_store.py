"""Canonical, append-only experience storage independent of memory architectures."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ...tools_and_guardrails.security import redact_secrets

SCHEMA_VERSION = 1
MAX_RECORD_CHARS = 64_000
LOCK_STALE_SECONDS = 60


class ExperienceStoreError(RuntimeError):
    """Canonical experience storage failed or is corrupt."""


@dataclass
class ExperienceRecord:
    """Normalized completed-task experience.

    Fields not supplied by the harness remain ``None`` or empty; the bridge
    never fabricates test results, verification, or outcomes.
    """

    experience_id: str
    schema_version: int = SCHEMA_VERSION
    session_id: str | None = None
    task_id: str | None = None
    workspace_id: str | None = None
    timestamp: str | None = None
    task_description: str = ""
    relevant_context: str = ""
    tool_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    execution_outcome: str = ""
    test_results: list[dict[str, Any]] = field(default_factory=list)
    verified_success: bool | None = None
    failure_reason: str | None = None
    run_status: str | None = None
    source_event_references: list[dict[str, Any]] = field(default_factory=list)
    # Set when persistence had to drop or shorten optional evidence. Defaults
    # keep older JSONL records loadable through ``from_dict``.
    truncated: bool = False
    truncated_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperienceRecord":
        if not isinstance(value, dict):
            raise ExperienceStoreError("experience must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ExperienceStoreError(
                "unknown experience fields: " + ", ".join(sorted(unknown))
            )
        try:
            record = cls(**value)
        except TypeError as exc:
            raise ExperienceStoreError(f"invalid experience: {exc}") from None
        if not isinstance(record.experience_id, str) or not record.experience_id:
            raise ExperienceStoreError("experience_id must be a non-empty string")
        if record.schema_version != SCHEMA_VERSION:
            raise ExperienceStoreError(
                f"unsupported experience schema {record.schema_version}"
            )
        if not isinstance(record.tool_actions, list):
            raise ExperienceStoreError("tool_actions must be a list")
        if not isinstance(record.tool_observations, list):
            raise ExperienceStoreError("tool_observations must be a list")
        if not isinstance(record.test_results, list):
            raise ExperienceStoreError("test_results must be a list")
        return record


def workspace_id(workdir: str | Path) -> str:
    """Return a stable, case-normalized workspace identity."""
    value = os.path.normcase(str(Path(workdir).resolve())).casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def stable_experience_id(
    session_id: str,
    task: str,
    *,
    run_index: int = 0,
) -> str:
    """Derive an idempotent canonical ID from session identity and run index."""
    name = f"{session_id}\x00{run_index}\x00{task.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


ACTION_ARG_CHARS = 4_000
OBSERVATION_CHARS = 8_000
EVIDENCE_EXCERPT_CHARS = 800
MAX_EVIDENCE_ITEMS = 100


def _encode_record(data: dict[str, Any]) -> str:
    """Encode exactly the way persistence does."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _redact_record(value: Any) -> Any:
    """Redact every string in the record before it becomes durable."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_redact_record(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_record(item) for key, item in value.items()}
    return value


def _bounded_record(record: ExperienceRecord) -> dict[str, Any]:
    """Return a persistable record, reducing optional evidence when needed.

    Essential fields (task, outcome, identifiers, status) are never dropped.
    Tool evidence is capped per item, deduplicated, then progressively halved
    until the encoded record fits :data:`MAX_RECORD_CHARS`. Every reduction is
    recorded in ``truncated``/``truncated_fields``.
    """
    data = _redact_record(record.to_dict())
    truncated: set[str] = set()

    def clip(value: Any, limit: int, name: str) -> str:
        text = str(value or "")
        if len(text) > limit:
            truncated.add(name)
            return text[:limit]
        return text

    data["task_description"] = clip(
        data.get("task_description"), 8_000, "task_description"
    )
    data["relevant_context"] = clip(
        data.get("relevant_context"), 12_000, "relevant_context"
    )
    if data.get("failure_reason"):
        data["failure_reason"] = clip(
            data["failure_reason"], 4_000, "failure_reason"
        )

    actions: list[dict[str, Any]] = []
    for action in (data.get("tool_actions") or []):
        if not isinstance(action, dict):
            continue
        item = dict(action)
        item["args"] = clip(
            item.get("args"), ACTION_ARG_CHARS, "tool_actions"
        )
        actions.append(item)
    data["tool_actions"] = actions[:MAX_EVIDENCE_ITEMS]

    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in (data.get("tool_observations") or []):
        if not isinstance(observation, dict):
            continue
        text = str(observation.get("text") or "")
        key = " ".join(text.split())
        if key and key in seen:
            truncated.add("tool_observations")
            continue
        if key:
            seen.add(key)
        item = dict(observation)
        item["text"] = clip(text, OBSERVATION_CHARS, "tool_observations")
        observations.append(item)
    data["tool_observations"] = observations[:MAX_EVIDENCE_ITEMS]

    evidence: list[dict[str, Any]] = []
    for result in (data.get("test_results") or []):
        if not isinstance(result, dict):
            continue
        item = dict(result)
        if "excerpt" in item:
            item["excerpt"] = clip(
                item["excerpt"], EVIDENCE_EXCERPT_CHARS, "test_results"
            )
        evidence.append(item)
    data["test_results"] = evidence[:MAX_EVIDENCE_ITEMS]
    data["source_event_references"] = list(
        data.get("source_event_references") or []
    )[:MAX_EVIDENCE_ITEMS]

    # Progressively halve optional evidence, keeping the most recent entries.
    def halve(name: str) -> bool:
        items = data.get(name) or []
        if not items:
            return False
        keep = len(items) // 2
        data[name] = items[len(items) - keep :] if keep else []
        truncated.add(name)
        return True

    reduce_context = ["relevant_context"]
    task_steps = [8_000, 4_000, 2_000, 500]
    # Placeholders so the flags themselves count toward the encoded size.
    data["truncated"] = False
    data["truncated_fields"] = []

    while len(_encode_record(data)) > MAX_RECORD_CHARS:
        if halve("tool_observations"):
            continue
        if halve("tool_actions"):
            continue
        if halve("test_results"):
            continue
        if reduce_context:
            name = reduce_context.pop(0)
            data[name] = ""
            truncated.add(name)
            continue
        if task_steps:
            limit = task_steps.pop(0)
            current = str(data.get("task_description") or "")
            if len(current) > limit:
                data["task_description"] = current[:limit]
                truncated.add("task_description")
            continue
        raise ExperienceStoreError(
            "experience exceeds the bounded record size even after truncation"
        )

    data["truncated"] = bool(truncated)
    data["truncated_fields"] = sorted(truncated)
    # The field names add a few dozen bytes; trim the description if the
    # record is right at the boundary, then fail honestly.
    while len(_encode_record(data)) > MAX_RECORD_CHARS:
        current = str(data.get("task_description") or "")
        if len(current) <= 200:
            raise ExperienceStoreError(
                "experience exceeds the bounded record size even after "
                "truncation"
            )
        data["task_description"] = current[: len(current) - 200]
        truncated.add("task_description")
        data["truncated_fields"] = sorted(truncated)
    return data


class ExperienceStore:
    """Per-workspace append-only JSONL experience store.

    The authoritative store is host-side and is not exposed to OpenShell tools.
    Writes use a lock file, flush, and fsync. A torn final line is ignored and
    truncated; malformed complete lines raise instead of being silently lost.
    """

    def __init__(self, root: str | Path, workspace: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.workspace_id = workspace_id(workspace)
        self.directory = self.root / "workspaces" / self.workspace_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "experiences.jsonl"
        self.lock_path = self.directory / "experiences.lock"
        self._thread_lock = threading.RLock()
        self._ids: set[str] | None = None

    def _load_ids(self) -> set[str]:
        if self._ids is None:
            self._ids = {record.experience_id for record in self.iter_records()}
        return self._ids

    def _acquire(self) -> None:
        deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                except OSError:
                    age = 0
                if age > LOCK_STALE_SECONDS:
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise ExperienceStoreError("experience store is locked")
                time.sleep(0.05)

    def _release(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ExperienceStoreError(f"could not release store lock: {exc}")

    def add(self, record: ExperienceRecord) -> bool:
        """Append a record, returning ``False`` when its ID already exists."""
        if not isinstance(record, ExperienceRecord):
            raise TypeError("record must be an ExperienceRecord")
        payload = _bounded_record(record)
        with self._thread_lock:
            if record.experience_id in self._load_ids():
                return False
            self._acquire()
            try:
                line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                self._release()
            self._ids.add(record.experience_id)
            return True

    def _read_lines(self) -> list[str]:
        """Read records and truncate only an unparseable final line."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if lines:
            last = lines[-1].strip()
            if last:
                try:
                    json.loads(last)
                except json.JSONDecodeError:
                    lines.pop()
                    if lines:
                        self.path.write_text(
                            "\n".join(lines) + "\n", encoding="utf-8"
                        )
                    else:
                        self.path.write_text("", encoding="utf-8")
        return lines

    def iter_records(self) -> Iterator[ExperienceRecord]:
        if not self.path.exists():
            return
        lines = self._read_lines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines):
                    # Torn final write: ignore it. Do not rewrite here.
                    continue
                raise ExperienceStoreError(
                    f"corrupt canonical experience at line {index}"
                ) from None
            yield ExperienceRecord.from_dict(value)

    def load(self) -> list[ExperienceRecord]:
        return list(self.iter_records())

    def count(self) -> int:
        return len(self.load())

    def health(self) -> dict[str, Any]:
        try:
            records = self.load()
            last = records[-1].timestamp if records else None
            return {
                "ok": True,
                "path": str(self.path),
                "workspace_id": self.workspace_id,
                "count": len(records),
                "last_timestamp": last,
            }
        except Exception as exc:
            return {
                "ok": False,
                "path": str(self.path),
                "workspace_id": self.workspace_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def __len__(self) -> int:
        return self.count()
