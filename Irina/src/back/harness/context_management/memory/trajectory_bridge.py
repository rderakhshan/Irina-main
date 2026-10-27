"""Convert Irina runs and durable sessions into canonical and upstream trajectories."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...loop.state import RunResult
from ...tools_and_guardrails.security import redact_secrets
from .experience_store import (
    ExperienceRecord,
    stable_experience_id,
    workspace_id,
)

_TEST_PATTERNS = (
    re.compile(r"\b(?:OK|PASSED|SUCCESS)\b", re.IGNORECASE),
    re.compile(r"\bRan\s+\d+\s+tests?\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+tests?\s+passed\b", re.IGNORECASE),
    re.compile(r"\bexit\s+(?:status|code)\s*[:=]\s*0\b", re.IGNORECASE),
    re.compile(r"\(exit\s+0\b", re.IGNORECASE),
)
_FAIL_PATTERNS = (
    re.compile(r"\bFAILED\b", re.IGNORECASE),
    re.compile(r"\b(?:exit\s+(?:status|code)\s*[:=]\s*[1-9])\b", re.IGNORECASE),
    re.compile(r"\(exit\s+[1-9]\b", re.IGNORECASE),
    re.compile(r"\bTraceback\s+\(most recent call last\)", re.IGNORECASE),
)


def _clip(value: str, limit: int) -> str:
    value = redact_secrets(str(value))
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _message_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            actions.append(
                {
                    "id": call_id,
                    "name": str(call.get("name", "")),
                    "args": redact_secrets(
                        str(call.get("args", {}))
                    )[:4_000],
                    "args_error": call.get("args_error"),
                }
            )
    return actions[:100]


def _message_observations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        observations.append(
            {
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "text": _clip(message.get("text", ""), 12_000),
            }
        )
    return observations[:100]


def _classify_evidence(observations: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    positive = False
    negative = False
    for observation in observations:
        text = str(observation.get("text", ""))
        name = str(observation.get("name", ""))
        if name not in {"bash", "shell", "powershell", "pwsh"}:
            continue
        failed = any(pattern.search(text) for pattern in _FAIL_PATTERNS)
        passed = any(pattern.search(text) for pattern in _TEST_PATTERNS) and not failed
        if passed or failed:
            evidence.append(
                {
                    "kind": "command",
                    "tool": name,
                    "passed": bool(passed),
                    "excerpt": _clip(text, 800),
                }
            )
        positive = positive or passed
        negative = negative or failed
    return (positive and not negative), evidence


def record_from_result(
    *,
    task: str,
    result: RunResult,
    workspace: str | Path,
    session_path: str | Path | None = None,
    session_id: str | None = None,
    run_index: int = 0,
    task_state: Any | None = None,
    verified_success: bool | None = None,
) -> ExperienceRecord:
    """Build a canonical record from a final ``RunResult``.

    Success is based on observed command/test evidence where possible. It is
    never inferred from the assistant's prose. A completed run without test
    evidence is stored with ``verified_success=False``.
    """
    messages = list(result.messages)
    actions = _message_calls(messages)
    observations = _message_observations(messages)
    observed_success, test_results = _classify_evidence(observations)
    if verified_success is None:
        verified_success = bool(result.completed and observed_success)
    elif verified_success and not observed_success:
        # The caller may have external validator evidence; accept it only when
        # explicitly supplied, but leave the observed tests empty.
        verified_success = True

    identifier = session_id
    if not identifier and session_path:
        identifier = Path(session_path).stem
    identifier = identifier or str(uuid.uuid4())
    task_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{workspace_id(workspace)}\x00{task.strip()}",
        )
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    context = ""
    if task_state is not None and callable(getattr(task_state, "render", None)):
        context = _clip(task_state.render(), 12_000)

    return ExperienceRecord(
        experience_id=stable_experience_id(
            identifier, task, run_index=run_index
        ),
        session_id=identifier,
        task_id=task_id,
        workspace_id=workspace_id(workspace),
        timestamp=timestamp,
        task_description=_clip(task, 8_000),
        relevant_context=context,
        tool_actions=actions,
        tool_observations=observations,
        execution_outcome=result.status.value,
        test_results=test_results,
        verified_success=verified_success,
        failure_reason=(
            _clip(result.reason, 4_000)
            if not result.completed
            else None
        ),
        run_status=result.status.value,
        source_event_references=[
            {
                "session_path": (
                    str(session_path) if session_path is not None else None
                ),
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "status": result.status.value,
            }
        ],
    )


def upstream_payload(record: ExperienceRecord) -> dict[str, Any]:
    """Convert a canonical record to the EvolveLab ``TrajectoryData`` shape."""
    trajectory: list[dict[str, Any]] = [
        {"type": "task", "content": record.task_description}
    ]
    for action in record.tool_actions:
        trajectory.append(
            {
                "type": "action",
                "tool_call_id": action.get("id"),
                "name": action.get("name"),
                "content": action.get("args"),
            }
        )
    for observation in record.tool_observations:
        trajectory.append(
            {
                "type": "observation",
                "tool_call_id": observation.get("tool_call_id"),
                "name": observation.get("name"),
                "content": observation.get("text"),
            }
        )
    return {
        "query": record.task_description,
        "trajectory": trajectory,
        "result": record.execution_outcome,
        "metadata": {
            "experience_id": record.experience_id,
            "session_id": record.session_id,
            "workspace_id": record.workspace_id,
            "is_correct": bool(record.verified_success),
            "task_success": bool(record.verified_success),
            "outcome": (
                "success" if record.verified_success else "failure"
            ),
            "test_results": record.test_results,
            "run_status": record.run_status,
        },
    }
