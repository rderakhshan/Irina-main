"""Stage 2: context compaction and session task state.

Concept: keep a long agent transcript inside a message budget without losing the
original task, verified tool outcomes, or unresolved errors. Older history is
summarized once; recent complete tool interactions are preserved.

Design constraints:
* No agent-loop rewrite. Compaction is called from the existing ``before_turn``
  hook and returns a replacement message list.
* Tool calls are never separated from their results. Pending or orphaned tool
  interactions stop compaction instead of being repaired by invention.
* The summary is a user-role message, so the provider's existing message schema
  is unchanged.
* Task state is derived from actual tool events and is kept separate from the
  conversation summary.
* Provider-reported input tokens calibrate the character heuristic.
* Secrets are redacted before transcript rendering and summary insertion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ..loop.runner import EventType, RunEvent
from ..providers.base import Provider, ProviderResponse
from ..tools_and_guardrails.security import redact_secrets

CHARS_PER_TOKEN = 4
KEEP_RECENT = 6

SUMMARY_PREFIX = "[Conversation so far, compacted]"
COMPACTION_INSTRUCTION = (
    "You compress agent transcripts. Preserve: the original task, every file "
    "created or edited and its purpose, key decisions, unresolved errors, and "
    "what remains to be done. Be dense and factual. Do not invent completed "
    "actions or test results. Clearly distinguish verified outcomes from "
    "planned work."
)

MAX_TRANSCRIPT_CHARS = 60000
MAX_SUMMARY_CHARS = 20000
MAX_TASK_STATE_CHARS = 4000
MAX_REDUCTION_ATTEMPTS = 4
CALIBRATION_CAP = 4.0

_calibration_ratio = 1.0


class CompactionError(RuntimeError):
    """Compaction could not produce a valid conversation within budget."""


def _serialized(messages: list[dict[str, Any]]) -> str:
    """Serialize messages deterministically for token estimation."""
    return json.dumps(
        messages,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate message tokens from a deterministic character count.

    The baseline is ``CHARS_PER_TOKEN`` characters per token, rounded up and
    scaled by the most conservative provider-observed ratio seen so far. This is
    an estimate, not the model's real tokenizer.

    Args:
        messages: Neutral transcript messages.

    Returns:
        Conservative estimated token count.
    """
    characters = len(_serialized(messages))
    baseline = math.ceil(characters / CHARS_PER_TOKEN)
    return math.ceil(baseline * _calibration_ratio)


def calibrate(estimated_tokens: int, actual_tokens: int) -> None:
    """Raise the conservative calibration ratio from provider usage.

    Args:
        estimated_tokens: Heuristic estimate for the request.
        actual_tokens: Provider-reported input tokens for that request.
    """
    global _calibration_ratio
    if estimated_tokens <= 0 or actual_tokens <= 0:
        return
    observed = actual_tokens / estimated_tokens
    _calibration_ratio = min(
        CALIBRATION_CAP, max(_calibration_ratio, observed)
    )


def reset_calibration() -> None:
    """Reset the heuristic calibration ratio to one."""
    global _calibration_ratio
    _calibration_ratio = 1.0


def calibration_ratio() -> float:
    """Return the current conservative calibration ratio."""
    return _calibration_ratio


def _clip(text: str, limit: int, marker: str = "[... truncated ...]") -> str:
    """Clip text to a character limit with an explicit marker."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n{marker}"


def _clean_item(value: Any) -> str:
    """Normalize and redact one task-state item."""
    return redact_secrets(" ".join(str(value).split())).strip()


@dataclass
class TaskState:
    """Session-level facts derived from actual tool events.

    Attributes:
        goal: Original task text.
        completed_steps: Verified tool actions completed.
        pending_steps: Manually recorded work that remains.
        modified_files: Files written or edited successfully.
        unresolved_errors: Tool errors, blocks, or unsuccessful run endings.
    """

    goal: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    unresolved_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the task state using the specification's field names."""
        return {
            "goal": self.goal,
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "modified_files": list(self.modified_files),
            "unresolved_errors": list(self.unresolved_errors),
        }

    @staticmethod
    def _append(
        items: list[str],
        value: Any,
        limit: int,
    ) -> None:
        """Append a unique, redacted item while keeping a bounded list."""
        cleaned = _clean_item(value)
        if not cleaned:
            return
        folded = cleaned.casefold()
        if any(item.casefold() == folded for item in items):
            return
        items.append(cleaned)
        if len(items) > limit:
            del items[: len(items) - limit]

    def add_completed(self, step: Any) -> None:
        """Record a verified completed step."""
        self._append(self.completed_steps, step, 50)

    def add_pending(self, step: Any) -> None:
        """Record a step that still needs work."""
        self._append(self.pending_steps, step, 20)

    def add_modified(self, path: Any) -> None:
        """Record a successfully modified file path."""
        self._append(self.modified_files, path, 50)

    def add_error(self, error: Any) -> None:
        """Record an unresolved error or block."""
        self._append(self.unresolved_errors, error, 20)

    def render(self) -> str:
        """Render a compact, human-readable state block."""
        lines = [f"Goal: {_clean_item(self.goal) or '(not set)'}"]
        sections = (
            ("Completed", self.completed_steps),
            ("Pending", self.pending_steps),
            ("Modified files", self.modified_files),
            ("Unresolved errors", self.unresolved_errors),
        )
        for title, items in sections:
            lines.append(f"{title}:")
            if items:
                lines.extend(f"- {item}" for item in items)
            else:
                lines.append("- (none)")
        return _clip(
            "\n".join(lines),
            MAX_TASK_STATE_CHARS,
            "[... task state truncated ...]",
        )


class TaskStateObserver:
    """Update a :class:`TaskState` from actual harness events.

    The observer records tool results, not model claims. A successful
    ``write_file`` or ``edit_file`` result adds the path to ``modified_files``.
    ``ERROR`` and ``BLOCKED`` results add unresolved errors.
    """

    def __init__(self, state: TaskState) -> None:
        self.state = state

    def __call__(self, event: RunEvent) -> None:
        """Apply one event to the task state."""
        if event.type is EventType.RUN_END:
            if event.status is not None and event.status.value != "completed":
                self.state.add_error(
                    f"run ended: {event.status.value}: {event.reason}"
                )
            return
        if event.type is not EventType.TOOL_END:
            return

        call = event.tool_call
        if call is None:
            return
        result = redact_secrets(str(event.result or ""))
        args = call.args if isinstance(call.args, dict) else {}
        if result.startswith(("ERROR:", "BLOCKED:")):
            self.state.add_error(f"{call.name}: {result}")
            return

        if call.name in {"write_file", "edit_file"}:
            path = str(args.get("path", "")).strip()
            if path:
                self.state.add_modified(path)
                self.state.add_completed(f"{call.name}: {path}")
            else:
                self.state.add_completed(call.name)
        elif call.name == "bash":
            command = " ".join(str(args.get("command", "")).split())
            self.state.add_completed(f"bash: {command[:200]}")
        elif call.name == "remember":
            self.state.add_completed("remember: project memory updated")
        elif call.name == "use_skill":
            self.state.add_completed(
                f"use_skill: {args.get('name', '')}".strip()
            )
        else:
            detail = args.get("path") or args.get("pattern") or ""
            self.state.add_completed(
                f"{call.name}: {detail}" if detail else call.name
            )


def _message_groups(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split a valid transcript into atomic message groups.

    An assistant message with tool calls is grouped with exactly its matching
    tool results. Pending calls, duplicate results, and orphaned results raise
    instead of being repaired.
    """
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise CompactionError("conversation messages must be dictionaries")
        role = message.get("role")
        if role == "tool":
            raise CompactionError("cannot compact an orphaned tool result")

        group = [message]
        if role == "assistant" and message.get("tool_calls"):
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list):
                raise CompactionError("assistant tool_calls must be a list")
            call_ids: list[str] = []
            for call in raw_calls:
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call.get("id")
                ):
                    raise CompactionError(
                        "assistant tool call is missing an id"
                    )
                call_ids.append(call["id"])
            if len(set(call_ids)) != len(call_ids):
                raise CompactionError("duplicate assistant tool call id")

            seen: set[str] = set()
            cursor = index + 1
            while (
                cursor < len(messages)
                and isinstance(messages[cursor], dict)
                and messages[cursor].get("role") == "tool"
            ):
                result = messages[cursor]
                call_id = result.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in call_ids:
                    raise CompactionError(
                        "cannot compact an orphaned tool result"
                    )
                if call_id in seen:
                    raise CompactionError("duplicate tool result")
                seen.add(call_id)
                group.append(result)
                cursor += 1
            if set(call_ids) - seen:
                raise CompactionError(
                    "cannot compact with pending tool calls"
                )
            index = cursor
        else:
            index += 1
        groups.append(group)
    return groups


def _select_slices(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int | None = None,
    reserve_tokens: int = 0,
    task_state: TaskState | None = None,
) -> tuple[
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Select the preserved task group, older history, and recent groups."""
    groups = _message_groups(messages)
    task_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        0,
    )
    task_group_index = next(
        (
            index
            for index, group in enumerate(groups)
            if any(message is messages[task_index] for message in group)
        ),
        None,
    )
    if task_group_index is None:
        raise CompactionError("could not locate the original task group")

    recent_indices: list[int] = []
    recent_count = 0
    for index in range(len(groups) - 1, -1, -1):
        if index == task_group_index:
            break
        recent_indices.insert(0, index)
        recent_count += len(groups[index])
        if recent_count >= KEEP_RECENT:
            break

    if not recent_indices:
        raise CompactionError("nothing newer than the original task to keep")

    older_indices = [
        index
        for index in range(len(groups))
        if index not in recent_indices and index != task_group_index
    ]
    if not older_indices and recent_indices:
        # KEEP_RECENT is a target. Move the preservation boundary older when
        # the recent slice consumed every non-task group, otherwise there is
        # nothing to summarize and compaction cannot reduce the context.
        older_indices = [recent_indices.pop(0)]

    if budget_tokens is not None:
        task_group_for_budget = groups[task_group_index]
        state_message = None
        if task_state is not None:
            state_message = {
                "role": "user",
                "text": f"[Task state]\n{task_state.render()}",
            }
        task_tokens = estimate_tokens(task_group_for_budget)
        state_tokens = (
            estimate_tokens([state_message]) if state_message is not None else 0
        )
        summary_allowance = max(128, min(512, budget_tokens // 3))
        margin = max(8, budget_tokens // 50)
        target = max(1, budget_tokens - margin)
        available = (
            target
            - reserve_tokens
            - int((task_tokens + state_tokens) * 2.0)
            - summary_allowance
        )
        while recent_indices:
            recent_messages = [
                message
                for index in recent_indices
                for message in groups[index]
            ]
            if int(estimate_tokens(recent_messages) * 2.0) <= available:
                break
            older_indices.append(recent_indices.pop(0))
        older_indices = sorted(set(older_indices))

    if not older_indices:
        raise CompactionError("no older history is available to summarize")

    task_group = groups[task_group_index]
    task_text = str(task_group[0].get("text", ""))
    older_messages = [
        message for index in older_indices for message in groups[index]
    ]
    recent_messages = [
        message for index in recent_indices for message in groups[index]
    ]
    return task_group, task_text, older_messages, recent_messages


def _render_transcript(
    messages: list[dict[str, Any]],
    original_task: str,
    task_state: TaskState | None,
) -> str:
    """Render older messages into a compact, redacted plain-text transcript."""
    parts = [
        "ORIGINAL TASK:",
        _clip(redact_secrets(original_task), 4000),
    ]
    if task_state is not None:
        parts.extend(
            ["TASK STATE:", _clip(task_state.render(), MAX_TASK_STATE_CHARS)]
        )
    parts.append("EARLIER MESSAGES:")

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = redact_secrets(str(message.get("text", "")))
        if role == "user":
            parts.append("USER:")
            parts.append(_clip(text, 3000))
        elif role == "assistant":
            parts.append("ASSISTANT:")
            if text.strip():
                parts.append(_clip(text, 3000))
            for call in message.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id", "")
                name = call.get("name", "")
                try:
                    args_text = json.dumps(
                        call.get("args", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                except (TypeError, ValueError):
                    args_text = str(call.get("args", {}))
                parts.append(
                    f"TOOL CALL id={call_id} name={name} "
                    f"args={_clip(args_text, 1000)}"
                )
        elif role == "tool":
            limit = (
                4000
                if text.startswith(("ERROR:", "BLOCKED:"))
                else 2000
            )
            parts.append(
                f"TOOL RESULT id={message.get('tool_call_id', '')} "
                f"name={message.get('name', '')}:"
            )
            parts.append(_clip(text, limit))
        else:
            parts.append(f"{str(role).upper()}:")
            parts.append(_clip(text, 1000))

    transcript = redact_secrets("\n".join(parts))
    return _clip(
        transcript,
        MAX_TRANSCRIPT_CHARS,
        "[... transcript truncated; task state preserved above ...]",
    )


def _build_compacted(
    task_group: list[dict[str, Any]],
    summary: str,
    task_state: TaskState | None,
    recent_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble preserved task, summary, task state, and recent messages."""
    messages = [dict(message) for message in task_group]
    messages.append(
        {
            "role": "user",
            "text": f"{SUMMARY_PREFIX}\n{summary.strip()}",
        }
    )
    if task_state is not None:
        state_text = task_state.render()
        if state_text.strip():
            messages.append(
                {"role": "user", "text": f"[Task state]\n{state_text}"}
            )
    messages.extend(dict(message) for message in recent_messages)
    return messages


def _emit_compaction(
    on_event: Callable[[RunEvent], None] | None,
    *,
    before: int,
    after: int | None,
    summarized: int,
    preserved: int,
    success: bool,
    reason: str,
) -> None:
    """Emit a context_compacted event without transcript contents."""
    if on_event is None:
        return
    on_event(
        RunEvent(
            type=EventType.CONTEXT_COMPACTED,
            turn=0,
            before_tokens=before,
            after_tokens=after,
            summarized_messages=summarized,
            preserved_messages=preserved,
            compaction_succeeded=success,
            reason=redact_secrets(reason),
        )
    )


def compact(
    model: str,
    messages: list[dict[str, Any]],
    budget_tokens: int,
    *,
    provider: Provider | None = None,
    task_state: TaskState | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    reserve_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Compact older history when the transcript exceeds ``budget_tokens``.

    Args:
        model: Model identifier used for the one summarization call.
        messages: Neutral transcript. It is not mutated.
        budget_tokens: Estimated context-token budget, including any reserve.
        provider: Existing provider used for the summary. Required when
            compaction is needed.
        task_state: Session state to preserve outside the summary.
        on_event: Optional callback for a ``context_compacted`` event.
        reserve_tokens: Estimated fixed system/tool context that is not part of
            ``messages``. It is counted against the same budget.

    Returns:
        The original list when it fits, otherwise a new message list.

    Raises:
        CompactionError: When the context cannot be compacted safely.
    """
    if budget_tokens <= 0:
        raise CompactionError("budget_tokens must be positive")

    reserve = max(0, int(reserve_tokens))
    if reserve >= budget_tokens:
        reason = "fixed system/tool context alone exceeds the budget"
        _emit_compaction(
            on_event,
            before=reserve,
            after=None,
            summarized=0,
            preserved=0,
            success=False,
            reason=reason,
        )
        raise CompactionError(reason)

    before = estimate_tokens(messages) + reserve
    if before <= budget_tokens:
        return list(messages)

    if provider is None:
        reason = "a provider is required to compact an oversized context"
        _emit_compaction(
            on_event,
            before=before,
            after=None,
            summarized=0,
            preserved=0,
            success=False,
            reason=reason,
        )
        raise CompactionError(reason)

    summarized = 0
    preserved = 0
    after: int | None = None
    try:
        task_group, original_task, older, recent = _select_slices(
            messages,
            budget_tokens=budget_tokens,
            reserve_tokens=reserve,
            task_state=task_state,
        )
        summarized = len(older)
        preserved = len(task_group) + len(recent)

        transcript = _render_transcript(
            older, original_task, task_state
        )
        summary_input = [{"role": "user", "text": transcript}]
        summary_estimate = estimate_tokens(
            [
                {"role": "system", "text": COMPACTION_INSTRUCTION},
                *summary_input,
            ]
        )
        try:
            response: ProviderResponse = provider.complete(
                model,
                COMPACTION_INSTRUCTION,
                summary_input,
                [],
            )
        except Exception as exc:
            raise CompactionError(
                "summarizer failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        calibrate(summary_estimate, response.usage.input_tokens)
        if response.finish_reason != "stop":
            raise CompactionError(
                "summarizer did not finish normally: "
                f"{response.finish_reason!r}"
            )
        summary = redact_secrets(str(response.text).strip())
        if not summary:
            raise CompactionError("summarizer returned an empty summary")
        summary = _clip(
            summary,
            MAX_SUMMARY_CHARS,
            "[... summary truncated ...]",
        )

        safety_margin = max(8, budget_tokens // 50)
        target = max(1, budget_tokens - safety_margin)
        current_summary = summary
        compacted: list[dict[str, Any]] = []
        for attempt in range(MAX_REDUCTION_ATTEMPTS + 1):
            compacted = _build_compacted(
                task_group, current_summary, task_state, recent
            )
            after = estimate_tokens(compacted) + reserve
            if after <= target:
                break
            if attempt >= MAX_REDUCTION_ATTEMPTS:
                break
            excess = after - target
            excess_chars = (
                int(
                    excess
                    * CHARS_PER_TOKEN
                    * max(_calibration_ratio, 1.0)
                )
                + 32
            )
            if len(current_summary) <= excess_chars + 64:
                break
            current_summary = (
                current_summary[:-excess_chars].rstrip()
                + "\n[... summary truncated ...]"
            )

        if after is None or after > target:
            raise CompactionError(
                "compacted context still exceeds the safety budget "
                f"({after} > {target})"
            )
    except CompactionError as exc:
        _emit_compaction(
            on_event,
            before=before,
            after=after,
            summarized=summarized,
            preserved=preserved,
            success=False,
            reason=str(exc),
        )
        raise

    reason = "compacted older history"
    if len(recent) < KEEP_RECENT:
        reason += " (preservation boundary moved to fit)"
    if current_summary != summary:
        reason += " (summary truncated to fit)"
    _emit_compaction(
        on_event,
        before=before,
        after=after,
        summarized=summarized,
        preserved=preserved,
        success=True,
        reason=reason,
    )
    return compacted


__all__ = [
    "CHARS_PER_TOKEN",
    "COMPACTION_INSTRUCTION",
    "CompactionError",
    "KEEP_RECENT",
    "SUMMARY_PREFIX",
    "TaskState",
    "TaskStateObserver",
    "calibrate",
    "calibration_ratio",
    "compact",
    "estimate_tokens",
    "reset_calibration",
]
