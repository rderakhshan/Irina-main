"""Stage 4: parallel, isolated fleet execution.

Concept: run several independent Harness jobs concurrently. Each job gets a
fresh Harness and therefore its own OpenShell sandbox, session, transcript, and
policy. One job failing never aborts the others.

Design constraints:
* Standard library only.
* Jobs are validated before any work starts.
* Workspaces may not overlap: two agents must never edit the same host tree.
* Results preserve input order even though execution is concurrent.
* Each job owns and cleans up its Harness unless the caller passed a shared
  backend; ``Harness.close`` already respects ownership.
* The optional review phase uses the same Harness and session as the build.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..loop.state import RunResult
from .harness import Harness

_JOB_KEYS = {"name", "workdir", "task", "review_task", "on_complete"}


class FleetError(ValueError):
    """Invalid fleet configuration."""


@dataclass(frozen=True)
class FleetJob:
    """One independent fleet task.

    Attributes:
        name: Stable display and result identifier; unique per fleet.
        workdir: Host workspace used for session identity.
        task: Initial task for the fresh Harness.
        review_task: Optional second task in the same session, used as a
            demanding review pass after the build completes.
        on_complete: Optional callback receiving the live Harness and final
            result. It may verify artifacts or download them before cleanup and
            may return metadata. Returning ``{"ok": False}`` marks the job as
            failed.
    """

    name: str
    workdir: str
    task: str
    review_task: str | None = None
    on_complete: (
        Callable[[Harness, RunResult], Mapping[str, Any] | None] | None
    ) = None


def _coerce_job(raw: Any) -> FleetJob:
    """Validate and convert one job mapping."""
    if isinstance(raw, FleetJob):
        job = raw
    elif isinstance(raw, Mapping):
        unknown = set(raw) - _JOB_KEYS
        if unknown:
            raise FleetError(
                "unknown fleet job field(s): " + ", ".join(sorted(unknown))
            )
        missing = {"name", "workdir", "task"} - set(raw)
        if missing:
            raise FleetError(
                "missing fleet job field(s): " + ", ".join(sorted(missing))
            )
        job = FleetJob(
            name=raw["name"],
            workdir=raw["workdir"],
            task=raw["task"],
            review_task=raw.get("review_task"),
            on_complete=raw.get("on_complete"),
        )
    else:
        raise FleetError("fleet jobs must be mappings or FleetJob instances")

    if not isinstance(job.name, str) or not job.name.strip():
        raise FleetError("fleet job name must be a non-empty string")
    if not isinstance(job.workdir, (str, os.PathLike)):
        raise FleetError(
            f"fleet job {job.name!r} workdir must be a path"
        )
    if not str(job.workdir).strip():
        raise FleetError(f"fleet job {job.name!r} workdir is empty")
    if not isinstance(job.task, str) or not job.task.strip():
        raise FleetError(f"fleet job {job.name!r} task is empty")
    if job.review_task is not None:
        if not isinstance(job.review_task, str) or not job.review_task.strip():
            raise FleetError(
                f"fleet job {job.name!r} review_task must be a non-empty "
                "string when provided"
            )
    if job.on_complete is not None and not callable(job.on_complete):
        raise FleetError(
            f"fleet job {job.name!r} on_complete must be callable"
        )
    return job


def _normalise(path: str | os.PathLike[str]) -> str:
    """Return a comparable absolute path."""
    return os.path.normcase(
        os.path.normpath(str(Path(path).expanduser().resolve()))
    )


def _inside(child: str, parent: str) -> bool:
    """Return whether ``child`` is equal to or below ``parent``."""
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def _validate_jobs(jobs: Sequence[FleetJob]) -> None:
    """Reject duplicate names and duplicate/overlapping workspaces."""
    names: set[str] = set()
    workspaces: list[tuple[str, str]] = []
    for job in jobs:
        if job.name in names:
            raise FleetError(f"duplicate fleet job name: {job.name!r}")
        names.add(job.name)
        workspaces.append((job.name, _normalise(job.workdir)))

    for index, (name, workspace) in enumerate(workspaces):
        for other_name, other in workspaces[index + 1 :]:
            if _inside(workspace, other) or _inside(other, workspace):
                raise FleetError(
                    "fleet jobs must use non-overlapping workspaces: "
                    f"{name!r} ({workspace}) and {other_name!r} ({other})"
                )


def _usage(result: RunResult | None) -> dict[str, int]:
    """Return token usage as a plain dictionary."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _metadata(
    job: FleetJob,
    harness: Harness | None,
    phases: list[RunResult],
    final: RunResult | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the structured result shared by success and failure paths."""
    turns = sum(int(getattr(phase, "turns", 0) or 0) for phase in phases)
    tool_calls = sum(
        int(getattr(phase, "tool_calls", 0) or 0) for phase in phases
    )
    input_tokens = sum(
        _usage(phase)["input_tokens"] for phase in phases
    )
    output_tokens = sum(
        _usage(phase)["output_tokens"] for phase in phases
    )
    session = getattr(harness, "session_path", None) if harness else None
    result: dict[str, Any] = {
        "name": job.name,
        "ok": bool(final is not None and final.completed),
        "report": final.text if final is not None else "",
        "status": final.status.value if final is not None else "error",
        "reason": final.reason if final is not None else "",
        "turns": turns,
        "tool_calls": tool_calls,
        "session": str(session) if session is not None else None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "phases": [
            {
                "status": phase.status.value,
                "turns": int(getattr(phase, "turns", 0) or 0),
                "tool_calls": int(getattr(phase, "tool_calls", 0) or 0),
            }
            for phase in phases
        ],
    }
    if extra is not None:
        result["metadata"] = dict(extra)
        if "ok" in extra:
            result["ok"] = bool(extra["ok"])
        if "report" in extra:
            result["report"] = str(extra["report"])
        extra_turns = int(extra.get("extra_turns", 0) or 0)
        extra_tool_calls = int(extra.get("extra_tool_calls", 0) or 0)
        result["turns"] += extra_turns
        result["tool_calls"] += extra_tool_calls
        extra_usage = extra.get("extra_usage")
        if isinstance(extra_usage, Mapping):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                result["usage"][key] += int(extra_usage.get(key, 0) or 0)
        extra_phases = extra.get("extra_phases")
        if isinstance(extra_phases, Sequence) and not isinstance(
            extra_phases, (str, bytes)
        ):
            result["phases"].extend(extra_phases)
    return result


def _run_one(
    job: FleetJob,
    make_harness: Callable[[FleetJob], Harness],
) -> dict[str, Any]:
    """Run one job in its own Harness and always attempt cleanup."""
    harness: Harness | None = None
    phases: list[RunResult] = []
    result = _metadata(job, None, phases, None, None)
    try:
        harness = make_harness(job)
        build = harness.run(job.task)
        phases.append(build)
        final = build
        if job.review_task is not None and build.completed:
            review = harness.continue_task(job.review_task)
            phases.append(review)
            final = review

        extra = None
        if job.on_complete is not None:
            extra = job.on_complete(harness, final)
            if extra is not None and not isinstance(extra, Mapping):
                raise FleetError(
                    f"fleet job {job.name!r} on_complete must return a "
                    "mapping or None"
                )
        result = _metadata(job, harness, phases, final, extra)
    except Exception as exc:
        result = _metadata(job, harness, phases, None, None)
        result["report"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "error"
        result["reason"] = str(exc)
    finally:
        if harness is not None:
            try:
                harness.close()
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_fleet(
    jobs: Sequence[Mapping[str, Any] | FleetJob],
    make_harness: Callable[[FleetJob], Harness],
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Run independent jobs concurrently and return ordered results.

    Args:
        jobs: Job mappings or :class:`FleetJob` instances.
        make_harness: Factory called once per job. It should return a fresh
            Harness whose workdir is ``job.workdir``.
        max_workers: Positive thread count. It is capped at the job count.

    Returns:
        One result per job, in input order. Each result contains ``name``,
        ``ok``, and ``report``, plus status, turn/tool-call counts, session,
        usage, and phase metadata.

    Raises:
        FleetError: If the fleet or any job is invalid.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise FleetError("max_workers must be a positive integer")
    if max_workers <= 0:
        raise FleetError("max_workers must be a positive integer")
    if not callable(make_harness):
        raise FleetError("make_harness must be callable")
    if jobs is None or isinstance(jobs, (str, bytes)):
        raise FleetError("jobs must be a sequence of job definitions")

    normalised = [_coerce_job(job) for job in jobs]
    if not normalised:
        return []
    _validate_jobs(normalised)

    workers = min(max_workers, len(normalised))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda job: _run_one(job, make_harness), normalised))


__all__ = ["FleetError", "FleetJob", "run_fleet"]
