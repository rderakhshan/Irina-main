"""Laya CPU model router.

Laya is loaded lazily and only on the trusted host. The router chooses one
catalog model identifier; it never generates tool arguments, executes tools, or
changes security policy.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..providers.catalog import ModelSpec


@dataclass(frozen=True)
class RoutingDecision:
    """One model-selection decision."""

    spec: ModelSpec
    reason: str
    source: str
    fallback: bool = False
    elapsed_seconds: float = 0.0


class LayaRouter:
    """Thin, lazy wrapper around Laya's choice-based Router."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        timeout_seconds: float | None = None,
        device: str = "cpu",
    ) -> None:
        if enabled is None:
            enabled = (os.environ.get("IRINA_LAYA_ENABLED", "true").strip().lower()
                       not in {"0", "false", "no", "off"})
        if timeout_seconds is None:
            try:
                timeout_seconds = float(
                    os.environ.get("IRINA_LAYA_TIMEOUT_SECONDS", "10")
                )
            except ValueError:
                timeout_seconds = 10.0
        self.enabled = bool(enabled)
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        # The mission requires CPU inference; never let an environment setting
        # silently move model routing onto a GPU.
        self.device = "cpu"
        self._router: Any | None = None
        self._lock = threading.Lock()
        self._warmed = False
        self.last_diagnostic: dict[str, Any] = {}

    def _ensure_router(self) -> Any:
        if self._router is not None:
            return self._router
        with self._lock:
            if self._router is not None:
                return self._router
            try:
                from laya import Router  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"Laya is not installed or cannot be imported: {type(exc).__name__}"
                ) from None
            self._router = Router(device=self.device)
            return self._router

    def _criteria(self, candidates: list[ModelSpec]) -> dict[str, str]:
        criteria: dict[str, str] = {}
        for spec in candidates:
            description = spec.description.strip() or (
                f"{spec.provider} model {spec.api_model}"
            )
            criteria[spec.id] = (
                f"{spec.provider} model {spec.api_model}. {description}"
            )
        return criteria

    def select(
        self,
        task: str,
        context: str,
        candidates: list[ModelSpec],
        fallback: ModelSpec,
    ) -> RoutingDecision:
        """Choose one actual configured model, falling back on every failure."""
        started = time.monotonic()
        if not self.enabled:
            return self._fallback(
                fallback, "Laya routing is disabled", started, source="disabled"
            )
        if not candidates:
            return self._fallback(
                fallback, "no eligible models were configured", started
            )
        if len(candidates) == 1:
            return RoutingDecision(
                spec=candidates[0],
                reason="only one eligible configured model",
                source="single",
                elapsed_seconds=time.monotonic() - started,
            )

        criteria = self._criteria(candidates)
        questions = {
            "model": {
                "type": "choice",
                "instructions": (
                    "Which configured model is the best fit for this request? "
                    "Choose the model whose strengths match the task and context."
                ),
                "criteria": criteria,
            }
        }
        state = {
            "request": str(task or ""),
            "recent_context": str(context or ""),
        }
        holder: dict[str, Any] = {}
        error: list[BaseException] = []

        def run() -> None:
            try:
                router = self._ensure_router()
                holder["value"] = router.predict(state, questions)
            except BaseException as exc:  # noqa: BLE001 - routed as a fallback
                error.append(exc)

        # The first call may have to download/load a checkpoint. Give that
        # startup work a bounded but generous window, then apply the configured
        # per-decision timeout once the router is warm.
        effective_timeout = (
            max(self.timeout_seconds, 180.0) if not self._warmed else self.timeout_seconds
        )
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(effective_timeout)
        if thread.is_alive():
            self.last_diagnostic = {
                "source": "laya",
                "fallback": True,
                "reason": "timeout",
                "selected": fallback.id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            return self._fallback(
                fallback,
                f"Laya timed out after {effective_timeout:.0f}s",
                started,
                source="timeout",
            )
        if error:
            self.last_diagnostic = {
                "source": "laya",
                "fallback": True,
                "reason": type(error[0]).__name__,
                "selected": fallback.id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            return self._fallback(
                fallback,
                f"Laya failed: {type(error[0]).__name__}",
                started,
                source="error",
            )

        result = holder.get("value")
        try:
            answer = result["answers"]["model"]["choice"]  # type: ignore[index]
        except (KeyError, TypeError):
            answer = None
        if not isinstance(answer, str) or not answer:
            return self._fallback(
                fallback, "Laya returned no model choice", started
            )
        selected = next(
            (spec for spec in candidates if spec.id == answer or spec.api_model == answer),
            None,
        )
        if selected is None:
            return self._fallback(
                fallback,
                f"Laya selected an unconfigured model {answer!r}",
                started,
            )

        self._warmed = True
        routing = result.get("routing") if isinstance(result, dict) else {}
        language = ""
        if isinstance(routing, dict):
            detection = routing.get("detection")
            if isinstance(detection, dict):
                language = str(detection.get("language") or "")
        reason = "Laya selected " + selected.id
        if language:
            reason += f" (input language: {language})"
        self.last_diagnostic = {
            "source": "laya",
            "fallback": False,
            "reason": reason,
            "selected": selected.id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return RoutingDecision(
            spec=selected,
            reason=reason,
            source="laya",
            elapsed_seconds=time.monotonic() - started,
        )

    def _fallback(
        self,
        fallback: ModelSpec,
        reason: str,
        started: float,
        *,
        source: str = "fallback",
    ) -> RoutingDecision:
        elapsed = time.monotonic() - started
        self.last_diagnostic = {
            "source": source,
            "fallback": True,
            "reason": reason,
            "selected": fallback.id,
            "elapsed_seconds": round(elapsed, 3),
        }
        return RoutingDecision(
            spec=fallback,
            reason=reason,
            source="fallback",
            fallback=True,
            elapsed_seconds=elapsed,
        )


_DEFAULT_ROUTER: LayaRouter | None = None
_DEFAULT_LOCK = threading.Lock()


def get_laya_router() -> LayaRouter:
    """Return the process-wide lazy router instance."""
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_ROUTER is None:
                _DEFAULT_ROUTER = LayaRouter()
    return _DEFAULT_ROUTER


__all__ = ["LayaRouter", "RoutingDecision", "get_laya_router"]
