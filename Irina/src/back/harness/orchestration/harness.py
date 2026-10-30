"""Stage 3: central Harness composition, sessions, and sub-agents.

Concept: ``Harness`` wires the existing loop, provider, tools, policy, context
engine, memory, skills, sessions, and OpenShell adapter. It owns coordination,
not a second implementation of any of those components.

Design constraints:
* One synchronous run at a time; standard library only.
* Sessions are trusted host-side JSONL and separate from working context.
* `safe` permissions are the default; OpenShell remains the execution boundary.
* Child agents share the parent backend, policy, and delegation budget.
* The existing ``RunResult`` contract is preserved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ..context_management.context import (
    TaskState,
    TaskStateObserver,
    compact,
    estimate_tokens,
)
from ..context_management.memory import build_system_prompt, memory_tools
from ..context_management.skills import (
    ensure_host_skills_dir,
    resolve_host_skills_dir,
    seed_skills,
    skill_tools,
    sync_skill_drafts,
)
from ..loop.runner import EventType, RunEvent, run_loop
from ..loop.state import RunResult
from ..loop.termination import TerminationPolicy
from ..providers.base import Provider
from ..providers.catalog import (
    PROVIDERS,
    ModelCatalog,
    ModelConfigurationError,
    ModelSpec,
    selected_provider_from_env,
)
from ..providers.deepseek import DEFAULT_MODEL
from ..providers.factory import ProviderFactory
from ..routing.laya_router import LayaRouter, RoutingDecision, get_laya_router
from .session import (
    SessionData,
    SessionError,
    SessionRecorder,
    latest,
    load,
    new_session,
    record_resume,
    workspace_session_dir,
)
from .subagent import DelegationBudget, subagent_tool
from ..tools_and_guardrails.sandbox import OpenShellSandbox, SandboxError
from ..tools_and_guardrails.extended_tools import extended_tools
from ..tools_and_guardrails.security import Policy, redact_secrets
from ..tools_and_guardrails.tools import Tool, core_tools


class HarnessError(RuntimeError):
    """A Harness configuration or lifecycle failure."""


class Harness:
    """Compose Irina's existing components behind one small API.

    Args:
        workdir: Approved host workspace used for session identity.
        model: Model identifier. Defaults to the provider's model.
        policy: Policy instance or mode string. Defaults to ``Policy("safe")``.
        extra_tools: Extra tools to merge after core tools. Duplicate names are
            rejected.
        system_extra: Extra system-prompt instructions.
        on_event: Optional user event callback.
        budget_tokens: Working-context token budget.
        max_turns: Maximum turns for a top-level run.
        session_path: Optional explicit session path.
        enable_subagents: Register ``spawn_agent``.
        persist: Create/append a durable top-level session.
        _depth: Internal agent depth; children are greater than zero.
        provider: Optional injected provider (tests or alternate adapter).
        provider_name: Optional explicit provider name for manual selection.
        provider_mode: ``auto``, ``default``, ``manual``, or a provider name.
        model_catalog: Optional pre-built catalog.
        laya_router: Optional router instance; defaults to the process-wide lazy
            Laya router.
        workspace_backend: Optional existing workspace/sandbox backend.
        sandbox: Optional existing OpenShell sandbox.
        sandbox_workspace: Workspace name inside a new sandbox.
        state_dir: Trusted host state root for sessions.
        max_seconds: Elapsed-time limit per run.
        max_tool_calls: Tool-call limit per run.
        max_total_tokens: Loop token limit per run.
        stream: Whether to use the provider streaming interface when available.
        completion_validator: Optional task completion policy.
        should_cancel: Optional cancellation probe.
        max_subagent_depth: Maximum nesting depth.
        max_subagent_tasks: Total delegated tasks across the tree.
        child_max_turns: Turn limit per child.
        child_max_seconds: Time limit per child.
        child_max_tool_calls: Tool-call limit per child.
        skills_dir: Optional host directory of ``<name>/SKILL.md`` packages to
            upload into the workspace before the system prompt is built.
        inputs: Optional host files/directories to upload into ``input/``.
        export_paths: Optional workspace paths to download on clean shutdown.
        artifacts_dir: Host directory that receives ``export_paths``.
        export_contents: Merge an exported directory's children into
            ``artifacts_dir`` instead of creating a subdirectory.
        force_export: Replace an existing exported artifact.
        sandbox_image: Optional approved OpenShell image, e.g. a TeX/Poppler
            image for document skills.
        sandbox_policy: Optional approved OpenShell policy YAML. Use the shipped
            network policy when web tools need approved destinations.
        child_budget_tokens: Context budget per child.
        delegation_budget: Optional shared delegated-work counter.
        owns_sandbox: Whether ``close`` should delete the sandbox.
    """

    def __init__(
        self,
        workdir: str | Path = ".",
        model: str | None = None,
        policy: Policy | str | None = None,
        extra_tools: Any = None,
        system_extra: str = "",
        on_event: Callable[[RunEvent], None] | None = None,
        budget_tokens: int = 600_000,
        max_turns: int = 120,
        session_path: str | Path | None = None,
        enable_subagents: bool = True,
        persist: bool = True,
        _depth: int = 0,
        *,
        provider: Provider | None = None,
        provider_name: str | None = None,
        provider_mode: str | None = None,
        model_catalog: ModelCatalog | None = None,
        laya_router: LayaRouter | None = None,
        workspace_backend: Any | None = None,
        sandbox: OpenShellSandbox | None = None,
        sandbox_workspace: str | None = None,
        state_dir: str | Path | None = None,
        max_seconds: float = 300.0,
        max_tool_calls: int = 200,
        max_total_tokens: int = 600_000,
        stream: bool = False,
        memory_manager: Any | None = None,
        completion_validator: (
            Callable[[Any, list[dict[str, Any]]], bool] | None
        ) = None,
        should_cancel: Callable[[], bool] | None = None,
        max_subagent_depth: int = 2,
        max_subagent_tasks: int = 4,
        child_max_turns: int = 40,
        child_max_seconds: float = 120.0,
        child_max_tool_calls: int = 100,
        child_budget_tokens: int | None = None,
        delegation_budget: DelegationBudget | None = None,
        owns_sandbox: bool | None = None,
        skills_dir: str | Path | None = None,
        inputs: list[str | Path] | None = None,
        export_paths: list[str | Path] | None = None,
        artifacts_dir: str | Path | None = None,
        export_contents: bool = False,
        force_export: bool = False,
        sandbox_image: str | None = None,
        sandbox_policy: str | Path | None = None,
    ) -> None:
        self.workdir = str(Path(workdir).resolve())
        self._provider_override = provider
        self._model_catalog = model_catalog
        self._factory: ProviderFactory | None = None
        self._laya = laya_router
        self._routing_mode = "manual"
        self.provider = provider
        self.provider_name = provider_name or ("custom" if provider else "")
        self.model_spec: ModelSpec | None = None
        self.last_routing: RoutingDecision | None = None
        self.stream = bool(stream)
        self.memory = memory_manager
        if provider is not None:
            self.model = (
                model
                or getattr(provider, "default_model", None)
                or DEFAULT_MODEL
            )
        else:
            self.model = ""
            self._configure_initial_provider(provider_mode, provider_name, model)
        if isinstance(policy, str):
            policy = Policy(policy)
        self.policy = policy or Policy("safe")
        self._extra_tools = self._normalise_extra_tools(extra_tools)
        self._user_on_event = on_event
        self._system_extra = system_extra
        self.budget_tokens = budget_tokens
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self._memory_context = ""
        self._run_index = 0
        self._completion_validator = completion_validator
        self._should_cancel = should_cancel
        self.enable_subagents = enable_subagents
        self.persist = bool(persist and _depth == 0)
        self._depth = _depth
        self._state_dir = state_dir
        self._session_dir = workspace_session_dir(
            self.workdir, state_dir=state_dir
        )
        self._session_path = (
            Path(session_path).resolve() if session_path is not None else None
        )
        self._session: SessionData | None = None
        self._session_loaded = False
        self._session_status: str | None = None
        self._recorder: SessionRecorder | None = None
        self._messages: list[dict[str, Any]] = []
        self._task_state: TaskState | None = None
        self._task_observer: TaskStateObserver | None = None
        self._last_result: RunResult | None = None
        self._skills_dir = resolve_host_skills_dir(skills_dir, self.workdir)
        self.seeded_skills: list[str] = []
        self.synced_skill_drafts: list[str] = []
        self._input_paths = [str(item) for item in (inputs or [])]
        self._export_paths = [str(item) for item in (export_paths or [])]
        self._artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve()
            if artifacts_dir not in {None, ""}
            else None
        )
        self._export_contents = bool(export_contents)
        self._force_export = bool(force_export)
        self.uploaded_inputs: list[str] = []
        self.exported_artifacts: list[Path] = []
        self.export_errors: list[str] = []
        self._exported_once = False
        self._sandbox_image = str(sandbox_image) if sandbox_image else None
        self._sandbox_policy = (
            str(Path(sandbox_policy).expanduser().resolve())
            if sandbox_policy not in {None, ""}
            else None
        )

        self._backend = workspace_backend if workspace_backend is not None else sandbox
        self._sandbox = (
            sandbox
            if sandbox is not None
            else (
                workspace_backend
                if isinstance(workspace_backend, OpenShellSandbox)
                else None
            )
        )
        self._sandbox_workspace = (
            sandbox_workspace
            or getattr(self._backend, "workspace", None)
        )
        if owns_sandbox is None:
            owns_sandbox = workspace_backend is None and sandbox is None
        self._owns_sandbox = bool(owns_sandbox)
        self._workspace: str | None = None
        self._tools: dict[str, Tool] = {}
        self._system = ""
        self._reserve_tokens = 0
        self._ready = False

        self.max_subagent_depth = max_subagent_depth
        self.max_subagent_tasks = max_subagent_tasks
        self._child_max_turns = child_max_turns
        self._child_max_seconds = child_max_seconds
        self._child_max_tool_calls = child_max_tool_calls
        self._child_budget_tokens = child_budget_tokens or budget_tokens
        self._delegation_budget = (
            delegation_budget
            if delegation_budget is not None
            else DelegationBudget(max_tasks=max_subagent_tasks)
        )

    @staticmethod
    def _normalise_extra_tools(extra_tools: Any) -> list[Tool]:
        """Normalise extra tools and validate their basic shape."""
        if extra_tools is None:
            return []
        if isinstance(extra_tools, dict):
            values = list(extra_tools.values())
        elif isinstance(extra_tools, Tool):
            values = [extra_tools]
        else:
            values = list(extra_tools)
        for item in values:
            if not hasattr(item, "name") or not hasattr(item, "spec"):
                raise HarnessError(
                    "extra tools must use the existing Tool interface"
                )
        return values

    @staticmethod
    def _normalise_routing_mode(mode: str | None) -> str:
        raw = (mode or os.environ.get("IRINA_PROVIDER") or "auto").strip().lower()
        if raw in {"auto", "default", "manual"}:
            return raw
        if raw in PROVIDERS:
            return "manual"
        raise ModelConfigurationError(
            "provider must be auto, default, manual, deepseek, openai, or gemini"
        )

    def _configure_initial_provider(
        self,
        provider_mode: str | None,
        provider_name: str | None,
        model: str | None,
    ) -> None:
        """Select an initial provider without invoking Laya.

        Automatic routing still happens at the start of each independent task;
        this initial choice only gives memory and session metadata a valid
        provider before the first route.
        """
        mode = self._normalise_routing_mode(provider_mode)
        explicit_provider = (provider_name or "").strip().lower() or None
        requested = (provider_mode or "").strip().lower()
        if explicit_provider is None and requested in PROVIDERS:
            explicit_provider = requested
        if explicit_provider is None:
            explicit_provider = selected_provider_from_env()
        catalog = self._model_catalog or ModelCatalog.from_env()
        factory = ProviderFactory(catalog)
        self._model_catalog = catalog
        self._factory = factory

        if model and mode == "auto" and explicit_provider is None:
            # An explicit model is a manual override, never an automatic route.
            mode = "manual"
        if explicit_provider is not None and mode == "auto":
            mode = "manual"

        if mode == "manual":
            spec = factory.resolve_manual(explicit_provider, model)
        elif mode == "default":
            spec = factory.require_spec(catalog.configured_default())
        else:
            candidates = factory.candidate_specs(
                require_tools=True, require_streaming=self.stream
            )
            if not candidates:
                raise ModelConfigurationError(
                    "no model is enabled and configured with credentials; set "
                    "DEEPSEEK_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY"
                )
            try:
                default_spec = catalog.configured_default()
            except ModelConfigurationError:
                default_spec = None
            if default_spec is not None and default_spec in candidates:
                spec = default_spec
            else:
                spec = candidates[0]
            factory.require_spec(spec)
        self._routing_mode = mode
        self._apply_spec(spec, emit=False)

    def _apply_spec(
        self,
        spec: ModelSpec,
        *,
        reason: str = "",
        source: str = "",
        emit: bool = True,
    ) -> None:
        if self._factory is None:
            raise HarnessError("model selection is not configured")
        provider = self._factory.get(spec)
        self.model_spec = spec
        self.provider = provider
        self.model = spec.api_model
        self.provider_name = spec.provider
        memory_model = getattr(self.memory, "_model", None) if self.memory else None
        if memory_model is not None:
            memory_model.provider = provider
            memory_model.model = self.model
        if emit:
            self._emit_model_event(reason=reason, source=source)

    def _emit_model_event(self, *, reason: str, source: str) -> None:
        if self._user_on_event is None or self.model_spec is None:
            return
        try:
            self._user_on_event(
                RunEvent(
                    type=EventType.MODEL,
                    turn=0,
                    provider=self.provider_name,
                    model=self.model,
                    model_id=self.model_spec.id,
                    routing_mode=self._routing_mode,
                    routing_reason=reason,
                )
            )
        except Exception:
            pass

    def _routing_context(self) -> str:
        relevant = [
            message
            for message in self._messages
            if message.get("role") in {"user", "assistant"}
            and str(message.get("text", "")).strip()
        ]
        return "\n".join(
            f"{message['role']}: {str(message.get('text', ''))[:1000]}"
            for message in relevant[-6:]
        )

    def _route_model(self, task: str) -> None:
        """Ask Laya for one configured model at an independent task boundary."""
        if self._provider_override is not None or self._factory is None:
            return
        candidates = self._factory.candidate_specs(
            require_tools=True, require_streaming=self.stream
        )
        fallback = self._factory.resolve_fallback(candidates)
        router = self._laya or get_laya_router()
        self._laya = router
        decision = router.select(
            task,
            self._routing_context(),
            candidates,
            fallback,
        )
        self.last_routing = decision
        self._routing_mode = "auto"
        self._apply_spec(
            decision.spec,
            reason=decision.reason,
            source=decision.source,
        )

    def _maybe_route(self, task: str) -> None:
        if self._routing_mode != "auto" or self._provider_override is not None:
            return
        unfinished = self._session_loaded and self._session_status not in {
            None,
            "completed",
        }
        if unfinished:
            return
        self._route_model(task)

    @property
    def model_label(self) -> str:
        """Return the selected provider and API model for display."""
        return f"{self.provider_name}:{self.model}"

    @property
    def routing_mode(self) -> str:
        """Return the current routing mode."""
        return self._routing_mode

    def set_model(
        self,
        *,
        mode: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> str:
        """Change model selection for the next task boundary.

        ``mode=auto`` keeps the current model until the next independent task,
        then Laya selects again. ``mode=default`` applies the configured
        default. A provider/model argument performs a validated manual switch.
        """
        if self._factory is None or self._model_catalog is None:
            raise HarnessError(
                "model selection is unavailable for an injected provider"
            )
        if mode is not None:
            normalised = self._normalise_routing_mode(mode)
            if normalised == "auto":
                self._routing_mode = "auto"
                self._emit_model_event(
                    reason="automatic routing enabled for the next task",
                    source="auto",
                )
                return self.model_label
            if normalised == "default":
                spec = self._factory.require_spec(
                    self._model_catalog.configured_default()
                )
                self._routing_mode = "default"
                self._apply_spec(spec, reason="configured default", source="default")
                return self.model_label
            spec = self._factory.resolve_manual(
                provider_name or selected_provider_from_env(),
                model,
            )
            self._routing_mode = "manual"
            self._apply_spec(spec, reason="manual selection", source="manual")
            return self.model_label
        if provider_name is not None or model is not None:
            spec = self._factory.resolve_manual(provider_name, model)
            self._routing_mode = "manual"
            self._apply_spec(spec, reason="manual selection", source="manual")
            return self.model_label
        return self.model_label

    @property
    def workspace(self) -> str:
        """Return the agent's approved workspace path, preparing lazily."""
        self._ensure_ready()
        assert self._workspace is not None
        return self._workspace

    @property
    def tools(self) -> dict[str, Tool]:
        """Return a copy of the registered tool map."""
        self._ensure_ready()
        return dict(self._tools)

    @property
    def system_prompt(self) -> str:
        """Return the assembled system prompt."""
        self._ensure_ready()
        return self._system

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the current working conversation."""
        return self._messages

    @property
    def task_state(self) -> TaskState | None:
        """Return the current session task state."""
        return self._task_state

    @property
    def session_path(self) -> Path | None:
        """Return the active session path, if any."""
        return self._session_path

    @property
    def session_status(self) -> str | None:
        """Return the last recorded run status, if any."""
        return self._session_status

    @property
    def session_data(self) -> SessionData | None:
        """Return the loaded session data, if any."""
        return self._session

    @property
    def last_result(self) -> RunResult | None:
        """Return the most recent run result."""
        return self._last_result

    @property
    def sandbox(self) -> OpenShellSandbox | None:
        """Return the OpenShell adapter owned or shared by this Harness.

        The adapter is available after :attr:`workspace` or a run has prepared
        the backend. Callers should not close a shared sandbox themselves.
        """
        return self._sandbox

    def _ensure_ready(self) -> None:
        """Start/attach the backend, build tools, and assemble the prompt."""
        if self._ready:
            return
        if self._backend is None:
            options: dict[str, Any] = {
                "workspace": self._sandbox_workspace or "irina"
            }
            if self._sandbox_image:
                options["image"] = self._sandbox_image
            if self._sandbox_policy:
                options["policy_path"] = self._sandbox_policy
            self._sandbox = OpenShellSandbox(**options)
            self._sandbox.start()
            self._backend = self._sandbox
            self._owns_sandbox = True
        else:
            if isinstance(self._backend, OpenShellSandbox):
                self._sandbox = self._backend
                if not self._backend.started:
                    self._backend.start()

        workspace = getattr(self._backend, "workspace", None)
        if not isinstance(workspace, str) or not workspace:
            raise HarnessError(
                "workspace backend does not expose an approved workspace path"
            )
        self._workspace = workspace

        if self._input_paths and self._depth == 0:
            from ..tools_and_guardrails.transfers import upload_inputs

            try:
                self.uploaded_inputs = upload_inputs(
                    self._input_paths, self._backend
                )
            except Exception as exc:
                raise HarnessError(
                    f"input upload failed: {type(exc).__name__}: {exc}"
                ) from None

        if self._skills_dir is not None and self._depth == 0:
            ensured = ensure_host_skills_dir(self._skills_dir)
            if ensured is not None:
                self._skills_dir = ensured
            try:
                self.seeded_skills = seed_skills(
                    self._skills_dir,
                    self._workspace,
                    workspace=self._backend,
                )
            except Exception:
                # Skills are optional; a bad host directory must not stop a run.
                self.seeded_skills = []

        tools = core_tools(self._workspace, self._backend)
        tools.extend(
            extended_tools(self._workspace, self._backend)
        )
        tools.extend(memory_tools(self._workspace, workspace=self._backend))
        tools.extend(skill_tools(self._workspace, workspace=self._backend))
        if self.enable_subagents:
            tools.append(
                subagent_tool(
                    self._make_child_harness,
                    depth=self._depth,
                    max_depth=self.max_subagent_depth,
                    budget=self._delegation_budget,
                )
            )

        tool_map: dict[str, Tool] = {}
        for item in tools:
            if item.name in tool_map:
                raise HarnessError(f"duplicate tool name: {item.name!r}")
            tool_map[item.name] = item
        for item in self._extra_tools:
            if item.name in tool_map:
                raise HarnessError(
                    f"extra tool {item.name!r} would overwrite a registered tool"
                )
            tool_map[item.name] = item
        self._tools = tool_map

        self._system = build_system_prompt(
            self._workspace,
            extra=self._system_extra,
            workspace=self._backend,
            platform_name=getattr(self._backend, "platform_name", None),
        )
        self._reserve_tokens = self._fixed_tokens(self._system, self._tools)
        self._ready = True

    def _emit_memory(
        self,
        *,
        phase: str,
        text: str,
        count: int | None = None,
        ok: bool | None = None,
    ) -> None:
        if self._user_on_event is None:
            return
        provider = getattr(
            getattr(self.memory, "config", None), "active_provider", ""
        )
        try:
            self._user_on_event(
                RunEvent(
                    type=EventType.MEMORY,
                    turn=0,
                    text=text,
                    memory_provider=str(provider or ""),
                    memory_phase=phase,
                    memory_count=count,
                    memory_ok=ok,
                )
            )
        except Exception:
            pass

    def _apply_memory_context(self, task: str) -> None:
        """Retrieve lower-trust memory and rebuild the system prompt if needed."""
        if self.memory is None:
            return
        try:
            context = self.memory.before_task(task)
        except Exception:
            context = ""
        self._memory_context = context or ""
        extra = self._system_extra
        if self._memory_context:
            extra = extra + "\n\n" + self._memory_context
        self._system = build_system_prompt(
            self._workspace,
            extra=extra,
            workspace=self._backend,
            platform_name=getattr(self._backend, "platform_name", None),
        )
        self._reserve_tokens = self._fixed_tokens(self._system, self._tools)
        if self._memory_context:
            count = len(getattr(self.memory, "last_retrieval", []) or [])
            self._emit_memory(
                phase="retrieve",
                text=(
                    f"Retrieved {count} relevant experience(s)"
                    if count
                    else "Retrieved historical memory"
                ),
                count=count,
                ok=True,
            )

    @staticmethod
    def _fixed_tokens(system: str, tools: dict[str, Tool]) -> int:
        """Estimate fixed system/tool context for compaction accounting."""
        specs = [item.spec for item in tools.values()]
        return estimate_tokens(
            [{"role": "system", "text": system}]
        ) + estimate_tokens(
            [{"role": "user", "text": json.dumps(specs, ensure_ascii=False)}]
        )

    def _make_child_harness(self, depth: int) -> "Harness":
        """Create a fresh child over the parent's backend and policy."""
        return Harness(
            workdir=self.workdir,
            model=self.model,
            policy=self.policy,
            extra_tools=self._extra_tools,
            system_extra=self._system_extra,
            on_event=None,
            budget_tokens=self._child_budget_tokens,
            max_turns=self._child_max_turns,
            session_path=None,
            enable_subagents=self.enable_subagents,
            persist=False,
            _depth=depth,
            provider=self.provider,
            provider_name=self.provider_name,
            workspace_backend=self._backend,
            sandbox=self._sandbox,
            sandbox_workspace=self._workspace,
            state_dir=self._state_dir,
            max_seconds=self._child_max_seconds,
            max_tool_calls=self._child_max_tool_calls,
            max_total_tokens=self.max_total_tokens,
            stream=self.stream,
            memory_manager=self.memory,
            completion_validator=None,
            should_cancel=self._should_cancel,
            max_subagent_depth=self.max_subagent_depth,
            max_subagent_tasks=self.max_subagent_tasks,
            child_max_turns=self._child_max_turns,
            child_max_seconds=self._child_max_seconds,
            child_max_tool_calls=self._child_max_tool_calls,
            child_budget_tokens=self._child_budget_tokens,
            delegation_budget=self._delegation_budget,
            owns_sandbox=False,
            sandbox_image=self._sandbox_image,
            sandbox_policy=self._sandbox_policy,
        )

    def _open_session(
        self,
        task: str,
        *,
        path: Path | None = None,
    ) -> None:
        """Create and open a new top-level session."""
        if self._task_state is None:
            self._task_state = TaskState(goal=task)
        safe_task = redact_secrets(task)
        label = safe_task[:32] if safe_task.strip() else "session"
        session_path = new_session(
            self.workdir,
            label,
            state_dir=self._state_dir,
            sandbox_name=getattr(self._sandbox, "name", None),
            sandbox_workspace=self._workspace,
            metadata={
                "model": self.model,
                "provider": self.provider_name,
                "model_id": self.model_spec.id if self.model_spec else self.model,
                "routing_mode": self._routing_mode,
                "harness": "stage3",
            },
            path=path if path is not None else self._session_path,
        )
        self._session_path = session_path
        self._session_loaded = True
        self._session_status = "running"
        self._recorder = SessionRecorder(session_path, self._task_state)

    def resume(self, path: str | Path | None = None) -> bool:
        """Resume a trusted top-level session for this workspace.

        Args:
            path: Optional explicit session path inside the trusted store. With
                no path, the newest resumable session is used.

        Returns:
            ``True`` when a session was loaded and the backend was restored.

        Raises:
            SessionError: For an invalid path, wrong workspace, or a sandbox
                that can no longer be restored.
        """
        if not self.persist:
            return False
        if self._session_loaded:
            if path is None:
                return self._session_status != "completed"
            if Path(path).resolve() != self._session_path:
                raise SessionError(
                    "this Harness is already attached to another session"
                )
            return self._session_status != "completed"

        if path is None:
            candidate = latest(
                self.workdir,
                state_dir=self._state_dir,
                resumable_only=True,
            )
            if candidate is None:
                return False
        else:
            candidate = Path(path).resolve()
            if not candidate.is_relative_to(self._session_dir.resolve()):
                raise SessionError(
                    "session path is outside the trusted workspace store"
                )

        data = load(candidate)
        if data.metadata.get("workspace") != self.workdir:
            raise SessionError(
                "session belongs to a different workspace and cannot be resumed"
            )
        if (
            data.metadata.get("depth", 0) != 0
            or data.metadata.get("parent") is not None
        ):
            raise SessionError("child-agent sessions are not top-level sessions")
        if data.status == "completed":
            return False

        metadata_workspace = data.metadata.get("sandbox_workspace")
        metadata_sandbox = data.metadata.get("sandbox_name")
        if self._backend is not None:
            backend_workspace = getattr(self._backend, "workspace", None)
            if (
                isinstance(metadata_workspace, str)
                and isinstance(backend_workspace, str)
                and metadata_workspace != backend_workspace
            ):
                raise SessionError(
                    "provided workspace does not match the session workspace"
                )
            backend_name = getattr(self._backend, "name", None)
            if (
                isinstance(metadata_sandbox, str)
                and isinstance(backend_name, str)
                and metadata_sandbox != backend_name
            ):
                raise SessionError(
                    "provided sandbox does not match the session sandbox"
                )
        elif isinstance(metadata_sandbox, str) and metadata_sandbox:
            try:
                self._sandbox = OpenShellSandbox.reattach(
                    metadata_sandbox,
                    workspace=metadata_workspace,
                )
            except SandboxError as exc:
                raise SessionError(
                    "cannot resume session: the original sandbox or workspace "
                    f"is unavailable: {exc}"
                ) from exc
            self._backend = self._sandbox
            self._sandbox_workspace = metadata_workspace
            self._owns_sandbox = True
        else:
            raise SessionError(
                "session has no restorable sandbox; pass a workspace_backend"
            )

        self._messages = list(data.messages)
        self._task_state = data.task_state
        self._session = data
        self._session_path = candidate
        self._session_status = data.status
        self._session_loaded = True
        self._ensure_ready()

        self._recorder = SessionRecorder(candidate, self._task_state)
        self._recorder.turns = data.turns
        self._recorder.tool_calls = data.tool_calls
        self._recorder.status = data.status
        self._recorder.seed_user_messages(self._messages)
        self._run_index = sum(
            1
            for record in data.records
            if record.get("type") == "run_end"
        )
        record_resume(candidate)
        return True

    def run(self, task: str) -> RunResult:
        """Run one task and return the existing structured ``RunResult``."""
        return self._execute(task, allow_completed=False)

    def continue_task(self, task: str) -> RunResult:
        """Append a follow-up task to the current session and run it.

        This is for explicit continuation, such as an interactive prompt loop or
        a second review phase. It never turns a completed run into a crash; it
        appends a new user message and a new ``run_end`` record to the same
        session. ``run()`` still refuses a completed session by default.
        """
        return self._execute(task, allow_completed=True)

    def _execute(self, task: str, *, allow_completed: bool) -> RunResult:
        """Shared execution path for initial and follow-up tasks."""
        if not isinstance(task, str) or not task.strip():
            raise HarnessError("task must be a non-empty string")

        if self.persist and not self._session_loaded and self._session_path:
            if self._session_path.exists():
                if not self.resume(self._session_path):
                    raise SessionError(
                        "explicit session is completed and cannot be resumed; "
                        "create a new Harness or session"
                    )

        if (
            self.persist
            and self._session_loaded
            and self._session_status == "completed"
            and not allow_completed
        ):
            raise SessionError(
                "session is already completed; create a new Harness for a new task"
            )

        self._ensure_ready()
        self._maybe_route(task)
        self._apply_memory_context(task)
        if self._task_state is None:
            self._task_state = TaskState(goal=task)
        elif not self._task_state.goal:
            self._task_state.goal = task

        if self.persist and not self._session_loaded:
            self._open_session(task)
        elif self._session_loaded:
            self._session_status = "running"
            if self._recorder is not None:
                self._recorder.status = "running"

        message = {"role": "user", "text": task}
        self._messages.append(message)
        if self._recorder is not None:
            self._recorder.record_user(task)

        self._task_observer = TaskStateObserver(self._task_state)

        def on_event(event: RunEvent) -> None:
            assert self._task_observer is not None
            self._task_observer(event)
            if self._recorder is not None:
                self._recorder.record_event(event)
            if self._user_on_event is not None:
                self._user_on_event(event)

        def before_turn(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if self._recorder is not None:
                self._recorder.sync_user_messages(current)
            return compact(
                self.model,
                current,
                self.budget_tokens,
                provider=self.provider,
                task_state=self._task_state,
                on_event=on_event,
                reserve_tokens=self._reserve_tokens,
            )

        termination = TerminationPolicy(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_seconds=self.max_seconds,
            max_total_tokens=self.max_total_tokens,
        )

        try:
            result = run_loop(
                provider=self.provider,
                model=self.model,
                system=self._system,
                messages=self._messages,
                tools=self._tools,
                on_event=on_event,
                before_tool=self.policy.check,
                policy=termination,
                completion_validator=self._completion_validator,
                before_turn=before_turn,
                should_cancel=self._should_cancel,
                stream=self.stream,
            )
        except KeyboardInterrupt as exc:
            if self._recorder is not None:
                self._recorder.record_error(exc)
            self._session_status = "interrupted"
            raise
        except Exception as exc:
            if self._recorder is not None:
                self._recorder.record_error(exc)
            self._session_status = "interrupted"
            raise

        if self._recorder is not None:
            self._recorder.record_run_end(result)
        self._session_status = result.status.value
        self._last_result = result
        self._ingest_memory(task, result)
        return result

    def _ingest_memory(self, task: str, result: RunResult) -> None:
        """Persist canonical experience and update memory without failing the run."""
        if self.memory is None or self._depth != 0:
            return
        try:
            from ..context_management.memory.trajectory_bridge import (
                record_from_result,
            )

            self._run_index += 1
            session_id = (
                self._session_path.stem if self._session_path else None
            )
            record = record_from_result(
                task=task,
                result=result,
                workspace=self.workdir,
                session_path=self._session_path,
                session_id=session_id,
                run_index=self._run_index,
                task_state=self._task_state,
            )
            result = self.memory.after_task(record)
            if result is not None:
                self._emit_memory(
                    phase="update",
                    text=(
                        "Experience saved"
                        if result.ok
                        else f"Experience update failed: {result.error}"
                    ),
                    count=1 if result.ok else 0,
                    ok=bool(result.ok),
                )
        except Exception as exc:
            # Memory must never corrupt an already-completed task.
            self._emit_memory(
                phase="update",
                text=f"Memory update failed: {type(exc).__name__}: {exc}",
                ok=False,
            )
            return

    def _export_path_exists(self, path: str) -> bool:
        try:
            listing = self._backend.list_files(f"{path}/**", limit=1)
            if listing.get("paths"):
                return True
        except Exception:
            pass
        try:
            self._backend.read_file(path, max_lines=1)
            return True
        except Exception:
            return False

    def export_now(self, *, force: bool | None = None) -> list[Path]:
        """Export configured workspace paths to the host immediately."""
        if self._artifacts_dir is None or not self._export_paths:
            raise HarnessError(
                "no artifact export is configured; use --export and "
                "--artifacts-dir"
            )
        from ..tools_and_guardrails.transfers import export_artifacts

        self.export_errors = []
        existing = [
            path for path in self._export_paths
            if self._export_path_exists(path)
        ]
        if not existing:
            self.exported_artifacts = []
            return []
        try:
            self.exported_artifacts = export_artifacts(
                existing,
                self._backend,
                self._artifacts_dir,
                contents=self._export_contents,
                force=(
                    self._force_export if force is None else bool(force)
                ),
            )
        except Exception as exc:
            self.export_errors = [f"{type(exc).__name__}: {exc}"]
            raise
        self._exported_once = True
        return list(self.exported_artifacts)

    def close(self) -> None:
        """Delete an owned sandbox and release the Harness.

        A shared child sandbox or a caller-provided backend is not deleted.
        """
        if (
            self._artifacts_dir is not None
            and self._export_paths
            and self._depth == 0
            and self._ready
            and not self._exported_once
        ):
            try:
                self.export_now()
            except Exception:
                pass
        if self._skills_dir is not None and self._depth == 0 and self._ready:
            try:
                self.synced_skill_drafts = sync_skill_drafts(
                    self._skills_dir,
                    self.workdir,
                    workspace=self._backend,
                    state_dir=self._state_dir,
                )
            except Exception:
                self.synced_skill_drafts = []
        self._ready = False
        if (
            self._owns_sandbox
            and self._sandbox is not None
            and self._sandbox.started
        ):
            self._sandbox.delete()

    def __enter__(self) -> "Harness":
        self._ensure_ready()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


__all__ = ["Harness", "HarnessError"]
