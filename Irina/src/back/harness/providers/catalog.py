"""Configured model catalog for routing and manual selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

PROVIDERS = ("deepseek", "openai")
DEFAULT_DESCRIPTIONS = {
    "deepseek": "DeepSeek: strong coding, mathematics, and long technical analysis.",
    "openai": "OpenAI: broad general reasoning, writing, and multimodal understanding.",
    "gemini": "Gemini: Google ecosystem tasks, long context, and multimodal inputs.",
}
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-3.5-flash",
}
KEY_ENV = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
MODEL_ENV = {
    "deepseek": ("DEEPSEEK_MODEL", "IRINA_MODEL"),
    "openai": ("OPENAI_MODEL",),
    "gemini": ("GEMINI_MODEL",),
}


@dataclass(frozen=True)
class ModelSpec:
    """One selectable provider/model entry."""

    id: str
    provider: str
    api_model: str
    enabled: bool = True
    supports_tools: bool = True
    supports_streaming: bool = True
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.api_model}"


class ModelConfigurationError(RuntimeError):
    """The configured model catalog or selection is invalid."""


def provider_has_credentials(provider: str) -> bool:
    """Return whether a provider has a non-empty credential variable."""
    for name in KEY_ENV.get(provider, ()):
        if os.environ.get(name, "").strip():
            return True
    return False


def credential_name(provider: str) -> str:
    names = KEY_ENV.get(provider, ())
    return " or ".join(names) if names else f"{provider.upper()}_API_KEY"


def _provider_default_model(provider: str) -> str:
    for name in MODEL_ENV.get(provider, ()):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return DEFAULT_MODELS.get(provider, "")


def _default_specs() -> list[ModelSpec]:
    specs = []
    for provider in PROVIDERS:
        api_model = _provider_default_model(provider)
        if not api_model:
            continue
        specs.append(
            ModelSpec(
                id=f"{provider}/{api_model}",
                provider=provider,
                api_model=api_model,
                description=DEFAULT_DESCRIPTIONS.get(provider, ""),
            )
        )
    return specs


def _parse_catalog(raw: str) -> list[ModelSpec]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelConfigurationError(
            f"IRINA_MODEL_CATALOG is not valid JSON: {exc}"
        ) from None
    if isinstance(payload, dict):
        payload = payload.get("models", [])
    if not isinstance(payload, list):
        raise ModelConfigurationError("IRINA_MODEL_CATALOG must be a list or {\"models\": [...]}")
    specs: list[ModelSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ModelConfigurationError(
                f"IRINA_MODEL_CATALOG entry {index} must be an object"
            )
        provider = str(item.get("provider", "")).strip().lower()
        if provider not in PROVIDERS:
            raise ModelConfigurationError(
                f"IRINA_MODEL_CATALOG entry {index} has unknown provider {provider!r}"
            )
        api_model = str(item.get("api_model", item.get("model", ""))).strip()
        if not api_model:
            raise ModelConfigurationError(
                f"IRINA_MODEL_CATALOG entry {index} is missing api_model"
            )
        model_id = str(item.get("id", f"{provider}/{api_model}")).strip()
        if not model_id:
            raise ModelConfigurationError(
                f"IRINA_MODEL_CATALOG entry {index} is missing id"
            )
        description = str(item.get("description", DEFAULT_DESCRIPTIONS.get(provider, "")))
        specs.append(
            ModelSpec(
                id=model_id,
                provider=provider,
                api_model=api_model,
                enabled=bool(item.get("enabled", True)),
                supports_tools=bool(item.get("supports_tools", True)),
                supports_streaming=bool(item.get("supports_streaming", True)),
                description=description,
            )
        )
    return specs


class ModelCatalog:
    """Enabled and configured model candidates."""

    def __init__(self, models: Iterable[ModelSpec]) -> None:
        self.models = list(models)
        if not self.models:
            raise ModelConfigurationError("the model catalog is empty")
        seen: set[str] = set()
        for spec in self.models:
            if spec.id in seen:
                raise ModelConfigurationError(f"duplicate model id {spec.id!r}")
            seen.add(spec.id)

    @classmethod
    def from_env(cls) -> "ModelCatalog":
        raw = os.environ.get("IRINA_MODEL_CATALOG", "").strip()
        models = _parse_catalog(raw) if raw else _default_specs()
        return cls(models)

    def get(self, model_id: str, provider: str | None = None) -> ModelSpec:
        """Resolve a catalog id or API model id, optionally within a provider."""
        wanted = str(model_id or "").strip()
        if not wanted:
            raise ModelConfigurationError("model identifier is empty")
        candidates = [
            spec
            for spec in self.models
            if provider is None or spec.provider == provider
        ]
        for spec in candidates:
            if wanted in {spec.id, spec.api_model, f"{spec.provider}:{spec.api_model}"}:
                return spec
        raise ModelConfigurationError(
            f"model {wanted!r} is not in the configured catalog"
        )

    def available(
        self,
        *,
        require_tools: bool = True,
        require_streaming: bool = False,
    ) -> list[ModelSpec]:
        """Return enabled models with credentials and required capabilities."""
        return [
            spec
            for spec in self.models
            if spec.enabled
            and provider_has_credentials(spec.provider)
            and (spec.supports_tools or not require_tools)
            and (spec.supports_streaming or not require_streaming)
        ]

    def provider_default(self, provider: str) -> ModelSpec:
        candidates = [
            spec
            for spec in self.models
            if spec.provider == provider and spec.enabled
        ]
        if not candidates:
            raise ModelConfigurationError(
                f"no enabled model is configured for provider {provider!r}"
            )
        requested = next(
            (
                os.environ.get(name, "").strip()
                for name in MODEL_ENV.get(provider, ())
                if os.environ.get(name, "").strip()
            ),
            "",
        )
        if requested:
            for spec in candidates:
                if requested in {spec.id, spec.api_model, f"{spec.provider}:{spec.api_model}"}:
                    return spec
        return candidates[0]

    def configured_default(self) -> ModelSpec:
        provider = (os.environ.get("IRINA_DEFAULT_PROVIDER") or "deepseek").strip().lower()
        if provider not in PROVIDERS:
            raise ModelConfigurationError(
                f"IRINA_DEFAULT_PROVIDER has unknown provider {provider!r}"
            )
        model = os.environ.get("IRINA_DEFAULT_MODEL", "").strip()
        if model:
            try:
                return self.get(model, provider=provider)
            except ModelConfigurationError:
                # Allow a default API model that is not in the catalog by using
                # the provider's configured model entry only when it matches.
                raise
        return self.provider_default(provider)

    def configured_fallback(self) -> ModelSpec:
        provider = (
            os.environ.get("IRINA_FALLBACK_PROVIDER")
            or os.environ.get("IRINA_DEFAULT_PROVIDER")
            or "deepseek"
        ).strip().lower()
        if provider not in PROVIDERS:
            raise ModelConfigurationError(
                f"IRINA_FALLBACK_PROVIDER has unknown provider {provider!r}"
            )
        model = os.environ.get("IRINA_FALLBACK_MODEL", "").strip()
        if model:
            return self.get(model, provider=provider)
        return self.provider_default(provider)


def routing_mode_from_env() -> str:
    """Return ``auto``, ``manual``, or ``default`` from IRINA_PROVIDER."""
    raw = (os.environ.get("IRINA_PROVIDER") or "auto").strip().lower()
    if raw in {"auto", "default", "manual"}:
        return raw
    if raw in PROVIDERS:
        return "manual"
    raise ModelConfigurationError(
        f"IRINA_PROVIDER must be auto, default, manual, or one of {', '.join(PROVIDERS)}"
    )


def selected_provider_from_env() -> str | None:
    """Return the concrete provider name in IRINA_PROVIDER, if any."""
    raw = (os.environ.get("IRINA_PROVIDER") or "").strip().lower()
    return raw if raw in PROVIDERS else None


__all__ = [
    "DEFAULT_DESCRIPTIONS",
    "DEFAULT_MODELS",
    "KEY_ENV",
    "ModelCatalog",
    "ModelConfigurationError",
    "ModelSpec",
    "PROVIDERS",
    "credential_name",
    "provider_has_credentials",
    "routing_mode_from_env",
    "selected_provider_from_env",
]
