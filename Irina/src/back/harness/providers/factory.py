"""Resolve provider/model identifiers to concrete adapters."""

from __future__ import annotations

from .base import Provider
from .catalog import (
    ModelCatalog,
    ModelConfigurationError,
    ModelSpec,
    credential_name,
    provider_has_credentials,
)
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider


class ProviderFactory:
    """Create only the provider adapters a run actually needs."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self.catalog = catalog
        self._cache: dict[str, Provider] = {}

    def candidate_specs(
        self,
        *,
        require_tools: bool = True,
        require_streaming: bool = False,
    ) -> list[ModelSpec]:
        return self.catalog.available(
            require_tools=require_tools,
            require_streaming=require_streaming,
        )

    def require_spec(self, spec: ModelSpec) -> ModelSpec:
        """Validate one catalog entry for manual selection."""
        if not spec.enabled:
            raise ModelConfigurationError(f"model {spec.id!r} is disabled")
        if not provider_has_credentials(spec.provider):
            raise ModelConfigurationError(
                f"provider {spec.provider!r} requires {credential_name(spec.provider)}"
            )
        return spec

    def resolve_manual(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        default_provider: str | None = None,
    ) -> ModelSpec:
        """Resolve an explicit provider/model choice and validate credentials."""
        provider_name = (provider or default_provider or "").strip().lower() or None
        model_name = (model or "").strip() or None
        if provider_name is not None:
            if provider_name not in {"deepseek", "openai", "gemini"}:
                raise ModelConfigurationError(
                    f"unknown provider {provider_name!r}; expected deepseek, openai, or gemini"
                )
            spec = (
                self.catalog.get(model_name, provider=provider_name)
                if model_name
                else self.catalog.provider_default(provider_name)
            )
        elif model_name is not None:
            spec = self.catalog.get(model_name)
        else:
            spec = self.catalog.configured_default()
        return self.require_spec(spec)

    def resolve_fallback(self, candidates: list[ModelSpec]) -> ModelSpec:
        """Return a usable fallback: configured first, then any candidate."""
        try:
            spec = self.catalog.configured_fallback()
            if spec.enabled and spec in candidates:
                return spec
        except ModelConfigurationError:
            pass
        if candidates:
            return candidates[0]
        raise ModelConfigurationError(
            "no model is enabled and configured with credentials; set "
            "DEEPSEEK_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY"
        )

    def get(self, spec: ModelSpec) -> Provider:
        """Return the cached adapter for one model's provider."""
        provider_name = spec.provider
        if provider_name in self._cache:
            return self._cache[provider_name]
        if provider_name == "deepseek":
            provider: Provider = DeepSeekProvider(model=spec.api_model)
        elif provider_name == "openai":
            provider = OpenAIProvider(model=spec.api_model)
        elif provider_name == "gemini":
            provider = GeminiProvider(model=spec.api_model)
        else:
            raise ModelConfigurationError(f"unsupported provider {provider_name!r}")
        self._cache[provider_name] = provider
        return provider


__all__ = ["ProviderFactory"]
