"""Day 1: provider adapters and contracts.

Concept: the execution loop talks to a provider through a narrow, vendor-neutral
contract. Concrete adapters translate neutral messages to a wire format.

Design constraints:
* Standard library only; no vendor SDKs.
* Contracts contain no vendor-specific fields.
* Importing the package does not require credentials or perform network I/O.
"""

from .base import (
    Provider,
    ProviderError,
    ProviderHTTPError,
    ProviderInvalidResponse,
    ProviderResponse,
    ProviderRetryExhausted,
    TokenUsage,
    ToolCall,
)
from .deepseek import API_ROOT, DEFAULT_MODEL, DeepSeekProvider

__all__ = [
    "API_ROOT",
    "DEFAULT_MODEL",
    "DeepSeekProvider",
    "Provider",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderInvalidResponse",
    "ProviderResponse",
    "ProviderRetryExhausted",
    "TokenUsage",
    "ToolCall",
]
