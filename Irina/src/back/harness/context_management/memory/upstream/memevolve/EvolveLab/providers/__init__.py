"""Memory providers for different frameworks.

Irina compatibility patch: upstream eagerly imported four providers here,
which made importing the dependency-free Lightweight provider require numpy,
scikit-learn, and sentence-transformers. The public names are preserved but
are imported lazily. See ../../PATCHES.md.
"""

from importlib import import_module

_EXPORTS = {
    "AgentKBProvider": "agent_kb_provider",
    "SkillWeaverProvider": "skillweaver_provider",
    "MobileEProvider": "mobilee_provider",
    "ExpeLProvider": "expel_provider",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
