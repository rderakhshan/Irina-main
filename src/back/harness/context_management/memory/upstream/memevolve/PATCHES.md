# Upstream Compatibility Notes

Upstream: `https://github.com/bingreeky/MemEvolve`
Commit: `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`
License: Apache-2.0

The source under `EvolveLab/`, `MemEvolve/`, `FlashOAgents/`, `base_agent.py`,
`evolve_cli.py`, and `utils.py` is copied verbatim. No upstream algorithms have
been rewritten.

Irina adds the upstream directory to its import path only inside the upstream
loader. The loader removes that path after imports. This is required because the
upstream code uses top-level imports such as `EvolveLab...`, `MemEvolve...`, and
`FlashOAgents...`.

Optional research dependencies are not installed by default. Providers that
cannot import are reported as dependency-blocked instead of being replaced with
mock implementations.

## Compatibility patch 1

`EvolveLab/providers/__init__.py` now exports the same four public provider
classes lazily instead of importing four provider modules eagerly. Upstream
imported numpy, scikit-learn, and sentence-transformers merely to import the
provider package. Lazy exports allow the dependency-free Lightweight provider
to load while preserving the public class names.
