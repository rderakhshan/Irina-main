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

## Compatibility patch 2

`FlashOAgents/__init__.py` and `MemEvolve/__init__.py` now export public names
lazily. Upstream imported every FlashOAgents submodule (web search, crawling,
media tools) and every evolution phase merely to import the package, so the
original evolution core could not be imported without `requests`, `crawl4ai`,
`pandas`, `biopython`, `python-pptx`, and other research-only packages.
`base_agent.py` now imports `WebSearchTool`, `CrawlPageTool`,
`VisualInspectorTool`, `AudioInspectorTool`, and `TextInspectorTool` inside the
agent classes that use them instead of at module import time. No upstream
algorithm or prompt was changed.

## Compatibility patch 3

`MemEvolve.core.memory_evolver.MemoryEvolver.create()` accepts an optional
`base_dir` and defaults to its own `work_dir`. Upstream always passed `"."` to
`MemorySystemCreator`, which wrote generated providers into the process working
directory (and could mutate the vendored source tree). Irina passes an isolated
copy of `EvolveLab/` under the evolution work directory, so generated code is
never written into the vendored source.
