"""Stage 4: build three real projects through a parallel Harness fleet.

Usage::

    python demos/stage4_projects.py
    python demos/stage4_projects.py --only artisan-coffee --workers 1

Each project runs two phases in one session: build, then a demanding review and
fix pass. The fleet gives every job its own OpenShell sandbox. ``on_complete``
runs mechanical checks, downloads the artifacts to the host project directory,
and records hashes before the sandbox is deleted.

The checks are deliberately separate from prose claims. Browser interaction is
not automated here; web results report static checks only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.back.harness import Harness, Policy, run_fleet
from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.loop.state import RunStatus
from src.back.harness.orchestration.cli import _configure_streams, _load_dotenv
from src.back.harness.providers.deepseek import DeepSeekProvider

SKILL_PATH = "skills/design-engineering/SKILL.md"

DESIGN_SKILL = """# Design Engineering

Build the requested interface as a real, self-contained product, not a
placeholder. Work inside the approved workspace. Use no external fonts, images,
frameworks, CDNs, or build tools unless the task explicitly allows them.

Quality bar for landing pages (all of these apply to artisan-coffee):
- a real design system as CSS custom properties;
- at least 9 distinct sections for a landing page;
- at least 1,200 words of real copy, no lorem ipsum;
- at least 4 hand-drawn inline SVG illustrations, one being a product artifact
  in the hero;
- at least 3 working interactive behaviors;
- responsive at 360, 768, and 1280;
- semantic HTML with focus states;
- a self-review pass before finishing that counts sections, words, SVGs, and
  interactions against these minimums and fixes any shortfall.

For games and tools, keep the same craft standard without forcing landing-page
word or section counts. Do not merely describe the work. Write the files,
inspect them, count the required elements, and fix the shortfalls before
reporting completion.
"""

REVIEW = (
    "Review every file you produced against the applicable project requirements "
    "as a demanding reviewer. List 12 concrete deficiencies or improvement "
    "opportunities, fix the valid deficiencies, and verify again."
)

ARTISAN_TASK = """Build a production-quality landing page for Artisan Coffee
Roasters in `index.html`. Work only in the approved workspace. First read
`skills/design-engineering/SKILL.md` with the skill tool.

The page must feel like a warm, editorial specialty-coffee brand: cream, deep
espresso, copper, and forest accents; expressive display type; generous
whitespace; real product voice. It must be self-contained: no external fonts,
images, frameworks, CDNs, or libraries.

Required content and sections (at least 9 distinct sections):
1. Sticky navigation with brand, section links, and a dark-mode control.
2. Hero with a hand-drawn inline SVG illustration of a coffee bag or pour-over
   as the product artifact.
3. Origin cards for six single-origin coffees, each with farm/region, tasting
   notes, roast level, and a real price.
4. Subscription comparison with three tiers and a working monthly/annual
   toggle that updates every displayed price.
5. Brew guide with working tabs for V60, French press, and espresso, including
   ratio, grind, temperature, and time.
6. Roastery/process story with concrete sourcing and roasting detail.
7. Testimonials from named-sounding customers with specific praise.
8. FAQ accordion with at least five useful questions.
9. Footer with contact, hours, location, and newsletter form.

Quality requirements:
- At least 1,200 words of visible, non-repeated copy. No lorem ipsum, no
  placeholder labels, no "insert text here".
- At least 4 hand-drawn inline SVG illustrations; one must be the hero product
  artifact. Use real paths/shapes, not empty boxes.
- At least 3 working interactions: billing toggle, brew tabs, FAQ accordion,
  and dark mode. Dark mode must persist in localStorage.
- CSS custom properties for the design system.
- Responsive layouts at 360, 768, and 1280 (use real media queries).
- Semantic HTML: header, nav, main, section, footer, headings, buttons, labels.
  Visible keyboard focus states.
- No horizontal overflow at 360px. Respect reduced motion.
- Include a short "Our promise" element with a real guarantee.
- Before finishing, count the sections, visible words, SVGs, and interactions
  and fix any shortfall. Write the file in several calls: a skeleton first, then
  append CSS, sections, and JavaScript with edit_file. Do not emit the entire
  file in one giant write_file call. Do not ask questions; finish the page."""

TASKMAN_TASK = """Create a small, production-quality command-line task manager in
`taskman.py` with tests in `test_taskman.py`. Work only in the approved
workspace. Use only the Python standard library.

Requirements:
- argparse CLI with subcommands: add, list, done, rm, stats.
- `add` takes a title, optional --priority low|normal|high, optional --due
  YYYY-MM-DD. Assign a stable integer id. Trim surrounding whitespace from the
  title and reject an empty title.
- `list` shows all tasks in an aligned table with id, status, priority, due
  date, and title. Support --all and --pending.
- `done` marks one or more ids complete; reject unknown ids clearly.
- `rm` removes one or more ids; reject unknown ids clearly.
- `stats` reports total, pending, completed, and counts by priority.
- Persist to JSON. Support --data PATH (default `tasks.json`). Create the file
  safely; reject malformed JSON without destroying it.
- Use a temporary file plus replace for atomic writes.
- Clear validation errors and nonzero exit codes for bad input.
- Human-readable output; no traceback for expected user errors.
- Write at least 10 unittest cases in `test_taskman.py` using subprocess and a
  temporary data file. Cover add, list, done, rm, stats, priorities, due dates,
  unknown ids, invalid JSON, and empty state.
- Write the files in several calls: a skeleton first, then append with edit_file.
  Do not emit either file in one giant write_file call.
- Run `python3 -m unittest -v test_taskman.py` and fix until every test passes.
- Do not ask questions; finish the implementation and tests."""

VIPER_TASK = """Build a polished Snake game called Viper in `index.html`.
Work only in the approved workspace. It must be self-contained: no external
libraries, fonts, images, or CDNs.

Requirements:
- HTML canvas game with a visible grid and crisp, intentional visual design.
- Arrow keys and WASD control the snake; prevent immediate reverse moves.
- requestAnimationFrame game loop with fixed-step movement and interpolation or
  a clean tick model.
- Food appears in random free cells; eating grows the snake and increases score.
- Speed increases every 5 foods eaten.
- Pause/resume (Space or P), restart, and a clear game-over state.
- Score and high score displayed; high score persists in localStorage.
- Start screen and game-over overlay with clear instructions.
- Responsive canvas layout that works at 360, 768, and 1280 widths.
- Keyboard focus and accessible labels; do not rely only on color.
- Touch/swipe controls as a bonus, but keyboard must work.
- Keep the game deterministic enough to test the core movement and collision
  logic; separate pure logic from rendering where practical.
- Write the file in several calls: a skeleton first, then append CSS and
  JavaScript with edit_file. Do not emit the entire file in one giant write_file
  call.
- Before finishing, verify the canvas, requestAnimationFrame, localStorage,
  keyboard listeners, game-over, pause, and restart paths. Do not ask questions;
  finish the game."""

TASKS = {
    "artisan-coffee": (ARTISAN_TASK, REVIEW, True),
    "taskman": (TASKMAN_TASK, REVIEW, False),
    "viper": (VIPER_TASK, REVIEW, True),
}


@dataclass
class Verification:
    """Mechanical checks and facts for one artifact."""

    checks: dict[str, bool] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _visible_text(html: str) -> str:
    """Return text outside script/style/tags for a word-count check."""
    body = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"&[a-zA-Z#0-9]+;", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def verify_artisan(project: Path) -> Verification:
    """Check the artisan landing page's mechanical requirements."""
    path = project / "index.html"
    if not path.is_file():
        return Verification({"index_exists": False})
    html = _read(path)
    lower = html.lower()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", _visible_text(html))
    sections = len(re.findall(r"<section\b", html, flags=re.IGNORECASE))
    svgs = len(re.findall(r"<svg\b", html, flags=re.IGNORECASE))
    checks = {
        "index_exists": True,
        "sections_9": sections >= 9,
        "words_1200": len(words) >= 1200,
        "svgs_4": svgs >= 4,
        "design_tokens": "--" in html and "var(" in html,
        "dark_mode": "localstorage" in lower,
        "billing_toggle": "monthly" in lower
        and "annual" in lower
        and ("toggle" in lower or "billing" in lower),
        "brew_tabs": "brew" in lower and "tab" in lower,
        "faq_accordion": "faq" in lower
        and ("accordion" in lower or "<details" in lower),
        "interactions_3": lower.count("addeventlistener") >= 3,
        "responsive": "@media" in lower
        and ("max-width" in lower or "min-width" in lower),
        "focus_visible": ":focus" in lower or ":focus-visible" in lower,
        "semantic": all(
            tag in lower
            for tag in ("<header", "<nav", "<main", "<section", "<footer")
        ),
        "no_external": not re.search(
            r"(?:src|href)=[\"']https?://", html, flags=re.IGNORECASE
        ),
    }
    facts = {
        "sections": sections,
        "visible_words": len(words),
        "svg_count": svgs,
        "addEventListener_count": lower.count("addeventlistener"),
        "bytes": len(html.encode("utf-8")),
    }
    return Verification(checks, facts)


def verify_viper(project: Path) -> Verification:
    """Check Viper's static implementation and required game behaviors."""
    path = project / "index.html"
    if not path.is_file():
        return Verification({"index_exists": False})
    html = _read(path)
    lower = html.lower()
    checks = {
        "index_exists": True,
        "canvas": "<canvas" in lower,
        "raf": "requestanimationframe" in lower,
        "keyboard": "keydown" in lower or "keyup" in lower,
        "local_storage": "localstorage" in lower,
        "score": "score" in lower,
        "food": "food" in lower,
        "game_over": "game over" in lower or "gameover" in lower,
        "pause": "pause" in lower,
        "restart": "restart" in lower,
        "responsive": "@media" in lower,
        "focus_visible": ":focus" in lower or ":focus-visible" in lower,
        "no_external": not re.search(
            r"(?:src|href)=[\"']https?://", html, flags=re.IGNORECASE
        ),
    }
    facts = {
        "bytes": len(html.encode("utf-8")),
        "canvas_count": lower.count("<canvas"),
        "listener_count": lower.count("addeventlistener"),
    }
    return Verification(checks, facts)


def verify_taskman(
    project: Path,
    *,
    tests_pass: bool | None = None,
    help_pass: bool | None = None,
) -> Verification:
    """Check taskman files and optionally the sandbox test results."""
    task = project / "taskman.py"
    tests = project / "test_taskman.py"
    checks: dict[str, bool] = {
        "taskman_exists": task.is_file(),
        "tests_exist": tests.is_file(),
    }
    facts: dict[str, Any] = {}
    if task.is_file():
        source = _read(task)
        lower = source.lower()
        checks.update(
            {
                "argparse": "argparse" in lower,
                "json_persistence": "json" in lower,
                "atomic_replace": ".replace(" in lower
                or "os.replace" in lower,
                "add": '"add"' in lower or "'add'" in lower,
                "list": '"list"' in lower or "'list'" in lower,
                "done": '"done"' in lower or "'done'" in lower,
                "rm": '"rm"' in lower or "'rm'" in lower,
                "stats": '"stats"' in lower or "'stats'" in lower,
            }
        )
        facts["taskman_bytes"] = len(source.encode("utf-8"))
    if tests.is_file():
        tests_source = _read(tests)
        test_count = len(
            re.findall(r"def test_", tests_source, flags=re.IGNORECASE)
        )
        checks["tests_10"] = test_count >= 10
        facts["test_functions"] = test_count
        facts["test_bytes"] = len(tests_source.encode("utf-8"))
    if tests_pass is not None:
        checks["tests_pass"] = tests_pass
    if help_pass is not None:
        checks["help_pass"] = help_pass
    return Verification(checks, facts)


def _artifacts(project: Path, files: Sequence[str]) -> list[dict[str, Any]]:
    """Return size and digest facts for downloaded files."""
    artifacts = []
    for relative in files:
        path = project / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        artifacts.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return artifacts


class JobPrinter:
    """Thread-safe, low-noise progress printer."""

    _lock = threading.Lock()

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, event: RunEvent) -> None:
        if event.type is EventType.TOOL_START:
            call = event.tool_call
            if call is None:
                return
            text = f"[{self.name}] {call.name}"
        elif event.type is EventType.RUN_END:
            status = event.status.value if event.status else "unknown"
            text = f"[{self.name}] run {status}"
        else:
            return
        with self._lock:
            print(text, flush=True)


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    task: str
    review: str
    use_skill: bool
    files: tuple[str, ...]
    verifier: Callable[[Path], Verification]


SPECS: dict[str, ProjectSpec] = {
    "artisan-coffee": ProjectSpec(
        name="artisan-coffee",
        task=TASKS["artisan-coffee"][0],
        review=TASKS["artisan-coffee"][1],
        use_skill=True,
        files=("index.html",),
        verifier=verify_artisan,
    ),
    "taskman": ProjectSpec(
        name="taskman",
        task=TASKS["taskman"][0],
        review=TASKS["taskman"][1],
        use_skill=False,
        files=("taskman.py", "test_taskman.py"),
        verifier=verify_taskman,
    ),
    "viper": ProjectSpec(
        name="viper",
        task=TASKS["viper"][0],
        review=TASKS["viper"][1],
        use_skill=True,
        files=("index.html",),
        verifier=verify_viper,
    ),
}


def _make_harness(
    spec: ProjectSpec,
    *,
    project: Path,
    model: str | None,
    max_turns: int,
    max_seconds: float,
    max_tokens: int,
    max_total_tokens: int,
) -> Harness:
    """Create and seed one project Harness, closing it if seeding fails."""
    provider = DeepSeekProvider(max_tokens=max_tokens)
    harness = Harness(
        workdir=project,
        model=model,
        provider=provider,
        policy=Policy("yolo"),
        max_turns=max_turns,
        max_seconds=max_seconds,
        max_total_tokens=max_total_tokens,
        on_event=JobPrinter(spec.name),
    )
    try:
        harness.workspace
        sandbox = harness.sandbox
        if sandbox is None:
            raise RuntimeError("Harness did not expose an OpenShell sandbox")
        if spec.use_skill:
            sandbox.write_file(SKILL_PATH, DESIGN_SKILL)
    except Exception:
        harness.close()
        raise
    return harness


def _verify_and_download(
    spec: ProjectSpec,
    harness: Harness,
    project: Path,
) -> Verification:
    """Run task-specific checks, then download artifacts before cleanup."""
    tests_pass = None
    help_pass = None
    test_exit = None
    help_exit = None
    test_tail = None
    if spec.name == "taskman":
        test = harness.sandbox.exec(
            ["python3", "-m", "unittest", "-v", "test_taskman.py"],
            timeout=180,
        )
        help_run = harness.sandbox.exec(
            ["python3", "taskman.py", "--help"],
            timeout=60,
        )
        test_exit = test.exit_code
        help_exit = help_run.exit_code
        tests_pass = test_exit == 0
        help_pass = help_exit == 0
        if not tests_pass:
            test_tail = test.stdout[-2000:] + test.stderr[-2000:]

    for relative in spec.files:
        harness.sandbox.download(relative, project / relative)
    if spec.use_skill:
        harness.sandbox.download(SKILL_PATH, project / SKILL_PATH)

    if spec.name == "taskman":
        verification = verify_taskman(
            project,
            tests_pass=tests_pass,
            help_pass=help_pass,
        )
        verification.facts["test_exit"] = test_exit
        verification.facts["help_exit"] = help_exit
        if test_tail:
            verification.facts["test_tail"] = test_tail
    else:
        verification = spec.verifier(project)

    downloaded = list(spec.files)
    if spec.use_skill:
        downloaded.append(SKILL_PATH)
    verification.facts["artifacts"] = _artifacts(project, downloaded)
    return verification


def _on_complete(
    spec: ProjectSpec,
) -> Callable[[Harness, Any], dict[str, Any]]:
    """Build verification/download plus one focused repair callback."""

    def complete(harness: Harness, final: Any) -> dict[str, Any]:
        project = Path(harness.workdir)
        try:
            verification = _verify_and_download(spec, harness, project)
        except Exception as exc:
            verification = Verification(
                {"verification_error": False},
                {"verification_error": f"{type(exc).__name__}: {exc}"},
            )

        repairs = []
        if not verification.ok and final.status not in {
            RunStatus.PROVIDER_ERROR,
            RunStatus.CANCELLED,
        }:
            failed = [
                name
                for name, passed in verification.checks.items()
                if not passed
            ]
            facts = {
                key: value
                for key, value in verification.facts.items()
                if key != "artifacts"
            }
            repair_task = (
                "The mechanical verification failed. Failed checks: "
                + (", ".join(failed) if failed else "(verification error)")
                + ". Verification facts: "
                + json.dumps(facts, ensure_ascii=False)[:6000]
                + ". Fix only these concrete deficiencies, re-run the project's "
                "tests or verification commands, and leave all previously "
                "working requirements intact. Do not rewrite successful work. "
                "Do not ask questions."
            )
            repair = harness.continue_task(repair_task)
            repairs.append(repair)
            try:
                verification = _verify_and_download(spec, harness, project)
            except Exception as exc:
                verification = Verification(
                    {"verification_error": False},
                    {"verification_error": f"{type(exc).__name__}: {exc}"},
                )

        failed = [
            name
            for name, passed in verification.checks.items()
            if not passed
        ]
        report = (
            "all mechanical checks passed"
            if not failed
            else "failed checks: " + ", ".join(failed)
        )
        extra: dict[str, Any] = {
            "ok": verification.ok,
            "report": report,
            "checks": verification.checks,
            "facts": verification.facts,
        }
        if repairs:
            extra.update(
                {
                    "extra_turns": sum(
                        int(getattr(item, "turns", 0) or 0)
                        for item in repairs
                    ),
                    "extra_tool_calls": sum(
                        int(getattr(item, "tool_calls", 0) or 0)
                        for item in repairs
                    ),
                    "extra_phases": [
                        {
                            "status": item.status.value,
                            "turns": int(getattr(item, "turns", 0) or 0),
                            "tool_calls": int(
                                getattr(item, "tool_calls", 0) or 0
                            ),
                        }
                        for item in repairs
                    ],
                    "extra_usage": {
                        "input_tokens": sum(
                            int(getattr(item.usage, "input_tokens", 0) or 0)
                            for item in repairs
                        ),
                        "output_tokens": sum(
                            int(getattr(item.usage, "output_tokens", 0) or 0)
                            for item in repairs
                        ),
                    },
                    "repair_status": [
                        item.status.value for item in repairs
                    ],
                }
            )
            extra["extra_usage"]["total_tokens"] = (
                extra["extra_usage"]["input_tokens"]
                + extra["extra_usage"]["output_tokens"]
            )
        return extra

    return complete


def _print_results(results: list[dict[str, Any]]) -> None:
    """Print a compact, truthful results table."""
    print("\nStage 4 results")
    print("=" * 72)
    for item in results:
        state = "PASS" if item.get("ok") else "FAIL"
        print(
            f"{item['name']:<18} {state:<4} "
            f"status={item.get('status'):<14} "
            f"turns={item.get('turns', 0):<3} "
            f"tools={item.get('tool_calls', 0):<3}"
        )
        metadata = item.get("metadata") or {}
        checks = metadata.get("checks") or {}
        if checks:
            failed = [key for key, value in checks.items() if not value]
            print(
                "  checks: "
                + ("all passed" if not failed else "failed: " + ", ".join(failed))
            )
        facts = metadata.get("facts") or {}
        if facts:
            print("  facts: " + json.dumps(facts, ensure_ascii=False))
        if item.get("session"):
            print(f"  session: {item['session']}")
        if not item.get("ok"):
            print(f"  report: {item.get('report', '')[:500]}")
    print("=" * 72)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated project names (default: all three).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel fleet workers (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override; otherwise IRINA_MODEL/provider default.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Turn limit per phase (default: 100).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=900.0,
        help="Time limit per phase (default: 900).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Provider output-token ceiling (default: 32768).",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=1_000_000,
        help="Per-phase loop token budget (default: 1000000).",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "stage4_results.json"),
        help="Results JSON path (default: stage4_results.json).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_streams()
    _load_dotenv()
    names = [
        name.strip()
        for name in (args.only.split(",") if args.only else SPECS.keys())
        if name.strip()
    ]
    unknown = [name for name in names if name not in SPECS]
    if unknown:
        print(f"Unknown project(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2
    if args.max_turns <= 0 or args.max_seconds <= 0:
        print("--max-turns and --max-seconds must be positive", file=sys.stderr)
        return 2
    if args.max_tokens <= 0:
        print("--max-tokens must be positive", file=sys.stderr)
        return 2
    if args.max_total_tokens <= 0:
        print("--max-total-tokens must be positive", file=sys.stderr)
        return 2

    specs = [SPECS[name] for name in names]
    jobs = []
    for spec in specs:
        project = ROOT / spec.name
        project.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "name": spec.name,
                "workdir": str(project),
                "task": spec.task,
                "review_task": spec.review,
                "on_complete": _on_complete(spec),
            }
        )

    started = time.monotonic()
    print(f"Running {len(jobs)} project(s) with {args.workers} worker(s)...")
    results = run_fleet(
        jobs,
        lambda job: _make_harness(
            SPECS[job.name],
            project=Path(job.workdir),
            model=args.model,
            max_turns=args.max_turns,
            max_seconds=args.max_seconds,
            max_tokens=args.max_tokens,
            max_total_tokens=args.max_total_tokens,
        ),
        max_workers=args.workers,
    )
    elapsed = time.monotonic() - started
    _print_results(results)
    print(f"elapsed={elapsed:.1f}s")
    print(
        "Note: browser interaction checks are not automated; web projects "
        "report static checks only."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"results={output}")
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
