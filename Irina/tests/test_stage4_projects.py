"""Stage 4 tests for the product-proof verifiers and demo configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from demos.stage4_projects import (
    SPECS,
    parse_args,
    verify_artisan,
    verify_taskman,
    verify_viper,
)


def artisan_html(words: int = 1250) -> str:
    copy = "word " * words
    return f"""<!doctype html>
<html><head><style>
:root {{ --ink: #222; --cream: #fff8ec; }}
body {{ color: var(--ink); background: var(--cream); }}
:focus-visible {{ outline: 2px solid var(--ink); }}
@media (max-width: 768px) {{ body {{ font-size: 16px; }} }}
</style></head>
<body>
<header><nav>Artisan</nav></header>
<main>
<section>one</section><section>two</section><section>three</section>
<section>four</section><section>five</section><section>six</section>
<section>seven</section><section>eight</section><section>{copy}</section>
<svg></svg><svg></svg><svg></svg><svg></svg>
<button>Monthly</button><button>Annual</button>
<div>Billing toggle</div>
<div>Brew guide tabs</div><div>FAQ accordion</div>
</main><footer>Footer</footer>
<script>
localStorage.setItem('theme', 'dark');
document.addEventListener('click', () => {{}});
document.addEventListener('keydown', () => {{}});
document.addEventListener('change', () => {{}});
</script>
</body></html>"""


VIPER_HTML = """<!doctype html>
<html><head><style>
:focus-visible { outline: 2px solid white; }
@media (max-width: 600px) { canvas { width: 100%; } }
</style></head>
<body>
<canvas></canvas>
<button>Restart</button><p>Score</p><p>Food</p><p>Game over</p>
<script>
const canvas = document.querySelector('canvas');
let score = 0;
localStorage.setItem('highScore', String(score));
function frame() { requestAnimationFrame(frame); }
requestAnimationFrame(frame);
document.addEventListener('keydown', (event) => {
  if (event.key === ' ') { pause(); }
});
function pause() {}
function restart() {}
</script>
</body></html>"""


TASKMAN_SOURCE = """import argparse, json, os
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command')
sub.add_parser('add'); sub.add_parser('list'); sub.add_parser('done')
sub.add_parser('rm'); sub.add_parser('stats')
def save(tasks):
    with open('tasks.json.tmp', 'w') as handle:
        json.dump(tasks, handle)
    os.replace('tasks.json.tmp', 'tasks.json')
"""

TASKMAN_TESTS = "\n".join(
    f"def test_{index}(): pass" for index in range(10)
)


class ArtisanVerifierTests(unittest.TestCase):
    def test_complete_page_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index.html").write_text(artisan_html(), encoding="utf-8")
            verification = verify_artisan(root)
            self.assertTrue(verification.ok, verification.checks)
            self.assertGreaterEqual(verification.facts["sections"], 9)
            self.assertGreaterEqual(verification.facts["visible_words"], 1200)
            self.assertGreaterEqual(verification.facts["svg_count"], 4)

    def test_short_page_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index.html").write_text("<html></html>", encoding="utf-8")
            verification = verify_artisan(root)
            self.assertFalse(verification.ok)
            self.assertFalse(verification.checks["sections_9"])

    def test_missing_page_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(verify_artisan(Path(temp)).ok)


class ViperVerifierTests(unittest.TestCase):
    def test_canvas_game_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index.html").write_text(VIPER_HTML, encoding="utf-8")
            verification = verify_viper(root)
            self.assertTrue(verification.ok, verification.checks)

    def test_missing_game_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(verify_viper(Path(temp)).ok)


class TaskmanVerifierTests(unittest.TestCase):
    def test_python_project_passes_with_test_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "taskman.py").write_text(TASKMAN_SOURCE, encoding="utf-8")
            (root / "test_taskman.py").write_text(
                TASKMAN_TESTS, encoding="utf-8"
            )
            verification = verify_taskman(
                root, tests_pass=True, help_pass=True
            )
            self.assertTrue(verification.ok, verification.checks)
            self.assertEqual(verification.facts["test_functions"], 10)

    def test_failing_tests_fail_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "taskman.py").write_text(TASKMAN_SOURCE, encoding="utf-8")
            (root / "test_taskman.py").write_text(
                TASKMAN_TESTS, encoding="utf-8"
            )
            self.assertFalse(
                verify_taskman(root, tests_pass=False, help_pass=True).ok
            )


class ConfigurationTests(unittest.TestCase):
    def test_all_three_projects_are_configured(self) -> None:
        self.assertEqual(
            set(SPECS), {"artisan-coffee", "taskman", "viper"}
        )
        self.assertTrue(SPECS["artisan-coffee"].use_skill)
        self.assertTrue(SPECS["viper"].use_skill)
        self.assertFalse(SPECS["taskman"].use_skill)

    def test_defaults_are_parallel_and_bounded(self) -> None:
        args = parse_args([])
        self.assertEqual(args.workers, 3)
        self.assertEqual(args.max_turns, 100)
        self.assertEqual(args.max_seconds, 900.0)
        self.assertEqual(args.max_tokens, 32768)
        self.assertEqual(args.max_total_tokens, 1_000_000)


if __name__ == "__main__":
    unittest.main()
