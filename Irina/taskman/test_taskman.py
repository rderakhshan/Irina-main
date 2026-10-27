"""End-to-end tests for the taskman command-line task manager.

The tests exercise the CLI through subprocess so argparse wiring, exit codes,
persistence, and human-readable output are all verified together.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TASKMAN = str(Path(__file__).resolve().with_name("taskman.py"))


class TaskmanTestCase(unittest.TestCase):
    """Base helper for running the CLI against a temporary data file."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_path = str(Path(self.tmpdir.name) / "tasks.json")

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run taskman.py with a temporary --data file prepended."""
        return subprocess.run(
            [sys.executable, TASKMAN, "--data", self.data_path, *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def read_tasks(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


class TaskmanTests(TaskmanTestCase):
    """Individual CLI behaviours required by the specification."""

    def test_add_creates_task_with_trimmed_title_priority_and_due(self):
        result = self.run_cli(
            "add", "  Buy milk  ", "--priority", "high", "--due", "2025-12-01"
        )
        self.assertEqual(0, result.returncode, result.stderr)
        tasks = self.read_tasks()
        self.assertEqual(1, len(tasks))
        self.assertEqual("Buy milk", tasks[0]["title"])
        self.assertEqual("high", tasks[0]["priority"])
        self.assertEqual("2025-12-01", tasks[0]["due"])
        self.assertEqual("pending", tasks[0]["status"])
        self.assertIn("Added task #1", result.stdout)

    def test_add_assigns_stable_increasing_ids(self):
        self.run_cli("add", "First")
        self.run_cli("add", "Second")
        tasks = self.read_tasks()
        self.assertEqual([1, 2], [task["id"] for task in tasks])

    def test_add_rejects_empty_title(self):
        result = self.run_cli("add", "   ")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("title must not be empty", result.stderr)
        self.assertFalse(os.path.exists(self.data_path))

    def test_add_rejects_invalid_due_date(self):
        result = self.run_cli("add", "Bad date", "--due", "2025-13-99")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid date", result.stderr)

    def test_add_rejects_invalid_priority(self):
        result = self.run_cli("add", "Bad priority", "--priority", "urgent")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid choice", result.stderr)

    def test_add_rejects_non_zero_padded_due_date(self):
        result = self.run_cli("add", "Sloppy date", "--due", "2025-1-1")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid date", result.stderr)
        self.assertFalse(os.path.exists(self.data_path))

    def test_list_all_includes_completed_tasks(self):
        self.run_cli("add", "Pending task")
        self.run_cli("add", "Finished task")
        self.run_cli("done", "2")
        result = self.run_cli("list", "--all")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Pending task", result.stdout)
        self.assertIn("Finished task", result.stdout)
        self.assertIn("done", result.stdout)

    def test_done_deduplicates_duplicate_ids(self):
        self.run_cli("add", "Only once")
        result = self.run_cli("done", "1", "1")
        self.assertEqual(0, result.returncode, result.stderr)
        tasks = self.read_tasks()
        self.assertEqual("done", tasks[0]["status"])
        self.assertEqual("Marked task(s) #1 as done.", result.stdout.strip())

    def test_list_shows_aligned_table_with_headers(self):
        self.run_cli("add", "First task", "--priority", "low", "--due", "2025-01-02")
        self.run_cli("add", "Second task")
        result = self.run_cli("list")
        self.assertEqual(0, result.returncode, result.stderr)
        for header in ("ID", "STATUS", "PRIORITY", "DUE", "TITLE"):
            self.assertIn(header, result.stdout)
        self.assertIn("First task", result.stdout)
        self.assertIn("Second task", result.stdout)
        self.assertIn("2025-01-02", result.stdout)

    def test_list_pending_filters_out_completed_tasks(self):
        self.run_cli("add", "Pending task")
        self.run_cli("add", "Finished task")
        self.run_cli("done", "2")
        result = self.run_cli("list", "--pending")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Pending task", result.stdout)
        self.assertNotIn("Finished task", result.stdout)

    def test_done_marks_multiple_ids_complete(self):
        self.run_cli("add", "One")
        self.run_cli("add", "Two")
        self.run_cli("add", "Three")
        result = self.run_cli("done", "1", "3")
        self.assertEqual(0, result.returncode, result.stderr)
        tasks = self.read_tasks()
        self.assertEqual(
            ["done", "pending", "done"],
            [task["status"] for task in tasks],
        )

    def test_done_rejects_unknown_id_without_modifying_data(self):
        self.run_cli("add", "Keep pending")
        result = self.run_cli("done", "99")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("unknown task id(s): 99", result.stderr)
        tasks = self.read_tasks()
        self.assertEqual("pending", tasks[0]["status"])

    def test_rm_removes_multiple_ids(self):
        self.run_cli("add", "Keep one")
        self.run_cli("add", "Remove two")
        self.run_cli("add", "Remove three")
        self.run_cli("add", "Keep four")
        result = self.run_cli("rm", "2", "3")
        self.assertEqual(0, result.returncode, result.stderr)
        tasks = self.read_tasks()
        self.assertEqual([1, 4], [task["id"] for task in tasks])
        self.assertNotIn("Remove two", result.stdout)
        self.assertNotIn("Remove three", result.stdout)

    def test_rm_rejects_unknown_id_without_modifying_data(self):
        self.run_cli("add", "Do not remove")
        result = self.run_cli("rm", "42")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("unknown task id(s): 42", result.stderr)
        self.assertEqual(1, len(self.read_tasks()))

    def test_stats_reports_totals_and_priority_counts(self):
        self.run_cli("add", "Low pending", "--priority", "low")
        self.run_cli("add", "Normal done", "--priority", "normal")
        self.run_cli("done", "2")
        self.run_cli("add", "High pending", "--priority", "high")
        result = self.run_cli("stats")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Total:      3", result.stdout)
        self.assertIn("Pending:    2", result.stdout)
        self.assertIn("Completed:  1", result.stdout)
        self.assertIn("  low      1", result.stdout)
        self.assertIn("  normal   1", result.stdout)
        self.assertIn("  high     1", result.stdout)

    def test_stats_and_list_handle_empty_state(self):
        list_result = self.run_cli("list")
        self.assertEqual(0, list_result.returncode, list_result.stderr)
        self.assertIn("No tasks.", list_result.stdout)

        stats_result = self.run_cli("stats")
        self.assertEqual(0, stats_result.returncode, stats_result.stderr)
        self.assertIn("Total:      0", stats_result.stdout)
        self.assertIn("Pending:    0", stats_result.stdout)
        self.assertIn("Completed:  0", stats_result.stdout)

    def test_malformed_json_is_rejected_and_left_unchanged(self):
        malformed = "{ this is not valid JSON\n"
        with open(self.data_path, "w", encoding="utf-8") as fh:
            fh.write(malformed)
        result = self.run_cli("list")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("malformed JSON", result.stderr)
        with open(self.data_path, "r", encoding="utf-8") as fh:
            self.assertEqual(malformed, fh.read())

    def test_valid_json_with_wrong_shape_is_rejected_and_left_unchanged(self):
        original = '{"not": "a list"}\n'
        with open(self.data_path, "w", encoding="utf-8") as fh:
            fh.write(original)
        result = self.run_cli("list")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("expected a JSON list", result.stderr)
        with open(self.data_path, "r", encoding="utf-8") as fh:
            self.assertEqual(original, fh.read())

    def test_empty_data_file_is_rejected_and_left_unchanged(self):
        with open(self.data_path, "w", encoding="utf-8") as fh:
            fh.write("")
        result = self.run_cli("add", "New task")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("malformed JSON", result.stderr)
        self.assertEqual(0, os.path.getsize(self.data_path))


if __name__ == "__main__":
    unittest.main()
