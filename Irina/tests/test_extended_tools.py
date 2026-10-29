"""Tests for the 18 web, document, and data tools."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.back.harness.orchestration.harness import Harness
from src.back.harness.loop.runner import run_loop
from src.back.harness.loop.state import RunStatus
from src.back.harness.loop.termination import TerminationPolicy
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.tools_and_guardrails import (
    LocalWorkspace,
    NETWORK_POLICY_PATH,
    OpenShellSandbox,
    Policy,
    SandboxError,
    SandboxUnavailable,
    core_tools,
    extended_tools,
)

EXTENDED_NAMES = [
    "web_search",
    "web_fetch",
    "http_request",
    "download_url",
    "read_pdf",
    "create_pdf",
    "edit_pdf",
    "compile_latex",
    "read_docx",
    "write_docx",
    "read_xlsx",
    "write_xlsx",
    "create_pptx",
    "sql_query",
    "database_schema",
    "read_data",
    "write_data",
    "execute_notebook",
]

EXISTING_NAMES = {
    "read_file",
    "write_file",
    "edit_file",
    "bash",
    "list_files",
    "grep",
    "remember",
    "use_skill",
    "spawn_agent",
}

TEST_IMAGE = os.environ.get("IRINA_TEST_IMAGE", "irina/irina-tools:1")


def call(name: str, **args: object) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args)


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.backend = LocalWorkspace(self.root)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_all_eighteen_tools_are_built_in_order(self) -> None:
        tools = extended_tools(str(self.root), self.backend)
        self.assertEqual([tool.name for tool in tools], EXTENDED_NAMES)
        self.assertEqual(len(tools), 18)

    def test_every_extended_schema_is_provider_compatible(self) -> None:
        for item in extended_tools(str(self.root), self.backend):
            schema = item.spec["schema"]
            self.assertEqual(schema["name"], item.name)
            self.assertTrue(schema["description"])
            parameters = schema["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertIsInstance(parameters["required"], list)
            for name, value in parameters["properties"].items():
                self.assertEqual(value["type"], "string", name)
                self.assertTrue(value["description"], name)

    def test_harness_registers_all_tools_in_one_registry(self) -> None:
        class Provider:
            default_model = "scripted"

        harness = Harness(
            workdir=str(self.root),
            provider=Provider(),
            workspace_backend=self.backend,
            persist=False,
            skills_dir="",
        )
        try:
            names = set(harness.tools)
        finally:
            harness.close()
        self.assertTrue(set(EXTENDED_NAMES).issubset(names))
        self.assertTrue(EXISTING_NAMES.issubset(names))
        self.assertEqual(len(names), len(EXISTING_NAMES) + 18)

    def test_no_tool_runs_on_the_host_without_openshell(self) -> None:
        tools = {item.name: item for item in extended_tools(str(self.root), self.backend)}
        result = tools["read_pdf"].run(path="missing.pdf")
        self.assertIn("SandboxUnavailable", result)
        self.assertIn("OpenShell", result)


class PolicyTests(unittest.TestCase):
    def test_read_only_tools_are_allowed(self) -> None:
        policy = Policy("read-only")
        for name, args in [
            ("web_search", {"query": "x"}),
            ("web_fetch", {"url": "https://example.com"}),
            ("read_pdf", {"path": "a.pdf"}),
            ("read_docx", {"path": "a.docx"}),
            ("read_xlsx", {"path": "a.xlsx"}),
            ("read_data", {"path": "a.csv"}),
            ("database_schema", {"database": "a.db"}),
            ("http_request", {"method": "GET", "url": "https://example.com"}),
        ]:
            self.assertIsNone(policy.check(call(name, **args)), name)

    def test_write_tools_require_approval_or_are_blocked(self) -> None:
        for name in [
            "create_pdf",
            "edit_pdf",
            "compile_latex",
            "write_docx",
            "write_xlsx",
            "create_pptx",
            "write_data",
            "download_url",
            "execute_notebook",
        ]:
            self.assertIn(
                "not allowed", Policy("read-only").check(call(name)) or ""
            )
            self.assertIn(
                "approval required", Policy("safe").check(call(name)) or ""
            )

        approved = Policy("safe", approver=lambda call, reason: True)
        self.assertIsNone(approved.check(call("create_pdf")))
        denied = Policy("safe", approver=lambda call, reason: False)
        self.assertIn("approval denied", denied.check(call("create_pdf")) or "")

    def test_sql_query_distinguishes_reads_from_writes_and_remote(self) -> None:
        read_only = Policy("read-only")
        select = call(
            "sql_query", database="app.db", query="SELECT * FROM users"
        )
        update = call(
            "sql_query", database="app.db", query="UPDATE users SET x = 1"
        )
        remote = call(
            "sql_query",
            database="postgresql://example/db",
            query="SELECT 1",
        )
        self.assertIsNone(read_only.check(select))
        self.assertIn("blocked", read_only.check(update) or "")
        self.assertIn("approval required", Policy("safe").check(update) or "")
        self.assertIn("approval required", Policy("safe").check(remote) or "")

    def test_http_methods_are_classified(self) -> None:
        policy = Policy("read-only")
        self.assertIsNone(
            policy.check(call("http_request", method="GET", url="https://x"))
        )
        self.assertIn(
            "blocked",
            policy.check(call("http_request", method="POST", url="https://x"))
            or "",
        )


class ScriptedAgent:
    """Provider that requests one tool, then returns a final answer."""

    def __init__(self, tool_calls: list[ToolCall], final_text: str) -> None:
        self.tool_calls = tool_calls
        self.final_text = final_text
        self.calls = 0

    def complete(self, model, system, messages, tools) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                text="",
                tool_calls=self.tool_calls,
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1),
            )
        return ProviderResponse(
            text=self.final_text,
            tool_calls=[],
            finish_reason="stop",
            usage=TokenUsage(1, 1),
        )


class OpenShellToolIntegrationTests(unittest.TestCase):
    sandbox: OpenShellSandbox
    tools: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.sandbox = OpenShellSandbox(
            image=TEST_IMAGE,
            workspace="tool-tests",
            policy_path=NETWORK_POLICY_PATH,
        )
        try:
            cls.sandbox.start()
        except SandboxUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from None
        except SandboxError as exc:
            raise unittest.SkipTest(f"test image unavailable: {exc}") from None
        items = core_tools(cls.sandbox.workspace, cls.sandbox)
        items.extend(extended_tools(cls.sandbox.workspace, cls.sandbox))
        cls.tools = {item.name: item for item in items}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sandbox.stop()

    def run_tool(self, name: str, **kwargs: object) -> str:
        return str(self.tools[name].run(**kwargs))

    def test_00_loop_discovers_and_invokes_an_extended_tool(self) -> None:
        provider = ScriptedAgent(
            [
                ToolCall(
                    id="fetch-call",
                    name="web_fetch",
                    args={"url": "https://example.com", "max_chars": "200"},
                )
            ],
            "The page is reachable.",
        )
        messages = [{"role": "user", "text": "Fetch example.com."}]
        result = run_loop(
            provider=provider,
            model="scripted",
            system="test",
            messages=messages,
            tools=self.tools,
            before_tool=Policy("yolo").check,
            policy=TerminationPolicy(),
        )
        self.assertIs(result.status, RunStatus.COMPLETED)
        observation = next(
            message["text"]
            for message in messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "fetch-call"
        )
        self.assertIn("Example Domain", observation)
        self.assertEqual(result.text, "The page is reachable.")

    def test_01_all_tools_and_existing_tools_are_registered(self) -> None:
        self.assertEqual(len(self.tools), 24)
        self.assertTrue(set(EXTENDED_NAMES).issubset(self.tools))
        for name in ["read_file", "write_file", "edit_file", "bash", "list_files", "grep"]:
            self.assertIn(name, self.tools)

    def test_02_web_search_and_fetch_use_the_network(self) -> None:
        search = json.loads(
            self.run_tool("web_search", query="NVIDIA OpenShell", max_results="2")
        )
        self.assertGreaterEqual(len(search["results"]), 1)
        self.assertTrue(search["results"][0]["url"].startswith("http"))
        page = json.loads(
            self.run_tool("web_fetch", url="https://example.com", max_chars="300")
        )
        self.assertEqual(page["status"], 200)
        self.assertIn("Example Domain", page["title"])

    def test_03_http_and_download(self) -> None:
        response = json.loads(
            self.run_tool(
                "http_request",
                method="GET",
                url="https://httpbin.org/get?probe=1",
            )
        )
        self.assertEqual(response["status"], 200)
        downloaded = json.loads(
            self.run_tool(
                "download_url",
                url="https://example.com",
                destination="downloads/example.html",
            )
        )
        self.assertGreater(downloaded["bytes"], 100)
        listing = self.run_tool("list_files", pattern="downloads/*")
        self.assertIn("downloads/example.html", listing)

    def test_04_network_denial_is_reported_without_bypass(self) -> None:
        result = self.run_tool("web_fetch", url="https://example.org")
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("network policy", result.lower())

    def test_05_request_timeout_is_reported(self) -> None:
        result = self.run_tool(
            "http_request",
            method="GET",
            url="https://httpbin.org/delay/10",
            timeout="5",
        )
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("timed out", result.lower())

    def test_10_pdf_create_read_merge_rotate_split(self) -> None:
        created = json.loads(
            self.run_tool(
                "create_pdf",
                path="pdf/one.pdf",
                content=json.dumps(
                    {"title": "Report", "paragraphs": ["first page"], "tables": [[["a", "b"], ["1", "2"]]]}
                ),
            )
        )
        self.assertEqual(created["pages"], 1)
        read = json.loads(self.run_tool("read_pdf", path="pdf/one.pdf"))
        self.assertEqual(read["pages"], 1)
        self.assertIn("first page", read["text"])
        self.assertIn("Report", read["text"])

        self.run_tool("create_pdf", path="pdf/two.pdf", content="second page")
        merged = json.loads(
            self.run_tool(
                "edit_pdf",
                operation="merge",
                inputs=json.dumps(["pdf/one.pdf", "pdf/two.pdf"]),
                output="pdf/merged.pdf",
            )
        )
        self.assertEqual(merged["pages"], 2)
        rotated = json.loads(
            self.run_tool(
                "edit_pdf",
                operation="rotate",
                input="pdf/merged.pdf",
                output="pdf/rotated.pdf",
                angle="90",
            )
        )
        self.assertEqual(rotated["pages"], 2)
        split = json.loads(
            self.run_tool(
                "edit_pdf",
                operation="split",
                input="pdf/merged.pdf",
                output="pdf/split",
            )
        )
        self.assertEqual(len(split["files"]), 2)
        extracted = json.loads(
            self.run_tool("read_pdf", path="pdf/merged.pdf", pages="2")
        )
        self.assertEqual(extracted["selected_pages"], [2])
        self.assertIn("second page", extracted["text"])

    def test_11_latex_compiles_inside_the_sandbox(self) -> None:
        self.run_tool(
            "write_file",
            path="latex/doc.tex",
            content=(
                "\\documentclass{article}\n"
                "\\begin{document}\nHello from LaTeX.\n"
                "\\end{document}\n"
            ),
        )
        result = json.loads(self.run_tool("compile_latex", path="latex/doc.tex"))
        self.assertEqual(result["pages"], 1)
        self.assertIn("doc.pdf", result["path"])
        read = json.loads(self.run_tool("read_pdf", path="latex/doc.pdf"))
        self.assertIn("Hello from LaTeX", read["text"])

    def test_12_docx_round_trip(self) -> None:
        content = json.dumps(
            {
                "title": "Notes",
                "sections": [{"heading": "Intro", "paragraphs": ["alpha"], "bullets": ["beta"]}],
                "tables": [[["k", "v"], ["a", "1"]]],
            }
        )
        written = json.loads(
            self.run_tool("write_docx", path="docs/notes.docx", content=content)
        )
        self.assertGreater(written["bytes"], 0)
        read = json.loads(self.run_tool("read_docx", path="docs/notes.docx"))
        texts = [item["text"] for item in read["paragraphs"]]
        self.assertIn("Notes", texts)
        self.assertIn("alpha", texts)
        self.assertIn("beta", texts)
        self.assertEqual(read["tables"][0][0], ["k", "v"])

    def test_13_xlsx_round_trip(self) -> None:
        content = json.dumps({"sheets": [{"name": "Data", "rows": [["a", "b"], [1, 2]]}]})
        written = json.loads(
            self.run_tool("write_xlsx", path="sheets/data.xlsx", content=content)
        )
        self.assertEqual(written["sheets"], ["Data"])
        read = json.loads(self.run_tool("read_xlsx", path="sheets/data.xlsx"))
        self.assertEqual(read["sheets"]["Data"]["rows"][0], ["a", "b"])
        self.assertEqual(read["sheets"]["Data"]["rows"][1], [1, 2])

    def test_14_pptx_is_readable(self) -> None:
        content = json.dumps(
            {"slides": [{"title": "One", "bullets": ["a", "b"]}, {"title": "Two", "bullets": ["c"]}]}
        )
        written = json.loads(
            self.run_tool("create_pptx", path="slides/deck.pptx", content=content)
        )
        self.assertEqual(written["slides"], 2)
        verification = self.run_tool(
            "bash",
            command=(
                "python3 -c \"from pptx import Presentation; "
                "p=Presentation('slides/deck.pptx'); "
                "print(len(p.slides), p.slides[0].shapes.title.text)\""
            ),
            timeout="30",
        )
        self.assertIn("2 One", verification)

    def test_20_sql_query_and_schema(self) -> None:
        created = json.loads(
            self.run_tool(
                "sql_query",
                database="db/app.db",
                query="create table users (id integer primary key, name text)",
            )
        )
        self.assertIn("engine", created)
        self.run_tool(
            "sql_query",
            database="db/app.db",
            query="insert into users (name) values (?)",
            params='["Ada"]',
        )
        schema = json.loads(
            self.run_tool("database_schema", database="db/app.db", table="users")
        )
        columns = schema["tables"]["users"]["columns"]
        self.assertEqual([column["name"] for column in columns], ["id", "name"])
        selected = json.loads(
            self.run_tool(
                "sql_query",
                database="db/app.db",
                query="select name from users where id = ?",
                params="[1]",
            )
        )
        self.assertEqual(selected["rows"], [["Ada"]])
        inserted = json.loads(
            self.run_tool(
                "sql_query",
                database="db/app.db",
                query="insert into users (name) values (?)",
                params='["Grace"]',
            )
        )
        self.assertEqual(inserted["rowcount"], 1)
        rejected = self.run_tool(
            "sql_query",
            database="db/app.db",
            query="delete from users",
            read_only="true",
        )
        self.assertIn("read_only=true", rejected)

    def test_21_data_formats(self) -> None:
        records = json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        for path in ("data/table.csv", "data/table.json", "data/table.parquet"):
            written = json.loads(self.run_tool("write_data", path=path, data=records))
            self.assertEqual(written["rows"], 2)
            read = json.loads(self.run_tool("read_data", path=path))
            self.assertEqual(read["rows"][0]["a"], 1)
            self.assertEqual(read["rows"][1]["b"], "y")

    def test_22_notebook_execution_and_errors(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "id": "ok",
                    "metadata": {},
                    "execution_count": None,
                    "source": "value = 6 * 7\nprint(value)\nvalue",
                    "outputs": [],
                },
                {
                    "cell_type": "code",
                    "id": "boom",
                    "metadata": {},
                    "execution_count": None,
                    "source": "1 / 0",
                    "outputs": [],
                },
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        self.run_tool("write_file", path="notebooks/run.ipynb", content=json.dumps(notebook))
        executed = json.loads(
            self.run_tool(
                "execute_notebook",
                path="notebooks/run.ipynb",
                output="notebooks/executed.ipynb",
                allow_errors="true",
            )
        )
        self.assertEqual(executed["executed_cells"], 2)
        self.assertIn("42", executed["cells"][0]["output"])
        self.assertTrue(any("ZeroDivisionError" in item["error"] for item in executed["errors"]))
        saved = self.run_tool("read_file", path="notebooks/executed.ipynb")
        self.assertIn("ZeroDivisionError", saved)

    def test_30_invalid_paths_are_rejected(self) -> None:
        self.assertIn("PermissionError", self.run_tool("read_pdf", path="../../etc/passwd"))
        self.assertIn(
            "PermissionError",
            self.run_tool("create_pdf", path="../escape.pdf", content="x"),
        )
        self.assertIn(
            "PermissionError",
            self.run_tool(
                "download_url",
                url="https://example.com",
                destination="../escape.html",
            ),
        )
        self.assertIn(
            "PermissionError",
            self.run_tool(
                "sql_query", database="../escape.db", query="select 1"
            ),
        )

    def test_31_malformed_documents_are_reported(self) -> None:
        self.run_tool("write_file", path="bad/broken.pdf", content="not a pdf")
        result = self.run_tool("read_pdf", path="bad/broken.pdf")
        self.assertTrue(result.startswith("ERROR:"))


class MissingDependencyTests(unittest.TestCase):
    """The built-in slim image must fail clearly, not silently, for documents."""

    sandbox: OpenShellSandbox
    tools: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.sandbox = OpenShellSandbox(image="python:3.12-slim", workspace="deps-tests")
        try:
            cls.sandbox.start()
        except (SandboxUnavailable, SandboxError) as exc:
            raise unittest.SkipTest(str(exc)) from None
        items = extended_tools(cls.sandbox.workspace, cls.sandbox)
        cls.tools = {item.name: item for item in items}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sandbox.stop()

    def test_missing_pdf_library_is_reported(self) -> None:
        result = self.tools["create_pdf"].run(path="x.pdf", content="hello")
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("reportlab", result)


if __name__ == "__main__":
    unittest.main()
