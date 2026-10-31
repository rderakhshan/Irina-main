"""Tests for Laya routing, the model catalog, and the OpenAI/Gemini adapters."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from src.back.harness.loop.runner import EventType, RunEvent
from src.back.harness.loop.state import RunStatus
from src.back.harness.orchestration.harness import Harness
from src.back.harness.orchestration.session import load
from src.back.harness.providers.base import (
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from src.back.harness.providers.catalog import (
    ModelCatalog,
    ModelConfigurationError,
    ModelSpec,
)
from src.back.harness.providers.gemini import GeminiProvider, _to_contents
from src.back.harness.providers.openai import OpenAIProvider, _to_input
from src.back.harness.providers.factory import ProviderFactory
from src.back.harness.routing.laya_router import LayaRouter, RoutingDecision
from src.back.harness.tools_and_guardrails import LocalWorkspace


class _StreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ScriptedProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.model = "fake-model"

    def complete(self, model, system, messages, tools):
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class CatalogTests(unittest.TestCase):
    def test_default_catalog_has_the_enabled_providers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "d",
                "OPENAI_API_KEY": "o",
                "GEMINI_API_KEY": "g",
            },
            clear=False,
        ):
            catalog = ModelCatalog.from_env()
            available = catalog.available()
        # Gemini stays available through its adapter but is not part of the
        # default routing catalog.
        self.assertEqual(
            {spec.provider for spec in available},
            {"deepseek", "openai"},
        )
        self.assertTrue(all(spec.supports_tools for spec in available))

    def test_missing_credentials_exclude_models(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "d",
                "OPENAI_API_KEY": "",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
            },
            clear=False,
        ):
            available = ModelCatalog.from_env().available()
        self.assertTrue(available)
        self.assertEqual({spec.provider for spec in available}, {"deepseek"})

    def test_capability_and_enabled_filters(self) -> None:
        catalog = ModelCatalog(
            [
                ModelSpec("a", "deepseek", "a", supports_tools=False),
                ModelSpec("b", "openai", "b", supports_streaming=False),
                ModelSpec("c", "gemini", "c", enabled=False),
                ModelSpec("d", "deepseek", "d"),
            ]
        )
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "d",
                "OPENAI_API_KEY": "o",
                "GEMINI_API_KEY": "g",
            },
            clear=False,
        ):
            self.assertEqual(
                [spec.id for spec in catalog.available()],
                ["b", "d"],
            )
            self.assertEqual(
                [spec.id for spec in catalog.available(require_streaming=True)],
                ["d"],
            )

    def test_manual_resolution_rejects_unknown_or_missing_credentials(self) -> None:
        catalog = ModelCatalog([ModelSpec("x", "openai", "x")])
        with self.assertRaises(ModelConfigurationError):
            catalog.get("missing")
        factory = ProviderFactory(catalog)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(ModelConfigurationError):
                factory.require_spec(catalog.get("x"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "o"}, clear=False):
            self.assertEqual(catalog.get("x").provider, "openai")


class LayaRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            ModelSpec("deepseek/d", "deepseek", "d", description="coding"),
            ModelSpec("openai/o", "openai", "o", description="writing"),
            ModelSpec("gemini/g", "gemini", "g", description="multimodal"),
        ]

    def test_failure_uses_fallback(self) -> None:
        router = LayaRouter(timeout_seconds=2)

        class Broken:
            def predict(self, state, questions):
                raise RuntimeError("no checkpoint")

        router._ensure_router = lambda: Broken()  # type: ignore[method-assign]
        decision = router.select("task", "", self.candidates, self.candidates[0])
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.spec, self.candidates[0])
        self.assertIn("RuntimeError", router.last_diagnostic["reason"])

    def test_checkpoint_temperature_warning_is_suppressed(self) -> None:
        from src.back.harness.routing.laya_router import (
            _quiet_checkpoint_temperature_warning,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _quiet_checkpoint_temperature_warning():
                warnings.warn(
                    "laya: this checkpoint ships invalid temperatures or values "
                    "outside [0.5, 5]; using choice:11+=0.10058280825614929 -> "
                    "0.5. Treat confidence from the affected entries as "
                    "uncalibrated.",
                    RuntimeWarning,
                )
                warnings.warn("unrelated runtime problem", RuntimeWarning)
        messages = [str(item.message) for item in caught]
        self.assertFalse(
            any("invalid temperatures" in message for message in messages)
        )
        self.assertTrue(any("unrelated runtime problem" in m for m in messages))

    def test_invalid_choice_uses_fallback(self) -> None:
        router = LayaRouter(timeout_seconds=2)

        class Invented:
            def predict(self, state, questions):
                return {"answers": {"model": {"choice": "invented"}}}

        router._ensure_router = lambda: Invented()  # type: ignore[method-assign]
        decision = router.select("task", "", self.candidates, self.candidates[2])
        self.assertTrue(decision.fallback)
        self.assertEqual(decision.spec, self.candidates[2])

    def test_calibration_path_is_passed_to_laya(self) -> None:
        import sys
        import types

        captured: dict = {}

        class FakeRouter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("laya")
        fake.Router = FakeRouter  # type: ignore[attr-defined]
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as handle:
            handle.write("{}")
            path = Path(handle.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        with patch.dict(
            os.environ, {"IRINA_LAYA_CALIBRATION": str(path)}, clear=False
        ), patch.dict(sys.modules, {"laya": fake}):
            router = LayaRouter()
            router._silence_checkpoint_download_progress = lambda: None
            router._ensure_router()
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(
            captured["agent_kwargs"]["calibration"], str(path.resolve())
        )

    def test_missing_calibration_file_is_refused(self) -> None:
        import sys
        import types

        fake = types.ModuleType("laya")
        fake.Router = lambda **kwargs: object()  # type: ignore[attr-defined]
        missing = Path(tempfile.gettempdir()) / "irina-missing-calibration.json"
        with patch.dict(
            os.environ, {"IRINA_LAYA_CALIBRATION": str(missing)}, clear=False
        ), patch.dict(sys.modules, {"laya": fake}):
            router = LayaRouter()
            router._silence_checkpoint_download_progress = lambda: None
            with self.assertRaises(RuntimeError):
                router._ensure_router()

    def test_no_calibration_keeps_plain_router_arguments(self) -> None:
        import sys
        import types

        captured: dict = {}

        class FakeRouter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("laya")
        fake.Router = FakeRouter  # type: ignore[attr-defined]
        with patch.dict(
            os.environ, {"IRINA_LAYA_CALIBRATION": ""}, clear=False
        ), patch.dict(sys.modules, {"laya": fake}):
            router = LayaRouter()
            router._silence_checkpoint_download_progress = lambda: None
            router._ensure_router()
        self.assertNotIn("agent_kwargs", captured)

    def test_timeout_uses_fallback(self) -> None:
        router = LayaRouter(timeout_seconds=0.2)
        router._warmed = True

        class Slow:
            def predict(self, state, questions):
                time.sleep(1.0)
                return {"answers": {"model": {"choice": "openai/o"}}}

        router._ensure_router = lambda: Slow()  # type: ignore[method-assign]
        start = time.monotonic()
        decision = router.select("task", "", self.candidates, self.candidates[0])
        self.assertTrue(decision.fallback)
        self.assertLess(time.monotonic() - start, 0.8)

    def test_valid_choice_selects_actual_candidate(self) -> None:
        router = LayaRouter(timeout_seconds=2)

        class Valid:
            def predict(self, state, questions):
                self.questions = questions
                return {
                    "answers": {"model": {"choice": "openai/o"}},
                    "routing": {"detection": {"language": "en"}},
                }

        fake = Valid()
        router._ensure_router = lambda: fake  # type: ignore[method-assign]
        decision = router.select("task", "ctx", self.candidates, self.candidates[0])
        self.assertFalse(decision.fallback)
        self.assertEqual(decision.spec, self.candidates[1])
        self.assertEqual(router.device, "cpu")
        self.assertIn("openai/o", fake.questions["model"]["criteria"])

    @unittest.skipUnless(
        os.environ.get("IRINA_LAYA_LIVE") == "1",
        "set IRINA_LAYA_LIVE=1 to run the real CPU checkpoint",
    )
    def test_real_laya_cpu_selection(self) -> None:
        router = LayaRouter(timeout_seconds=60)
        decision = router.select(
            "Write a Python function to parse a CSV file and handle errors.",
            "",
            self.candidates,
            self.candidates[0],
        )
        self.assertIn(decision.spec, self.candidates)
        self.assertFalse(decision.fallback)
        self.assertTrue(router._warmed)


class HarnessRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.backend = LocalWorkspace(self.root)
        self.catalog = ModelCatalog(
            [
                ModelSpec("deepseek/d", "deepseek", "d", description="coding"),
                ModelSpec("openai/o", "openai", "o", description="writing"),
            ]
        )
        self.env = {
            "DEEPSEEK_API_KEY": "test-deepseek",
            "OPENAI_API_KEY": "test-openai",
            "GEMINI_API_KEY": "test-gemini",
            # Neutralise a value loaded from the developer's .env by the CLI
            # tests, which otherwise forces manual routing here.
            "IRINA_PROVIDER": "auto",
            "IRINA_DEFAULT_PROVIDER": "deepseek",
            "IRINA_FALLBACK_PROVIDER": "deepseek",
        }

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _harness(self, router) -> Harness:
        return Harness(
            workdir=str(self.root),
            workspace_backend=self.backend,
            persist=False,
            skills_dir="",
            provider_mode="auto",
            model_catalog=self.catalog,
            laya_router=router,
        )

    def test_auto_route_selects_model_and_keeps_follow_up_model(self) -> None:
        class Router:
            def __init__(self):
                self.calls = 0

            def select(self, task, context, candidates, fallback):
                self.calls += 1
                return RoutingDecision(candidates[1], "scripted route", "laya")

        router = Router()
        with patch.dict(os.environ, self.env, clear=False):
            harness = self._harness(router)
            harness._maybe_route("independent task")
            self.assertEqual(harness.model_spec.id, "openai/o")
            self.assertEqual(router.calls, 1)
            harness._session_loaded = True
            harness._session_status = "interrupted"
            harness._maybe_route("follow-up")
            self.assertEqual(router.calls, 1)
            self.assertEqual(harness.model_spec.id, "openai/o")
        harness.close()

    def test_auto_route_selects_available_provider_when_default_lacks_key(self) -> None:
        class Router:
            def select(self, task, context, candidates, fallback):
                return RoutingDecision(candidates[0], "scripted", "laya")

        env = dict(self.env)
        env["DEEPSEEK_API_KEY"] = ""
        env["OPENAI_API_KEY"] = "test-openai"
        env["IRINA_DEFAULT_PROVIDER"] = "deepseek"
        with patch.dict(os.environ, env, clear=False):
            harness = Harness(
                workdir=str(self.root),
                workspace_backend=self.backend,
                persist=False,
                skills_dir="",
                provider_mode="auto",
                model_catalog=self.catalog,
                laya_router=Router(),
            )
            self.assertEqual(harness.provider_name, "openai")
        harness.close()

    def test_manual_missing_credentials_is_clear(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": ""},
            clear=False,
        ):
            with self.assertRaises(ModelConfigurationError):
                Harness(
                    workdir=str(self.root),
                    workspace_backend=self.backend,
                    persist=False,
                    skills_dir="",
                    provider_mode="openai",
                    model_catalog=self.catalog,
                )

    def test_manual_selection_bypasses_laya(self) -> None:
        class Router:
            def __init__(self):
                self.calls = 0

            def select(self, *args, **kwargs):
                self.calls += 1
                return RoutingDecision(args[2][1], "route", "laya")

        router = Router()
        with patch.dict(os.environ, self.env, clear=False):
            harness = self._harness(router)
            harness.set_model(provider_name="openai")
            self.assertEqual(harness.routing_mode, "manual")
            self.assertEqual(harness.model, "o")
            harness._maybe_route("another task")
            self.assertEqual(router.calls, 0)
        harness.close()

    def test_model_event_and_session_metadata(self) -> None:
        events: list[RunEvent] = []

        class Router:
            def select(self, task, context, candidates, fallback):
                return RoutingDecision(candidates[0], "selected", "laya")

        provider = _ScriptedProvider(
            [ProviderResponse("done", [], "stop", TokenUsage(1, 1))]
        )
        with patch.dict(os.environ, self.env, clear=False):
            harness = Harness(
                workdir=str(self.root),
                provider=provider,
                model="d",
                provider_name="deepseek",
                workspace_backend=self.backend,
                policy="yolo",
                on_event=events.append,
                persist=True,
                state_dir=self.root / "state",
                skills_dir="",
            )
            result = harness.run("hello")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            path = harness.session_path
            self.assertIsNotNone(path)
            data = load(path)
            self.assertEqual(data.metadata["provider"], "deepseek")
            self.assertEqual(data.metadata["model"], "d")
            self.assertIn("routing_mode", data.metadata)
            harness.close()
        self.assertFalse(any(event.type is EventType.MODEL for event in events))


class OpenAITests(unittest.TestCase):
    def test_complete_normalizes_text_tools_usage_and_continuation(self) -> None:
        payload = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "a.txt"}',
                },
                {"type": "reasoning", "encrypted_content": "opaque"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch(
            "src.back.harness.providers.openai.post_json",
            return_value=payload,
        ) as post:
            provider = OpenAIProvider(api_key="key", model="gpt-test")
            response = provider.complete(
                "gpt-test",
                "system",
                [
                    {"role": "user", "text": "read it"},
                    {"role": "tool", "tool_call_id": "call_0", "text": "ok"},
                ],
                [
                    {
                        "schema": {
                            "name": "read_file",
                            "description": "read",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ],
            )
        self.assertEqual(response.text, "hello")
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].args, {"path": "a.txt"})
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.usage, TokenUsage(10, 5))
        self.assertIn("reasoning_items", response.continuation or {})
        body = post.call_args.args[1]
        self.assertEqual(body["instructions"], "system")
        self.assertEqual(body["tools"][0]["type"], "function")
        self.assertEqual(body["tools"][0]["name"], "read_file")
        self.assertIn(
            {"type": "function_call_output", "call_id": "call_0", "output": "ok"},
            body["input"],
        )

    def test_malformed_arguments_are_recoverable(self) -> None:
        payload = {
            "id": "resp_2",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "read_file",
                    "arguments": "{",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with patch(
            "src.back.harness.providers.openai.post_json",
            return_value=payload,
        ):
            response = OpenAIProvider(api_key="key").complete(
                "gpt-test", "", [{"role": "user", "text": "x"}], []
            )
        self.assertIsNotNone(response.tool_calls[0].args_error)
        self.assertEqual(response.tool_calls[0].args, {})

    def test_streaming_emits_text_tool_and_finish(self) -> None:
        lines = [
            b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
            b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_s","name":"read_file","arguments":""}}\n\n',
            b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\"path\\":\\"a\\"}"}\n\n',
            b'data: {"type":"response.completed","response":{"id":"resp_s","status":"completed","output":[{"type":"function_call","call_id":"call_s","name":"read_file","arguments":"{\\"path\\":\\"a\\"}"}],"usage":{"input_tokens":2,"output_tokens":3}}}\n\n',
        ]
        with patch(
            "src.back.harness.providers.openai.open_stream",
            return_value=_StreamResponse(lines),
        ):
            events = list(
                OpenAIProvider(api_key="key").stream(
                    "gpt-test", "", [{"role": "user", "text": "x"}], []
                )
            )
        types = [event.type for event in events]
        self.assertIn("text_delta", [item.value for item in types])
        self.assertIn("finish", [item.value for item in types])
        self.assertEqual(events[-1].finish_reason, "tool_calls")
        self.assertEqual(events[-1].usage, TokenUsage(2, 3))

    def test_missing_key_is_only_checked_when_provider_is_created(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                OpenAIProvider()


class GeminiTests(unittest.TestCase):
    def test_complete_normalizes_text_function_call_and_continuation(self) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hello"},
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "a.txt"},
                                },
                                "thoughtSignature": "sig",
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
            },
        }
        with patch(
            "src.back.harness.providers.gemini.post_json",
            return_value=payload,
        ) as post:
            provider = GeminiProvider(api_key="key", model="gemini-test")
            response = provider.complete(
                "gemini-test", "system", [{"role": "user", "text": "x"}], []
            )
        self.assertEqual(response.text, "hello")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].args, {"path": "a.txt"})
        self.assertTrue(response.tool_calls[0].id)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.usage, TokenUsage(8, 4))
        self.assertIn("gemini_parts", response.continuation or {})
        body = post.call_args.args[1]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "system")

    def test_malformed_function_arguments_are_recoverable(self) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "read_file", "args": []}}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        with patch(
            "src.back.harness.providers.gemini.post_json",
            return_value=payload,
        ):
            response = GeminiProvider(api_key="key").complete(
                "gemini-test", "", [{"role": "user", "text": "x"}], []
            )
        self.assertIsNotNone(response.tool_calls[0].args_error)

    def test_streaming_emits_text_tool_and_finish(self) -> None:
        lines = [
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n\n',
            b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"a"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3}}\n\n',
        ]
        with patch(
            "src.back.harness.providers.gemini.open_stream",
            return_value=_StreamResponse(lines),
        ):
            events = list(
                GeminiProvider(api_key="key").stream(
                    "gemini-test", "", [{"role": "user", "text": "x"}], []
                )
            )
        self.assertIn(
            "text_delta", [event.type.value for event in events]
        )
        self.assertEqual(events[-1].finish_reason, "tool_calls")
        self.assertEqual(events[-1].usage, TokenUsage(2, 3))

    def test_contents_group_tool_responses_and_preserve_state(self) -> None:
        contents = _to_contents(
            [
                {"role": "user", "text": "read"},
                {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [
                        {"id": "c1", "name": "read_file", "args": {"path": "a"}}
                    ],
                    "provider_state": {
                        "gemini_parts": [
                            {
                                "functionCall": {"name": "read_file", "args": {"path": "a"}}
                            }
                        ]
                    },
                },
                {"role": "tool", "tool_call_id": "c1", "name": "read_file", "text": "ok"},
                {"role": "tool", "tool_call_id": "c2", "name": "grep", "text": "found"},
            ]
        )
        self.assertEqual(contents[-1]["role"], "user")
        self.assertEqual(len(contents[-1]["parts"]), 2)
        self.assertIn("functionResponse", contents[-1]["parts"][0])

    def test_missing_key_is_only_checked_when_provider_is_created(self) -> None:
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                GeminiProvider()


class EnvironmentFileTests(unittest.TestCase):
    def test_cli_exposes_provider_and_model_flags(self) -> None:
        from src.back.harness.orchestration import cli

        args = cli.build_parser().parse_args(
            ["--provider", "gemini", "--model", "gemini-3.5-flash"]
        )
        self.assertEqual(args.provider, "gemini")
        self.assertEqual(args.model, "gemini-3.5-flash")
        self.assertEqual(
            cli.build_parser().parse_args(["--provider", "auto"]).provider,
            "auto",
        )

    def test_env_example_documents_new_configuration(self) -> None:
        text = Path(".env.example").read_text(encoding="utf-8")
        for key in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_BASE_URL",
            "GEMINI_BASE_URL",
            "IRINA_PROVIDER",
            "IRINA_DEFAULT_PROVIDER",
            "IRINA_DEFAULT_MODEL",
            "IRINA_FALLBACK_PROVIDER",
            "IRINA_FALLBACK_MODEL",
            "DEEPSEEK_MODEL",
            "OPENAI_MODEL",
            "GEMINI_MODEL",
            "IRINA_LAYA_ENABLED",
            "IRINA_LAYA_DEVICE",
            "IRINA_LAYA_TIMEOUT_SECONDS",
        ]:
            self.assertIn(key, text)
        self.assertNotIn("sk-", text)

    def test_local_env_retains_deepseek_and_adds_new_keys(self) -> None:
        path = Path(".env")
        if not path.is_file():
            self.skipTest("local .env is absent")
        text = path.read_text(encoding="utf-8")
        self.assertIn("DEEPSEEK_API_KEY", text)
        for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "IRINA_PROVIDER"]:
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main()
