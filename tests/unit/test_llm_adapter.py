"""
Group 5 — LLM Adapter tests for bash_agent.llm.

T-23  Backend routing matrix (P0)
T-24  OpenRouter payload normalization (P0)
T-25  Gemini payload stripping + cost monkey-patch (P0)
T-26  Client cache identity (P2)

The adapter decides which API every request hits and how payloads are
shaped per backend. All tests are strictly offline:

  * ``bash_agent.config`` binds env vars at import time, so mutating
    os.environ after import has NO effect. Tests patch the config
    attributes directly (e.g. ``config.GEMINI_API_KEY``).
  * Payload tests seed ``llm._CLIENT_CACHE`` with FakeLLMClient (T-00c);
    the fake records call kwargs for extra_body assertions.
  * The base class replaces ``bash_agent.llm.OpenAI`` with a recording
    stand-in, so ANY code path that tries to build a real client fails
    loudly inside the test process instead of issuing an HTTP request —
    even when ambient API keys exist in the environment.

Every test isolates ``llm._CLIENT_CACHE`` via mock.patch.dict(clear=True)
so cache state never leaks between tests.
"""
import json
import unittest
from unittest import mock

from bash_agent import config, llm
from tests.helpers.fakes import FakeLLMClient, make_fake_response


class _OpenAIRecorder:
    """
    Stands in for openai.OpenAI. Records construction kwargs and appends
    itself to `instances`. Has no chat/embeddings attributes, so any
    unexpected real-client construction surfaces as a loud AttributeError,
    never as network traffic.
    """

    instances = []
    init_kwargs_log = []

    def __init__(self, **kwargs):
        type(self).init_kwargs_log.append(kwargs)
        type(self).instances.append(self)


class _OfflineLLMTestCase(unittest.TestCase):
    """Base class: isolated client cache + poisoned OpenAI constructor."""

    def setUp(self):
        cache_patcher = mock.patch.dict(llm._CLIENT_CACHE, {}, clear=True)
        cache_patcher.start()
        self.addCleanup(cache_patcher.stop)

        openai_patcher = mock.patch.object(llm, "OpenAI", _OpenAIRecorder)
        openai_patcher.start()
        self.addCleanup(openai_patcher.stop)

        _OpenAIRecorder.instances.clear()
        _OpenAIRecorder.init_kwargs_log.clear()

    @staticmethod
    def _seed(backend: str, response=None) -> FakeLLMClient:
        """Seed llm._CLIENT_CACHE[backend] with a FakeLLMClient; return it."""
        fake = FakeLLMClient(responses=[response] if response is not None else [])
        llm._CLIENT_CACHE[backend] = fake
        return fake


# ---------------------------------------------------------------------------
# T-23 — Backend routing matrix
# ---------------------------------------------------------------------------

class TestBackendRouting(_OfflineLLMTestCase):
    """T-23: get_backend() routes google/* models to gemini only when
    GEMINI_API_KEY is available; everything else goes to openrouter."""

    def _route(self, model, gemini_key):
        with mock.patch.object(config, "GEMINI_API_KEY", gemini_key):
            return llm.get_backend(model)

    def test_routing_matrix(self):
        cases = [
            # (model, GEMINI_API_KEY, expected backend)
            ("google/gemini-3-flash-preview", "fake-g-key", "gemini"),
            ("google/gemini-3-flash-preview", None, "openrouter"),
            ("openai/gpt-5-mini", "fake-g-key", "openrouter"),
            ("openai/gpt-5-mini", None, "openrouter"),
            ("deepseek/deepseek-v4-pro", "fake-g-key", "openrouter"),
            ("anthropic/claude-x", None, "openrouter"),
            (None, "fake-g-key", "openrouter"),   # None-safe
            ("", "fake-g-key", "openrouter"),     # empty-string safe
        ]
        for model, key, expected in cases:
            with self.subTest(model=model, gemini_key=key):
                self.assertEqual(self._route(model, key), expected)

    def test_gemini_routing_requires_key_not_just_prefix(self):
        """A google/ prefix alone must NOT select gemini when key is unset."""
        self.assertEqual(
            self._route("google/gemini-3-flash-preview", None), "openrouter"
        )


# ---------------------------------------------------------------------------
# T-24 — OpenRouter payload normalization
# ---------------------------------------------------------------------------

class TestOpenRouterPayloadNormalization(_OfflineLLMTestCase):
    """T-24: create_chat_completion() on the openrouter backend injects
    reasoning effort and MODEL_PROVIDERS whitelists into extra_body."""

    MESSAGES = [{"role": "user", "content": "hello"}]

    def _call(self, model, reasoning_effort=None, max_tokens=None):
        scripted = make_fake_response(content="ok")
        fake = self._seed("openrouter", scripted)
        result = llm.create_chat_completion(
            model=model,
            messages=self.MESSAGES,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return result, fake.calls[-1]

    def test_whitelisted_model_gets_reasoning_and_provider(self):
        _, kwargs = self._call(
            "deepseek/deepseek-v4-pro", reasoning_effort="low"
        )
        ebody = kwargs["extra_body"]
        self.assertEqual(ebody["reasoning"]["effort"], "low")
        self.assertEqual(ebody["provider"]["only"], ["deepseek"])
        # OpenRouter keeps the full provider-prefixed slug
        self.assertEqual(kwargs["model"], "deepseek/deepseek-v4-pro")

    def test_non_whitelisted_model_has_no_provider_key(self):
        _, kwargs = self._call(
            "openai/gpt-5-mini", reasoning_effort="high"
        )
        ebody = kwargs["extra_body"]
        self.assertNotIn("provider", ebody)
        self.assertEqual(ebody, {"reasoning": {"effort": "high"}})

    def test_reasoning_none_adds_no_reasoning_key(self):
        """reasoning_effort=None must not inject a 'reasoning' key — but a
        whitelisted model still gets its provider whitelist."""
        _, kwargs = self._call(
            "deepseek/deepseek-v4-pro", reasoning_effort=None
        )
        ebody = kwargs["extra_body"]
        self.assertNotIn("reasoning", ebody)
        self.assertEqual(ebody["provider"]["only"], ["deepseek"])

    def test_no_reasoning_no_whitelist_means_no_extra_body(self):
        _, kwargs = self._call(
            "openai/gpt-5-mini", reasoning_effort=None
        )
        self.assertNotIn("extra_body", kwargs)

    def test_messages_and_max_tokens_passthrough(self):
        _, kwargs = self._call(
            "deepseek/deepseek-v4-pro",
            reasoning_effort="medium",
            max_tokens=512,
        )
        self.assertIs(kwargs["messages"], self.MESSAGES)
        self.assertEqual(kwargs["max_tokens"], 512)

    def test_scripted_response_returned_unmodified(self):
        scripted = make_fake_response(content="pong")
        self._seed("openrouter", scripted)
        result = llm.create_chat_completion(
            model="openai/gpt-5-mini", messages=self.MESSAGES
        )
        self.assertIs(result, scripted)


# ---------------------------------------------------------------------------
# T-24a — OpenRouter App Attribution headers
# ---------------------------------------------------------------------------

class TestOpenRouterAttributionHeaders(_OfflineLLMTestCase):
    """T-24a: create_chat_completion()/create_embedding() on the openrouter
    backend must attach the App Attribution headers (HTTP-Referer,
    X-OpenRouter-Title, X-OpenRouter-Categories) via extra_headers, while
    the gemini backend stays header-free."""

    MESSAGES = [{"role": "user", "content": "hello"}]

    def _call_chat(self, model="openai/gpt-5-mini", **kw):
        scripted = make_fake_response(content="ok")
        fake = self._seed("openrouter", scripted)
        llm.create_chat_completion(
            model=model, messages=self.MESSAGES, **kw
        )
        return fake.calls[-1]

    def _call_embedding(self, model="perplexity/pplx-embed-v1-4b"):
        fake = self._seed("openrouter")
        llm.create_embedding(model=model, input_texts=["hello"])
        return fake.embedding_calls[-1]

    def test_attribution_headers_present_on_openrouter_chat(self):
        kwargs = self._call_chat()
        headers = kwargs.get("extra_headers", {})
        self.assertEqual(headers.get("HTTP-Referer"), config.APP_URL)
        self.assertEqual(headers.get("X-OpenRouter-Title"), config.APP_TITLE)
        self.assertEqual(headers.get("X-OpenRouter-Categories"), config.APP_CATEGORIES)

    def test_attribution_headers_present_with_extra_body(self):
        kwargs = self._call_chat(reasoning_effort="low")
        headers = kwargs.get("extra_headers", {})
        self.assertIn("HTTP-Referer", headers)
        self.assertIn("X-OpenRouter-Title", headers)
        self.assertIn("X-OpenRouter-Categories", headers)
        # extra_body still works alongside extra_headers
        self.assertEqual(kwargs["extra_body"]["reasoning"]["effort"], "low")

    def test_attribution_headers_present_on_openrouter_embedding(self):
        kwargs = self._call_embedding()
        headers = kwargs.get("extra_headers", {})
        self.assertEqual(headers.get("HTTP-Referer"), config.APP_URL)
        self.assertEqual(headers.get("X-OpenRouter-Title"), config.APP_TITLE)
        self.assertEqual(headers.get("X-OpenRouter-Categories"), config.APP_CATEGORIES)

    def test_no_attribution_headers_on_gemini_chat(self):
        scripted = make_fake_response(content="ok")
        fake = self._seed("gemini", scripted)
        with mock.patch.object(config, "GEMINI_API_KEY", "fake-g-key"):
            llm.create_chat_completion(
                model="google/gemini-3-flash-preview", messages=self.MESSAGES
            )
        kwargs = fake.calls[-1]
        self.assertNotIn("extra_headers", kwargs)

    def test_no_attribution_headers_on_gemini_embedding(self):
        fake = self._seed("gemini")
        with mock.patch.object(config, "GEMINI_API_KEY", "fake-g-key"):
            llm.create_embedding(
                model="google/gemini-3-flash-preview", input_texts=["hi"]
            )
        kwargs = fake.embedding_calls[-1]
        self.assertNotIn("extra_headers", kwargs)

    def test_header_values_match_config_constants(self):
        self.assertEqual(config.APP_URL, "https://github.com/geraldnilles/Bash-Agent")
        self.assertEqual(config.APP_TITLE, "Bash Agent")
        self.assertEqual(config.APP_CATEGORIES, "cli-agent")


# ---------------------------------------------------------------------------
# T-25 — Gemini payload stripping + cost monkey-patch
# ---------------------------------------------------------------------------

class TestGeminiPayloadStrippingAndCostPatch(_OfflineLLMTestCase):
    """T-25: create_chat_completion() on the gemini backend strips the
    google/ prefix, drops OpenRouter-only extras, and monkey-patches
    model_dump() to inject usage.cost from calculate_gemini_cost()."""

    MESSAGES = [{"role": "user", "content": "hello"}]

    def setUp(self):
        super().setUp()
        # Force gemini routing regardless of ambient environment
        key_patcher = mock.patch.object(config, "GEMINI_API_KEY", "test-key")
        key_patcher.start()
        self.addCleanup(key_patcher.stop)

    def _gemini_call(self, model="google/gemini-3-flash-preview", **resp_kwargs):
        scripted = make_fake_response(**resp_kwargs)
        fake = self._seed("gemini", scripted)
        result = llm.create_chat_completion(
            model=model,
            messages=self.MESSAGES,
            extra_body={"provider": {"only": ["x"]}},   # OpenRouter-only junk
            reasoning_effort="high",                    # OpenRouter-only junk
        )
        return result, fake.calls[-1]

    def test_model_stripped_and_extras_dropped(self):
        _, kwargs = self._gemini_call()
        self.assertEqual(kwargs["model"], "gemini-3-flash-preview")
        self.assertNotIn("extra_body", kwargs)

    def test_model_dump_cost_injected_from_token_counts(self):
        resp, _ = self._gemini_call(
            prompt_tokens=2_000_000, completion_tokens=1_000_000, cost=123.45
        )
        dumped = resp.model_dump()
        # flash tier: (2e6/1e6)*0.075 + (1e6/1e6)*0.30 == 0.45
        self.assertAlmostEqual(dumped["usage"]["cost"], 0.45, places=9)
        # Original usage fields preserved; bogus fake cost overwritten
        self.assertEqual(dumped["usage"]["prompt_tokens"], 2_000_000)
        self.assertEqual(dumped["usage"]["completion_tokens"], 1_000_000)
        self.assertIn("choices", dumped)

    def test_model_dump_json_reflects_patched_cost(self):
        resp, _ = self._gemini_call(prompt_tokens=1_000_000, completion_tokens=0)
        dumped = json.loads(resp.model_dump_json(indent=2))
        self.assertAlmostEqual(dumped["usage"]["cost"], 0.075, places=9)

    def test_openrouter_responses_are_not_cost_patched(self):
        scripted = make_fake_response(cost=0.42)
        self._seed("openrouter", scripted)
        result = llm.create_chat_completion(
            model="openai/gpt-5-mini", messages=self.MESSAGES
        )
        self.assertAlmostEqual(result.model_dump()["usage"]["cost"], 0.42)

    def test_missing_usage_left_unpatched(self):
        class _NoUsageResp:
            def model_dump(self):
                return {"choices": []}

        self._seed("gemini", _NoUsageResp())
        result = llm.create_chat_completion(
            model="google/gemini-3-flash-preview", messages=self.MESSAGES
        )
        self.assertIsInstance(result, _NoUsageResp)
        self.assertNotIn("usage", result.model_dump())

    def test_calculate_gemini_cost_tier_selection(self):
        # Per-1M-in + per-1M-out totals by tier substring
        cases = [
            ("gemini-3-flash-preview", 0.375),      # flash tier
            ("gemini-3.1-flash-lite-0827", 0.15),   # flash-lite tier
            ("totally-unknown-model", 0.75),        # default tier
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                cost = llm.calculate_gemini_cost(model, 1_000_000, 1_000_000)
                self.assertAlmostEqual(cost, expected, places=9)

    def test_calculate_gemini_cost_zero_and_proportional(self):
        self.assertEqual(llm.calculate_gemini_cost("any-model", 0, 0), 0.0)
        cost = llm.calculate_gemini_cost("gemini-3-flash", 500_000, 250_000)
        self.assertAlmostEqual(cost, 0.0375 + 0.075, places=9)


# ---------------------------------------------------------------------------
# T-26 — Client cache identity
# ---------------------------------------------------------------------------

class TestClientCacheIdentity(_OfflineLLMTestCase):
    """T-26: get_llm_client() caches per backend; seeding bypasses
    construction entirely. llm.OpenAI is already replaced by
    _OpenAIRecorder via the base class."""

    def test_same_backend_returns_same_client(self):
        c1 = llm.get_llm_client("openrouter")
        c2 = llm.get_llm_client("openrouter")
        self.assertIs(c1, c2)
        self.assertEqual(len(_OpenAIRecorder.instances), 1)  # built once

    def test_different_backends_return_different_clients(self):
        c = llm.get_llm_client("openrouter")
        g = llm.get_llm_client("gemini")
        self.assertIsNot(c, g)
        self.assertEqual(len(_OpenAIRecorder.instances), 2)

    def test_construction_uses_config_credentials_and_urls(self):
        with mock.patch.object(config, "OPENROUTER_API_KEY", "or-key"), \
             mock.patch.object(config, "GEMINI_API_KEY", "g-key"):
            llm.get_llm_client("openrouter")
            llm.get_llm_client("gemini")
        or_kwargs, g_kwargs = _OpenAIRecorder.init_kwargs_log
        self.assertEqual(or_kwargs["api_key"], "or-key")
        self.assertEqual(or_kwargs["base_url"], config.OPENROUTER_BASE_URL)
        self.assertEqual(g_kwargs["api_key"], "g-key")
        self.assertEqual(g_kwargs["base_url"], config.GEMINI_BASE_URL)

    def test_seeded_cache_bypasses_construction(self):
        sentinel = object()
        llm._CLIENT_CACHE["openrouter"] = sentinel
        self.assertIs(llm.get_llm_client("openrouter"), sentinel)
        self.assertEqual(_OpenAIRecorder.instances, [])

    def test_default_backend_is_openrouter(self):
        sentinel = object()
        llm._CLIENT_CACHE["openrouter"] = sentinel
        self.assertIs(llm.get_llm_client(), sentinel)


if __name__ == "__main__":
    unittest.main()
