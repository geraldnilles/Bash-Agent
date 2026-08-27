"""
Group 5 — LLM Adapter tests for bash_agent.llm.

T-24  OpenRouter payload normalization (P0)
T-24a OpenRouter App Attribution headers (P0)
T-26  Client cache identity (P2)

All requests are routed through the single OpenRouter backend; the
adapter shapes payloads (reasoning effort, provider whitelists,
attribution headers) per request. All tests are strictly offline:

  * ``bash_agent.config`` binds env vars at import time, so mutating
    os.environ after import has NO effect. Tests patch the config
    attributes directly (e.g. ``config.OPENROUTER_API_KEY``).
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

        # Some Agent-instantiating tests set a module-level session id via
        # llm.set_session_id(); isolate it so payload assertions are exact.
        session_patcher = mock.patch.object(llm, "_SESSION_ID", None)
        session_patcher.start()
        self.addCleanup(session_patcher.stop)

        _OpenAIRecorder.instances.clear()
        _OpenAIRecorder.init_kwargs_log.clear()

    @staticmethod
    def _seed(backend: str, response=None) -> FakeLLMClient:
        """Seed llm._CLIENT_CACHE[backend] with a FakeLLMClient; return it."""
        fake = FakeLLMClient(responses=[response] if response is not None else [])
        llm._CLIENT_CACHE[backend] = fake
        return fake


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
# Session id injection (cache-routing)
# ---------------------------------------------------------------------------

class TestSessionIdInjection(_OfflineLLMTestCase):
    """When a session id is set via llm.set_session_id(), every OpenRouter
    payload (chat, embeddings) carries it in extra_body; when unset, nothing
    is added."""

    MESSAGES = [{"role": "user", "content": "hello"}]

    def setUp(self):
        super().setUp()
        self.scripted = make_fake_response(content="ok")
        self.fake = FakeLLMClient([self.scripted] * 5)
        llm._CLIENT_CACHE["openrouter"] = self.fake

    def test_chat_injects_session_id_into_extra_body(self):
        llm.set_session_id("sess-1234")
        llm.create_chat_completion(model="m", messages=self.MESSAGES)
        self.assertEqual(
            self.fake.calls[-1]["extra_body"].get("session_id"), "sess-1234"
        )

    def test_chat_session_id_coexists_with_reasoning_and_provider(self):
        llm.set_session_id("sess-1234")
        with mock.patch.object(config, "MODEL_PROVIDERS", {"m": ["p"]}):
            llm.create_chat_completion(
                model="m", messages=self.MESSAGES, reasoning_effort="low"
            )
        ebody = self.fake.calls[-1]["extra_body"]
        self.assertEqual(
            ebody,
            {
                "session_id": "sess-1234",
                "reasoning": {"effort": "low"},
                "provider": {"only": ["p"]},
            },
        )

    def test_chat_no_session_id_means_no_key(self):
        llm.create_chat_completion(model="m", messages=self.MESSAGES)
        self.assertNotIn("session_id", self.fake.calls[-1].get("extra_body", {}))

    def test_embedding_injects_session_id_into_extra_body(self):
        llm.set_session_id("sess-1234")
        llm.create_embedding(model="e", input_texts=["x"])
        self.assertEqual(
            self.fake.embedding_calls[-1]["extra_body"],
            {"session_id": "sess-1234"},
        )

    def test_getter_roundtrip(self):
        self.assertIsNone(llm.get_session_id())
        llm.set_session_id("abc")
        self.assertEqual(llm.get_session_id(), "abc")


# ---------------------------------------------------------------------------
# T-24a — OpenRouter App Attribution headers
# ---------------------------------------------------------------------------

class TestOpenRouterAttributionHeaders(_OfflineLLMTestCase):
    """T-24a: create_chat_completion()/create_embedding() must attach the
    App Attribution headers (HTTP-Referer, X-OpenRouter-Title,
    X-OpenRouter-Categories) via extra_headers on every request."""

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

    def test_header_values_match_config_constants(self):
        self.assertEqual(config.APP_URL, "https://github.com/geraldnilles/Bash-Agent")
        self.assertEqual(config.APP_TITLE, "Bash Agent")
        self.assertEqual(config.APP_CATEGORIES, "cli-agent")


# ---------------------------------------------------------------------------
# T-26 — Client cache identity
# ---------------------------------------------------------------------------

class TestClientCacheIdentity(_OfflineLLMTestCase):
    """T-26: get_llm_client() caches one shared OpenRouter client; seeding
    _CLIENT_CACHE bypasses construction entirely. llm.OpenAI is already
    replaced by _OpenAIRecorder via the base class."""

    def test_same_client_returned_across_calls(self):
        c1 = llm.get_llm_client()
        c2 = llm.get_llm_client()
        self.assertIs(c1, c2)
        self.assertEqual(len(_OpenAIRecorder.instances), 1)  # built once

    def test_construction_uses_config_credentials_and_url(self):
        with mock.patch.object(config, "OPENROUTER_API_KEY", "or-key"):
            llm.get_llm_client()
        (or_kwargs,) = _OpenAIRecorder.init_kwargs_log
        self.assertEqual(or_kwargs["api_key"], "or-key")
        self.assertEqual(or_kwargs["base_url"], config.OPENROUTER_BASE_URL)

    def test_seeded_cache_bypasses_construction(self):
        sentinel = object()
        llm._CLIENT_CACHE["openrouter"] = sentinel
        self.assertIs(llm.get_llm_client(), sentinel)
        self.assertEqual(_OpenAIRecorder.instances, [])


if __name__ == "__main__":
    unittest.main()
