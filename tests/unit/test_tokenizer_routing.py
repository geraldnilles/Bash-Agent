"""
Unit tests for the model-aware token-estimation routing
(bash_agent.tokenizer.count_tokens).

Verifies that ``count_tokens`` selects a *model-specific* offline tokenizer
when one exists, rather than the single generic fallback LiteLLM chose before:

  * ``deepseek/...``              -> bundled ``deepseek_tokenizer`` (exact)
  * OpenAI-family (incl. leading
    ``openai/`` provider prefix)  -> tiktoken.encoding_for_model
  * everything else               -> LiteLLM provider-neutral path
  * no tokenizer available        -> ~3.5 chars/token heuristic

The routing is validated structurally (which code path fires) and
numerically (different model families yield different counts on a
discriminating sample).  Only stdlib ``unittest`` + mocking; never hits the
network (LiteLLM's HF download is already disabled by
``bash_agent.tokenizer``) and both ``tiktoken`` and ``deepseek_tokenizer``
are bundled deps available offline.
"""
import unittest
from unittest import mock
from unittest.mock import patch

from bash_agent.tokenizer import (
    clear_token_cache,
    count_tokens,
    _route_model_specific,
    _route_deepseek,
    _route_openai_family,
)

# A sample with UTF-8/emoji that tokenizes *very* differently between the
# families (cl100k vs deepseek BPE vs o200k), so numeric assertions are
# meaningful and stable across the bunded offline tokenizers.
_DISCRIMINATING_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 20
    + "Emoji 🦊🚀 café déjà vu — “quoted” naïve résumé. "
    + "def fn(param):\n    return param + 1  # → ⇔ ← ñ é\n" * 30
)


class ModelRoutingCase(unittest.TestCase):
    """Structural: which of the tokenizer helpers fires for a given slug."""

    def setUp(self):
        clear_token_cache()

    def test_deepseek_slug_routes_to_deepseek_tokenizer(self):
        # _route_deepseek returns an int when deepseek_tokenizer is usable;
        # if it can't load, it returns None and LiteLLM/char fallback applies.
        n = _route_deepseek(_DISCRIMINATING_TEXT)
        if n is not None:
            self.assertIsInstance(n, int)
            self.assertGreater(n, 0)
        # On this host it must be importable (bundled dep).
        self.assertIsNotNone(n)

    def test_openai_4o_routes_via_tiktoken(self):
        n = _route_openai_family("gpt-4o", _DISCRIMINATING_TEXT)
        self.assertIsNotNone(n)
        self.assertIsInstance(n, int)

    def test_deepseek_and_openai_give_different_counts(self):
        """The two families must NOT collapse onto the same number now."""
        ds = count_tokens(_DISCRIMINATING_TEXT, model="deepseek/deepseek-v4-flash-0731")
        gpt4o = count_tokens(_DISCRIMINATING_TEXT, model="gpt-4o")
        self.assertNotEqual(ds, gpt4o)

    def test_provider_prefixed_openai_matches_bare(self):
        """openai/gpt-4o must use the same tokenizer as bare gpt-4o
        (LiteLLM's own catalog lacks provider-prefixed names)."""
        prefixed = count_tokens(_DISCRIMINATING_TEXT, model="openai/gpt-4o")
        bare = count_tokens(_DISCRIMINATING_TEXT, model="gpt-4o")
        self.assertEqual(prefixed, bare)

    def test_cl100k_backed_and_gpt4o_family_differ(self):
        """gpt-4o (o200k) must NOT equal gpt-3.5-turbo (cl100k) anymore."""
        gpt4o = count_tokens(_DISCRIMINATING_TEXT, model="gpt-4o")
        gpt35 = count_tokens(_DISCRIMINATING_TEXT, model="gpt-3.5-turbo")
        self.assertNotEqual(gpt4o, gpt35)

    def test_litellm_path_still_used_for_others(self):
        """An Anthropic slug with no family tokenizer falls through to the
        LiteLLM path (count differs from the DeepSeek tokenizer)."""
        from unittest.mock import patch as _patch
        # Force a deterministic, offline LiteLLM result by patching the
        # helper we *would* call; assert it fires for a non-family slug.
        fake_lm = mock.Mock()
        fake_lm.token_counter = mock.Mock(return_value=1234)

        with _patch("bash_agent.tokenizer._load_litellm", return_value=fake_lm):
            # A slug that doesn't trigger deepseek/openai routing
            n = count_tokens("Some text here", model="anthropic/claude-3-7-sonnet")
            self.assertEqual(n, 1234)
            fake_lm.token_counter.assert_called_once_with(
                model="anthropic/claude-3-7-sonnet", text="Some text here"
            )


class EmptyAndNoneCase(unittest.TestCase):
    """Baseline guard-rails unchanged: empty text -> 0, model None -> default."""

    def test_empty_text_zero(self):
        self.assertEqual(count_tokens("", model="gpt-4o"), 0)
        self.assertEqual(count_tokens("", model=None), 0)

    def test_model_none_uses_default_model(self):
        from bash_agent.config import DEFAULT_MODEL
        clear_token_cache()
        a = count_tokens("Hello world", model=None)
        b = count_tokens("Hello world", model=DEFAULT_MODEL)
        self.assertEqual(a, b)


class CacheCase(unittest.TestCase):
    def test_caching_cache_hits(self):
        clear_token_cache()
        import bash_agent.tokenizer as tk

        text = "cache test payload"
        model = "gpt-4o"
        # force a fresh computation
        n = count_tokens(text, model=model)
        self.assertIn((model, text), tk._TEXT_TOKEN_CACHE)
        self.assertEqual(count_tokens(text, model=model), n)

        clear_token_cache()
        self.assertEqual(tk._TEXT_TOKEN_CACHE, {})


if __name__ == "__main__":
    unittest.main()
