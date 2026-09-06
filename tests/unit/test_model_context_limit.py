"""
Model-specific context limit (implements ROADMAP 'Model Specific Context Limit').

Replace the fixed character-based ceiling with one that is a fraction of
the selected model's reported context_length (tokens) from the OpenRouter
/models catalog:

    context_limit_tokens = int(context_length_tokens / 4)

Text is counted with the model's real local tokenizer (not a chars/8
heuristic). The /4 keeps the agent comfortably under the model's usable
window, leaving generous headroom for its own output, tool-call shaping
and API overhead.

Contract pinned here:

  ContextManager
    * ContextManager(uuid, context_limit=N) uses N for its SCRATCHPAD warning
      threshold and for its 80%-hysteresis trim target instead of the module
      constant CONTEXT_LIMIT.
    * Omitting context_limit keeps the historical module-fallback behaviour
      (existing pruning / warning tests stay valid unchanged).

  Agent
    * _fetch_model_context_limit() derives int(ctx_tokens / 4) into
      model_context_limit_tokens from the catalog entry whose id ==
      self.model, stores the full window in model_full_context_tokens, or
      sets both None on catalog failure / model miss (so __init__ always
      has a safe fallback).
    * __init__ resolves self.context_limit (None -> config.CONTEXT_LIMIT)
      and hands it to ContextManager so pruning matches the model.

Seams (offline only): ContextManager tests pass the ceiling directly; Agent
tests stub the three network probes exactly like helpers._make_agent.
"""
import contextlib
import io
import unittest
import uuid as uuid_module
from unittest import mock

from bash_agent.context import ContextManager
from bash_agent.config import CONTEXT_LIMIT as MODULE_CONTEXT_LIMIT
from tests.helpers.fakes import chdir_tmp, _make_agent, _stub_model_context_info


# ---------------------------------------------------------------------------
# ContextManager instance-level limit
# ---------------------------------------------------------------------------

def _total_length(history):
    return sum(ContextManager._content_tokens(m.get("content", "")) for m in history)


class InstanceLimitCase(unittest.TestCase):
    """ContextManager(..., context_limit=N) honours N instead of CONTEXT_LIMIT.

    Accounting is measured in TOKENS. ``u`` encodes linearly in the DeepSeek
    tokenizer (2 chars = 1 token), so ``tokfill`` provides EXACT token counts.
    """

    INSTANCE_LIMIT = 1000
    WARN_THRESHOLD = int(INSTANCE_LIMIT * (99 / 100.0))   # 990

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self.out = io.StringIO()
        self._out_cm = contextlib.redirect_stdout(self.out)
        self._out_cm.__enter__()
        self.uid = str(uuid_module.uuid4())
        self.cm = ContextManager(self.uid, context_limit=self.INSTANCE_LIMIT)

    def tearDown(self):
        self._out_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def sys_msg(self):
        return {"role": "system", "content": "s" * 60}

    def _current(self):
        return sum(
            ContextManager._content_tokens(m["content"])
            for m in self.cm.history if isinstance(m.get("content"), str)
        )

    def test_warning_threshold_uses_instance_limit(self):
        # Seed the history to sit EXACTLY one token below the warn threshold
        # (990), then a one-token user add crosses 990 and must warn.
        self.cm.history = [self.sys_msg()]
        filler = self.WARN_THRESHOLD - self._current() - 1  # 989 total
        self.cm.history.append({"role": "user", "content": self._fill_tokens(filler)})
        self.assertEqual(self._current(), self.WARN_THRESHOLD - 1)  # 989 < 990

        # A +1 token add (989 -> 990) would sit EXACTLY on the threshold and
        # the guard is strictly-greater, so use "uuuu" (4 chars == 2 tokens)
        # to push 989 -> 991 > 990 and fire the warning.
        self.cm.add_message("user", "uuuu")  # +2 tokens -> 991 > 990
        self.assertTrue(self.cm._warning_sent)
        self.assertIn(
            "back up any important findings",
            " ".join(m["content"] for m in self.cm.history),
        )

    def test_trim_target_uses_instance_limit(self):
        # Build a history that exactly meets the INSTANCE_LIMIT (1000), then
        # cross beyond via add_message: the crossing user add fires the
        # deferred warning (no trim yet); the FOLLOWING assistant add
        # confirms the warning and runs hysteresis, bringing history down to
        # at most the instance ceiling.
        self.cm.history = [self.sys_msg()]
        filler = self.INSTANCE_LIMIT - self._current() - 1  # total 999
        self.cm.history.append({"role": "user", "content": self._fill_tokens(filler)})
        self.assertEqual(self._current(), self.INSTANCE_LIMIT - 1)

        self.cm.add_message("user", "y")          # 1000+ -> warning deferred
        self.assertTrue(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)
        self.assertEqual(self.out.getvalue().count("Initiating hysteresis cleanup"), 0)

        self.cm.add_message("assistant", "ack")   # assistant confirms -> trims
        self.assertTrue(self.cm._warning_confirmed)
        self.assertIn("Initiating hysteresis cleanup", self.out.getvalue())
        self.assertLessEqual(self._current(), self.INSTANCE_LIMIT)

    def test_no_limit_keeps_module_fallback(self):
        cm = ContextManager(self.uid)
        self.assertEqual(cm.context_limit, MODULE_CONTEXT_LIMIT)

    @staticmethod
    def _fill_tokens(tokens):
        """Plain string of EXACTLY `tokens` tokens (u encodes 2 chars/token)."""
        return "u" * (2 * tokens)


# ---------------------------------------------------------------------------
# Agent context-length fetch + constructor wiring
# ---------------------------------------------------------------------------

class AgentContextDerivationCase(unittest.TestCase):
    """_fetch_model_context_limit math and None-fallback."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self.agent = _make_agent()

    def tearDown(self):
        self._chdir_cm.__exit__(None, None, None)

    def _catalog(self, context_length):
        return [{"id": self.agent.model, "context_length": context_length}]

    def test_derives_quarter_model_window(self):
        self.agent._get_models_catalog = lambda: self._catalog(4096)
        self.agent._fetch_model_context_limit()
        self.assertEqual(
            self.agent.model_context_limit_tokens,
            int(4096 / 4),   # 1024 tokens
        )

    def test_catalog_miss_sets_none(self):
        self.agent._get_models_catalog = lambda: [{"id": "other/model"}]
        self.agent._fetch_model_context_limit()
        self.assertIsNone(self.agent.model_context_limit_tokens)

    def test_empty_catalog_sets_none(self):
        self.agent._get_models_catalog = lambda: []
        self.agent._fetch_model_context_limit()
        self.assertIsNone(self.agent.model_context_limit_tokens)

    def test_missing_context_length_field_sets_none(self):
        self.agent._get_models_catalog = lambda: [{"id": self.agent.model}]
        self.agent._fetch_model_context_limit()
        self.assertIsNone(self.agent.model_context_limit_tokens)


class AgentConstructorWiringCase(unittest.TestCase):
    """Agent.__init__ propagates resolved limit to ContextManager, falls back."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()

    def tearDown(self):
        self._chdir_cm.__exit__(None, None, None)

    def test_default_stub_falls_back_to_config_limit(self):
        # _make_agent stubs _fetch_model_context_limit to set None
        agent = _make_agent()
        self.assertEqual(agent.context_limit, MODULE_CONTEXT_LIMIT)
        self.assertEqual(agent.context.context_limit, agent.context_limit)

    def test_resolved_limit_is_passed_to_context_manager(self):
        # Replicate _make_agent scaffolding but with a context stub that
        # reports a real derived ceiling (e.g. 4096 tokens -> 1024 budget tokens).
        from bash_agent.agent import Agent
        from tests.helpers.fakes import FakeSandbox

        def custom_ctx(self):
            self.model_context_limit_tokens = int(4096 / 4)
            self.model_full_context_tokens = 4096

        with mock.patch.object(Agent, "_check_model_capabilities", return_value=None):
            with mock.patch.object(Agent, "_fetch_model_reasoning_info", _custom_full_reasoning):
                with mock.patch.object(Agent, "_fetch_model_context_limit", custom_ctx):
                    with mock.patch("bash_agent.agent.cleanup_tmp_folder", return_value=None):
                        agent = Agent()
        agent.sandbox = FakeSandbox()

        self.assertEqual(agent.context_limit, 1024)
        self.assertEqual(agent.context.context_limit, 1024)
        self.assertEqual(agent.context.context_limit, agent.context_limit)


def _custom_full_reasoning(self):
    """Reasoning stub giving full production-default fallback attributes."""
    self.reasoning_supported_efforts = ["high", "medium", "low", "minimal", "none"]
    self.reasoning_mandatory = False
    self.reasoning_default_effort = "medium"


if __name__ == "__main__":
    unittest.main()
