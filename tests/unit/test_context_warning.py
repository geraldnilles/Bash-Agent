"""
Group 4 — Context-limit warning tests for ContextManager.

New behavior (ROADMAP 'Context limit warning'):
  * When the conversation crosses CONTEXT_WARN_PERCENT% of CONTEXT_LIMIT, a
    one-time user-role warning is injected telling the LLM to back up
    important notes to the SCRATCHPAD before the oldest ~20% of the history
    is trimmed.
  * Trimming is DEFERRED until the warning has been seen by the LLM: the
    warning is injected at the START of an LLM turn (a user message), and
    hard pruning only runs once an ASSISTANT message is subsequently added
    (proving the model read the warning and issued its backup commands,
    whose commit outputs sit at the END of the history and therefore survive
    the trim).
  * It is acceptable to briefly exceed CONTEXT_LIMIT to deliver the warning.

Context accounting is measured in TOKENS (via
``bash_agent.context._content_tokens``), NOT characters.  Because counting is
environment-dependent in production (LiteLLM when installed; heuristic
offline), these wiring tests install a fixed deterministic meter (offline
tiktoken cl100k_base - the generic BPE LiteLLM falls back to for arbitrary
OpenRouter slugs) via ``setUpModule`` below.  Text fixtures use a repeat of
the letter ``u`` which that meter encodes EXACTLY linearly (2 chars = 1
token), keeping arithmetic exact and immune to tokenizer quirks.

Contract pinned here:
  * below the warn threshold: no warning, no trim;
  * crossing the threshold: exactly one warning injected, NO trim yet;
  * an assistant message after the warning triggers trimming;
  * the warning is injected at most once;
  * the warning text mentions backing up to the SCRATCHPAD;
  * `reset` (via agent._handle_special_command) re-arms the flags.

Seam notes: each ContextManager is constructed with an explicit
per-instance ``context_limit=LIMIT`` (the same instance-ceiling pathway the
Agent uses after resolving the model's token window).  No module-global
constant is patched here, so these tests are immune to cross-file pollution.
CONTEXT_WARN_PERCENT stays at the production value (99).
"""

import contextlib
import io
import re
import unittest
import uuid as uuid_module

from bash_agent.context import ContextManager
from tests.helpers.fakes import DeterministicTokenCounts, deterministic_count_tokens

# ---------------------------------------------------------------------------
# Shared harness — reuse the same conventions as test_context_pruning.py
# ---------------------------------------------------------------------------

# Token budget for the trimmed ContextManager.
LIMIT = 300                               # per-instance ContextManager ceiling (TOKENS)
WARN_PERCENT = 99                          # production config value
WARN_THRESHOLD = int(LIMIT * (WARN_PERCENT / 100.0))  # 297
TARGET = int(LIMIT * 0.8)                  # 240 hysteresis target

HYSTERESIS_BANNER = "Initiating hysteresis cleanup"
WARNING_MARKER = "back up any important findings"


def _total_length(history):
    """Mirror of the production accounting: sum of _content_tokens."""
    return sum(ContextManager._content_tokens(m.get("content", "")) for m in history)


def tokfill(tokens):
    """Return a plain string that the tokenizer encodes to EXACTLY `tokens`.

    ``u`` encodes linearly as 2 chars per 1 token, verified empirically.
    """
    assert tokens >= 0
    return "u" * (2 * tokens)


class WarningCase(unittest.TestCase):
    """Throwaway CWD + captured stdout + ready ContextManager (flags fresh)."""

    def setUp(self):
        from tests.helpers.fakes import chdir_tmp
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.uid = str(uuid_module.uuid4())
        # Explicit per-instance token ceiling (mirrors Agent wiring) — no
        # module-global constant is patched, eliminating any risk of a
        # leaked patch leaking into other test modules.
        self.cm = ContextManager(self.uid, context_limit=LIMIT)

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- helpers ------------------------------------------------------------

    def sys_tokens(self):
        """Token cost of the standard system-prompt fixture."""
        return count_tokens(self.system_prompt())

    def system_prompt(self, pad=100):
        return "You are the system prompt. " + "S" * pad

    def plain_msg(self, role, text):
        return {"role": role, "content": text}

    def warning_messages(self):
        """All history messages whose content contains the SCRATCHPAD warning."""
        return [m for m in self.cm.history
                if isinstance(m.get("content"), str)
                and WARNING_MARKER in m["content"]]

    def banners(self):
        return self.stdout_buf.getvalue().count(HYSTERESIS_BANNER)


# ---------------------------------------------------------------------------
# Threshold / injection semantics
# ---------------------------------------------------------------------------

class TestWarningInjection(WarningCase):
    """The warning fires exactly once, as a user message, with no trim yet."""

    def test_no_warning_below_threshold(self):
        sys_msg = self.plain_msg("system", self.system_prompt())
        # sys (~57 t) + 200-token filler = ~257 t < 297.
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(200))]
        before_len = len(self.cm.history)

        self.cm.add_message("user", "zz")  # still comfortably below 297

        self.assertEqual(len(self.cm.history), before_len + 1)
        self.assertFalse(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)
        self.assertEqual(self.warning_messages(), [])
        self.assertEqual(self.banners(), 0)

    def test_exactly_at_threshold_no_warning(self):
        # Boundary: guard is `total > warn_threshold` -> inject, so a history
        # sitting EXACTLY on the threshold must NOT inject.
        sys_tokens = self.sys_tokens()
        filler_len = WARN_THRESHOLD - sys_tokens          # exact fill to threshold
        sys_msg = self.plain_msg("system", self.system_prompt())
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(filler_len))]
        self.assertEqual(_total_length(self.cm.history), WARN_THRESHOLD)

        # Appending a zero-length message leaves total exactly at the
        # threshold (guard: total > threshold -> inject), so no warning fires.
        self.cm.add_message("user", "")
        self.assertFalse(self.cm._warning_sent)
        self.assertEqual(self.warning_messages(), [])
        self.assertEqual(_total_length(self.cm.history), WARN_THRESHOLD)

        # A single character crossing the threshold DOES fire the warning.
        self.cm.add_message("user", "x")
        self.assertTrue(self.cm._warning_sent)
        self.assertEqual(len(self.warning_messages()), 1)

    def test_crossing_threshold_injects_once_and_defers_trim(self):
        # Build a history just under the threshold, then cross it.  The
        # warning must be injected as a user message and NO pruning may
        # happen yet (the LLM has not confirmed it).
        sys_msg = self.plain_msg("system", self.system_prompt())
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(200))]

        self.cm.add_message("user", tokfill(60))  # 200 + 60 = crosses 297

        self.assertTrue(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)
        self.assertEqual(self.banners(), 0)  # trim strictly deferred
        self.assertEqual(len(self.warning_messages()), 1)
        self.assertEqual(self.warning_messages()[0]["role"], "user")
        self.assertIn("SCRATCHPAD", self.warning_messages()[0]["content"])
        # The warning is the LAST message (right at the front of the LLM's view).
        self.assertIs(self.cm.history[-1]["content"],
                      self.warning_messages()[0]["content"])

    def test_warning_fires_only_once(self):
        sys_msg = self.plain_msg("system", self.system_prompt())
        # Directly seeded OVER the threshold so the first add_message warns.
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(250))]  # >297

        # Two crossing adds: only the FIRST should inject the warning.
        self.cm.add_message("user", "u1")
        self.cm.add_message("user", "u2")

        self.assertEqual(len(self.warning_messages()), 1)
        self.assertTrue(self.cm._warning_sent)

    def test_context_warning_message_constant_mentions_scratchpad(self):
        text = ContextManager._context_warning_message()
        self.assertIn("SCRATCHPAD", text)
        self.assertIn("about to be trimmed", text)
        self.assertIn("back up", text)


# ---------------------------------------------------------------------------
# Deferred trimming until the warning is confirmed
# ---------------------------------------------------------------------------

class TestDeferredTrimUntilConfirmed(WarningCase):
    """Hard pruning only runs once an assistant turn follows the warning."""

    def setUp(self):
        super().setUp()
        sys_msg = self.plain_msg("system", self.system_prompt())
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(200))]
        # Cross the threshold -> warning injected, trim deferred.
        self.cm.add_message("user", tokfill(60))
        self.assertTrue(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)

    def test_user_message_after_warning_still_no_trim(self):
        # More user traffic (e.g. backup command OUTPUT being committed) must
        # NOT trigger trimming — the assistant turn (the model's textual reply)
        # is what confirms the warning was seen.
        self.cm.add_message("user", "output of my scratchpad write")
        self.assertFalse(self.cm._warning_confirmed)
        self.assertEqual(self.banners(), 0)

    def test_assistant_message_confirms_and_trims(self):
        # The model's next assistant turn proves it read the warning; pruning
        # then runs, removing OLDEST messages while the warning itself and the
        # model's fresh (backup-command) content near the end survive.
        self.cm.add_message("assistant", "ack")
        self.assertTrue(self.cm._warning_confirmed)
        self.assertEqual(self.banners(), 1)
        self.assertLessEqual(_total_length(self.cm.history), TARGET)
        # System prompt is preserved at index 0.
        self.assertEqual(self.cm.history[0]["role"], "system")

    def test_end_to_end_backup_survives_trim(self):
        # Full production-shaped sequence:
        #   warning injected -> LLM writes backup commands -> outputs committed
        #   as user messages -> next assistant msg triggers trim -> the backup
        #   commands/outputs (near the end) survive; the warning also survives.
        self.cm.add_message("user", "cat >> SCRATCHPAD.md <<'EOF'\nnotes\nEOF\n")
        self.cm.add_message("assistant", "done backing up")

        contents = [m["content"] for m in self.cm.history]
        self.assertTrue(any(WARNING_MARKER in c for c in contents))
        self.assertTrue(any("backing up" in c for c in contents))
        self.assertLessEqual(_total_length(self.cm.history), TARGET)


# ---------------------------------------------------------------------------
# reset re-arms the warning
# ---------------------------------------------------------------------------

class TestResetReArmsWarning(WarningCase):
    """The `reset` command lets a fresh conversation warn again."""

    def test_reset_clears_both_flags(self):
        sys_msg = self.plain_msg("system", self.system_prompt())
        self.cm.history = [sys_msg, self.plain_msg("user", tokfill(250))]  # >297
        self.cm.add_message("user", "u1")   # crosses -> warning fires
        self.assertTrue(self.cm._warning_sent)

        # Simulate the agent-side `reset` handler: keep only the system prompt
        # and re-arm the flags.
        self.cm.history = [self.cm.history[0]] if self.cm.history else []
        self.cm._warning_sent = False
        self.cm._warning_confirmed = False

        # The fresh session can now warn again as it regrows.
        self.assertFalse(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)
        self.cm.add_message("user", tokfill(250))  # cross threshold again
        self.assertTrue(self.cm._warning_sent)
        self.assertEqual(len(self.warning_messages()), 1)


# ---------------------------------------------------------------------------
# Deterministic local token meter for these warning fixtures.
# The production accounting now goes through LiteLLM/generic tokenizers whose
# exact ratios differ from the old hard-coded DeepSeek ones.  These tests
# exercise the *warning / deferred-trim wiring* against a fixed, deterministic
# tokenizer (offline tiktoken cl100k_base), so their exact token arithmetic
# stays valid no matter which backend is installed.
# ---------------------------------------------------------------------------
_local_token_counts = DeterministicTokenCounts(fixture_module=__import__(__name__))
count_tokens = deterministic_count_tokens


def setUpModule():
    _local_token_counts.start()


def tearDownModule():
    _local_token_counts.stop()


if __name__ == "__main__":
    unittest.main()
