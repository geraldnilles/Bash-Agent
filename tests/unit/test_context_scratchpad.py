"""
Group 4 — Context management tests for ContextManager.

T-21  Scratchpad hashing and VISIBLE math (P1)

ContextManager.get_scratchpad_block() is the only channel through which the
agent's persistent memory reaches the LLM. Its contract:

  * first call emits the full file content wrapped in the session's
    SCRATCHPAD.md fence with a VISIBLE_{pct}% header,
  * a second call with an UNCHANGED file returns "" — the md5 hash cache
    (last_scratchpad_hash) exists so the scratchpad is injected only when it
    actually changed; without it every turn would carry a duplicate block,
  * a changed file re-emits (the cache is content-addressed: reverting to a
    previously-seen content re-emits too, because equality — not monotonic
    change — is what is checked),
  * an oversized file (> SCRATCHPAD_LIMIT) is truncated to EXACTLY
    SCRATCHPAD_LIMIT characters, and the VISIBLE percentage is computed from
    the PRE-TRUNCATION length (e.g. LIMIT=80k over a 100k original ->
    VISIBLE_80%). Computing it from the truncated body instead would always
    yield 100 and silently lie to the LLM about how much it is seeing,
  * the truncation error suffix is appended AFTER the END fence with the
    exact wording "[ERROR]: Scratchpad truncated. Please clean it up using
    bash commands." — that wording is duplicated in remove_old_scratchpads'
    cleanup regex, so drift in either copy breaks scratchpad cleanup.

Seam notes:
  * context.py binds the limit via `from bash_agent.config import
    SCRATCHPAD_LIMIT`, so tests patch bash_agent.context.SCRATCHPAD_LIMIT —
    patching bash_agent.config.SCRATCHPAD_LIMIT would NOT be seen (same
    subtlety as CONTEXT_LIMIT in test_context_pruning.py).
  * ContextManager.__init__ creates .bash_agent_tmp/SCRATCHPAD.md in the
    CWD, so every test runs inside chdir_tmp (T-00a).
"""

import contextlib
import hashlib
import io
import os
import unittest
import uuid as uuid_module
from unittest import mock

from bash_agent.context import ContextManager
from tests.helpers.fakes import chdir_tmp

# Tiny limit so oversized fixtures stay human-checkable.
LIMIT = 1000

TRUNCATION_ERROR = (
    "[ERROR]: Scratchpad truncated. Please clean it up using bash commands."
)


def scratchpad_block(uid, visible, body, error=False):
    """
    Build the EXACT string get_scratchpad_block() must emit.

    Mirrors context.py's f-string byte for byte:
      * leading \n before the START marker,
      * body sandwiched between single newlines,
      * NO newline before the END marker's closing fence,
      * non-truncated blocks end with a bare "\n",
      * truncated blocks splice the ERROR banner directly onto the END
        marker ("---END...---\n[ERROR]: ...\n") -- which is precisely the
        shape remove_old_scratchpads' cleanup regex expects.
    """
    base = (
        f"\n---START_SCRATCHPAD.md-VISIBLE_{visible}%-{uid}---\n"
        f"{body}\n"
        f"---END_SCRATCHPAD.md-{uid}---"
    )
    if error:
        return base + f"\n{TRUNCATION_ERROR}\n"
    return base + "\n"


class ScratchpadCase(unittest.TestCase):
    """
    Shared harness: throwaway CWD (ContextManager writes the scratchpad
    there), captured stdout, and SCRATCHPAD_LIMIT patched in the *context*
    module namespace.
    """

    LIMIT_PATCH = None  # subclasses opt into patching the limit

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.uid = str(uuid_module.uuid4())
        self._patches = []
        if self.LIMIT_PATCH is not None:
            p = mock.patch("bash_agent.context.SCRATCHPAD_LIMIT", self.LIMIT_PATCH)
            p.start()
            self._patches.append(p)
        self.cm = ContextManager(self.uid)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- helpers ------------------------------------------------------------

    def write_scratchpad(self, text):
        with open(self.cm.scratchpad_path, "w") as f:
            f.write(text)

    def read_scratchpad(self):
        with open(self.cm.scratchpad_path, "r") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Hash caching — inject only on change
# ---------------------------------------------------------------------------

class TestHashCaching(ScratchpadCase):
    """The md5 cache suppresses re-injection of unchanged scratchpads."""

    def test_first_call_emits_block(self):
        # __init__ seeds the file with a minimal header; hash cache is None.
        self.assertIsNone(self.cm.last_scratchpad_hash)
        block = self.cm.get_scratchpad_block()
        self.assertEqual(
            block, scratchpad_block(self.uid, 100, "# Project Scratchpad\n\n")
        )
        self.assertIsNotNone(self.cm.last_scratchpad_hash)

    def test_unchanged_file_second_call_returns_empty(self):
        self.write_scratchpad("plan: test the agent\n")
        self.assertNotEqual(self.cm.get_scratchpad_block(), "")
        self.assertEqual(self.cm.get_scratchpad_block(), "")

    def test_hash_equals_md5_of_file_content(self):
        self.write_scratchpad("some notes\n")
        self.cm.get_scratchpad_block()
        expected = hashlib.md5(b"some notes\n").hexdigest()
        self.assertEqual(self.cm.last_scratchpad_hash, expected)

    def test_changed_file_emits_new_block(self):
        self.write_scratchpad("version one\n")
        first = self.cm.get_scratchpad_block()
        self.write_scratchpad("version two\n")
        second = self.cm.get_scratchpad_block()
        self.assertNotEqual(second, "")
        self.assertIn("version two", second)
        self.assertNotIn("version one", second)
        self.assertNotEqual(first, second)

    def test_cache_is_content_addressed_not_monotonic(self):
        # Reverting to previously-seen content must re-emit: the check is
        # hash EQUALITY with the last injection, not "file changed at all".
        self.write_scratchpad("alpha\n")
        self.assertNotEqual(self.cm.get_scratchpad_block(), "")
        self.write_scratchpad("beta\n")
        self.assertNotEqual(self.cm.get_scratchpad_block(), "")
        self.write_scratchpad("alpha\n")
        block = self.cm.get_scratchpad_block()
        self.assertNotEqual(block, "")
        self.assertIn("alpha", block)

    def test_repeated_changes_emit_each_time(self):
        for i in range(3):
            self.write_scratchpad(f"iteration {i}\n")
            block = self.cm.get_scratchpad_block()
            self.assertIn(f"iteration {i}", block)

    def test_empty_file_emits_empty_body_then_caches(self):
        # An emptied scratchpad is a legitimate state (the LLM cleaned it up);
        # the empty body must still be fenced and then hash-cached.
        self.write_scratchpad("")
        block = self.cm.get_scratchpad_block()
        self.assertEqual(block, scratchpad_block(self.uid, 100, ""))
        self.assertEqual(self.cm.get_scratchpad_block(), "")


# ---------------------------------------------------------------------------
# VISIBLE math and truncation (SCRATCHPAD_LIMIT patched to 1000)
# ---------------------------------------------------------------------------

class TestOversizeTruncation(ScratchpadCase):
    """Oversized scratchpads are truncated with honest VISIBLE math."""

    LIMIT_PATCH = LIMIT

    def test_body_truncated_to_exactly_limit(self):
        # Whole-block equality pins BOTH the exact LIMIT-character body and
        # the surrounding wire format in one shot.
        self.write_scratchpad("x" * (LIMIT + 500))
        block = self.cm.get_scratchpad_block()
        self.assertEqual(
            block, scratchpad_block(self.uid, 66, "x" * LIMIT, error=True)
        )

    def test_visible_percent_from_pre_truncation_length(self):
        # 1000 kept of 4000 original -> VISIBLE_25%, NOT VISIBLE_100%.
        # This is the regression the TEST_PLAN calls out: computing the
        # percentage from the truncated body always yields 100.
        self.write_scratchpad("y" * (LIMIT * 4))
        block = self.cm.get_scratchpad_block()
        self.assertIn(f"VISIBLE_25%-{self.uid}", block)
        self.assertNotIn("VISIBLE_100%", block)

    def test_plan_example_80k_of_100k_is_visible_80(self):
        # The exact arithmetic from TEST_PLAN.md T-21.
        with mock.patch("bash_agent.context.SCRATCHPAD_LIMIT", 80_000):
            self.write_scratchpad("z" * 100_000)
            block = self.cm.get_scratchpad_block()
        self.assertIn("VISIBLE_80%", block)
        self.assertNotIn("VISIBLE_100%", block)

    def test_just_over_limit_rounds_down(self):
        # int((1000/1001)*100) == 99 — the percentage floors.
        self.write_scratchpad("w" * (LIMIT + 1))
        self.assertIn("VISIBLE_99%", self.cm.get_scratchpad_block())

    def test_exactly_at_limit_is_not_truncated(self):
        self.write_scratchpad("q" * LIMIT)
        block = self.cm.get_scratchpad_block()
        self.assertIn("VISIBLE_100%", block)
        self.assertNotIn(TRUNCATION_ERROR, block)

    def test_error_suffix_appended_after_end_fence(self):
        self.write_scratchpad("e" * (LIMIT + 1))
        block = self.cm.get_scratchpad_block()
        suffix = f"---END_SCRATCHPAD.md-{self.uid}---\n{TRUNCATION_ERROR}\n"
        self.assertTrue(block.endswith(suffix))

    def test_error_suffix_exact_wording(self):
        # remove_old_scratchpads' regex matches this wording verbatim;
        # drift here would orphan ERROR banners in the history forever.
        self.write_scratchpad("e" * (LIMIT + 1))
        self.assertIn(f"\n{TRUNCATION_ERROR}", self.cm.get_scratchpad_block())

    def test_hash_covers_full_pre_truncation_content(self):
        # The cache key is the md5 of the ORIGINAL content, not the truncated
        # body: an unchanged oversized file must not re-inject every turn,
        # and a one-char append (invisible in the truncated body) must.
        self.write_scratchpad("a" * (LIMIT + 500))
        self.assertNotEqual(self.cm.get_scratchpad_block(), "")
        self.assertEqual(self.cm.get_scratchpad_block(), "")
        self.write_scratchpad("a" * (LIMIT + 500) + "!")  # beyond the cut
        block = self.cm.get_scratchpad_block()
        self.assertNotEqual(block, "")

    def test_truncated_block_is_removed_by_remove_old_scratchpads(self):
        # End-to-end coupling: whatever get_scratchpad_block() emits
        # (including the ERROR suffix) must be fully stripped from history
        # by remove_old_scratchpads(), or stale scratchpads would pile up.
        self.write_scratchpad("b" * (LIMIT + 500))
        block = self.cm.get_scratchpad_block()
        self.cm.add_message("user", f"prior turn{block}tail text")
        self.cm.remove_old_scratchpads()
        msg = self.cm.history[0]["content"]
        self.assertNotIn("START_SCRATCHPAD.md", msg)
        self.assertNotIn("END_SCRATCHPAD.md", msg)
        self.assertNotIn(TRUNCATION_ERROR, msg)
        self.assertEqual(msg, "prior turntail text")

    def test_emitted_block_matches_expected_skeleton(self):
        # Pin the full wire format for the truncated path in one assertion.
        self.write_scratchpad("n" * (LIMIT * 2))
        block = self.cm.get_scratchpad_block()
        self.assertEqual(
            block, scratchpad_block(self.uid, 50, "n" * LIMIT, error=True)
        )


if __name__ == "__main__":
    unittest.main()
