"""
Scratchpad injection tests for ContextManager.

T-21  Scratchpad block emission and one-shot semantics (P1)

The scratchpad is auto-injected EXACTLY ONCE at the start of a fresh
session (Agent.run() calls get_scratchpad_block() one time before the
first user message). Contrast with the old behavior that re-injected on
every change. Contract pinned here:

  * get_scratchpad_block() returns a fenced block wrapping the CURRENT file
    content with a VISIBLE_{pct}% header,
  * an oversized file (SCRATCHPAD_LIMIT patched small) is truncated to
    EXACTLY SCRATCHPAD_LIMIT characters with an [ERROR] suffix, and the
    VISIBLE percentage is computed from the PRE-TRUNCATION length,
  * calling it a second time returns a NEW block (no hash caching — the
    caller (Agent.run) decides to call it only once),
  * empty file yields an empty fenced body.

Seam notes: context.py binds SCRATCHPAD_LIMIT via `from bash_agent.config
import`, so tests patch bash_agent.context.SCRATCHPAD_LIMIT.
"""

import contextlib
import io
import os
import unittest
import uuid as uuid_module
from unittest import mock

from bash_agent.context import ContextManager
from tests.helpers.fakes import chdir_tmp

LIMIT = 1000
TRUNCATION_ERROR = "[ERROR]: Scratchpad truncated. Please clean it up using bash commands."


def scratchpad_block(uid, visible, body, error=False):
    base = (
        f"\n---START_SCRATCHPAD.md-VISIBLE_{visible}%-{uid}---\n"
        f"{body}\n"
        f"---END_SCRATCHPAD.md-{uid}---"
    )
    if error:
        return base + f"\n{TRUNCATION_ERROR}\n"
    return base + "\n"


class ScratchpadOneShotCase(unittest.TestCase):
    """Shared harness in a throwaway CWD."""

    LIMIT_PATCH = None

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

    def write_scratchpad(self, text):
        with open(self.cm.scratchpad_path, "w") as f:
            f.write(text)


class TestScratchpadEmission(ScratchpadOneShotCase):
    """Normal (under-limit) files are fenced verbatim each call."""

    def test_emits_fenced_block_with_content(self):
        # content is read verbatim (trailing \\n included) and an extra \\n
        # is appended by the emitter, so the body carries "keep me\n\n".
        self.write_scratchpad("# Notes\n\nkeep me\n")
        block = self.cm.get_scratchpad_block()
        self.assertEqual(
            block, scratchpad_block(self.uid, 100, "# Notes\n\nkeep me\n")
        )

    def test_no_hash_cache_second_call_emits_again(self):
        # One-shot semantics live in the CALLER (Agent.run), not here.
        self.write_scratchpad("x")
        b1 = self.cm.get_scratchpad_block()
        b2 = self.cm.get_scratchpad_block()
        self.assertEqual(b1, b2)
        self.assertNotEqual(b1, "")

    def test_empty_file_emits_empty_fenced_body(self):
        self.write_scratchpad("")
        self.assertEqual(
            self.cm.get_scratchpad_block(), scratchpad_block(self.uid, 100, "")
        )

    def test_scratchpad_created_if_absent(self):
        self.assertTrue(os.path.exists(self.cm.scratchpad_path))


class TestOversizeTruncation(ScratchpadOneShotCase):
    LIMIT_PATCH = LIMIT

    def test_body_truncated_to_exactly_limit(self):
        self.write_scratchpad("x" * (LIMIT + 500))
        self.assertEqual(
            self.cm.get_scratchpad_block(),
            scratchpad_block(self.uid, 66, "x" * LIMIT, error=True),
        )

    def test_visible_from_pre_truncation_length(self):
        self.write_scratchpad("y" * (LIMIT * 4))
        block = self.cm.get_scratchpad_block()
        self.assertIn(f"VISIBLE_25%-{self.uid}", block)
        self.assertNotIn("VISIBLE_100%", block)

    def test_error_suffix_appended_after_end_fence(self):
        self.write_scratchpad("e" * (LIMIT + 1))
        block = self.cm.get_scratchpad_block()
        self.assertTrue(
            block.endswith(f"---END_SCRATCHPAD.md-{self.uid}---\n{TRUNCATION_ERROR}\n")
        )


if __name__ == "__main__":
    unittest.main()
