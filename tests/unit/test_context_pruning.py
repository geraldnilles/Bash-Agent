"""
Group 4 — Context management tests for ContextManager.

T-18  Multimodal content length accounting (P1)

ContextManager._content_length is the yardstick every pruning decision is
measured with: _trim_context_if_needed() sums it across the whole history to
decide when CONTEXT_LIMIT is breached, and agent.py reuses the same static
method for its context-size bookkeeping. Its contract:

  * plain strings count their character length,
  * multimodal lists count each {"type": "text"} part by its text length,
  * each {"type": "image_url"} part costs a flat 6400 characters
    (~800 tokens x 8 chars/token) REGARDLESS of the encoded payload size,
  * bare strings nested inside a list count their own length,
  * junk items inside a list (ints, None, unknown dict types) contribute 0,
  * anything that is neither a str nor a list (None, int, dict, tuple)
    measures 0.

Drift in these numbers silently shifts WHEN trimming kicks in, so the exact
constants are pinned below. Per the plan this is a static-method test with
zero setup: no sandbox, no LLM, no filesystem, no mocks.
"""

import unittest

from bash_agent.context import ContextManager


def clen(content):
    """Shorthand so assertions read like the spec lines in TEST_PLAN.md."""
    return ContextManager._content_length(content)


# ---------------------------------------------------------------------------
# T-18 — Multimodal content length accounting
# ---------------------------------------------------------------------------

class TestPlainStrings(unittest.TestCase):
    """Legacy string content is measured verbatim."""

    def test_empty_string_is_zero(self):
        self.assertEqual(clen(""), 0)

    def test_plain_string_counts_characters(self):
        self.assertEqual(clen("hello"), 5)

    def test_protocol_fenced_string_counts_every_character(self):
        text = "---START_BASH_OUTPUT-x---\nline1\nline2\n"
        self.assertEqual(clen(text), len(text))


class TestTextParts(unittest.TestCase):
    """{"type": "text"} parts are measured by their text payload."""

    def test_single_text_part(self):
        self.assertEqual(clen([{"type": "text", "text": "abc"}]), 3)

    def test_multiple_text_parts_sum(self):
        parts = [
            {"type": "text", "text": "abc"},
            {"type": "text", "text": "de"},
        ]
        self.assertEqual(clen(parts), 5)

    def test_text_part_missing_text_key_counts_zero(self):
        # .get("text", "") default — a malformed part must not raise.
        self.assertEqual(clen([{"type": "text"}]), 0)

    def test_unknown_dict_types_contribute_nothing(self):
        # Unknown part types must neither raise nor leak their payload size.
        self.assertEqual(clen([{"type": "mystery", "payload": "12345"}]), 0)


class TestImageParts(unittest.TestCase):
    """Each image_url part costs a flat 6400 chars (~800 tokens)."""

    IMAGE_CHARS = 6400  # pinned constant from context.py

    def test_single_image_part_costs_exactly_6400(self):
        img = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        self.assertEqual(clen(img), self.IMAGE_CHARS)

    def test_n_image_parts_cost_n_times_6400(self):
        imgs = [
            {"type": "image_url", "image_url": {"url": f"data:x{i}"}}
            for i in range(3)
        ]
        self.assertEqual(clen(imgs), 3 * self.IMAGE_CHARS)

    def test_image_cost_ignores_payload_size(self):
        # Real data URLs are enormous base64 blobs. The flat-rate estimate
        # must NOT scale with the payload — a naive len(url) here would
        # massively overstate context pressure and trigger premature pruning.
        tiny = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}}]
        huge = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 500_000},
            }
        ]
        self.assertEqual(clen(tiny), clen(huge))
        self.assertEqual(clen(huge), self.IMAGE_CHARS)


class TestMixedContent(unittest.TestCase):
    """Lists mixing text parts, image parts, and bare strings."""

    def test_mixed_parts_and_bare_strings_sum(self):
        content = [
            {"type": "text", "text": "hello"},                       #    5
            {"type": "image_url", "image_url": {"url": "data:…"}},   # 6400
            "bare string!",                                          #   12
            {"type": "text", "text": "!?"},                          #    2
        ]
        self.assertEqual(clen(content), 5 + 6400 + 12 + 2)

    def test_empty_list_is_zero(self):
        self.assertEqual(clen([]), 0)

    def test_junk_items_inside_list_are_skipped(self):
        # Non-dict, non-string items inside a list are ignored, not fatal.
        self.assertEqual(
            clen(["ab", 42, None, {"type": "text", "text": "cd"}]),
            4,
        )


class TestNonStringNonList(unittest.TestCase):
    """Anything that is not a str or list measures 0 (never raises)."""

    def test_none_is_zero(self):
        self.assertEqual(clen(None), 0)

    def test_int_is_zero(self):
        self.assertEqual(clen(42), 0)

    def test_plain_dict_is_zero(self):
        # A bare dict (e.g. a whole message passed by mistake) is not a list.
        self.assertEqual(clen({"role": "user", "content": "hi"}), 0)

    def test_tuple_is_zero(self):
        # Only lists are recognized — tuples fall through to the 0 branch.
        self.assertEqual(clen(("a", "b")), 0)


class TestPruningRelevance(unittest.TestCase):
    """
    Why the constants matter: they define the exchange rate between
    modalities that drives _trim_context_if_needed().
    """

    def test_one_image_outweighs_a_chunky_text_message(self):
        chunky_output = "x" * 2000  # typical large tool output
        one_image = [{"type": "image_url", "image_url": {"url": "u"}}]
        self.assertGreater(clen(one_image), clen(chunky_output))

    def test_documented_token_exchange_rate(self):
        # Source comment: "each Image is roughly 800 tokens. Or 6400 characters"
        # i.e. 8 chars/token — keep the two numbers consistent.
        self.assertEqual(6400 // 8, 800)


# ---------------------------------------------------------------------------
# T-19 — Hysteresis pruning ladder (_trim_context_if_needed)
# ---------------------------------------------------------------------------

import contextlib
import io
import re
import uuid as uuid_module
from unittest import mock

from tests.helpers.fakes import bash_block, output_block, chdir_tmp

# Tiny limit so ladders complete in a handful of passes yet stay human-checkable.
LIMIT = 2000
TARGET = int(LIMIT * 0.8)  # documented hysteresis target: prune down to 80%

DELETED_MARKER = "[BASH_OUTPUT DELETED TO SAVE CONTEXT]"
TRUNCATED_MARKER = "...[TRUNCATED]"
HYSTERESIS_BANNER = "Initiating hysteresis cleanup"


def _total_length(history):
    """Mirror of the production accounting: sum of _content_length over history."""
    return sum(ContextManager._content_length(m.get("content", "")) for m in history)


class PruningCase(unittest.TestCase):
    """
    Shared harness for ladder tests:
      * throwaway CWD (ContextManager.__init__ writes .bash_agent_tmp/SCRATCHPAD.md),
      * captured stdout (production prints [System] banners),
      * CONTEXT_LIMIT patched to LIMIT in the *context* module namespace
        (context.py binds it via `from bash_agent.config import CONTEXT_LIMIT`,
        so patching bash_agent.config.CONTEXT_LIMIT would NOT be seen).
    """

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.uid = str(uuid_module.uuid4())
        self._limit_patch = mock.patch("bash_agent.context.CONTEXT_LIMIT", LIMIT)
        self._limit_patch.start()
        self.cm = ContextManager(self.uid)

    def tearDown(self):
        self._limit_patch.stop()
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- message builders ---------------------------------------------------

    def system_prompt(self, pad=100):
        return "You are the system prompt. " + "S" * pad

    def output_msg(self, role, body):
        return {"role": role, "content": output_block(self.uid, 0, body)}

    def command_msg(self, role, script):
        return {"role": role, "content": bash_block(self.uid, script)}

    def plain_msg(self, role, text):
        return {"role": role, "content": text}

    # -- assertion helpers --------------------------------------------------

    def banners(self):
        return self.stdout_buf.getvalue().count(HYSTERESIS_BANNER)

    def command_bodies(self, msg):
        """Extract script bodies from BASH/PYTHON command fences in a message."""
        pat = (
            rf"(---START_(?:BASH|PYTHON)_COMMAND-{self.uid}---\n?)"
            rf"(.*?)"
            rf"(\n?---END_(?:BASH|PYTHON)_COMMAND-{self.uid}---)"
        )
        return re.findall(pat, msg["content"], flags=re.DOTALL)


class TestHysteresisGuard(PruningCase):
    """No trimming may occur until the STRICT limit is reached/exceeded."""

    def test_no_trim_below_limit(self):
        sys_msg = self.plain_msg("system", self.system_prompt())
        filler = self.plain_msg("user", "f" * (LIMIT - 200))
        self.cm.history = [sys_msg, filler]
        before = _total_length(self.cm.history)
        self.assertLess(before, LIMIT)

        self.cm._trim_context_if_needed()

        self.assertEqual(_total_length(self.cm.history), before)
        self.assertEqual(len(self.cm.history), 2)
        self.assertEqual(self.banners(), 0)

    def test_no_trim_exactly_at_limit(self):
        # Boundary: the guard is `total <= CONTEXT_LIMIT -> return`, so a
        # history sitting EXACTLY on the limit must be left untouched.
        sys_msg = self.plain_msg("system", self.system_prompt())
        filler_len = LIMIT - _total_length([sys_msg])
        self.cm.history = [sys_msg, self.plain_msg("user", "f" * filler_len)]
        self.assertEqual(_total_length(self.cm.history), LIMIT)

        self.cm._trim_context_if_needed()

        self.assertEqual(_total_length(self.cm.history), LIMIT)
        self.assertEqual(len(self.cm.history), 2)
        self.assertEqual(self.banners(), 0)

    def test_trim_triggers_immediately_past_limit(self):
        sys_msg = self.plain_msg("system", self.system_prompt())
        filler_len = LIMIT - _total_length([sys_msg])
        self.cm.history = [
            sys_msg,
            self.plain_msg("user", "f" * filler_len),
            self.plain_msg("assistant", "y"),
        ]
        self.assertEqual(_total_length(self.cm.history), LIMIT + 1)

        self.cm._trim_context_if_needed()

        # Banner emitted exactly once for the whole trimming episode...
        self.assertEqual(self.banners(), 1)
        # ...and the episode ended under the hysteresis target.
        self.assertLessEqual(_total_length(self.cm.history), TARGET)


class TestOutputDeletionLadder(PruningCase):
    """
    Ladder rung 1: oldest OUTPUT blocks are hollowed out first.

    History layout (oldest -> newest): system, out, out, out, cmd, plain.
    Sized so deleting the TWO OLDEST outputs lands under TARGET, proving:
      * deletions proceed oldest-first (the NEWEST output survives intact),
      * command blocks are NOT truncated while outputs suffice,
      * termination happens with total <= 80% of the limit.
    """

    def setUp(self):
        super().setUp()
        self.out_body = "o" * 480
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            self.output_msg("user", self.out_body),
            self.output_msg("user", self.out_body),
            self.output_msg("user", self.out_body),
            self.command_msg("assistant", "c" * 300),
            self.plain_msg("user", "p" * 50),
        ]
        # Size-window preconditions. A "deletion" does not reclaim the whole
        # body: the OUTPUT fences and the 37-char marker survive, so the
        # hollowed message is exactly output_block(uid, 0, DELETED_MARKER).
        # Compute the real per-deletion saving and require that two
        # deletions reach the target while one does not.
        hollow = output_block(self.uid, 0, DELETED_MARKER)
        full = output_block(self.uid, 0, self.out_body)
        savings = len(full) - len(hollow)
        total0 = _total_length(self.cm.history)
        self.assertGreater(total0, LIMIT, "fixture must start over the strict limit")
        self.assertGreater(total0 - savings, TARGET,
                           "one deletion must NOT reach the target")
        self.assertLessEqual(total0 - 2 * savings, TARGET,
                             "two deletions must reach the target")

    def test_oldest_outputs_deleted_first_and_newest_survives(self):
        before = [dict(m) for m in self.cm.history]

        self.cm._trim_context_if_needed()

        # Rung 1 hit the two oldest outputs...
        for i in (1, 2):
            content = self.cm.history[i]["content"]
            self.assertIn(DELETED_MARKER, content)
            self.assertNotIn(self.out_body, content)
            # Fences survive the hollowing-out (protocol shape preserved).
            self.assertIn(f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---", content)
            self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", content)
        # ...while the NEWEST output kept its full body: proof of oldest-first.
        self.assertIn(self.out_body, self.cm.history[3]["content"])
        self.assertNotIn(DELETED_MARKER, self.cm.history[3]["content"])
        # Each hollowed message is byte-for-byte the expected shape:
        # fences + newline-wrapped marker, nothing else reclaimed.
        self.assertEqual(
            self.cm.history[1]["content"], output_block(self.uid, 0, DELETED_MARKER)
        )

    def test_commands_not_truncated_while_outputs_suffice(self):
        self.cm._trim_context_if_needed()
        for msg in self.cm.history:
            self.assertNotIn(TRUNCATED_MARKER, msg["content"])
        # The command message is byte-identical to how it was built.
        self.assertEqual(self.cm.history[4]["content"], bash_block(self.uid, "c" * 300))

    def test_terminates_under_target_with_system_intact(self):
        original_system = self.cm.history[0]["content"]
        self.cm._trim_context_if_needed()
        self.assertLessEqual(_total_length(self.cm.history), TARGET)
        self.assertEqual(self.cm.history[0]["content"], original_system)
        self.assertEqual(self.banners(), 1)


class TestCommandTruncationLadder(PruningCase):
    """
    Ladder rung 2: with NO outputs present, command scripts are truncated
    to their first 80 characters plus the ...[TRUNCATED] marker, oldest
    first, without dropping any message.
    """

    SCRIPT = "c" * 1000

    def setUp(self):
        super().setUp()
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            self.command_msg("user", self.SCRIPT),
            self.command_msg("assistant", self.SCRIPT),
            self.plain_msg("user", "p" * 300),
        ]
        total0 = _total_length(self.cm.history)
        self.assertGreater(total0, LIMIT, "fixture must start over the strict limit")

    def test_both_commands_truncated_to_80_chars_plus_marker(self):
        self.cm._trim_context_if_needed()

        expected_body = self.SCRIPT[:80] + TRUNCATED_MARKER
        for i in (1, 2):
            bodies = self.command_bodies(self.cm.history[i])
            self.assertEqual(len(bodies), 1, "exactly one command fence expected")
            self.assertEqual(bodies[0][1], expected_body)
            self.assertIn(TRUNCATED_MARKER, self.cm.history[i]["content"])

    def test_truncation_happens_without_dropping_messages(self):
        n_before = len(self.cm.history)
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), n_before)
        self.assertLessEqual(_total_length(self.cm.history), TARGET)

    def test_plain_tail_and_system_untouched(self):
        original_plain = self.cm.history[3]["content"]
        original_system = self.cm.history[0]["content"]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[3]["content"], original_plain)
        self.assertEqual(self.cm.history[0]["content"], original_system)


class TestWholesaleDropFailsafe(PruningCase):
    """
    Ladder rung 3: when nothing is block-trimmable (plain prose only), the
    oldest non-system message is dropped WHOLE, repeatedly, until the
    hysteresis target is met.
    """

    def setUp(self):
        super().setUp()
        self.a, self.b, self.c = "a" * 900, "b" * 900, "c" * 900
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            self.plain_msg("user", self.a),
            self.plain_msg("assistant", self.b),
            self.plain_msg("user", self.c),
        ]
        self.assertGreater(_total_length(self.cm.history), LIMIT)

    def test_oldest_messages_dropped_until_under_target(self):
        self.cm._trim_context_if_needed()

        # 2827 chars total (incl. 127-char system prompt) -> drop "a" -> 1927
        # (still > 1600) -> drop "b" -> 1027 (<= 1600) -> stop.
        self.assertEqual(len(self.cm.history), 2)
        self.assertEqual(self.cm.history[1]["content"], self.c)
        self.assertLessEqual(_total_length(self.cm.history), TARGET)

    def test_system_prompt_never_dropped(self):
        original_system = self.cm.history[0]["content"]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[0]["content"], original_system)


class TestDegenerateHistoriesTerminate(PruningCase):
    """The trim loop must always terminate and never touch index 0."""

    def test_single_oversized_system_message_is_left_alone(self):
        # Pathological: ONLY the system prompt exists and it alone exceeds
        # the limit. The failsafe must bail out (len(history) == 1) instead
        # of popping index 0 or looping forever.
        big = "S" * (LIMIT * 2)
        self.cm.history = [self.plain_msg("system", big)]
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)
        self.assertEqual(self.cm.history[0]["content"], big)

    def test_repeated_trims_are_stable_once_under_target(self):
        # Hysteresis contract: once pruned under the target, subsequent
        # calls are no-ops (no further mutation, no extra banners).
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            self.plain_msg("user", "x" * 900),
            self.plain_msg("assistant", "y" * 900),
            self.plain_msg("user", "z" * 900),
        ]
        self.cm._trim_context_if_needed()
        snapshot = [dict(m) for m in self.cm.history]
        banners_after_first = self.banners()

        self.cm._trim_context_if_needed()

        self.assertEqual(self.cm.history, snapshot)
        self.assertEqual(self.banners(), banners_after_first)


if __name__ == "__main__":
    unittest.main()
