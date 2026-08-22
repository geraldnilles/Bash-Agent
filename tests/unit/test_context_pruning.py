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


if __name__ == "__main__":
    unittest.main()
