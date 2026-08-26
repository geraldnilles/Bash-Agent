"""
Group 4 — Context management tests for ContextManager.

T-18  Multimodal content length accounting (P1)

ContextManager._content_length is the yardstick every pruning decision is
measured with: _trim_context_if_needed() sums it across the whole history to
decide when CONTEXT_LIMIT is breached, and agent.py reuses the same static
method for its context-size bookkeeping. Its contract:

  * plain strings count their character length,
  * multimodal lists count each {"type": "text"} part by its text length,
  * {"type": "image_url"} parts are charged by IMAGE RESOLUTION:
    1000 tokens per megapixel (rounded, then x8 chars/token), derived by
    decoding the data URL.  Undecodable payloads / non-data URLs fall back
    to a flat 6400 chars (~800 tokens).
  * {"type": "input_audio"} parts are charged by CLIP LENGTH:
    400 tokens per minute, derived by parsing the MP3 frame headers
    (falling back to a 128kbps CBR estimate from payload bytes when header
    parsing fails).  Unparseable payloads fall back to a flat 50000 chars.
  * In NO case does the calculation scale with the raw base64 payload
    length — a naive len(url)/len(data) would massively overstate context
    pressure and trigger premature pruning.
  * bare strings nested inside a list count their own length,
  * junk items inside a list (ints, None, unknown dict types) contribute 0,
  * anything that is neither a str nor a list (None, int, dict, tuple)
    measures 0.

Drift in these numbers silently shifts WHEN trimming kicks in, so the exact
rates are pinned below. Per the plan this is a static-method test with
zero setup: no sandbox, no LLM, no filesystem, no mocks.
"""

import base64
import io
import unittest

from PIL import Image

from bash_agent.context import ContextManager


def clen(content):
    """Shorthand so assertions read like the spec lines in TEST_PLAN.md."""
    return ContextManager._content_length(content)


# ---------------------------------------------------------------------------
# T-18 — Multimodal content length accounting
# ---------------------------------------------------------------------------

def png_data_url(width, height, pixel_bytes=16):
    """Encode a real PNG of the given dimensions as a data URL."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def mp3_b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


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
    """image_url parts are charged at 1000 tokens per megapixel."""

    def test_exactly_one_megapixel_costs_8000_chars(self):
        # 1000x1000 px = 1.0 MP -> 1000 tokens -> 8000 chars (x8).
        img = [{"type": "image_url", "image_url": {"url": png_data_url(1000, 1000)}}]
        self.assertEqual(clen(img), 8000)

    def test_two_megapixels_cost_16000_chars(self):
        img = [{"type": "image_url", "image_url": {"url": png_data_url(2000, 1000)}}]
        self.assertEqual(clen(img), 16000)

    def test_half_megapixel_costs_4000_chars(self):
        img = [{"type": "image_url", "image_url": {"url": png_data_url(1000, 500)}}]
        self.assertEqual(clen(img), 4000)

    def test_subpixel_resolution_rounds_down_to_0(self):
        # 8x6 = 48 px is essentially 0 MP; the rounded estimate is 0 chars.
        img = [{"type": "image_url", "image_url": {"url": png_data_url(8, 6)}}]
        self.assertEqual(clen(img), 0)

    def test_n_image_parts_cost_n_times_single(self):
        imgs = [
            {"type": "image_url", "image_url": {"url": png_data_url(1000, 1000)}}
            for _ in range(3)
        ]
        self.assertEqual(clen(imgs), 3 * 8000)

    def test_undecodable_data_url_falls_back_to_flat_6400(self):
        # Whatever the "payload" is, junk base64 must never raise and never
        # scale with its length; legacy flat rate applies instead.
        tiny = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}}]
        huge = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 500_000},
            }
        ]
        self.assertEqual(clen(tiny), 6400)
        self.assertEqual(clen(huge), 6400)
        self.assertEqual(clen(tiny), clen(huge))

    def test_non_data_url_falls_back_to_flat_6400(self):
        # Plain http(s) URLs carry no embedded pixels -> flat estimate.
        img = [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
        self.assertEqual(clen(img), 6400)

    def test_missing_image_url_key_counts_fallback(self):
        # Malformed part must not raise; flat estimate applies.
        self.assertEqual(clen([{"type": "image_url"}]), 6400)

    def test_legacy_bare_string_image_url_counts_fallback(self):
        # Older clients use {"image_url": "http..."} instead of the
        # nested {"image_url": {"url": ...}} dict; must not raise and
        # falls back to the flat rate (no pixel data in a bare URL).
        self.assertEqual(
            clen([{"type": "image_url", "image_url": "https://x/y.png"}]),
            6400,
        )

    def test_decodable_bare_string_data_url_still_scales(self):
        # Even in the legacy bare-string shape, a decodable data URL is
        # charged by its resolution.
        self.assertEqual(
            clen([{"type": "image_url", "image_url": png_data_url(1000, 1000)}]),
            8000,
        )

    def test_dynamic_cost_does_not_scale_with_png_payload_size(self):
        # Two same-resolution images with very different file sizes (e.g.
        # solid color vs noisy) must cost the SAME: the resolution, not
        # the byte length, drives the estimate.
        buf_plain = io.BytesIO()
        Image.new("RGB", (100, 100), (0, 0, 0)).save(buf_plain, format="PNG")
        buf_noisy = io.BytesIO()
        Image.new("RGB", (100, 100), (0, 0, 0))
        # stack many identical PNGs is overkill; use a large random-looking
        # payload inside a DIFFERENT but still decodable PNG by increasing
        # color noise:
        from PIL import Image as _I
        noisy = _I.new("RGB", (100, 100))
        px = noisy.load()
        for y in range(100):
            for x in range(100):
                px[x, y] = (x * y % 256, (x + y) % 256, (x * 7 + y * 3) % 256)
        noisy.save(buf_noisy, format="PNG")
        b64_plain = base64.b64encode(buf_plain.getvalue()).decode()
        b64_noisy = base64.b64encode(buf_noisy.getvalue()).decode()
        self.assertLess(len(b64_plain), len(b64_noisy))
        p1 = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_plain}"}}]
        p2 = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_noisy}"}}]
        self.assertEqual(clen(p1), clen(p2))


class TestAudioParts(unittest.TestCase):
    """input_audio parts are charged at 400 tokens per minute."""

    # A 60-second, mono 128kbps MP3 (the exact profile transcribe.py's
    # convert_to_mp3 emits). At 44100Hz/1152 samples per MPEG1 L3 frame:
    # frame_len = (1152/8)*128000/44100 = 417.96 bytes/frame and
    # frames/sec ≈ 38.28, so 60s ≈ 2297 frames ≈ 960KB.
    MP3_FALLBACK_CHARS = 50000  # legacy flat fallback for unparseable audio

    @classmethod
    def setUpClass(cls):
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not available to build MP3 fixtures")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            cls._mp3_path = tf.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi",
                 "-i", "anullsrc=r=44100:cl=mono",
                 "-t", "60", "-ac", "1", "-b:a", "128k",
                 cls._mp3_path],
                check=True, capture_output=True,
            )
            with open(cls._mp3_path, "rb") as f:
                cls._sixty_sec_mp3 = f.read()
        finally:
            try:
                os.unlink(cls._mp3_path)
            except Exception:
                pass

    def _audio_part(self, b64_payload):
        return [
            {
                "type": "input_audio",
                "input_audio": {"data": b64_payload, "format": "mp3"},
            }
        ]

    def test_fifty_nine_to_sixty_second_mp3_costs_about_3200(self):
        # 400 tokens/min * 1 minute = 400 tokens = 3200 chars. The header
        # walk can shave a frame or two, so tolerate ±150 chars.
        cost = clen(self._audio_part(mp3_b64(self._sixty_sec_mp3)))
        self.assertTrue(3050 <= cost <= 3350, f"cost was {cost}")

    def test_two_minutes_double_the_length_contribution(self):
        six = self._sixty_sec_mp3
        two_min = six + six  # concatenated frames -> ~120s of audio
        c1 = clen(self._audio_part(mp3_b64(six)))
        c2 = clen(self._audio_part(mp3_b64(two_min)))
        self.assertTrue(2 * c1 - 400 <= c2 <= 2 * c1 + 400, f"{c1} vs {c2}")

    def test_unparseable_payload_falls_back_to_flat_50000(self):
        # Junk base64 never raises and never scales with its length.
        tiny = self._audio_part("a")
        huge = self._audio_part("A" * 500_000)
        self.assertEqual(clen(tiny), self.MP3_FALLBACK_CHARS)
        self.assertEqual(clen(huge), self.MP3_FALLBACK_CHARS)

    def test_missing_input_audio_key_counts_fallback(self):
        self.assertEqual(clen([{"type": "input_audio"}]), self.MP3_FALLBACK_CHARS)

    def test_legacy_bare_string_input_audio_counts_fallback(self):
        # Legacy shape {"input_audio": "base64..."} must not raise;
        # junk payload falls back to the flat estimate.
        self.assertEqual(
            clen([{"type": "input_audio", "input_audio": "junk"}]),
            self.MP3_FALLBACK_CHARS,
        )

    def test_decodable_bare_string_audio_scales_with_length(self):
        # Even in the legacy bare-string shape, a real MP3 payload is
        # charged by its clip length (≈3200 chars for 60s).
        cost = clen(
            [{"type": "input_audio", "input_audio": mp3_b64(self._sixty_sec_mp3)}]
        )
        self.assertTrue(3050 <= cost <= 3350, f"cost was {cost}")

    def test_empty_payload_falls_back_to_flat_50000(self):
        self.assertEqual(clen(self._audio_part("")), self.MP3_FALLBACK_CHARS)

    def test_text_plus_audio_mixed_content(self):
        # Fallback flat rate for junk payload "DATA".
        content = [
            {"type": "text", "text": "hello"},                  #      5
            {"type": "input_audio",
             "input_audio": {"data": "DATA", "format": "mp3"}}, # 50000
            "bare!",                                            #      5
        ]
        self.assertEqual(clen(content), 5 + self.MP3_FALLBACK_CHARS + 5)


class TestMixedContent(unittest.TestCase):
    """Lists mixing text parts, image parts, and bare strings."""

    def test_mixed_parts_and_bare_strings_sum(self):
        content = [
            {"type": "text", "text": "hello"},                       #    5
            {"type": "image_url",
             "image_url": {"url": png_data_url(1000, 1000)}},        # 8000
            "bare string!",                                          #   12
            {"type": "text", "text": "!?"},                          #    2
        ]
        self.assertEqual(clen(content), 5 + 8000 + 12 + 2)

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

    def test_one_megapixel_image_outweighs_a_chunky_text_message(self):
        chunky_output = "x" * 2000  # typical large tool output
        one_image = [{"type": "image_url", "image_url": {"url": png_data_url(1000, 1000)}}]
        self.assertGreater(clen(one_image), clen(chunky_output))

    def test_documented_token_exchange_rates(self):
        # 8 chars/token exchange rate, 400 tokens/min audio, and
        # 1000 tokens/MP image are the documented rates driving pruning.
        import bash_agent.context as context_module

        self.assertEqual(context_module.CHARS_PER_TOKEN, 8)
        self.assertEqual(context_module.AUDIO_TOKENS_PER_MINUTE, 400)
        self.assertEqual(context_module.IMAGE_TOKENS_PER_MEGAPIXEL, 1000)


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


# ---------------------------------------------------------------------------
# T-20 — Image-bearing messages dropped wholesale (P0)
# ---------------------------------------------------------------------------

IMAGE_DROP_BANNER = "Dropped an old image-bearing message"


def text_part(text):
    return {"type": "text", "text": text}


def image_part():
    # Payload size is irrelevant to accounting (flat 6400/image — pinned by
    # T-18); a small blob keeps the fixture readable.
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 64},
    }


def image_msg(role, caption="describe this screenshot"):
    """A realistic multimodal message as built by _commit_execution_feedback."""
    return {"role": role, "content": [text_part(caption), image_part()]}


def audio_msg(role, caption="transcribe this recording"):
    """A realistic audio-bearing multimodal message as built by
    _commit_execution_feedback (input_audio part)."""
    return {
        "role": role,
        "content": [
            text_part(caption),
            {
                "type": "input_audio",
                "input_audio": {"data": "SUQzBAAAAA==", "format": "mp3"},
            },
        ],
    }


class MultimodalPruningCase(PruningCase):
    """Shared pruning harness plus multimodal-specific assertion helpers."""

    def list_messages(self, history=None):
        history = self.cm.history if history is None else history
        return [m for m in history if isinstance(m.get("content"), list)]


class TestImageMessageDroppedWholesale(MultimodalPruningCase):
    """
    Core regression guard (commit 78773ca): under pruning pressure a
    list-content (multimodal) message must be REMOVED whole. Before the
    fix, the regex ladder hit the list and raised TypeError ("expected
    string or bytes-like object"), crashing the entire trim loop.
    """

    def setUp(self):
        super().setUp()
        self.sys_content = self.system_prompt()
        self.img = image_msg("user", "what does this stack trace show?")
        self.out = self.output_msg("assistant", "o" * 300)
        self.tail = self.plain_msg("user", "t" * 300)
        self.cm.history = [
            self.plain_msg("system", self.sys_content),
            dict(self.img),
            self.out,
            self.tail,
        ]
        # ~128 + ~6425 (flat-rate image) + ~445 + 300 >> LIMIT.
        self.assertGreater(_total_length(self.cm.history), LIMIT,
                           "fixture must start over the strict limit")

    def test_list_message_removed_entirely(self):
        self.cm._trim_context_if_needed()
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(len(self.cm.history), 3)

    def test_string_neighbors_untouched_by_the_drop(self):
        # Popping the flat-rate image alone frees plenty of space; the
        # surrounding string messages must come through without any
        # ladder treatment (no deletion marker, no truncation).
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[0]["content"], self.sys_content)
        self.assertEqual(self.cm.history[1]["content"], self.out["content"])
        self.assertNotIn(DELETED_MARKER, self.cm.history[1]["content"])
        self.assertNotIn(TRUNCATED_MARKER, self.cm.history[1]["content"])
        self.assertEqual(self.cm.history[2]["content"], self.tail["content"])
        self.assertLessEqual(_total_length(self.cm.history), TARGET)

    def test_dedicated_banner_printed_not_failsafe(self):
        # The multimodal branch has its own banner; assert we took THAT
        # path and not the generic oldest-message failsafe drop.
        self.cm._trim_context_if_needed()
        out = self.stdout_buf.getvalue()
        self.assertEqual(out.count(IMAGE_DROP_BANNER), 1)
        self.assertEqual(out.count(HYSTERESIS_BANNER), 1)
        self.assertNotIn("Dropping oldest conversational message", out)


class TestAudioMessageDroppedWholesale(MultimodalPruningCase):
    """Mirror of the image regression guard: a list-content message carrying
    an input_audio part must be removed ENTIRELY under pruning pressure —
    never regex-laddered (lists would crash the ladder) and never
    partially truncated."""

    def setUp(self):
        super().setUp()
        self.sys_content = self.system_prompt()
        self.aud = audio_msg("user", "transcribe this meeting")
        self.out = self.output_msg("assistant", "o" * 300)
        self.tail = self.plain_msg("user", "t" * 300)
        self.cm.history = [
            self.plain_msg("system", self.sys_content),
            dict(self.aud),
            self.out,
            self.tail,
        ]
        # ~128 + ~50005 (flat-rate audio) + ~445 + 300 >> LIMIT.
        self.assertGreater(_total_length(self.cm.history), LIMIT,
                           "fixture must start over the strict limit")

    def test_list_message_removed_entirely(self):
        self.cm._trim_context_if_needed()
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(len(self.cm.history), 3)

    def test_string_neighbors_untouched_by_the_drop(self):
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[0]["content"], self.sys_content)
        self.assertEqual(self.cm.history[1]["content"], self.out["content"])
        self.assertNotIn(DELETED_MARKER, self.cm.history[1]["content"])
        self.assertNotIn(TRUNCATED_MARKER, self.cm.history[1]["content"])
        self.assertEqual(self.cm.history[2]["content"], self.tail["content"])
        self.assertLessEqual(_total_length(self.cm.history), TARGET)

    def test_dedicated_banner_printed_not_failsafe(self):
        # The multimodal branch has its own banner; assert we took THAT
        # path and not the generic oldest-message failsafe drop.
        self.cm._trim_context_if_needed()
        out = self.stdout_buf.getvalue()
        self.assertEqual(out.count(IMAGE_DROP_BANNER), 1)
        self.assertEqual(out.count(HYSTERESIS_BANNER), 1)
        self.assertNotIn("Dropping oldest conversational message", out)


class TestMultipleImagesDroppedOldestFirst(MultimodalPruningCase):
    """Two image-bearing messages: both popped, oldest first."""

    def setUp(self):
        super().setUp()
        self.tail = self.plain_msg("user", "t" * 300)
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            image_msg("user", "first screenshot"),
            image_msg("assistant", "second screenshot"),
            self.tail,
        ]
        self.assertGreater(_total_length(self.cm.history), LIMIT)

    def test_both_images_popped_and_tail_survives(self):
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 2)
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(self.cm.history[1]["content"], self.tail["content"])
        self.assertLessEqual(_total_length(self.cm.history), TARGET)

    def test_one_banner_per_dropped_image(self):
        self.cm._trim_context_if_needed()
        self.assertEqual(
            self.stdout_buf.getvalue().count(IMAGE_DROP_BANNER), 2)


class TestMixedLadderStringsAndImages(MultimodalPruningCase):
    """
    Both treatments in one trim session: the image message is popped
    wholesale while string OUTPUT blocks around it receive the normal
    deletion ladder until the hysteresis target is met.

    Fixture arithmetic (uuid is always 36 chars): sys=128, output block
    with an n-char body = n+145, hollowed block = 182, image msg = 6425.
      initial: 128 + 845 + 6425 + 845 + 845            = 9088 > 2000
      pass 1 (oldest output hollowed):                 = 8425
      pass 2 (image popped):                           = 2000 > 1600
      pass 3 (second output hollowed):                 = 1337 <= 1600 STOP
    """

    def setUp(self):
        super().setUp()
        self.old_out = self.output_msg("user", "o" * 700)
        self.mid_out = self.output_msg("user", "m" * 700)
        self.new_out = self.output_msg("assistant", "n" * 700)
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            self.old_out,
            image_msg("assistant", "mid-conversation screenshot"),
            self.mid_out,
            self.new_out,
        ]
        self.assertGreater(_total_length(self.cm.history), LIMIT)

    def test_image_popped_while_old_outputs_deleted(self):
        self.cm._trim_context_if_needed()
        contents = [m["content"] for m in self.cm.history]
        self.assertFalse(any(isinstance(c, list) for c in contents))
        self.assertIn(DELETED_MARKER, self.cm.history[1]["content"])
        self.assertIn(DELETED_MARKER, self.cm.history[2]["content"])
        self.assertNotIn(DELETED_MARKER, self.cm.history[3]["content"])

    def test_banners_report_both_treatments(self):
        self.cm._trim_context_if_needed()
        out = self.stdout_buf.getvalue()
        self.assertEqual(out.count(IMAGE_DROP_BANNER), 1)
        self.assertGreaterEqual(out.count("Context trimmed an old block"), 1)

    def test_terminates_under_target_with_system_intact(self):
        original_system = self.cm.history[0]["content"]
        self.cm._trim_context_if_needed()
        self.assertLessEqual(_total_length(self.cm.history), TARGET)
        self.assertEqual(self.cm.history[0]["content"], original_system)


class TestMultimodalEdgeCases(MultimodalPruningCase):
    """Degenerate inputs must terminate and respect the index-0 invariant."""

    def test_list_content_at_index_zero_is_never_popped(self):
        # The trim loop starts at index 1 for EVERY branch, including the
        # multimodal wholesale drop. A degenerate history whose only entry
        # is an oversized list-content system prompt must be left alone
        # rather than popping index 0 or looping forever.
        big_sys = image_msg("system", "pathological oversized system prompt")
        self.cm.history = [big_sys]
        self.assertGreater(_total_length(self.cm.history), LIMIT)
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)
        self.assertIs(self.cm.history[0], big_sys)

    def test_any_non_string_content_dropped_wholesale(self):
        # The guard is `not isinstance(content, str)` — not an image_url
        # check — so ANY non-string content (e.g. legacy bare-string lists,
        # which _content_length still measures) takes the wholesale path.
        bare_list_msg = {"role": "user", "content": ["x" * 5000]}
        tail = self.plain_msg("assistant", "t" * 300)
        self.cm.history = [
            self.plain_msg("system", self.system_prompt()),
            bare_list_msg,
            tail,
        ]
        self.assertGreater(_total_length(self.cm.history), LIMIT)
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 2)
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(self.cm.history[1]["content"], tail["content"])
        self.assertEqual(
            self.stdout_buf.getvalue().count(IMAGE_DROP_BANNER), 1)


if __name__ == "__main__":
    unittest.main()
