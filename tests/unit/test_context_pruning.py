"""
Group 4 — Context management tests for ContextManager.

T-18  Multimodal content TOKEN accounting (P0)
T-19  Hysteresis pruning ladder (P0)
T-20  Multimodal wholesale drop (P0)

The ContextManager accounts *ALL* conversational pressure in provider
TOKENS (see bash_agent/tokenizer.py).  Plain text is tokenized with the
model's real local tokenizer; multimodal parts are priced by stated rates
(1000 tokens/MP image, 400 tokens/min audio).  There is NO character
arithmetic left in these tests.

To stay deterministic in token-space every text fixture is built with
``tokfill(n)``: a pure ``'u'*n`` string that the DeepSeek tokenizer encodes
to EXACTLY ``n`` tokens (verified: 2 characters -> 1 token, no context
sensitivity).  This lets us dial an exact conversation weight without
guessing.

The pruning tests assert the *behavioural contract*:

  * under the strict per-instance ceiling nothing is touched or announced,
  * once the ceiling is crossed the trim loop runs until total tokens fall
    to <= 80% of the ceiling (hysteresis target),
  * hollowing works oldest-first on plain string OUTPUT messages while
    preserving the protocol fence shape and replacing only the body,
  * when only command scripts remain they are truncated body-first to
    80 characters (plus the marker),
  * when only plain conversational prose remains the oldest message is
    dropped,
  * list-content (image / audio) messages are never fed to the regex
    ladder (that would raise TypeError) — they are dropped whole,
  * the system prompt at index 0 is never removed,
  * the trim loop always terminates on pathological inputs.
"""
import base64
import contextlib
import io
import os
import re
import unittest
import uuid as uuid_module
from unittest import mock

from PIL import Image

from bash_agent.context import ContextManager
from bash_agent.tokenizer import count_tokens
from tests.helpers.fakes import bash_block, chdir_tmp, output_block

# ---------------------------------------------------------------------------
# T-18 — content accounting
# ---------------------------------------------------------------------------


def tokfill(n: int) -> str:
    """Return a plain string tokenized to EXACTLY ``n`` tokens.

    ``u`` is a 1-byte token in the DeepSeek tokenizer; each pair of
    characters encodes as exactly one token (2 chars per token), verified
    empirically.  All text fixtures stack ``u`` so the number of tokens is
    dialed directly without any external rate.
    """
    if n < 0:
        raise ValueError(n)
    return "u" * (2 * n)


def clen(content):
    """Shorthand: token length of one message-content value."""
    return ContextManager._content_tokens(content)


# ---------------------------------------------------------------------------
# T-18 helpers
# ---------------------------------------------------------------------------

def png_data_url(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def mp3_b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


# ---------------------------------------------------------------------------
# T-18 — plain text tokens
# ---------------------------------------------------------------------------

class TestPlainStrings(unittest.TestCase):
    def test_empty_string_is_zero(self):
        self.assertEqual(clen(""), 0)

    def test_short_lowercase_string_tokenizes(self):
        # "hello" compresses into one BPE token, not 5 chars.
        self.assertEqual(clen("hello"), 1)

    def test_protocol_fenced_string_tokenizes(self):
        text = "---START_BASH_OUTPUT-x---\nline1\nline2\n"
        self.assertEqual(clen(text), count_tokens(text))

    def test_tokfill_n_produces_n_tokens(self):
        for n in (0, 1, 7, 200, 5000):
            self.assertEqual(clen(tokfill(n)), n)


class TestTextParts(unittest.TestCase):
    def test_single_text_part(self):
        self.assertEqual(clen([{"type": "text", "text": "abc"}]), 1)

    def test_multiple_text_parts_sum(self):
        parts = [
            {"type": "text", "text": "abc"},
            {"type": "text", "text": "de"},
        ]
        self.assertEqual(clen(parts), 2)

    def test_text_part_missing_text_key_counts_zero(self):
        self.assertEqual(clen([{"type": "text"}]), 0)


# ---------------------------------------------------------------------------
# T-18 — image url accounting (tokens per MP, flat fallback; never scales
#        with base64 payload size)
# ---------------------------------------------------------------------------

class TestImageParts(unittest.TestCase):
    def test_exactly_one_megapixel_costs_1000_tokens(self):
        img = [{"type": "image_url",
                "image_url": {"url": png_data_url(1000, 1000)}}]
        self.assertEqual(clen(img), 1000)

    def test_two_megapixels_cost_2000_tokens(self):
        img = [{"type": "image_url",
                "image_url": {"url": png_data_url(2000, 1000)}}]
        self.assertEqual(clen(img), 2000)

    def test_half_megapixel_costs_500_tokens(self):
        img = [{"type": "image_url",
                "image_url": {"url": png_data_url(1000, 500)}}]
        self.assertEqual(clen(img), 500)

    def test_subpixel_resolution_counts_zero_tokens(self):
        img = [{"type": "image_url",
                "image_url": {"url": png_data_url(8, 6)}}]
        self.assertEqual(clen(img), 0)

    def test_n_images_are_n_times_single(self):
        imgs = [
            {"type": "image_url",
             "image_url": {"url": png_data_url(1000, 1000)}}
            for _ in range(3)
        ]
        self.assertEqual(clen(imgs), 3 * 1000)

    def test_undecodable_url_falls_back_to_flat_800(self):
        tiny = [{"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,a"}}]
        huge = [{"type": "image_url",
                 "image_url": {
                     "url": "data:image/png;base64," + "A" * 500_000}}]
        self.assertEqual(clen(tiny), 800)
        self.assertEqual(clen(huge), 800)
        self.assertEqual(clen(tiny), clen(huge))

    def test_non_data_url_falls_back_to_flat_800(self):
        img = [{"type": "image_url",
                "image_url": {"url": "https://example.com/x.png"}}]
        self.assertEqual(clen(img), 800)


# ---------------------------------------------------------------------------
# T-18 — audio accounting (tokens / minute, flat fallback; not base64 size)
# ---------------------------------------------------------------------------

class TestAudioParts(unittest.TestCase):
    MP3_FALLBACK_TOKENS = 6000

    @classmethod
    def setUpClass(cls):
        import subprocess
        import tempfile
        import shutil

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
        return [{"type": "input_audio",
                 "input_audio": {"data": b64_payload, "format": "mp3"}}]

    def test_sixty_sec_mp3_costs_about_400_tokens(self):
        cost = clen(self._audio_part(mp3_b64(self._sixty_sec_mp3)))
        self.assertTrue(385 <= cost <= 415, f"cost was {cost}")

    def test_two_minutes_roughly_double(self):
        one = self._sixty_sec_mp3
        two = one + one
        c1 = clen(self._audio_part(mp3_b64(one)))
        c2 = clen(self._audio_part(mp3_b64(two)))
        self.assertTrue(2 * c1 - 40 <= c2 <= 2 * c1 + 40, f"{c1} vs {c2}")

    def test_unparseable_payload_is_flat_6000(self):
        tiny = self._audio_part("a")
        huge = self._audio_part("A" * 500_000)
        self.assertEqual(clen(tiny), self.MP3_FALLBACK_TOKENS)
        self.assertEqual(clen(huge), self.MP3_FALLBACK_TOKENS)
        self.assertEqual(clen(tiny), clen(huge))

    def test_missing_audio_key_counts_fallback(self):
        self.assertEqual(clen([{"type": "input_audio"}]),
                         self.MP3_FALLBACK_TOKENS)

    def test_empty_payload_counts_fallback(self):
        self.assertEqual(clen(self._audio_part("")), self.MP3_FALLBACK_TOKENS)


# ---------------------------------------------------------------------------
# T-18 — lists with mixed shapes
# ---------------------------------------------------------------------------

class TestMixedContent(unittest.TestCase):
    def test_mixed_parts_and_bare_strings_sum(self):
        content = [
            {"type": "text", "text": "hello"},                  #  1
            {"type": "image_url",
             "image_url": {"url": png_data_url(1000, 1000)}},  # 1000
            "bare string!",                                     #  3
            {"type": "text", "text": "!?"},                     #  2
        ]
        self.assertEqual(clen(content), 1 + 1000 + 3 + 2)

    def test_empty_list_is_zero(self):
        self.assertEqual(clen([]), 0)

    def test_junk_items_are_skipped(self):
        self.assertEqual(clen(["ab", 42, None, {"type": "text", "text": "cd"}]), 2)


# ---------------------------------------------------------------------------
# T-18 — anything not str/list is counted as 0
# ---------------------------------------------------------------------------

class TestNonStringNonList(unittest.TestCase):
    def test_none_is_zero(self):
        self.assertEqual(clen(None), 0)

    def test_int_is_zero(self):
        self.assertEqual(clen(42), 0)

    def test_plain_dict_is_zero(self):
        self.assertEqual(clen({"role": "user", "content": "hi"}), 0)

    def test_tuple_is_zero(self):
        self.assertEqual(clen(("a", "b")), 0)


# ---------------------------------------------------------------------------
# pruning constants and shared harness
# ---------------------------------------------------------------------------

# Token budget for the trimmed ContextManager.  Small enough that a handful
# of fixture messages already exercise the ladder, yet consistent so that
# any drift in `_content_tokens` is caught immediately.
TOKEN_LIMIT = 1000
TARGET = int(TOKEN_LIMIT * 0.8)  # documented hysteresis target

DELETED_MARKER = "[BASH_OUTPUT DELETED TO SAVE CONTEXT]"
TRUNCATED_MARKER = "...[TRUNCATED]"
HYSTERESIS_BANNER = "Initiating hysteresis cleanup"
IMAGE_DROP_BANNER = "Dropped an old image-bearing message"
FAILSAFE_BANNER = "Dropping oldest conversational message"


def total_tokens(history) -> int:
    return sum(ContextManager._content_tokens(m.get("content", ""))
               for m in history)


def text_part(text):
    return {"type": "text", "text": text}


def image_part():
    # account for 800 fallback tokens; content payload irrelevant
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 64}}


def image_msg(role, caption="describe this screenshot"):
    return {"role": role, "content": [text_part(caption), image_part()]}


def audio_msg(role, caption="transcribe this recording"):
    return {"role": role,
            "content": [
                text_part(caption),
                {"type": "input_audio",
                 "input_audio": {"data": "SUQzBAAAAA==", "format": "mp3"}},
            ]}


class PruningCase(unittest.TestCase):
    """Shared harness: temp CWD, captured stdout, per-instance token limit."""

    limit = TOKEN_LIMIT
    target = TARGET

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self._chdir_cm.__enter__()
        self._cleanup_chdir = True
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.uid = str(uuid_module.uuid4())
        self.cm = ContextManager(self.uid, context_limit=self.limit)
        self.addCleanup(self.cleanup_trims)

    def cleanup_trims(self):
        # exit the redirect streams / chdir in reverse setup order
        try:
            self._stdout_cm.__exit__(None, None, None)
            self._chdir_cm.__exit__(None, None, None)
        except Exception:
            pass

    # message builders
    def system_prompt(self, pad=100):
        return "You are the system prompt. " + "S" * pad

    def output_msg(self, role, body_tokens):
        return {"role": role,
                "content": output_block(self.uid, 0, tokfill(body_tokens))}

    def command_msg(self, role, script_tokens):
        # c is 4 chars / token.  Keep it readable and use 'u' for exact.
        return {"role": role,
                "content": bash_block(self.uid, tokfill(script_tokens))}

    def plain_msg(self, role, token_count):
        return {"role": role, "content": tokfill(token_count)}

    # stdout observations
    def stdout_text(self):
        return self.stdout_buf.getvalue()

    def banners(self):
        return self.stdout_text().count(HYSTERESIS_BANNER)


# ---------------------------------------------------------------------------
# T-19 — hysteresis guard
# ---------------------------------------------------------------------------

class TestHysteresisGuard(PruningCase):
    """Guards: no action until TOTAL strictly exceeds the per-instance ceiling."""

    def test_no_trim_below_limit(self):
        # sys (56) + plain user (400 tokens) comfortably below ceiling.
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},
            self.plain_msg("user", 400),
        ]
        total0 = total_tokens(self.cm.history)
        self.assertLess(total0, self.limit)
        self.cm._trim_context_if_needed()
        self.assertEqual(total_tokens(self.cm.history), total0)
        self.assertEqual(self.banners(), 0)
        self.assertEqual(len(self.cm.history), 2)

    def test_exactly_at_limit_no_trim(self):
        # sys=56; pad the user message so total == 1000 exactly.  The guard
        # is `total <= limit -> return`, so AT the limit nothing runs.
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},
            self.plain_msg("user", self.limit - 56),
        ]
        self.assertEqual(total_tokens(self.cm.history), self.limit)
        before = [dict(m) for m in self.cm.history]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history, before)
        self.assertEqual(self.banners(), 0)

    def test_one_token_over_limit_triggers_trim(self):
        # push sys + user to exactly 1001 => crosses the strict guard.
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},
            self.plain_msg("user", self.limit - 55),   # 56+945=1001
        ]
        self.assertEqual(total_tokens(self.cm.history), self.limit + 1)
        self.cm._trim_context_if_needed()
        self.assertEqual(self.banners(), 1)
        self.assertLessEqual(total_tokens(self.cm.history), self.target)

    def test_trim_when_crosses_limit(self):
        # 3 x output block of 300 token bodies already far above limit.
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},
            self.output_msg("user", 300),
            self.output_msg("user", 300),
            self.output_msg("user", 300),
        ]
        total0 = total_tokens(self.cm.history)
        self.assertGreater(total0, self.limit)
        self.cm._trim_context_if_needed()
        self.assertEqual(self.banners(), 1)
        self.assertLessEqual(total_tokens(self.cm.history), self.target)

    def test_hysteresis_banner_names_context_in_tokens(self):
        self.cm.history = [
            self.plain_msg("system", 0),
            self.output_msg("user", 300),
            self.output_msg("user", 300),
            self.output_msg("user", 300),
        ]
        self.cm._trim_context_if_needed()
        banner = self.stdout_text()
        self.assertIn("tokens", banner)
        self.assertIn(str(self.target), banner)


# ---------------------------------------------------------------------------
# T-19 — Output hollowing ladder (string messages)
# ---------------------------------------------------------------------------

class TestOutputDeletionLadder(PruningCase):
    """
    Ladder rung 1: hollow oldest OUTPUT blocks first.

    Token fixture (limit=1000, target=800):
      [system(56), user output body 200t (~271), user output body 200t (~271),
       assistant output body 300t (~371), plain tail 100t]

      initial total = 56 + 271+271+371+100   = 1069 > 1000
      pass 1: hollow idx1 (full 271 -> marker ~85) : total ~ 883 > 800
      pass 2: hollow idx2 (full 271 -> marker ~85) : total ~ 697 <= 800 STOP
      => exactly two OLDEST user outputs get hollowed; assistant body & tail intact.
    """

    def setUp(self):
        super().setUp()
        self.body_old = 200
        self.body_new = 300
        self.tail_tokens = 100
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},   # 56
            self.output_msg("user", self.body_old),                # idx1
            self.output_msg("user", self.body_old),                # idx2
            self.output_msg("assistant", self.body_new),           # idx3
            self.plain_msg("user", self.tail_tokens),              # idx4
        ]
        self.total0 = total_tokens(self.cm.history)
        self.assertGreater(self.total0, self.limit,
                           "fixture must start over the strict limit")

    def test_exactly_two_oldest_outputs_hollowed(self):
        self.cm._trim_context_if_needed()
        hollowed = [
            i for i, m in enumerate(self.cm.history)
            if isinstance(m.get("content"), str)
            and DELETED_MARKER in m["content"]
        ]
        # Hollowing ladder removed the body from exactly indexes 1 and 2.
        self.assertEqual(hollowed, [1, 2])
        # Hollowed messages keep the OUTPUT fences + marker (protocol shape).
        for i in (1, 2):
            self.assertIn(f"---START_BASH_OUTPUT-EXIT_CODE_0-",
                          self.cm.history[i]["content"])
            self.assertIn(DELETED_MARKER, self.cm.history[i]["content"])
            self.assertNotIn("uuuuu", self.cm.history[i]["content"])

    def test_newest_output_and_tail_untouched(self):
        orig_new = self.cm.history[3]["content"]
        orig_tail = self.cm.history[4]["content"]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[3]["content"], orig_new)
        self.assertEqual(self.cm.history[4]["content"], orig_tail)
        self.assertNotIn(DELETED_MARKER, self.cm.history[3]["content"])

    def test_drops_below_target_after_two_hollows(self):
        self.cm._trim_context_if_needed()
        self.assertLessEqual(total_tokens(self.cm.history), self.target)
        # Only the 'hollowed' pathway prints; not a wholesale 'drop'
        self.assertNotIn(FAILSAFE_BANNER, self.stdout_text())
        self.assertNotIn(TRUNCATED_MARKER, self.stdout_text())

    def test_system_prompt_untouched(self):
        original = self.cm.history[0]["content"]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[0]["content"], original)
        self.assertEqual(len(self.cm.history), 5)

# ---------------------------------------------------------------------------
# T-19 — command truncation
# ---------------------------------------------------------------------------

class TestCommandTruncationLadder(PruningCase):
    """
    Ladder rung 2: with NO outputs present, command scripts are truncated
    to 80 chars (plus the marker), oldest first, without dropping any message.

    Token arithmetic (limit=1000, target=800):
      sys (56) + cmd script(550 tok, full cost ~612) twice:
        start  = 56 + 612 + 612             = 1280 > 1000
        pass 1 (hollow oldest cmd to ~147): = 815  > 800  -> continue
        pass 2 (hollow second cmd to ~147): = 350  <= 800 -> stop
    """

    def setUp(self):
        super().setUp()
        self.script_tokens = 600          # 600 "u"-pairs => body 1200 chars
        self.cm.history = [
            {"role": "system", "content": self.system_prompt()},  # 56
            self.command_msg("user", self.script_tokens),          # full ~612
            self.command_msg("assistant", self.script_tokens),     # full ~612
        ]
        self.assertTrue(total_tokens(self.cm.history) > self.limit,
                        "fixture must start over the strict limit")

    def test_commands_truncated_but_keep_fences(self):
        self.cm._trim_context_if_needed()
        truncated = [m for m in self.cm.history
                     if isinstance(m.get("content"), str)
                     and TRUNCATED_MARKER in m["content"]]
        self.assertEqual(len(truncated), 2)
        self.assertLessEqual(total_tokens(self.cm.history), self.target)

    def test_no_messages_dropped(self):
        n = len(self.cm.history)
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), n)
        self.assertNotIn(FAILSAFE_BANNER, self.stdout_text())


class TestWholesaleDropFailsafe(PruningCase):
    """Only plain messages -> oldest dropped until target reached."""

    def setUp(self):
        super().setUp()
        self.cm.history = [
            self.plain_msg("system", 100),
            self.plain_msg("user", 700),
            self.plain_msg("assistant", 700),
            self.plain_msg("user", 700),
        ]
        self.assertGreater(total_tokens(self.cm.history), self.limit)

    def test_drop_happens_until_under_target(self):
        self.cm._trim_context_if_needed()
        self.assertLessEqual(total_tokens(self.cm.history), self.target)
        self.assertIn(FAILSAFE_BANNER, self.stdout_text())
        # system prompt retained at index 0
        self.assertEqual(self.cm.history[0]["role"], "system")

    def test_system_never_dropped_even_with_oversize(self):
        self.cm.history = [self.plain_msg("system", 10_000)]
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)
        self.assertEqual(self.cm.history[0]["role"], "system")


# ---------------------------------------------------------------------------
# T-19 — degenerate histories terminate
# ---------------------------------------------------------------------------

class TestDegenerateHistoriesTerminate(PruningCase):
    def test_single_oversized_system_is_untouched(self):
        big = tokfill(5000)
        self.cm.history = [{"role": "system", "content": big}]
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)

    def test_repeated_trims_stable(self):
        self.cm.history = [
            self.plain_msg("system", 0),
            self.plain_msg("user", 500),
            self.plain_msg("user", 800),
            self.plain_msg("user", 700),
        ]
        self.cm._trim_context_if_needed()
        snapshot = [dict(m) for m in self.cm.history]
        first_banners = self.banners()
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history, snapshot)
        self.assertEqual(self.banners(), first_banners)


# ---------------------------------------------------------------------------
# T-20 — multimodal drop
# ---------------------------------------------------------------------------

class MultimodalPruningCase(PruningCase):
    def setUp(self):
        super().setUp()

    def list_messages(self):
        return [m for m in self.cm.history
                if isinstance(m.get("content"), list)]


class TestImageMessageDroppedWholesale(MultimodalPruningCase):
    """
    Core regression guard (commit 78773ca): under pruning pressure a
    list-content (multimodal) message must be REMOVED whole. Before the
    fix, the regex ladder hit the list and raised TypeError ("expected
    string or bytes-like object"), crashing the entire trim loop.

    Token fixture (limit=1000, target=800):
      sys(0) + image list(804) + assistant plain(420 tokens) + user plain(300)
      total 1524 > 1000.  Popping the 804-token list message yields
      720 tokens <= 800 -- exactly ONE list message dropped, the two
      surrounding plain-string messages survive untouched.
    """
    def setUp(self):
        super().setUp()
        self.cm.history = [
            {"role": "system", "content": ""},
            image_msg("user", "what does this stack trace show?"),   # ~804
            self.plain_msg("assistant", 420),
            self.plain_msg("user", 300),
        ]
        self.assertGreater(total_tokens(self.cm.history), self.limit,
                           "fixture must start over the strict limit")

    def test_list_message_removed_entirely(self):
        n = len(self.cm.history)
        self.assertEqual(len(self.list_messages()), 1)
        self.cm._trim_context_if_needed()
        self.assertLessEqual(total_tokens(self.cm.history), self.target)
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(len(self.cm.history), n - 1)   # only list popped
        self.assertEqual(self.cm.history[0]["role"], "system")

    def test_string_neighbors_untouched_by_the_drop(self):
        sys_content = self.cm.history[0]["content"]
        assistant_content = self.cm.history[2]["content"]
        tail_content = self.cm.history[3]["content"]
        self.cm._trim_context_if_needed()
        # list drop alone brought us under target, so the two plain string
        # messages must pass through byte-for-byte.
        self.assertEqual(self.cm.history[0]["content"], sys_content)
        self.assertEqual(self.cm.history[1]["content"], assistant_content)
        self.assertEqual(self.cm.history[2]["content"], tail_content)
        self.assertNotIn(DELETED_MARKER, assistant_content)
        self.assertNotIn(TRUNCATED_MARKER, assistant_content)

    def test_dedicated_banner_printed_not_failsafe(self):
        self.cm._trim_context_if_needed()
        out = self.stdout_text()
        self.assertEqual(out.count(IMAGE_DROP_BANNER), 1)
        self.assertEqual(out.count(HYSTERESIS_BANNER), 1)
        self.assertNotIn(FAILSAFE_BANNER, out)


class TestAudioMessageDroppedWholesale(MultimodalPruningCase):
    """
    Mirror of the image regression guard: an audio-bearing list-content
    message (~6000 flat-token cost) is removed ENTIRELY under pruning
    pressure -- never regex-laddered (lists would crash the ladder).
    """
    def setUp(self):
        super().setUp()
        self.cm.history = [
            {"role": "system", "content": ""},
            audio_msg("user", "transcribe this meeting"),   # ~6004
            self.plain_msg("assistant", 420),
            self.plain_msg("user", 300),
        ]
        self.assertGreater(total_tokens(self.cm.history), self.limit,
                           "fixture must start over the strict limit")

    def test_audio_message_removed_entirely(self):
        n = len(self.cm.history)
        self.assertEqual(len(self.list_messages()), 1)
        self.cm._trim_context_if_needed()
        self.assertLessEqual(total_tokens(self.cm.history), self.target)
        self.assertEqual(self.list_messages(), [])
        self.assertEqual(len(self.cm.history), n - 1)
        self.assertEqual(self.cm.history[0]["role"], "system")

    def test_string_neighbors_untouched(self):
        assistant_content = self.cm.history[2]["content"]
        tail_content = self.cm.history[3]["content"]
        self.cm._trim_context_if_needed()
        self.assertEqual(self.cm.history[1]["content"], assistant_content)
        self.assertEqual(self.cm.history[2]["content"], tail_content)
        self.assertNotIn(DELETED_MARKER, assistant_content)

    def test_dedicated_banner_printed_not_failsafe(self):
        self.cm._trim_context_if_needed()
        out = self.stdout_text()
        self.assertEqual(out.count(IMAGE_DROP_BANNER), 1)
        self.assertNotIn(FAILSAFE_BANNER, out)


class TestMultimodalEdgeCases(MultimodalPruningCase):
    """Degenerate inputs must terminate and respect the index-0 invariant."""

    def test_list_content_at_index_zero_is_never_popped(self):
        # Trim loop starts at index 1 for EVERY branch, including the
        # multimodal wholesale drop. A degenerate history whose only entry
        # is an oversized list-content system prompt must be left alone
        # rather than popping index 0 or looping forever.
        # An oversized list-content SYSTEM prompt: add a second huge text part
        # so the single message itself far exceeds the per-instance ceiling.
        big_sys = {
            "role": "system",
            "content": [
                text_part("pathological oversized system prompt"),
                text_part(tokfill(2000)),          # +2000 token text part
                image_part(),                       # 800 token image
            ],
        }
        self.cm.history = [big_sys]
        self.assertGreater(total_tokens(self.cm.history), self.limit)
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)
        self.assertIs(self.cm.history[0], big_sys)

    def test_any_non_string_non_system_gets_whole_drop(self):
        # The pruner takes the wholesale path for ANY list content (regardless
        # of whether a real image is present), dropping it as a unit with the
        # image-drop banner rather than feeding it to the regex string ladder.
        bare_list_msg = {"role": "user", "content": [tokfill(5000)]}
        tail = self.plain_msg("assistant", 300)
        self.cm.history = [
            {"role": "system", "content": ""},
            bare_list_msg,
            tail,
        ]
        self.assertGreater(total_tokens(self.cm.history), self.limit)
        self.cm._trim_context_if_needed()
        self.assertLessEqual(total_tokens(self.cm.history), self.target)
        self.assertEqual(self.list_messages(), [])
        # system preserved at index 0, tail at index 1
        self.assertEqual(self.cm.history[0]["role"], "system")
        self.assertEqual(self.cm.history[1]["content"], tail["content"])
        self.assertIn(IMAGE_DROP_BANNER, self.stdout_text())


if __name__ == "__main__":
    unittest.main()
