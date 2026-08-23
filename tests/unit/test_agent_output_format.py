"""
Group 3 — Output formatting tests for Agent._format_output.

T-16  Output truncation formatting (P0)

_format_output is the last thing sandbox output passes through before it is
committed back into the conversation, so its contract IS the protocol's
output half:

  * small outputs pass through byte-for-byte under a VISIBLE_100% header,
  * outputs longer than config.OUTPUT_LIMIT are replaced by their first and
    last OUTPUT_LIMIT//2 characters joined by TRUNCATION_BANNER,
    a single-line unicode/emoji fence marker in the START/END style that
    includes the session UUID,
  * the VISIBLE_% header reports floor(OUTPUT_LIMIT / original_len * 100),
    computed from the PRE-truncation length,
  * the exit code survives in the header either way, and
  * truncated results gain a trailing coaching line telling the model how to
    avoid truncation (grep/head/tail/sed/awk).

All tests are direct calls on the pure method — no sandbox, no LLM, no I/O
beyond the throwaway CWD that Agent construction touches. Boundary tests
patch bash_agent.agent.OUTPUT_LIMIT (the name _format_output actually reads)
to keep the arithmetic human-checkable.
"""
import contextlib
import io
import re
import unittest
import uuid
from unittest import mock

from bash_agent.config import OUTPUT_LIMIT
from bash_agent.agent import TRUNCATION_BANNER

from tests.helpers.fakes import (
    output_block,
    chdir_tmp,
    _make_agent,
)


class OutputFormatCase(unittest.TestCase):
    """Shared harness: temp CWD + captured stdout + ready-made agent."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.uid = str(uuid.uuid4())
        # Capture production prints; individual tests may assert on them.
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.agent = _make_agent(uuid_str=self.uid)

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def fmt(self, *args, **kwargs):
        """Direct call into the pure method under test."""
        return self.agent._format_output(*args, **kwargs)


# ---------------------------------------------------------------------------
# T-16 — Output truncation formatting
# ---------------------------------------------------------------------------

class TestSmallOutputsPassThrough(OutputFormatCase):
    """Below the limit, output must survive byte-for-byte at VISIBLE_100%."""

    def test_small_output_round_trips_through_helper(self):
        """A typical ls-style output equals the helper-built fence exactly."""
        out = (
            "total 172\n"
            "drwxr-xr-x 2 gerald gerald 4096 .\n"
            "-rw-r--r-- 1 gerald gerald 848 pyproject.toml"
        )
        result = self.fmt(0, out)
        # Byte-for-byte agreement with the protocol-valid fence builder.
        self.assertEqual(result, output_block(self.uid, 0, out))
        self.assertIn("VISIBLE_100%", result)
        # No truncation marker, no coaching line.
        self.assertNotIn("Truncated", result)
        self.assertNotIn("WARNING", result)

    def test_empty_output_still_well_formed(self):
        """Commands may produce nothing; the fence must still be valid."""
        result = self.fmt(0, "")
        self.assertEqual(result, output_block(self.uid, 0, ""))
        self.assertIn("VISIBLE_100%", result)

    def test_interior_newlines_kept_trailing_newlines_stripped(self):
        """rstrip('\\n') only trims the tail; interior blank lines survive."""
        result = self.fmt(0, "a\n\nb\n\n\n")
        self.assertEqual(result, output_block(self.uid, 0, "a\n\nb"))

    def test_exit_codes_survive_in_header(self):
        """EXIT_CODE_N is preserved verbatim, including signal negatives."""
        for code in (0, 1, 2, 127, 255, -9):
            with self.subTest(exit_code=code):
                result = self.fmt(code, "ok")
                self.assertIn(f"EXIT_CODE_{code}-VISIBLE_100%", result)

    def test_header_layout_pins_field_order_and_uuid(self):
        """START fence reads START_<TYPE>_OUTPUT-EXIT_CODE_n-VISIBLE_p%-<uuid>."""
        first_line = self.fmt(3, "hi").splitlines()[0]
        m = re.match(
            r"^---START_BASH_OUTPUT-EXIT_CODE_3-VISIBLE_100%-(.+?)---$",
            first_line,
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), self.uid)

    def test_python_flavor_names_the_fences(self):
        """cmd_type drives both fence names; BASH must not leak in."""
        result = self.fmt(0, "42", cmd_type="PYTHON")
        self.assertIn(
            f"---START_PYTHON_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---",
            result,
        )
        self.assertIn(f"---END_PYTHON_OUTPUT-{self.uid}---", result)
        self.assertNotIn("BASH", result)


class TestTruncatedOutputs(OutputFormatCase):
    """Above the limit: head+tail halves, floored VISIBLE math, advice line."""

    def test_large_output_head_tail_split(self):
        """25k chars -> first/last 5k kept, middle dropped, VISIBLE_40%."""
        head, mid, tail = "A" * 8000, "M" * 9000, "Z" * 8000
        out = head + mid + tail  # exactly 25,000 chars
        result = self.fmt(0, out)

        expected_body = (
            "A" * 5000
            + TRUNCATION_BANNER.format(uuid=self.uid)
            + "Z" * 5000
        )
        expected_core = output_block(self.uid, 0, expected_body, visible=40)
        # The truncated block is the helper-built fence plus a warning suffix.
        self.assertTrue(result.startswith(expected_core))

        suffix = result[len(expected_core):]
        self.assertTrue(suffix.startswith("\n\n"))
        self.assertIn("⚠️ WARNING", suffix)
        self.assertIn("Only 40%", suffix)
        self.assertIn("was truncated", suffix)
        self.assertIn("grep, head, tail, sed, awk", suffix)

        # Middle region really gone; halves really contiguous; one marker.
        self.assertNotIn("MMMMMMM", result)
        self.assertIn("A" * 5000, result)
        self.assertIn("Z" * 5000, result)
        self.assertEqual(result.count("OUTPUT_TRUNCATED_HERE"), 1)

    def test_truncation_point_arithmetic_pins_boundary_chars(self):
        """The split is exactly [:5000] + [-5000:] of the original.

        Sentinel X sits at index 4999: it belongs to the head half. The tail
        is the final 5,000 Q's; everything between is discarded.
        """
        out = "P" * 4999 + "X" + "Q" * 15000  # exactly 20,000 chars
        result = self.fmt(0, out)
        expected_body = (
            "P" * 4999
            + "X"
            + TRUNCATION_BANNER.format(uuid=self.uid)
            + "Q" * 5000
        )
        self.assertTrue(
            result.startswith(output_block(self.uid, 0, expected_body, visible=50))
        )

    def test_visible_percentage_is_floored_ratio_of_original_length(self):
        """VISIBLE_% = int(OUTPUT_LIMIT / original_len * 100), hand-computed."""
        cases = {
            10_001: 99,     # int(99.9900...) — one char over the limit
            12_345: 81,     # int(81.0036...)
            20_000: 50,
            25_000: 40,
            50_000: 20,
            100_000: 10,
            123_456: 8,     # int(8.1000...)
        }
        for size, pct in cases.items():
            with self.subTest(size=size, expected=pct):
                result = self.fmt(0, "c" * size)
                self.assertIn(f"VISIBLE_{pct}%", result)
                self.assertNotIn("VISIBLE_100%", result)

    def test_at_limit_output_is_not_truncated(self):
        """Boundary: len == OUTPUT_LIMIT uses strict '>', so it passes through
        untouched at VISIBLE_100% with the real production constant."""
        out = "x" * OUTPUT_LIMIT
        result = self.fmt(0, out)
        self.assertEqual(result, output_block(self.uid, 0, out))
        self.assertNotIn("Truncated", result)
        self.assertNotIn("WARNING", result)

    def test_over_limit_by_one_char_is_truncated(self):
        """Boundary from above with a small patched limit: 1,001 chars against
        OUTPUT_LIMIT=1000 -> halves of 500, VISIBLE_99%."""
        with mock.patch("bash_agent.agent.OUTPUT_LIMIT", 1000):
            out = "y" * 1001
            result = self.fmt(0, out)
        expected_body = "y" * 500 + TRUNCATION_BANNER.format(uuid=self.uid) + "y" * 500
        self.assertTrue(
            result.startswith(output_block(self.uid, 0, expected_body, visible=99))
        )

    def test_failure_output_truncated_keeps_exit_code_and_edges(self):
        """A big failing log keeps EXIT_CODE_2, its first/last lines, drops
        the middle, and still gets the coaching line."""
        out = "\n".join(f"error line {i}: E" for i in range(2000))  # ~32k chars
        result = self.fmt(2, out)

        m = re.search(r"EXIT_CODE_2-VISIBLE_(\d+)%", result)
        self.assertIsNotNone(m)
        self.assertLess(int(m.group(1)), 100)

        self.assertIn("error line 0:", result)      # head edge survives
        self.assertIn("error line 1999:", result)   # tail edge survives
        self.assertNotIn("error line 1000:", result)  # middle is gone
        self.assertIn("⚠️ WARNING", result)
        self.assertIn("grep, head, tail, sed, awk", result)


if __name__ == "__main__":
    unittest.main()
