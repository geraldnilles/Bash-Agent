"""
Group 8 — pure-helper tests for bash_agent.memo.

T-40  memo helpers (P2)

Recording itself is hardware-dependent (PipeWire microphones, pw-record,
Ctrl+C timing) and stays untested BY DESIGN per TEST_PLAN.md. What CAN be
pinned offline is the parsing/formatting logic around it:

  * get_sources            — parses canned `pactl list sources short` output;
                             monitor filtering; malformed/blank line tolerance
  * find_source            — node-ID vs case-insensitive substring matching,
                             ambiguity resolution (first match wins), error
  * format_timestamp       — deterministic under a frozen clock
  * format_duration        — seconds arithmetic across all three tiers
                             (sub-minute, minutes, hours)
  * fmt_size               — human-readable byte scaling incl. TB fallback
  * pick_source_interactive— stdin-driven selection (EOF/blank/valid paths)

All subprocess boundaries are patched at bash_agent.memo.subprocess.run;
nothing here touches audio hardware.
"""

import contextlib
import io
import os
import tempfile
import types
import unittest
from unittest import mock

from bash_agent import memo

# ---------------------------------------------------------------------------
# Canned fixtures
# ---------------------------------------------------------------------------

PACTL_OUTPUT = (
    # id, name, driver, spec — tab separated like real `pactl ... short`
    "54\talsa_input.pci-0000_00_1f.3.analog-stereo\tpipeewire\tfront-left\n"
    "61\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\t\t\n"
    "72\talsa_input.usb-Blue_Microphones_Yeti_Stereo-00.analog-stereo\textra\n"
    "\n"
    "garbage-line-without-tab\n"
    "99\tusb_mic_with_54_in_name\n"
)


def make_result(stdout_text):
    """Stand-in for subprocess.CompletedProcess (only .stdout is read)."""
    return types.SimpleNamespace(stdout=stdout_text, stderr="", returncode=0)

SOURCES = [
    ("54", "alsa_input.pci.analog-stereo"),
    ("61", "alsa_output.pci.analog-stereo.monitor"),
    ("72", "alsa_input.usb-yeti"),
]


class TestGetSources(unittest.TestCase):
    def setUp(self):
        self.run_mock = mock.patch(
            "bash_agent.memo.subprocess.run",
            return_value=make_result(PACTL_OUTPUT),
        )
        self.run_mock.start()
        self.addCleanup(self.run_mock.stop)

    def test_invokes_pactl_correctly(self):
        memo.get_sources()
        args, kwargs = memo.subprocess.run.call_args
        self.assertEqual(args[0], ["pactl", "list", "sources", "short"])
        self.assertTrue(kwargs.get("capture_output"))
        self.assertEqual(kwargs.get("text"), True)
        self.assertIsNotNone(kwargs.get("timeout"))

    def test_filters_monitor_sinks_by_default(self):
        sources = memo.get_sources()
        names = [name for _, name in sources]
        self.assertNotIn("alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
                         "".join(names))
        self.assertIn(("54", "alsa_input.pci-0000_00_1f.3.analog-stereo"),
                      sources)
        self.assertIn(("72", "alsa_input.usb-Blue_Microphones_Yeti_Stereo-00.analog-stereo"),
                      sources)

    def test_include_monitors_keeps_everything(self):
        sources = memo.get_sources(include_monitors=True)
        ids = [nid for nid, _ in sources]
        self.assertIn("54", ids)
        self.assertIn("61", ids)
        self.assertIn("72", ids)
        self.assertIn("99", ids)

    def test_skips_blank_and_malformed_lines(self):
        sources = memo.get_sources(include_monitors=True)
        self.assertEqual(len(sources), 4)  # 54, 61, 72, 99 — junk dropped

    def test_empty_pactl_output(self):
        with mock.patch("bash_agent.memo.subprocess.run",
                        return_value=make_result("")):
            self.assertEqual(memo.get_sources(), [])


class TestFindSource(unittest.TestCase):
    def test_numeric_spec_matches_node_id(self):
        srcs = [("99", "usb_mic_with_54_in_name"), ("54", "other")]
        self.assertEqual(memo.find_source("54", srcs), ("54", "other"))

    def test_substring_match_case_insensitive(self):
        result = memo.find_source("YETI", SOURCES)
        self.assertEqual(result[1], "alsa_input.usb-yeti")

    def test_node_id_match_takes_precedence_over_name(self):
        # "54" appears inside a NAME, but an exact node-ID hit must win.
        srcs = [("10", "mic54"), ("54", "internal")]
        self.assertEqual(memo.find_source("54", srcs), ("54", "internal"))

    def test_multiple_matches_returns_first_and_prints_choices(self):
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            result = memo.find_source("alsa_input", SOURCES)
        out = stdout_buf.getvalue()
        self.assertEqual(result, ("54", "alsa_input.pci.analog-stereo"))
        self.assertIn("Multiple sources match", out)
        self.assertIn("Using the first match:", out)
        # Both candidates were shown
        self.assertIn("alsa_input.pci.analog-stereo", out)
        self.assertIn("alsa_input.usb-yeti", out)

    def test_no_match_raises_value_error(self):
        with self.assertRaises(ValueError) as cm:
            memo.find_source("nonexistent-mic", SOURCES)
        self.assertIn("No source found matching 'nonexistent-mic'",
                      str(cm.exception))

    def test_numeric_spec_with_no_id_falls_through_to_substring_then_errors(self):
        with self.assertRaises(ValueError):
            memo.find_source("777", SOURCES)


class TestFormatTimestamp(unittest.TestCase):
    def test_iso_compact_format_under_frozen_clock(self):
        import datetime as _dt
        frozen = _dt.datetime(2026, 4, 28, 15, 30, 45)
        fake_dtmod = types.SimpleNamespace(
            datetime=types.SimpleNamespace(now=lambda: frozen))
        with mock.patch("bash_agent.memo.datetime", fake_dtmod):
            self.assertEqual(memo.format_timestamp(), "2026-04-28T153045")


class TestFormatDuration(unittest.TestCase):
    def test_examples_across_all_tiers(self):
        cases = {
            0: "0s",
            5: "5s",
            59: "59s",
            60: "1m00s",
            65: "1m05s",          # canonical plan example
            3599: "59m59s",
            3600: "1h00m00s",
            3723: "1h02m03s",
            90061: "25h01m01s",   # hours beyond 24 kept raw
        }
        for secs, expected in cases.items():
            with self.subTest(secs=secs):
                self.assertEqual(memo.format_duration(secs), expected)

    def test_float_seconds_truncated_not_rounded(self):
        self.assertEqual(memo.format_duration(59.9), "59s")
        self.assertEqual(memo.format_duration(60.9), "1m00s")

    def test_negative_seconds_clamps_via_int(self):
        # int(-0.5) == 0 -> smallest tier renders "0s"
        self.assertEqual(memo.format_duration(-0.5), "0s")


class TestFmtSize(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    @property
    def tmpdir(self):
        return self._tmp.name

    def _file_of_size(self, nbytes, name="blob.bin"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.truncate(nbytes)  # sparse — logical size without disk cost
        return path

    def test_bytes_kilobytes_megabytes_gigabytes(self):
        self.assertEqual(memo.fmt_size(self._file_of_size(512)), "512.0 B")
        self.assertEqual(memo.fmt_size(self._file_of_size(2048)), "2.0 KB")
        self.assertEqual(memo.fmt_size(self._file_of_size(5 * 1024 * 1024)),
                         "5.0 MB")
        self.assertEqual(
            memo.fmt_size(self._file_of_size(3 * 1024 ** 3)), "3.0 GB")

    def test_terabyte_fallback_beyond_table(self):
        # Avoid materializing multi-TB sparseness on odd filesystems: patch size.
        with mock.patch("bash_agent.memo.os.path.getsize",
                        return_value=2 * 1024 ** 4):
            self.assertEqual(memo.fmt_size("/unused/path.bin"), "2.0 TB")


class TestPickSourceInteractive(unittest.TestCase):
    """pick_source_interactive is stdin/stdout driven; both are patched here.

    The helper catches ValueError/EOFError/KeyboardInterrupt from input() and
    falls back to sources[0], so the EOF test feeds an exhausted answer queue
    (scripted() converts StopIteration into EOFError).
    """

    def _run_pick(self, answers):
        out = io.StringIO()

        def scripted(prompt=""):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        with mock.patch("builtins.input", side_effect=scripted), \
                contextlib.redirect_stdout(out):
            result = memo.pick_source_interactive(SOURCES)
        return result, out.getvalue()

    def test_enter_accepts_default_first_source(self):
        result, _ = self._run_pick(iter([""]))
        self.assertEqual(result, SOURCES[0])

    def test_valid_index_selects_that_source(self):
        result, out = self._run_pick(iter(["2"]))
        self.assertEqual(result, SOURCES[1])
        self.assertIn("[61]", out)  # menu listed before the prompt

    def test_eof_falls_back_to_first_source(self):
        result, _ = self._run_pick(iter([]))
        self.assertEqual(result, SOURCES[0])

    def test_keyboard_interrupt_falls_back_to_first_source(self):
        def interrupted(prompt=""):
            raise KeyboardInterrupt

        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=interrupted), \
                contextlib.redirect_stdout(out):
            self.assertEqual(memo.pick_source_interactive(SOURCES), SOURCES[0])

    def test_out_of_range_reprompts_then_accepts_valid_choice(self):
        result, out = self._run_pick(iter(["9", "1"]))
        self.assertEqual(result, SOURCES[0])
        self.assertIn("Please enter a number between 1 and 3", out)

    def test_garbage_input_returns_default_without_error(self):
        # int("abc") -> ValueError caught by helper -> returns sources[0]
        result, _ = self._run_pick(iter(["abc"]))
        self.assertEqual(result, SOURCES[0])


if __name__ == "__main__":
    unittest.main()
