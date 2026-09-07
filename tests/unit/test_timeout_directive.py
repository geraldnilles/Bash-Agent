"""
Group 11 — Per-command timeout directive (bash_agent.agent).

T-46  Pure helper: first-line `# timeout: N` directive parsing (P1)

A block may declare its own grace period up to config.MAX_COMMAND_TIMEOUT on
its very first line:

    # timeout: 240
    <slow commands...>

extract_timeout_directive() is the pure string transformer AGENT-A owns. It
scrubs a VALID directive from the body and returns the effective integer
timeout, yet never mutates the script for malformed / misplaced directives
(they degrade to the default timeout and gain a teaching note instead).

T-47  Agent wiring: _execute_script forwards the per-call timeout (P1)

_execute_script must scrub the directive, dispatch to
sandbox.execute(..., timeout=eff) / execute_python(..., timeout=eff) ONLY
when a valid directive is present, and append any advisory note to the
formatted output using the existing trailing-warning pattern.

All tests are offline:
  * T-46 calls the pure function directly — no Agent, no sandbox.
  * T-47 goes through _make_agent with the shared tests/helpers FakeSandbox,
    which accepts the optional timeout kwarg and records it in its
    ``timeouts_used`` list (("BASH"|"PYTHON", value_or_None)).
"""
import contextlib
import io
import unittest
import uuid

from bash_agent.agent import extract_timeout_directive

from tests.helpers.fakes import (
    FakeSandbox,
    bash_block,
    chdir_tmp,
    _make_agent,
)


# ---------------------------------------------------------------------------
# T-46 — pure helper
# ---------------------------------------------------------------------------

class TestExtractTimeoutDirective(unittest.TestCase):
    """Direct calls on extract_timeout_directive — pure string logic."""

    # ----- valid directives ------------------------------------------------

    def test_valid_directive_scrubbed_and_timeout_returned(self):
        script, eff, note = extract_timeout_directive("# timeout: 240\necho hi")
        self.assertEqual(script, "echo hi")
        self.assertEqual(eff, 240)
        self.assertIsNone(note)

    def test_valid_directive_alone_produces_empty_scrubbed_script(self):
        script, eff, note = extract_timeout_directive("# timeout: 60")
        self.assertEqual(script, "")
        self.assertEqual(eff, 60)
        self.assertIsNone(note)

    def test_lower_bound_is_default_timeout(self):
        script, eff, note = extract_timeout_directive("# timeout: 60")
        self.assertEqual(eff, 60)

    def test_upper_bound_is_max_command_timeout(self):
        script, eff, note = extract_timeout_directive("# timeout: 600")
        self.assertEqual(eff, 600)

    def test_extra_whitespace_after_colon_accepted(self):
        script, eff, note = extract_timeout_directive("# timeout:  120\nls")
        self.assertEqual(script, "ls")
        self.assertEqual(eff, 120)

    def test_directive_key_is_case_insensitive(self):
        script, eff, note = extract_timeout_directive("# TIMEOUT: 90")
        self.assertEqual(eff, 90)
        self.assertIsNone(note)

    # ----- invalid / malformed --------------------------------------------

    def test_directive_on_line_two_is_not_a_directive(self):
        script, eff, note = extract_timeout_directive(
            "sleep 5\n# timeout: 300"
        )
        # Script untouched (no data loss)
        self.assertEqual(script, "sleep 5\n# timeout: 300")
        self.assertIsNone(eff)
        # ...but a teaching note is returned teaching first-line placement
        self.assertIsNotNone(note)
        self.assertIn("VERY FIRST", note)

    def test_leading_blank_line_makes_it_non_directive(self):
        script, eff, note = extract_timeout_directive("\n# timeout: 300")
        self.assertEqual(script, "\n# timeout: 300")
        self.assertIsNone(eff)
        self.assertIsNotNone(note)

    def test_non_comment_leading_line_is_not_a_directive(self):
        script, eff, note = extract_timeout_directive('print("# timeout: 999")')
        self.assertEqual(script, 'print("# timeout: 999")')
        self.assertIsNone(eff)
        self.assertIsNone(note)

    def test_non_integer_value_is_not_a_directive(self):
        script, eff, note = extract_timeout_directive("# timeout: abc")
        self.assertEqual(script, "# timeout: abc")
        self.assertIsNone(eff)
        self.assertIn("positive base-10 integer", note)

    def test_space_before_colon_is_not_a_directive(self):
        script, eff, note = extract_timeout_directive("# timeout : 200")
        self.assertEqual(script, "# timeout : 200")
        self.assertIsNone(eff)
        self.assertIsNotNone(note)

    def test_empty_script_is_untouched(self):
        script, eff, note = extract_timeout_directive("")
        self.assertEqual(script, "")
        self.assertIsNone(eff)
        self.assertIsNone(note)

    # ----- clamped ---------------------------------------------------------

    def test_below_default_is_clamped_up_with_note(self):
        script, eff, note = extract_timeout_directive("# timeout: 5\ncmd")
        self.assertEqual(script, "cmd")
        self.assertEqual(eff, 60)
        self.assertEqual(note, "# timeout: N is clamped to the minimum of 60s (requested 5s).")

    def test_above_max_is_clamped_down_with_note(self):
        script, eff, note = extract_timeout_directive("# timeout: 7000\ncmd")
        self.assertEqual(script, "cmd")
        self.assertEqual(eff, 600)
        self.assertEqual(note, "# timeout: N is clamped to the maximum of 600s (requested 7000s).")

    def test_absent_directive_returns_none_timeout_no_note(self):
        script, eff, note = extract_timeout_directive("echo hello")
        self.assertEqual(script, "echo hello")
        self.assertIsNone(eff)
        self.assertIsNone(note)


# ---------------------------------------------------------------------------
# T-47 — Agent wiring (shared FakeSandbox records timeouts_used)
# ---------------------------------------------------------------------------

class TimeoutWiringCase(unittest.TestCase):
    """Agent + shared FakeSandbox so _execute_script dispatch can be asserted."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.uid = str(uuid.uuid4())
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.agent = _make_agent(uuid_str=self.uid)
        self.fake_sb = FakeSandbox(execute_result=(0, "ran"))
        self.agent.sandbox = self.fake_sb

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def test_bash_directive_scrubs_and_forwards_timeout(self):
        formatted = self.agent._execute_script("BASH", "# timeout: 240\necho slow")
        # scrubbed script reached the sandbox
        self.assertEqual(self.fake_sb.executed_scripts, ["echo slow"])
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", 240)])
        # no advisory note for a valid directive
        self.assertNotIn("SYSTEM WARNING", formatted)
        self.assertIn("ran", formatted)

    def test_python_directive_forwards_timeout(self):
        self.agent._execute_script("PYTHON", "# timeout: 300\nprint('ok')")
        self.assertEqual(self.fake_sb.executed_python_scripts, ["print('ok')"])
        self.assertEqual(self.fake_sb.timeouts_used, [("PYTHON", 300)])

    def test_no_directive_never_passes_timeout(self):
        self.agent._execute_script("BASH", "echo hello")
        self.assertEqual(self.fake_sb.executed_scripts, ["echo hello"])
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", None)])

    def test_no_directive_python_never_passes_timeout(self):
        self.agent._execute_script("PYTHON", "print('hi')")
        self.assertEqual(self.fake_sb.timeouts_used, [("PYTHON", None)])

    def test_clamped_directive_forwards_clamped_value_and_note(self):
        formatted = self.agent._execute_script("BASH", "# timeout: 5\nfast")
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", 60)])
        self.assertIn("clamped to the minimum of 60s", formatted)

    def test_malformed_directive_runs_original_and_warns(self):
        formatted = self.agent._execute_script(
            "BASH", "# timeout : 200\necho still-runs"
        )
        # original untouched, no timeout forwarded
        self.assertEqual(self.fake_sb.executed_scripts, ["# timeout : 200\necho still-runs"])
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", None)])
        # teaching note appended after the output fence
        self.assertIn("[SYSTEM WARNING]", formatted)

    def test_misplaced_directive_runs_original_and_warns(self):
        formatted = self.agent._execute_script(
            "BASH", "echo first\n# timeout: 300"
        )
        self.assertEqual(
            self.fake_sb.executed_scripts, ["echo first\n# timeout: 300"]
        )
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", None)])
        self.assertIn("[SYSTEM WARNING]", formatted)

    def test_regression_misplaced_directive_no_data_loss_full_flow(self):
        """AGENT-E regression: a misplaced/malformed directive surviving the
        parser reaches the sandbox UNCHANGED (no data loss) and executes with
        the default timeout, while the OUTPUT gains a teaching warning."""
        # Misplaced (line 2+) directive inside a real fenced body
        body = "echo real-command\n# timeout: 999"
        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, body)
        )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        # Original script reached the sandbox verbatim - no data loss
        self.assertEqual(
            self.fake_sb.executed_scripts,
            ["echo real-command\n# timeout: 999"],
        )
        # No per-call timeout forwarded -> default remains
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", None)])
        # Teaching warning appended on the committed OUTPUT
        users = [m for m in self.agent.context.history if m["role"] == "user"]
        self.assertEqual(len(users), 1)
        self.assertIn("[SYSTEM WARNING]", users[0]["content"])

    def test_full_parse_and_execute_with_directive(self):
        """End-to-end through the parser: directive survives the block body
        (parser keeps it inside the script) and is honored at dispatch."""
        from tests.helpers.fakes import output_block

        body = "# timeout: 240\necho slow"
        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, body)
        )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(self.fake_sb.executed_scripts, ["echo slow"])
        self.assertEqual(self.fake_sb.timeouts_used, [("BASH", 240)])

        # committed message carries the OUTPUT block with the scrubbed result
        users = [m for m in self.agent.context.history if m["role"] == "user"]
        self.assertEqual(len(users), 1)
        self.assertIn(
            output_block(self.uid, 0, "ran"),
            users[0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
