"""
Group 2 — Special Commands tests for agent._handle_special_command.

T-06  `exit` terminates the process (P0)
T-07  `reset` clears history but preserves system prompt (P0)
T-08  `request-write` approval flows (P1)
T-09  `ask-user` captures stdin answer (P1)
T-10  `copy-to-clipboard` exits after copying (P1)
T-11  Non-special scripts fall through to execution (P0)

These tests pin the interception layer that sits between block parsing and
sandbox execution. Special commands must be handled BEFORE any sandbox round
trip, must produce protocol-valid OUTPUT blocks, and — for the terminating
commands (`exit`, `copy-to-clipboard`) — must flush debug state before
sys.exit() so a resumed session never loses the final turn.

All tests are offline:
  * Agent construction goes through helpers.fakes._make_agent (no network
    probe, FakeSandbox injected).
  * T-08 swaps in a REAL bash_agent.sandbox.Sandbox instance because its
    request_write() prompts via builtins.input(); the prompt is answered by
    patching builtins.input — no other seam is needed.
  * sys.exit is wrapped (never suppressed) so SystemExit still propagates and
    ordering can be recorded.

Every test runs inside a throwaway CWD (chdir_tmp) and captures stdout so the
production prints ([System] ..., [AGENT REQUEST] ...) stay out of test output.
"""
import contextlib
import io
import os
import sys
import unittest
import uuid
from unittest import mock

from tests.helpers.fakes import (
    bash_block,
    python_block,
    chdir_tmp,
    _make_agent,
    FakeSandbox,
)
from bash_agent.sandbox import Sandbox


class SpecialCommandCase(unittest.TestCase):
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

    def last_message_text(self) -> str:
        """Text of the most recent context message (str or multimodal list)."""
        content = self.agent.context.history[-1]["content"]
        if isinstance(content, str):
            return content
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )

    def attach_real_sandbox(self) -> Sandbox:
        """Swap in a real (offline-safe) Sandbox for input()-driven flows."""
        sb = Sandbox(
            scratchpad_path=os.path.join(self.tmpdir, ".bash_agent_tmp", "SCRATCHPAD.md"),
            uuid=self.uid,
        )
        self.agent.sandbox = sb
        return sb


# ---------------------------------------------------------------------------
# T-06 — `exit` terminates the process
# ---------------------------------------------------------------------------

class TestExitCommand(SpecialCommandCase):
    def test_exit_via_parse_and_execute_raises_system_exit_zero(self):
        """A bash block containing exactly `exit` must raise SystemExit(0)
        from parse_and_execute, and the sandbox must never be touched."""
        with mock.patch("sys.exit", side_effect=sys.exit):
            with self.assertRaises(SystemExit) as cm:
                self.agent.parse_and_execute(bash_block(self.uid, "exit"))
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(self.agent.sandbox.executed_scripts, [])

    def test_debug_history_flushed_before_exit(self):
        """Order matters for resumability: the debug log flush must complete
        before sys.exit() fires."""
        events = []
        real_exit = sys.exit

        def fake_exit(code=0):
            events.append("exit")
            real_exit(code)

        with mock.patch.object(
            self.agent, "_log_debug_history", side_effect=lambda: events.append("flush")
        ), mock.patch("sys.exit", side_effect=fake_exit):
            with self.assertRaises(SystemExit):
                self.agent.parse_and_execute(bash_block(self.uid, "exit"))
        self.assertEqual(events, ["flush", "exit"])

    def test_python_block_exit_also_terminates(self):
        """_handle_special_command ignores cmd_type: a PYTHON block containing
        exactly `exit` is intercepted identically."""
        with mock.patch("sys.exit", side_effect=sys.exit):
            with self.assertRaises(SystemExit) as cm:
                self.agent.parse_and_execute(python_block(self.uid, "exit"))
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(self.agent.sandbox.executed_python_scripts, [])

    def test_direct_handle_exits_with_code_zero(self):
        """Unit-level pin: _handle_special_command itself raises SystemExit(0)."""
        with mock.patch("sys.exit", side_effect=sys.exit):
            with self.assertRaises(SystemExit) as cm:
                self.agent._handle_special_command("BASH", "exit")
        self.assertEqual(cm.exception.code, 0)


# ---------------------------------------------------------------------------
# T-07 — `reset` clears history but preserves system prompt
# ---------------------------------------------------------------------------

class TestResetCommand(SpecialCommandCase):
    def _seed_history(self):
        system_msg = {"role": "system", "content": "SYSTEM PROMPT"}
        user_msg = {"role": "user", "content": "please do things"}
        assistant_msg = {"role": "assistant", "content": "doing things"}
        self.agent.context.history = [system_msg, user_msg, assistant_msg]
        return system_msg

    def test_reset_direct_call_keeps_only_system_prompt(self):
        """After reset, exactly one message remains and it IS the original
        system prompt object (preserved, not re-created)."""
        system_msg = self._seed_history()
        handled, out = self.agent._handle_special_command("BASH", "reset")
        self.assertTrue(handled)
        self.assertEqual(len(self.agent.context.history), 1)
        self.assertIs(self.agent.context.history[0], system_msg)
        self.assertIn("EXIT_CODE_0", out)
        self.assertIn("Context history has been reset.", out)

    def test_reset_via_parse_and_execute_preserves_system_prompt(self):
        """Pipeline-level behavior: reset wipes everything except index 0.
        The confirmation OUTPUT block is then committed as the next user
        message by _commit_execution_feedback, so the post-turn history is
        [system, user-feedback] — length 2, not 1."""
        system_msg = self._seed_history()
        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "reset")
        )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(len(self.agent.context.history), 2)
        self.assertIs(self.agent.context.history[0], system_msg)
        self.assertEqual(self.agent.context.history[0]["role"], "system")
        self.assertIn("Context history has been reset.", self.last_message_text())

    def test_reset_empty_history_is_safe_no_op(self):
        """Reset on an empty history list must not raise IndexError; it stays
        a no-op returning EXIT_CODE_0."""
        self.agent.context.history = []
        try:
            handled, out = self.agent._handle_special_command("BASH", "reset")
        except IndexError:
            self.fail("reset on empty history raised IndexError")
        self.assertTrue(handled)
        self.assertIn("EXIT_CODE_0", out)
        self.assertEqual(self.agent.context.history, [])

    def test_reset_empty_history_via_pipeline(self):
        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "reset")
        )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertIn("Context history has been reset.", self.last_message_text())


# ---------------------------------------------------------------------------
# T-08 — `request-write` approval flows
# ---------------------------------------------------------------------------

class TestRequestWrite(SpecialCommandCase):
    def test_approval_appends_absolute_path_and_exit_zero(self):
        sb = self.attach_real_sandbox()
        with mock.patch("builtins.input", return_value="y"):
            handled, out = self.agent._handle_special_command(
                "BASH", "request-write /etc/hosts"
            )
        self.assertTrue(handled)
        self.assertIn("EXIT_CODE_0", out)
        self.assertIn("/etc/hosts", sb.approved_write_paths)
        self.assertIn("Write access granted.", out)

    def test_denial_returns_exit_code_one(self):
        sb = self.attach_real_sandbox()
        with mock.patch("builtins.input", return_value="n"):
            handled, out = self.agent._handle_special_command(
                "BASH", "request-write /etc/hosts"
            )
        self.assertTrue(handled)
        self.assertIn("EXIT_CODE_1", out)
        self.assertNotIn("/etc/hosts", sb.approved_write_paths)
        self.assertIn("denied", out.lower())

    def test_free_text_answer_echoed_back(self):
        """A free-text answer denies the request but the user's words are
        echoed back inside the OUTPUT block so the LLM can read them."""
        sb = self.attach_real_sandbox()
        answer = "use ~/notes instead of /var/log"
        with mock.patch("builtins.input", return_value=answer):
            handled, out = self.agent._handle_special_command(
                "BASH", "request-write /var/log"
            )
        self.assertTrue(handled)
        self.assertIn("EXIT_CODE_1", out)
        self.assertIn(answer, out)
        self.assertNotIn("/var/log", sb.approved_write_paths)

    def test_relative_path_resolved_against_cwd(self):
        sb = self.attach_real_sandbox()
        expected = os.path.abspath(os.path.join(self.tmpdir, "data/out.txt"))
        with mock.patch("builtins.input", return_value="y"):
            handled, out = self.agent._handle_special_command(
                "BASH", "request-write data/out.txt"
            )
        self.assertTrue(handled)
        self.assertIn(expected, sb.approved_write_paths)
        self.assertNotIn("data/out.txt", sb.approved_write_paths)  # stored absolute

    def test_end_to_end_commit_includes_grant_message(self):
        """Full pipeline: the grant confirmation lands in the committed user
        feedback message with a valid EXIT_CODE_0 marker."""
        sb = self.attach_real_sandbox()
        with mock.patch("builtins.input", return_value="y"):
            executed, feedback = self.agent.parse_and_execute(
                bash_block(self.uid, "request-write /etc/hosts")
            )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        text = self.last_message_text()
        self.assertIn("EXIT_CODE_0", text)
        self.assertIn("Write access granted.", text)
        self.assertIn("/etc/hosts", sb.approved_write_paths)


# ---------------------------------------------------------------------------
# T-09 — `ask-user` captures stdin answer
# ---------------------------------------------------------------------------

class TestAskUser(SpecialCommandCase):
    def test_question_printed_and_answer_wrapped_in_output_block(self):
        answer = "42"
        with mock.patch("builtins.input", return_value=answer):
            handled, out = self.agent._handle_special_command(
                "BASH", "ask-user What is the answer?"
            )
        printed = self.stdout_buf.getvalue()
        self.assertTrue(handled)
        self.assertIn("[Question to User]", printed)
        self.assertIn("What is the answer?", printed)
        # Properly fenced OUTPUT block carrying the answer
        self.assertTrue(out.startswith(
            f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---"
        ))
        self.assertTrue(out.endswith(f"---END_BASH_OUTPUT-{self.uid}---"))
        self.assertIn(answer, out)

    def test_eof_fallback_message(self):
        """When stdin is closed (CI-like environments), the EOF branch must
        substitute the fallback string instead of crashing."""
        with mock.patch("builtins.input", side_effect=EOFError):
            handled, out = self.agent._handle_special_command(
                "BASH", "ask-user Anyone there?"
            )
        self.assertTrue(handled)
        self.assertIn("[User provided no response / EOF]", out)
        self.assertIn("EXIT_CODE_0", out)

    def test_multi_word_question_preserved(self):
        question = "Should I delete build artifacts, cache dirs, AND logs?"
        with mock.patch("builtins.input", return_value="yes"):
            handled, out = self.agent._handle_special_command(
                "PYTHON", f"ask-user {question}"
            )
        self.assertTrue(handled)
        self.assertIn(question, self.stdout_buf.getvalue())
        self.assertIn("yes", out)

    def test_answer_committed_as_user_feedback(self):
        with mock.patch("builtins.input", return_value="blue"):
            executed, feedback = self.agent.parse_and_execute(
                bash_block(self.uid, "ask-user favorite color?")
            )
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertIn("blue", self.last_message_text())


# ---------------------------------------------------------------------------
# T-10 — `copy-to-clipboard` exits after copying
# ---------------------------------------------------------------------------

class TestCopyToClipboard(SpecialCommandCase):
    def test_files_passed_verbatim_then_exit_zero(self):
        recorder = mock.Mock(name="copy_recorder")
        events = []
        real_exit = sys.exit

        def fake_exit(code=0):
            events.append("exit")
            real_exit(code)

        file_list = "README.md, docs/api.md"
        with mock.patch(
            "bash_agent.agent.copy_project_to_clipboard",
            side_effect=lambda files: (events.append("copy"), recorder(files)),
        ), mock.patch("sys.exit", side_effect=fake_exit):
            with self.assertRaises(SystemExit) as cm:
                self.agent.parse_and_execute(
                    bash_block(self.uid, f"copy-to-clipboard {file_list}")
                )
        self.assertEqual(cm.exception.code, 0)
        recorder.assert_called_once_with(file_list)  # verbatim, unmodified
        self.assertEqual(events, ["copy", "exit"])   # copy BEFORE termination
        self.assertEqual(self.agent.sandbox.executed_scripts, [])

    def test_single_file_list(self):
        recorder = mock.Mock(name="copy_recorder")
        with mock.patch(
            "bash_agent.agent.copy_project_to_clipboard", recorder
        ), mock.patch("sys.exit", side_effect=sys.exit):
            with self.assertRaises(SystemExit):
                self.agent.parse_and_execute(
                    bash_block(self.uid, "copy-to-clipboard main.py")
                )
        recorder.assert_called_once_with("main.py")


# ---------------------------------------------------------------------------
# T-11 — Non-special scripts fall through to execution
# ---------------------------------------------------------------------------

class TestFallThrough(SpecialCommandCase):
    LOOKALIKES = [
        "echo hi",
        "request-writeup.sh",          # prefix lookalike of request-write
        "ask-user2 --batch",           # prefix lookalike of ask-user
        "copy-to-clipboard_backup.sh", # prefix lookalike of copy-to-clipboard
        "resets",                      # near-match of reset
        "EXIT",                        # case-sensitive exact match required
    ]

    def test_plain_script_not_handled_and_dispatched(self):
        fake_sb = FakeSandbox(execute_result=(0, "hi"))
        self.agent.sandbox = fake_sb
        handled, out = self.agent._handle_special_command("BASH", "echo hi")
        self.assertFalse(handled)
        self.assertEqual(out, "")
        executed, feedback = self.agent.parse_and_execute(bash_block(self.uid, "echo hi"))
        self.assertTrue(executed)
        self.assertEqual(fake_sb.executed_scripts[-1], "echo hi")

    def test_prefix_lookalikes_not_intercepted(self):
        """Guards against a future refactor switching ==/startswith semantics:
        none of these strings may trigger special-command handling; each must
        reach the sandbox verbatim."""
        for script in self.LOOKALIKES:
            with self.subTest(script=script):
                fake_sb = FakeSandbox(execute_result=(0, ""))
                self.agent.sandbox = fake_sb
                handled, _ = self.agent._handle_special_command("BASH", script)
                self.assertFalse(handled)
                self.agent.parse_and_execute(bash_block(self.uid, script))
                self.assertEqual(fake_sb.executed_scripts[-1], script)

    def test_exit_with_arguments_falls_through(self):
        """Only the bare string `exit` terminates; `exit 0` is an ordinary
        shell command and must be executed, not intercepted."""
        fake_sb = FakeSandbox(execute_result=(0, ""))
        self.agent.sandbox = fake_sb
        handled, _ = self.agent._handle_special_command("BASH", "exit 0")
        self.assertFalse(handled)
        self.agent.parse_and_execute(bash_block(self.uid, "exit 0"))
        self.assertEqual(fake_sb.executed_scripts[-1], "exit 0")

    def test_multiline_script_containing_exit_line_falls_through(self):
        """A multi-line script whose LAST line is `exit` does not match the
        exact-equality check and is executed as a whole."""
        script = "echo step-one\nexit"
        fake_sb = FakeSandbox(execute_result=(0, "step-one"))
        self.agent.sandbox = fake_sb
        handled, _ = self.agent._handle_special_command("BASH", script)
        self.assertFalse(handled)
        self.agent.parse_and_execute(bash_block(self.uid, script))
        self.assertEqual(fake_sb.executed_scripts[-1], script)


if __name__ == "__main__":
    unittest.main()
