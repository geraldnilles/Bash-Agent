"""
E2E — special (intercepted) commands driven through the REAL Agent loop.

Journey: a scripted fake LLM emits each built-in command block inside a full
`Agent.run()` session. The REAL dispatcher (`_handle_special_command`),
context bookkeeping, history persistence, and termination paths all run; only
the LLM and the sandbox subprocess are faked.

Commands covered:
  * request-write  -> sandbox approval/denial + OUTPUT block
  * ask-user       -> prompts on stdin, answer committed as user feedback
  * exit           -> SystemExit(0) from inside the loop
  * reset          -> history truncated to the system prompt
  * copy-to-clipboard -> copy_project_to_clipboard then SystemExit(0)

Replaces tests/unit/test_agent_special_commands.py.
"""
import unittest
from unittest import mock

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    FakeSandbox,
    cmd_start,
    out_start,
    out_end,
)

import re

UUID_RE = re.compile(r"current session UUID is: ([0-9a-fA-F-]+)")


def uid_of(messages):
    content = messages[0].get("content", "") if messages else ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    m = UUID_RE.search(content)
    return m.group(1) if m else "unknown"


class SpecialCmdCase(E2ECase):
    """E2ECase + non-orchestrator agent that runs the full loop."""

    def make_loop_agent(self, **kw):
        agent = self.make_agent(is_orchestrator=False, **kw)
        agent.sandbox = FakeSandbox(execute_result=(0, "ran-output"))
        # Skip warmup/scratchpad so history shape is minimal and the sandbox
        # records ONLY the scripted command turns (not the 2 warmup scripts).
        agent.resumed_session = True
        return agent

    def seed_dynamic(self, turns):
        """Seed a fake LLM whose Nth turn is `turns[N](uid, n)` (str content)."""
        counters = {"n": 0}

        def responder(**kwargs):
            uid = uid_of(kwargs["messages"])
            n = counters["n"]
            counters["n"] = n + 1
            if n < len(turns):
                return turns[n](uid, n)
            return make_fake_response(content=bash_block(uid, "exit"), cost=0.0)

        # FakeLLMClient POPS one response per call, so seed enough copies of
        # the dynamic responder to cover every turn (it dispatches on a
        # shared counter).
        self.seed_llm([responder] * 12)
        return counters


class TestRequestWrite(SpecialCmdCase):
    def test_approved_request_write_runs_and_records(self):
        agent = self.make_loop_agent()
        agent.sandbox = FakeSandbox()
        agent.sandbox.stub_request_write("/tmp/legit/out.txt", (True, "Write access granted."))

        def r(uid, n):
            return make_fake_response(
                content="requesting\n\n"
                + bash_block(uid, "request-write /tmp/legit/out.txt"),
                cost=0.0,
            )
        self.seed_dynamic([r, lambda uid, n: make_fake_response(
            content=bash_block(uid, "exit"), cost=0.0)])
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("needs write")
        self.assertEqual(cm.exception.code, 0)
        # The OUTPUT block for the request-write landed in history.
        text = self.messages_text(agent)
        self.assertIn(out_start("BASH", 0, agent.uuid), text)
        self.assertIn("Write access granted.", text)

    def test_denied_request_write_reports_exit_code_1(self):
        agent = self.make_loop_agent()
        agent.sandbox = FakeSandbox()
        agent.sandbox.stub_request_write("/etc/passwd", (False, "Write access denied."))

        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "request-write /etc/passwd"), cost=0.0),
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0),
        ])
        with runaway_guard(30):
            with self.assertRaises(SystemExit):
                agent.run("needs write")
        text = self.messages_text(agent)
        self.assertIn(out_start("BASH", 1, agent.uuid), text)
        self.assertIn("Write access denied.", text)


class TestAskUser(SpecialCmdCase):
    def test_question_and_answer_committed_as_feedback(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "ask-user What is the answer?"), cost=0.0),
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0),
        ])
        with mock.patch("builtins.input", return_value="42"):
            with runaway_guard(30):
                with self.assertRaises(SystemExit) as cm:
                    agent.run("ask me")
        self.assertEqual(cm.exception.code, 0)
        text = self.messages_text(agent)
        self.assertIn("[Question to User] What is the answer?", self.stdout_buf.getvalue())
        self.assertIn(out_start("BASH", 0, agent.uuid), text)
        self.assertIn("42", text)

    def test_eof_yields_fallback_answer(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "ask-user Anyone there?"), cost=0.0),
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0),
        ])
        with mock.patch("builtins.input", side_effect=EOFError):
            with runaway_guard(30):
                with self.assertRaises(SystemExit):
                    agent.run("ask me")
        self.assertIn("[User provided no response / EOF]", self.messages_text(agent))


class TestReset(SpecialCmdCase):
    def test_reset_keeps_only_system_prompt(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "reset"), cost=0.0),
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0),
        ])
        with runaway_guard(30):
            with self.assertRaises(SystemExit):
                agent.run("then reset")
        # After reset (turn 1) the history is only the system prompt; the exit
        # turn (turn 2) then appends nothing because `exit` raises immediately.
        h = self.history(agent)
        # Reset preserves ONLY the system prompt; everything before is gone.
        self.assertEqual(h[0]["role"], "system")
        self.assertNotIn("then reset", h[0]["content"])
        # The sole surviving user message is the reset command's own OUTPUT.
        user_msgs = [m for m in h if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertIn("Context history has been reset.", user_msgs[0]["content"])


class TestExit(SpecialCmdCase):
    def test_exit_terminates_with_zero_and_no_sandbox_exec(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content="bye\n\n" + bash_block(uid, "exit"), cost=0.0),
        ])
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("just exit")
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(agent.sandbox.executed_scripts, [])


class TestCopyToClipboard(SpecialCmdCase):
    def test_copy_then_exit_zero(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "copy-to-clipboard README.md,docs/a.md"),
                cost=0.0),
        ])
        recorder = mock.Mock(name="copy_recorder")
        with mock.patch(
            "bash_agent.agent.copy_project_to_clipboard", recorder
        ), runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("copy files")
        self.assertEqual(cm.exception.code, 0)
        # Verbatim file spec forwarded to the real copy helper.
        self.assertEqual(recorder.call_args.args[0], "README.md,docs/a.md")
        self.assertEqual(agent.sandbox.executed_scripts, [])

    def test_copy_with_ignore_flag_forwarded(self):
        agent = self.make_loop_agent()
        self.seed_dynamic([
            lambda uid, n: make_fake_response(
                content=bash_block(uid, "copy-to-clipboard --ignore *.log"), cost=0.0),
        ])
        recorder = mock.Mock(name="copy_recorder")
        with mock.patch(
            "bash_agent.agent.copy_project_to_clipboard", recorder
        ), runaway_guard(30):
            with self.assertRaises(SystemExit):
                agent.run("copy all minus logs")
        # files=None, ignore forwarded as a kwarg.
        self.assertIsNone(recorder.call_args.args[0])
        self.assertEqual(recorder.call_args.kwargs.get("ignore"), "*.log")


if __name__ == "__main__":
    unittest.main()
