"""
Group 3 — Execution Pipeline tests for parse_and_execute / _execute_script.

T-12  Full happy-path turn (P0)

These tests pin the smallest end-to-end proof that
parse -> dispatch -> format -> commit works as a unit:

  * A protocol-valid LLM response containing one fenced block is parsed,
  * dispatched to the right sandbox method (execute vs execute_python),
  * the raw result is wrapped in a protocol-valid OUTPUT fence carrying the
    exit code and VISIBLE_% header, and
  * exactly one user message is committed to the context containing that
    fence (plus whatever scratchpad co-commit logic adds).

All tests are offline:
  * Agent construction goes through helpers.fakes._make_agent (no network
    probe, FakeSandbox injected).
  * The sandbox is a FakeSandbox returning canned (exit_code, output) tuples;
    real-sandbox behavior is covered separately by Group 7 integration tests.

Every test runs inside a throwaway CWD (chdir_tmp) and captures stdout so the
production prints ([System] ..., colored output blocks) stay out of test runs.
"""
import contextlib
import io
import unittest
import uuid

from tests.helpers.fakes import (
    bash_block,
    python_block,
    output_block,
    chdir_tmp,
    _make_agent,
    FakeSandbox,
)


class PipelineCase(unittest.TestCase):
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

    def user_messages(self):
        """All committed user-role messages, in order."""
        return [m for m in self.agent.context.history if m["role"] == "user"]


# ---------------------------------------------------------------------------
# T-12 — Full happy-path turn
# ---------------------------------------------------------------------------

class TestFullHappyPathTurn(PipelineCase):
    def test_single_bash_block_round_trip(self):
        """One valid bash block -> sandbox executes it verbatim -> the commit
        carries an EXIT_CODE_0 OUTPUT fence and the call returns (True, "")."""
        fake_sb = FakeSandbox(execute_result=(0, "hello"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "echo hello")
        )

        # Return contract
        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        # Dispatch: the sandbox saw the script exactly as fenced
        self.assertEqual(fake_sb.executed_scripts, ["echo hello"])
        self.assertEqual(fake_sb.executed_python_scripts, [])

        # Commit: exactly ONE user message appended (system prompt is index 0)
        users = self.user_messages()
        self.assertEqual(len(users), 1)

        content = users[0]["content"]
        # The formatted output block must appear verbatim, exit code intact
        self.assertIn(output_block(self.uid, 0, "hello"), content)
        self.assertIn("EXIT_CODE_0", content)
        self.assertIn("VISIBLE_100%", content)
        # ...and the closing fence must match the session UUID
        self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", content)

    def test_single_python_block_dispatches_to_execute_python(self):
        """Same happy path, PYTHON flavor: proves cmd_type drives the choice
        between sandbox.execute and sandbox.execute_python."""
        fake_sb = FakeSandbox(execute_python_result=(0, "42"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            python_block(self.uid, "print(42)")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(fake_sb.executed_python_scripts, ["print(42)"])
        self.assertEqual(fake_sb.executed_scripts, [])

        users = self.user_messages()
        self.assertEqual(len(users), 1)
        self.assertIn(
            output_block(self.uid, 0, "42", cmd_type="PYTHON"),
            users[0]["content"],
        )
        self.assertIn("EXIT_CODE_0", users[0]["content"])

    def test_nonzero_exit_code_is_preserved_in_fence(self):
        """A failing command still commits feedback — the model needs the
        EXIT_CODE_N header to react — and returns (True, "")."""
        fake_sb = FakeSandbox(execute_result=(2, "ls: cannot access 'x'"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "ls x")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        users = self.user_messages()
        self.assertEqual(len(users), 1)
        self.assertIn(
            output_block(self.uid, 2, "ls: cannot access 'x'"),
            users[0]["content"],
        )
        self.assertIn("EXIT_CODE_2", users[0]["content"])

    def test_empty_output_still_commits_a_fenced_block(self):
        """Silent success (`touch file`) must still produce a well-formed
        empty OUTPUT fence so the model knows the command succeeded."""
        fake_sb = FakeSandbox(execute_result=(0, ""))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "touch file.txt")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        content = self.user_messages()[0]["content"]
        self.assertIn(
            f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---",
            content,
        )
        self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", content)


if __name__ == "__main__":
    unittest.main()
