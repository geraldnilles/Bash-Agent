"""
Group 43 - Hierarchical Orchestrator / Subagent mode.

T-43a Sandbox isolation for is_orchestrator=True (P1)
T-43b Subagent `exit` raises SubagentExit (never sys.exit) (P0)
T-43c `subagent` command returns exactly the last 2 messages (P1)
T-43d Subagent cost is folded into the parent's session_cost (P1)
T-43e Nested subagents are refused (P1)
T-43f Orchestrator skips warmup exchanges (P1)

These tests pin the multi-agent contract:

  * When is_orchestrator=True the Sandbox masks the home tree via a tmpfs
    over /home (masking without binding .bash_agent_tmp/ back in would also
    hide the orchestrator's own working directory), binds .bash_agent_tmp/
    back in read-write, anchors its working directory inside
    .bash_agent_tmp/, restricts approved write paths
    to .bash_agent_tmp/, and rejects request-write() with a delegation hint.
  * A subagent calling `exit` raises SubagentExit (caught by the orchestrator)
    instead of terminating the whole process via sys.exit.
  * The orchestrator's `subagent` command runs a nested Agent and returns the
    last 2 messages of that Agent's conversation.
  * The nested Agent's cost is added to the orchestrator's session_cost.
  * A subagent may not spawn nested subagents.

Offline seams (per tests/AGENTS.md):
  * Agent construction patches the capability/reasoning/context probes on the
    Agent class so no network call happens; FakeSandbox replaces the real one.
  * subprocess.run is replaced inside bash_agent.sandbox for argv inspection.
"""
import contextlib
import io
import os
import subprocess
import unittest
from unittest import mock

from bash_agent.sandbox import Sandbox
from bash_agent.agent import Agent, SubagentExit

from tests.helpers.fakes import (
    chdir_tmp,
    bash_block,
    FakeSandbox,
    _make_agent,
    _stub_model_reasoning_info,
    _stub_model_context_info,
)


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


class SandboxIsolationTests(unittest.TestCase):
    """T-43a: orchestrator sandbox isolation."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    @staticmethod
    def _has_seq(cmd, seq):
        n = len(seq)
        return any(cmd[i:i + n] == seq for i in range(len(cmd) - n + 1))

    @staticmethod
    def _bind_pairs(cmd):
        pairs = []
        for i, tok in enumerate(cmd):
            if tok == "--bind":
                pairs.append((cmd[i + 1], cmd[i + 2]))
        return pairs

    def _run_execute(self, sandbox, script="true"):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _FakeCompleted()

        with mock.patch("bash_agent.sandbox.subprocess.run", new=fake_run):
            sandbox.execute(script)
        return calls[0]

    def test_orchestrator_sandbox_posture(self):
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        cmd = self._run_execute(sb)
        # The home tree is hidden with a tmpfs -- NOT by masking the whole
        # namespace (which would also hide .bash_agent_tmp, nested under
        # /home, and make every command die with status=200/CHDIR).
        self.assertTrue(self._has_seq(cmd, ["--tmpfs", "/home"]))
        # The one directory the orchestrator may use is bound back in writable.
        scratch = os.path.abspath(".bash_agent_tmp")
        self.assertEqual(self._bind_pairs(cmd), [(scratch, scratch)])
        # ...and it is the working directory too.
        self.assertTrue(self._has_seq(cmd, ["--chdir", scratch]))
        self.assertEqual(
            sb.approved_write_paths, [os.path.abspath(".bash_agent_tmp")]
        )

    def test_default_sandbox_is_not_orchestrator(self):
        sb = Sandbox("sp", uuid="u")
        cmd = self._run_execute(sb)
        # A subagent keeps /home visible (no tmpfs mask) and only punches
        # the project tree back in as writable.
        self.assertFalse(self._has_seq(cmd, ["--tmpfs", "/home"]))
        self.assertEqual(
            self._bind_pairs(cmd), [(os.path.abspath("."), os.path.abspath("."))]
        )
        self.assertEqual(sb.approved_write_paths, [os.path.abspath(".")])

    def test_orchestrator_request_write_rejected(self):
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        ok, msg = sb.request_write("/etc")
        self.assertFalse(ok)
        self.assertIn("subagent", msg.lower())
        self.assertEqual(
            sb.approved_write_paths, [os.path.abspath(".bash_agent_tmp")]
        )


class SubagentLifecycleTests(unittest.TestCase):
    """T-43b..f: lifecycle, output extraction, budget, nesting, warmup."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def test_subagent_exit_raises_subagent_exit(self):
        agent = _make_agent()
        agent.is_subagent = True
        with self.assertRaises(SubagentExit):
            agent._handle_special_command("BASH", "exit")

    def test_top_level_exit_still_calls_sys_exit(self):
        agent = _make_agent()
        agent.is_subagent = False
        with self.assertRaises(SystemExit) as cm:
            agent._handle_special_command("BASH", "exit")
        self.assertEqual(cm.exception.code, 0)

    def test_orchestrator_skips_warmup(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.context.history = [{"role": "system", "content": "s"}]
        agent._run_warmup_exchanges()
        self.assertEqual(len(agent.context.history), 1)

    def test_nested_subagent_refused(self):
        agent = _make_agent()
        agent.is_orchestrator = False
        handled, out = agent._handle_subagent_command("BASH", "subagent \"do it\"")
        self.assertTrue(handled)
        self.assertIn("nested", out.lower())

    def test_subagent_command_requires_prompt(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        handled, out = agent._handle_subagent_command("BASH", "subagent")
        self.assertTrue(handled)
        self.assertIn("requires a task prompt", out)

    def _run_subagent(self, agent, fake_history, fake_cost):
        """Helper: patch Agent so the spawned subagent is fully faked."""
        created = {}

        real_agent_cls = Agent

        class _FakeSubAgent:
            def __init__(self, **kwargs):
                self.uuid = "abcdef1234567890"
                self.session_cost = fake_cost
                self.kwargs = kwargs
                ctx = mock.Mock()
                ctx.history = fake_history
                self.context = ctx

            def run(self, initial_task=None):
                created["task"] = initial_task
                # Simulate the subagent issuing `exit`.
                raise SubagentExit()

        def _factory(*args, **kwargs):
            return _FakeSubAgent(**kwargs)

        with mock.patch("bash_agent.agent.Agent", side_effect=_factory):
            handled, out = agent._handle_subagent_command(
                "BASH", 'subagent "inspect files"'
            )
        return handled, out, created

    def test_returns_last_two_messages_and_cost(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.subagent_model = "some/model"
        agent.session_cost = 0.0
        agent.budget = 1.0
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "mid"},
            {"role": "assistant", "content": "final answer"},
            {"role": "user", "content": "output block"},
        ]
        handled, out, created = self._run_subagent(agent, history, 0.0123)
        self.assertTrue(handled)
        # Only the final 2 messages appear.
        self.assertIn("final answer", out)
        self.assertIn("output block", out)
        self.assertNotIn("mid", out)
        self.assertNotIn("sys", out)
        # Cost folded into the parent.
        self.assertAlmostEqual(agent.session_cost, 0.0123, places=9)
        self.assertEqual(created["task"], "inspect files")

    def test_subagent_gets_subagent_model_and_role_flags(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.subagent_model = "sub/model"
        agent.budget = 1.0
        captured = {}

        class _FakeSubAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.uuid = "abcdef1234567890"
                self.session_cost = 0.0
                m = mock.Mock()
                m.history = []
                self.context = m

            def run(self, initial_task=None):
                raise SubagentExit()

        with mock.patch("bash_agent.agent.Agent", side_effect=lambda **kw: _FakeSubAgent(**kw)):
            agent._handle_subagent_command("BASH", 'subagent "x"')
        self.assertEqual(captured.get("model"), "sub/model")
        self.assertTrue(captured.get("is_subagent"))
        self.assertFalse(captured.get("is_orchestrator"))


if __name__ == "__main__":
    unittest.main()
