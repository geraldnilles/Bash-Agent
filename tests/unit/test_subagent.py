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
        # T-43d: the run is retained in the per-agent ledger so the end-of-
        # session breakdown can report each subagent's individual spend.
        self.assertEqual(len(agent.subagent_runs), 1)
        run = agent.subagent_runs[0]
        self.assertEqual(run["uuid"], "abcdef1234567890")
        self.assertEqual(run["task"], "inspect files")
        self.assertAlmostEqual(run["cost"], 0.0123, places=9)

    def test_print_cost_breakdown_with_subagent(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.subagent_model = "some/model"
        agent.budget = 1.0
        agent.session_cost = 0.0
        history = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "final answer"},
            {"role": "user", "content": "output block"},
        ]
        self._run_subagent(agent, history, 0.0123)
        # Simulate the orchestrator's own turn spend.
        agent._orchestrator_cost = 0.05
        agent.session_cost = 0.05 + 0.0123

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent._print_cost_breakdown()
        out = buf.getvalue()
        self.assertIn("COST BREAKDOWN", out)
        self.assertIn("Orchestrator (own turns):   $0.0500", out)
        self.assertIn("Subagents: 1 spawned", out)
        self.assertIn("abcdef12", out)
        self.assertIn("spent $0.0123", out)
        self.assertIn("task=\"inspect files\"", out)
        self.assertIn("TOTAL SESSION SPEND:        $0.0623", out)

    def test_print_cost_breakdown_no_subagents(self):
        agent = _make_agent()
        agent.session_cost = 0.0031
        agent._orchestrator_cost = 0.0031

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent._print_cost_breakdown()
        out = buf.getvalue()
        self.assertIn("COST BREAKDOWN", out)
        self.assertIn("No subagents spawned.", out)
        self.assertIn("Orchestrator (own turns):   $0.0031", out)

    def _spawn_capture(self, agent, script):
        """Spawn a fully-faked subagent and return the kwargs it received."""
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

        with mock.patch("bash_agent.agent.Agent",
                        side_effect=lambda **kw: _FakeSubAgent(**kw)):
            handled, out = agent._handle_subagent_command("BASH", script)
        return handled, out, captured

    def test_default_subagent_budget_is_25_percent_of_parent(self):
        # No override -> default subagent budget = 25% of the OVERALL budget.
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.0
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(agent, 'subagent "x"')
        self.assertAlmostEqual(captured["budget"], 0.25, places=9)

    def test_per_command_budget_override_applied(self):
        # `--budget 0.05` overrides the 25% default.
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.0
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(
            agent, 'subagent --budget 0.05 "run tests"')
        self.assertAlmostEqual(captured["budget"], 0.05, places=9)

    def test_short_b_flag_alias_applies(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.0
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(
            agent, 'subagent -b 0.07 "run tests"')
        self.assertAlmostEqual(captured["budget"], 0.07, places=9)

    def test_global_subagent_budget_beats_25pct_default(self):
        # --subagent-budget (self.subagent_budget) beats the 25% default but
        # loses to a per-command override.
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.0
        agent.subagent_budget = 0.10
        _, _, captured = self._spawn_capture(agent, 'subagent "x"')
        self.assertAlmostEqual(captured["budget"], 0.10, places=9)

    def test_invalid_budget_value_is_error(self):
        for bad in ["abc", "0", "-1"]:
            agent = _make_agent()
            agent.is_orchestrator = True
            agent.budget = 1.0
            agent.session_cost = 0.0
            handled, out = agent._handle_subagent_command(
                "BASH", f'subagent --budget {bad} "x"')
            self.assertTrue(handled)
            self.assertIn("--budget", out)

    def test_missing_budget_value_is_error(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.0
        handled, out = agent._handle_subagent_command(
            "BASH", "subagent --budget")
        self.assertTrue(handled)
        self.assertIn("--budget", out)

    def test_budget_clamped_to_remaining(self):
        # Requesting more than remains clamps to the remaining parent budget.
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.90
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(
            agent, 'subagent --budget 0.50 "x"')
        self.assertAlmostEqual(captured["budget"], 0.10, places=9)

    def test_budget_floored_at_minimum(self):
        # A nearly-exhausted parent still grants the 0.01 floor.
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 1.0
        agent.session_cost = 0.999
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(agent, 'subagent "x"')
        self.assertAlmostEqual(captured["budget"], 0.01, places=9)

    def test_no_parent_budget_yields_unbounded_subagent(self):
        agent = _make_agent()
        agent.is_orchestrator = True
        agent.budget = 0.0
        agent.session_cost = 0.0
        agent.subagent_budget = None
        _, _, captured = self._spawn_capture(agent, 'subagent "x"')
        self.assertIsNone(captured["budget"])

    def test_per_command_budget_honored_when_parent_unbudgeted(self):
        # An EXPLICIT per-command --budget must be honored even when the parent
        # itself has no budget (None or 0); with no finite parent there is
        # nothing to clamp against, so the requested value is used as-is.
        for parent_budget in (None, 0.0):
            with self.subTest(parent_budget=parent_budget):
                agent = _make_agent()
                agent.is_orchestrator = True
                agent.budget = parent_budget
                agent.session_cost = 0.0
                agent.subagent_budget = None
                _, _, captured = self._spawn_capture(
                    agent, 'subagent --budget 0.05 "run tests"')
                self.assertAlmostEqual(captured["budget"], 0.05, places=9)

    def test_global_subagent_budget_honored_when_parent_unbudgeted(self):
        # Same precedence for the global --subagent-budget when no per-command
        # override is given: an unbudgeted parent adopts the requested value.
        for parent_budget in (None, 0.0):
            with self.subTest(parent_budget=parent_budget):
                agent = _make_agent()
                agent.is_orchestrator = True
                agent.budget = parent_budget
                agent.session_cost = 0.0
                agent.subagent_budget = 0.08
                _, _, captured = self._spawn_capture(agent, 'subagent "x"')
                self.assertAlmostEqual(captured["budget"], 0.08, places=9)

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


class AgentBudgetDefaultTests(unittest.TestCase):
    """Default budget depends on the agent's role at construction time.

    An orchestrator must fund its own turns AND every subagent it delegates to,
    so it defaults to the larger DEFAULT_ORCHESTRATOR_BUDGET (0.20); a
    standalone agent (or a subagent, which normally gets an explicit budget
    from its parent) falls back to DEFAULT_BUDGET (0.10).
    """

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()

    def tearDown(self):
        self._chdir_cm.__exit__(None, None, None)

    def test_orchestrator_default_budget(self):
        from bash_agent.config import DEFAULT_ORCHESTRATOR_BUDGET
        agent = _make_agent(is_orchestrator=True)
        self.assertEqual(agent.budget, DEFAULT_ORCHESTRATOR_BUDGET)
        self.assertEqual(agent.budget, 0.20)

    def test_standalone_default_budget(self):
        from bash_agent.config import DEFAULT_BUDGET
        agent = _make_agent(is_orchestrator=False)
        self.assertEqual(agent.budget, DEFAULT_BUDGET)
        self.assertEqual(agent.budget, 0.10)

    def test_subagent_default_budget_falls_back_to_standalone(self):
        from bash_agent.config import DEFAULT_BUDGET
        agent = _make_agent(is_orchestrator=False, is_subagent=True)
        self.assertEqual(agent.budget, DEFAULT_BUDGET)

    def test_explicit_budget_wins_for_orchestrator(self):
        agent = _make_agent(is_orchestrator=True, budget=1.5)
        self.assertEqual(agent.budget, 1.5)

    def test_remaining_budget_helper(self):
        agent = _make_agent(is_orchestrator=False, budget=1.0)
        agent.session_cost = 0.30
        self.assertAlmostEqual(agent.remaining_budget(), 0.70, places=9)
        agent.session_cost = 2.0
        self.assertEqual(agent.remaining_budget(), 0.0)  # never negative

    def test_remaining_budget_zero_when_unbudgeted(self):
        agent = _make_agent(is_orchestrator=False, budget=0.0)
        self.assertEqual(agent.remaining_budget(), 0.0)


if __name__ == "__main__":
    unittest.main()
