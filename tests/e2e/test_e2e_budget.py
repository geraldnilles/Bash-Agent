"""
E2E — session budget accounting, exhaustion, and save-before-break.

Journey: a full `Agent.run()` loop where scripted LLM responses carry USD costs
that cumulatively cross the session budget. The REAL accounting
(`_record_response_usage` -> `session_cost`), the REAL budget gate
(`_handle_turn_budget_and_stats`), the REAL persistence (`save_history`) and
the REAL cost-breakdown printer all run.

Key invariants (regression guards):
  * the final turn's OUTPUT reaches DISK even though the loop then breaks,
    so a later `--resume` never loads a dangling assistant command fence,
  * a budget break prints the COST BREAKDOWN for a top-level agent but NOT for
    a subagent,
  * `--budget`/ctor budget sizes the session and `remaining_budget()` tracks it.

Replaces tests/unit/test_budget_persistence.py.
"""
import unittest

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    FakeSandbox,
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


class BudgetCase(E2ECase):
    def make_budget_agent(self, budget, is_subagent=False):
        agent = self.make_agent(
            is_orchestrator=False, budget=budget
        )
        agent.resumed_session = True  # skip warmup/scratchpad
        agent.is_subagent = is_subagent
        agent.sandbox = FakeSandbox(execute_result=(0, "SCRIPT-RAN-OK"))
        return agent

    def seed_echo_turns(self, n, cost_each):
        """Seed a fake LLM producing n turns, each running `echo i`.

        FakeLLMClient pops one queued item per call, so we enqueue many copies
        of a dynamic responder that dispatches on a shared turn counter.
        """
        state = {"n": 0}

        def responder(**kwargs):
            uid = uid_of(kwargs["messages"])
            i = state["n"]
            state["n"] = i + 1
            script = "echo %d" % i if i < n else "exit"
            return make_fake_response(
                content=bash_block(uid, script),
                cost=cost_each if i < n else 0.0,
                prompt_tokens=10, completion_tokens=5,
            )

        self.seed_llm([responder] * (n + 2))


class TestBudgetExhaustion(BudgetCase):
    def test_output_persisted_before_budget_break(self):
        agent = self.make_budget_agent(budget=0.10)
        self.seed_echo_turns(2, cost_each=0.06)  # 2nd turn crosses 0.10
        with runaway_guard(30):
            agent.run("budget path")
        self.assertGreaterEqual(agent.session_cost, 0.10)
        state = self.on_disk()
        self.assertEqual(state["history"], self.history(agent))
        # Final on-disk message is the OUTPUT of the last executed command.
        last = state["history"][-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-", last["content"])
        self.assertIn("SCRIPT-RAN-OK", last["content"])
        self.assertEqual(state["history"][-2]["role"], "assistant")

    def test_cost_breakdown_printed_on_exhaustion(self):
        agent = self.make_budget_agent(budget=0.10)
        self.seed_echo_turns(2, cost_each=0.06)
        with runaway_guard(30):
            agent.run("budget path")
        out = self.stdout_buf.getvalue()
        self.assertIn("COST BREAKDOWN", out)
        self.assertIn("TOTAL SESSION SPEND:", out)

    def test_no_breakdown_printed_for_subagent(self):
        agent = self.make_budget_agent(budget=0.10, is_subagent=True)
        self.seed_echo_turns(2, cost_each=0.06)
        with runaway_guard(30):
            agent.run("subagent budget path")
        out = self.stdout_buf.getvalue()
        self.assertNotIn("COST BREAKDOWN", out)

    def test_resume_sees_output_not_dangling_command(self):
        from bash_agent.context import ContextManager
        from unittest import mock
        agent = self.make_budget_agent(budget=0.10)
        self.seed_echo_turns(2, cost_each=0.06)
        with runaway_guard(30):
            agent.run("budget path")

        resumed = ContextManager("brand-new")
        with mock.patch("bash_agent.context.HISTORY_FILE", self.hist_file):
            self.assertTrue(resumed.load_history())
        self.assertEqual(resumed.uuid, agent.uuid)
        self.assertEqual(resumed.history[-1]["role"], "user")
        self.assertIn("SCRIPT-RAN-OK", resumed.history[-1]["content"])


class TestBudgetAccounting(BudgetCase):
    def test_remaining_budget_tracks_spend(self):
        agent = self.make_budget_agent(budget=0.50)
        self.assertEqual(agent.remaining_budget(), 0.50)
        agent.session_cost = 0.20
        self.assertAlmostEqual(agent.remaining_budget(), 0.30, places=9)
        agent.session_cost = 0.75  # over budget
        self.assertEqual(agent.remaining_budget(), 0.0)

    def test_unbudgeted_remaining_is_zero(self):
        agent = self.make_budget_agent(budget=None)
        # A standalone with no explicit budget gets DEFAULT_BUDGET; force none.
        agent.budget = None
        self.assertEqual(agent.remaining_budget(), 0.0)

    def test_default_budgets_by_role(self):
        # Constructor default budget differs by role.
        standalone = self.make_agent(is_orchestrator=False)
        self.assertAlmostEqual(standalone.budget, 0.10, places=9)
        orch = self.make_agent(is_orchestrator=True)
        self.assertAlmostEqual(orch.budget, 0.20, places=9)


if __name__ == "__main__":
    unittest.main()
