"""
Group 4 (extension) — Budget-exhaustion persistence regression.

T-23  Executed output reaches disk before the budget break (P0)

Background / bug being pinned:

    Agent.run() persists history at several points, but historically did NOT
    persist after a turn's command had executed. The order was:

        add_message("assistant", <command>)   -> save_history()
        parse_and_execute(...)                -> commits OUTPUT to memory
        _handle_turn_budget_and_stats()       -> break if budget exceeded
        (no further save)

    So when the budget was exhausted on a turn, the assistant's command fence
    was on disk but its matching BASH/PYTHON_OUTPUT block was only in RAM.
    A subsequent `--resume` session then loaded a DANGLING command with no
    output — a protocol discontinuity (the model is asked to continue
    immediately after a command whose result it never sees), and the output
    was lost permanently.

Fix: run() now calls context.save_history() unconditionally right after
parse_and_execute() (covering both the executed-output path and the
parse-error feedback path) and BEFORE the budget check can `break`.

Contract pinned here:
  * when the budget is exhausted on a turn, the on-disk history.json is
    identical to the in-memory context (the executed OUTPUT included),
  * the last on-disk message is the user OUTPUT message for the last command,
  * a resumed Agent loading that file sees the OUTPUT block (no dangling
    command), i.e. memory and disk agree after reload.

All offline: FakeLLMClient (no network) + FakeSandbox (no real sandbox); the
protocol warmup is stubbed out so only the scripted turns run.
"""

import contextlib
import io
import json
import os
import unittest
import uuid as uuid_module
from unittest import mock

from bash_agent.agent import Agent
from bash_agent.context import ContextManager
from bash_agent import llm

from tests.helpers.fakes import (
    FakeLLMClient,
    FakeSandbox,
    make_fake_response,
    bash_block,
    chdir_tmp,
    _make_agent,
)


def new_uid():
    return str(uuid_module.uuid4())


class BudgetPersistenceCase(unittest.TestCase):
    """Shared harness: throwaway CWD, captured stdout, patched seams."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

        # context.py binds HISTORY_FILE at import time; patch where it looks.
        self.hist_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "history.json")
        self._patches = [
            mock.patch("bash_agent.context.HISTORY_FILE", self.hist_file),
            mock.patch.object(Agent, "_run_warmup_exchanges", lambda self: None),
        ]
        for p in self._patches:
            p.start()

        self._old_client = llm._CLIENT_CACHE.get("openrouter")

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        if self._old_client is None:
            llm._CLIENT_CACHE.pop("openrouter", None)
        else:
            llm._CLIENT_CACHE["openrouter"] = self._old_client
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def seed_llm(self, responses):
        fake = FakeLLMClient(responses=responses)
        fake.seed_cache("openrouter")
        return fake

    def load_disk(self):
        with open(self.hist_file, "r", encoding="utf-8") as f:
            return json.load(f)


class TestBudgetExhaustionPersistsOutput(BudgetPersistenceCase):

    def test_executed_output_saved_before_budget_break(self):
        """The final turn's OUTPUT must be on disk even though the loop broke."""
        # Two turns; the second pushes the running cost over the budget.
        # _make_agent forces a known uuid, so the command fences must carry
        # that same uuid for _extract_blocks() to match them.
        fixed = new_uid()
        responses = [
            make_fake_response(content=bash_block(fixed, "echo one"),
                               cost=0.05, prompt_tokens=10, completion_tokens=5),
            make_fake_response(content=bash_block(fixed, "echo two"),
                               cost=0.06, prompt_tokens=10, completion_tokens=5),
        ]
        self.seed_llm(responses)
        agent = _make_agent(uuid_str=fixed, budget=0.10, model="deepseek/deepseek-v3")
        agent.sandbox = FakeSandbox(execute_result=(0, "SCRIPT-RAN-OK"))
        with contextlib.suppress(SystemExit):
            agent.run("budget persistence test")

        # Budget really was exhausted after turn 2.
        self.assertGreaterEqual(agent.session_cost, 0.10)
        # Both scripted commands executed.
        self.assertEqual(agent.sandbox.executed_scripts, ["echo one", "echo two"])

        state = self.load_disk()
        mem = agent.context.history
        disk = state["history"]

        # THE REGRESSION: disk must match memory (no lost final OUTPUT).
        self.assertEqual(
            len(disk), len(mem),
            "on-disk history diverged from memory at budget break: "
            "disk=%d memory=%d" % (len(disk), len(mem)),
        )
        self.assertEqual(disk, mem)

        # The final on-disk message is the OUTPUT for the last command.
        last = disk[-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("_BASH_" + "OUTPUT-EXIT_CODE_0-VISIBLE_100%-", last["content"])
        self.assertIn("SCRIPT-RAN-OK", last["content"])
        self.assertIn("echo two", disk[-2]["content"])
        self.assertEqual(disk[-2]["role"], "assistant")

    def test_resume_sees_output_not_dangling_command(self):
        """A resumed Agent must see the OUTPUT block, not a dangling command."""
        fixed = new_uid()
        responses = [
            make_fake_response(content=bash_block(fixed, "echo one"),
                               cost=0.05, prompt_tokens=10, completion_tokens=5),
            make_fake_response(content=bash_block(fixed, "echo two"),
                               cost=0.06, prompt_tokens=10, completion_tokens=5),
        ]
        self.seed_llm(responses)
        agent = _make_agent(uuid_str=fixed, budget=0.10, model="deepseek/deepseek-v3")
        agent.sandbox = FakeSandbox(execute_result=(0, "SECOND-OUTPUT-OK"))
        with contextlib.suppress(SystemExit):
            agent.run("first session")

        # Reload from disk exactly like --resume does.
        resumed = ContextManager("brand-new-uuid")
        self.assertTrue(resumed.load_history())
        self.assertEqual(resumed.uuid, fixed)
        self.assertEqual(resumed.history, agent.context.history)

        # The last entry is the OUTPUT (not a bare assistant command).
        self.assertEqual(resumed.history[-1]["role"], "user")
        self.assertIn("SECOND-OUTPUT-OK", resumed.history[-1]["content"])
        # No dangling command as the final message.
        self.assertFalse(
            resumed.history[-1]["content"].strip().startswith(
                "---START_" + "BASH_COMMAND-"
            )
        )

    def test_parse_error_feedback_also_persisted(self):
        """The parse-error feedback path must also reach disk."""
        fixed = new_uid()
        # A response with NO valid fenced block -> executed=False + feedback.
        responses = [
            make_fake_response(content="I will think about it.",
                               cost=0.05, prompt_tokens=10, completion_tokens=5),
            make_fake_response(content="still thinking",
                               cost=0.06, prompt_tokens=10, completion_tokens=5),
        ]
        self.seed_llm(responses)
        agent = _make_agent(uuid_str=fixed, budget=0.10, model="deepseek/deepseek-v3")
        agent.sandbox = FakeSandbox(execute_result=(0, "unused"))
        with contextlib.suppress(SystemExit):
            agent.run("parse error path")

        state = self.load_disk()
        self.assertEqual(state["history"], agent.context.history)
        # Feedback (a user message) is the last entry.
        self.assertEqual(state["history"][-1]["role"], "user")


    def test_cost_breakdown_printed_on_budget_exhaustion(self):
        """The budget-exhaustion break must print the cost breakdown."""
        fixed = new_uid()
        responses = [
            make_fake_response(content=bash_block(fixed, "echo one"),
                               cost=0.05, prompt_tokens=10, completion_tokens=5),
            make_fake_response(content=bash_block(fixed, "echo two"),
                               cost=0.06, prompt_tokens=10, completion_tokens=5),
        ]
        self.seed_llm(responses)
        agent = _make_agent(uuid_str=fixed, budget=0.10, model="deepseek/deepseek-v3")
        agent.sandbox = FakeSandbox(execute_result=(0, "SCRIPT-RAN-OK"))
        with contextlib.suppress(SystemExit):
            agent.run("budget breakdown path")

        # Budget really was exhausted -> the loop broke.
        self.assertGreaterEqual(agent.session_cost, 0.10)
        out = self.stdout_buf.getvalue()
        self.assertIn("COST BREAKDOWN", out)
        self.assertIn("TOTAL SESSION SPEND:", out)

    def test_cost_breakdown_not_printed_for_subagent_budget_break(self):
        """A subagent hitting its budget must NOT print the breakdown."""
        fixed = new_uid()
        responses = [
            make_fake_response(content=bash_block(fixed, "echo one"),
                               cost=0.05, prompt_tokens=10, completion_tokens=5),
            make_fake_response(content=bash_block(fixed, "echo two"),
                               cost=0.06, prompt_tokens=10, completion_tokens=5),
        ]
        self.seed_llm(responses)
        agent = _make_agent(uuid_str=fixed, budget=0.10, model="deepseek/deepseek-v3")
        agent.is_subagent = True
        agent.sandbox = FakeSandbox(execute_result=(0, "SCRIPT-RAN-OK"))
        with contextlib.suppress(SystemExit):
            agent.run("subagent budget path")

        self.assertGreaterEqual(agent.session_cost, 0.10)
        self.assertNotIn("COST BREAKDOWN", self.stdout_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
