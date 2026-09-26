"""
E2E — finish_reason recovery inside a full Agent loop.

Journey: scripted LLM responses use non-`stop` finish reasons that the REAL
`_get_llm_response()` must recover from mid-session:

  * finish_reason="tool_calls" -> the agent injects a permanent correction
    pair (assistant marker + user warning) and asks again; the model's
    follow-up valid response is what the loop executes.
  * finish_reason="length" with reasoning text -> the agent ingests the
    partial chain-of-thought, issues ONE immediate recovery request with the
    lowest reasoning effort, and returns that answer. The temporary
    <thinking>/warning pair MUST be popped so history is not polluted.

Replaces tests/unit/test_agent_llm_response.py.
"""
import unittest

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    FakeSandbox,
    out_start,
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


class RecoveryCase(E2ECase):
    def make_loop_agent(self):
        agent = self.make_agent(is_orchestrator=False)
        agent.resumed_session = True  # skip warmup/scratchpad
        agent.sandbox = FakeSandbox(execute_result=(0, "RECOVERED-RAN"))
        return agent


class TestToolCallsRecovery(RecoveryCase):
    def test_tool_calls_corrected_then_executed(self):
        agent = self.make_loop_agent()
        state = {"n": 0}

        def responder(**kwargs):
            uid = uid_of(kwargs["messages"])
            n = state["n"]
            state["n"] = n + 1
            if n == 0:
                return make_fake_response(
                    content="", finish_reason="tool_calls", cost=0.001)
            if n == 1:
                # Model's corrected follow-up: a real bash block.
                return make_fake_response(
                    content=bash_block(uid, "echo recovered"), cost=0.001)
            return make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0)

        fake = self.seed_llm([responder] * 6)
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("recover from tool_calls")
        self.assertEqual(cm.exception.code, 0)
        # The correction pair stays in history permanently.
        text = self.messages_text(agent)
        self.assertIn("[Invalid tool_calls attempt]", text)
        self.assertIn("finish_reason: tool_calls", text)
        # The corrected command really executed.
        self.assertIn(out_start("BASH", 0, agent.uuid), text)
        self.assertIn("RECOVERED-RAN", text)
        # No <thinking> artifacts from the tool_calls path.
        self.assertNotIn("<thinking>", text)


class TestLengthRecovery(RecoveryCase):
    def test_length_recovery_pops_temporary_thoughts(self):
        agent = self.make_loop_agent()
        state = {"n": 0}
        baseline_len = {"v": None}

        def responder(**kwargs):
            uid = uid_of(kwargs["messages"])
            n = state["n"]
            state["n"] = n + 1
            if n == 0:
                return make_fake_response(
                    content="", finish_reason="length",
                    reasoning="partial chain-of-thought", cost=0.001)
            if n == 1:
                # Recovery answer.
                return make_fake_response(
                    content="done thinking\n\n" + bash_block(uid, "echo post-length"),
                    cost=0.001)
            return make_fake_response(content=bash_block(uid, "exit"), cost=0.0)

        self.seed_llm([responder] * 6)
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("recover from length")
        self.assertEqual(cm.exception.code, 0)
        text = self.messages_text(agent)
        # Temporary recovery artifacts must NOT survive.
        self.assertNotIn("<thinking>", text)
        self.assertNotIn("You exceeded your max token budget", text)
        # The post-recovery command executed.
        self.assertIn(out_start("BASH", 0, agent.uuid), text)
        self.assertIn("RECOVERED-RAN", text)


if __name__ == "__main__":
    unittest.main()
