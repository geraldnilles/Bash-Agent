"""
Group 9 — LLM finish-reason handling in ``Agent._get_llm_response``.

T-43  finish_reason == "tool_calls": correction stays in history (P0)
T-43a finish_reason == "length": temporary recovery still pops (P0, regression)

Behavior pinned by these tests:

* When the model returns ``finish_reason == "tool_calls"``, the agent appends
  a ``[Invalid tool_calls attempt]`` assistant message plus a ``[SYSTEM WARNING]``
  user message showing the desired BASH/PYTHON syntax, then simply re-runs the
  LLM with that corrected history.  The correction is permanently KEPT in the
  conversation (unlike the ``length`` recovery, which pops its temporary
  messages) so the model can see what went wrong on the next call.

* The ``finish_reason == "length"`` thinking-token recovery is unchanged: it
  injects temporary ``<thinking>`` + warning messages, fetches a follow-up,
  and pops both temporary messages before returning.

All tests are offline:
  * Agent construction goes through helpers.fakes._make_agent (no network
    probe, FakeSandbox injected).
  * The LLM boundary is faked via llm._CLIENT_CACHE seeding (FakeLLMClient).
  * Every test runs inside a throwaway CWD (chdir_tmp) and captures stdout.
"""
import contextlib
import io
import unittest
from unittest import mock

from bash_agent import llm

from tests.helpers.fakes import (
    FakeLLMClient,
    _make_agent,
    bash_block,
    chdir_tmp,
    make_fake_response,
)


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------

class LlmResponseCase(unittest.TestCase):
    """Shared harness: temp CWD + captured stdout + ready-made agent."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()

        self._stdout_cm = contextlib.redirect_stdout(io.StringIO())
        self._stdout_cm.__enter__()

        self.agent = _make_agent()
        self.uid = self.agent.uuid

        # Seed the module-level client cache; every test restores prior state.
        self._old_client = llm._CLIENT_CACHE.get("openrouter")

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)
        if self._old_client is None:
            llm._CLIENT_CACHE.pop("openrouter", None)
        else:
            llm._CLIENT_CACHE["openrouter"] = self._old_client

    def seed_llm(self, responses):
        """Seed the module-level client cache with a scripted fake."""
        fake = FakeLLMClient(responses=responses)
        llm._CLIENT_CACHE["openrouter"] = fake
        return fake

    def fence_warning(self):
        """Rebuild the expected SYSTEM WARNING user message piecewise so no
        contiguous protocol marker literal appears in this source module."""
        d = "-" * 3
        start_bash = d + "START_" + "BASH_COMMAND-" + self.uid + d
        end_bash = d + "END_" + "BASH_COMMAND-" + self.uid + d
        start_py = d + "START_" + "PYTHON_COMMAND-" + self.uid + d
        end_py = d + "END_" + "PYTHON_COMMAND-" + self.uid + d
        return (
            "⚠️ [SYSTEM WARNING] Invalid finish_reason: tool_calls. The model attempted to use native tool_calls, "
            "which is not supported in this environment. Please use BASH and PYTHON code blocks instead.\n"
            f"{start_bash}\n[bash commands go here]\n{end_bash}\n"
            f"or\n"
            f"{start_py}\n[python code goes here]\n{end_py}"
        )


# ---------------------------------------------------------------------------
# T-43 — tool_calls: correction kept in history, loop continues
# ---------------------------------------------------------------------------

class TestToolCallsCorrectionStaysInHistory(LlmResponseCase):
    def test_correction_persists_and_followup_returned(self):
        """Respond 1: finish_reason=tool_calls. Respond 2: valid text + fence.
        The correction pair must remain in history and the second (model's
        follow-up) response is returned to the caller."""
        followup = "Corrected response:\n\n" + bash_block(self.uid, "echo ok")
        fake = self.seed_llm([
            make_fake_response(content="", finish_reason="tool_calls"),
            make_fake_response(content=followup, finish_reason="stop"),
        ])

        agent_msg = self.agent._get_llm_response()

        # Follow-up content returned to the caller
        self.assertEqual(agent_msg, followup)

        # Exactly two LLM calls: the failed attempt + the follow-up. There is
        # NO hidden third recovery call (the old implementation issued one).
        self.assertEqual(len(fake.calls), 2)

        # Correction stays in history: [system, assistant correction, user warning]
        h = self.agent.context.history
        self.assertEqual(
            [m["role"] for m in h],
            ["system", "assistant", "user"],
        )
        self.assertEqual(h[1]["content"], "[Invalid tool_calls attempt]")
        self.assertIn("SYSTEM WARNING", h[2]["content"])
        self.assertIn("finish_reason: tool_calls", h[2]["content"])
        self.assertEqual(h[2]["content"], self.fence_warning())

    def test_tool_info_included_when_message_carries_tool_calls(self):
        """If the message object exposes tool_calls, the assistant correction
        records them so the history tells the model exactly what it tried."""
        msg = type("Msg", (), {"content": "", "tool_calls": [{"id": "tc1"}], "model_extra": None})()
        choice = type("Choice", (), {"message": msg, "finish_reason": "tool_calls"})()
        resp = type(
            "Resp",
            (),
            {
                "choices": [choice],
                "usage": type("U", (), {"cost": 0.0, "prompt_tokens": 5, "completion_tokens": 5})(),
                "provider": None,
                "model_dump": lambda self=None: {
                    "choices": [{"message": {"content": ""}, "finish_reason": "tool_calls"}],
                    "usage": {"cost": 0.0, "prompt_tokens": 5, "completion_tokens": 5},
                },
                "model_dump_json": lambda indent=None, self=None: '{}',
            },
        )()
        self.seed_llm([
            resp,
            make_fake_response(content="ok now", finish_reason="stop"),
        ])

        agent_msg = self.agent._get_llm_response()
        self.assertEqual(agent_msg, "ok now")
        h = self.agent.context.history
        self.assertIn("Attempted tool_calls", h[1]["content"])
        self.assertIn("tc1", h[1]["content"])

    def test_no_thinking_recovery_artifacts(self):
        """The tool_calls path must never inject <thinking> blocks — those are
        exclusive to the finish_reason == 'length' recovery."""
        fake = self.seed_llm([
            make_fake_response(content="", finish_reason="tool_calls"),
            make_fake_response(content="done", finish_reason="stop"),
        ])
        self.agent._get_llm_response()
        history_text = "\n".join(m["content"] for m in self.agent.context.history)
        self.assertNotIn("<thinking>", history_text)
        self.assertEqual(len(fake.calls), 2)


# ---------------------------------------------------------------------------
# T-43a — regression: length recovery still pops its temporary messages
# ---------------------------------------------------------------------------

class TestLengthRecoveryStillPops(LlmResponseCase):
    def test_temporary_thoughts_removed_after_followup(self):
        """finish_reason='length' + reasoning text triggers the thinking
        recovery. The temporary assistant/user pair must be popped so the
        returned history is identical to the pre-call state."""
        self.seed_llm([
            make_fake_response(
                content="",
                finish_reason="length",
                reasoning="partial chain-of-thought",
            ),
            make_fake_response(content="immediate answer", finish_reason="stop"),
        ])

        # Baseline: agent constructed with just the system prompt
        h_before = list(self.agent.context.history)
        agent_msg = self.agent._get_llm_response()

        self.assertEqual(agent_msg, "immediate answer")
        # History unchanged: temporary thinking + warning were popped
        self.assertEqual(self.agent.context.history, h_before)
        history_text = "\n".join(m["content"] for m in self.agent.context.history)
        self.assertNotIn("<thinking>", history_text)
        self.assertNotIn("SYSTEM WARNING", history_text)


if __name__ == "__main__":
    unittest.main()
