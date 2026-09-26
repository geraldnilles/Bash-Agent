"""
E2E — `--resume` flow: restore history from disk and re-bind the UUID.

Journey: session A runs a full loop and persists history.json; session B is
constructed with `resume=True`, loads the SAME file, re-binds its UUID from
disk, skips warmup/scratchpad injection, and continues the conversation. Every
follow-up OUTPUT fence must carry the RESTORED uuid, proving the parser,
formatter, and sandbox uuid all re-bound coherently.

Replaces tests/unit/test_context_persistence.py.
"""
import json
import os
import unittest
from unittest import mock

from bash_agent.context import ContextManager

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


class ResumeCase(E2ECase):
    def exit_responder(self, note=""):
        def responder(**kwargs):
            uid = uid_of(kwargs["messages"])
            content = (note + "\n\n" if note else "") + bash_block(uid, "exit")
            return make_fake_response(content=content, cost=0.001)
        return responder


class TestResumeFlow(ResumeCase):
    def test_resume_rebinds_uuid_and_continues(self):
        # --- Session A -----------------------------------------------------
        agent_a = self.make_agent(is_orchestrator=False)
        agent_a.resumed_session = True  # keep history minimal (no warmup)
        agent_a.sandbox = FakeSandbox(execute_result=(0, "A-RAN"))
        self.seed_llm([self.exit_responder("bye A")])
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm_a:
                agent_a.run("session A task")
        self.assertEqual(cm_a.exception.code, 0)
        uuid_a = agent_a.uuid
        len_a = len(self.history(agent_a))
        self.assertTrue(os.path.exists(self.hist_file))

        # --- Session B: resume from disk ----------------------------------
        agent_b = self.make_agent(is_orchestrator=False, resume=True)
        self.assertTrue(agent_b.resumed_session)
        self.assertEqual(agent_b.uuid, uuid_a, "UUID must be re-bound from disk")
        self.assertEqual(len(agent_b.context.history), len_a)

        agent_b.sandbox = FakeSandbox(execute_result=(0, "B-RAN"))
        self.seed_llm([self.exit_responder()])
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm_b:
                agent_b.run("session B follow-up")
        self.assertEqual(cm_b.exception.code, 0)

        # Follow-up task appended after prior history.
        self.assertGreater(len(self.history(agent_b)), len_a)
        followup_task = agent_b.context.history[len_a]
        self.assertEqual(followup_task["role"], "user")
        self.assertIn("session B follow-up", followup_task["content"])
        # No scratchpad injection on resume (it would be a NEW user message).
        self.assertNotIn("SCRATCHPAD.md-VISIBLE", followup_task["content"])

    def test_resume_does_not_run_warmup(self):
        # Write a hand-crafted 3-message history to disk.
        stored_uuid = "11111111-2222-3333-4444-555555555555"
        state = {
            "uuid": stored_uuid,
            # The system prompt must carry the session UUID so the fake LLM
            # (and the real parser) can build/validate correctly-fenced blocks.
            "history": [
                {"role": "system",
                 "content": "current session UUID is: %s" % stored_uuid},
                {"role": "user", "content": "old task"},
                {"role": "assistant", "content": "old reply"},
            ],
        }
        os.makedirs(os.path.dirname(self.hist_file), exist_ok=True)
        with open(self.hist_file, "w") as f:
            json.dump(state, f)

        agent = self.make_agent(is_orchestrator=False, resume=True)
        self.assertEqual(agent.uuid, stored_uuid)
        self.assertEqual(len(agent.context.history), 3)
        agent.sandbox = FakeSandbox(execute_result=(0, "X"))
        self.seed_llm([self.exit_responder()])
        with runaway_guard(30):
            with self.assertRaises(SystemExit):
                agent.run("new task")
        # Exactly one appended user task (no warmup pairs, no scratchpad).
        h = self.history(agent)
        # old 3 + task + assistant(exit) = 5
        self.assertEqual(len(h), 5)
        self.assertEqual(h[3]["role"], "user")
        self.assertIn("new task", h[3]["content"])

    def test_context_manager_load_persist_roundtrip(self):
        cm = ContextManager("orig-uuid")
        cm.add_message("system", "sys")
        cm.add_message("user", "hello")
        with mock.patch("bash_agent.context.HISTORY_FILE", self.hist_file):
            cm.save_history()
            self.assertTrue(os.path.exists(self.hist_file))
            fresh = ContextManager("different-uuid")
            self.assertTrue(fresh.load_history())
            self.assertEqual(fresh.uuid, "orig-uuid")
            self.assertEqual(fresh.history, cm.history)

    def test_load_missing_file_returns_false(self):
        cm = ContextManager("u")
        with mock.patch("bash_agent.context.HISTORY_FILE",
                        os.path.join(self.tmpdir, "nope.json")):
            self.assertFalse(cm.load_history())


if __name__ == "__main__":
    unittest.main()
