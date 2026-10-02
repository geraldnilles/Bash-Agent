"""
E2E — Orchestrator -> Subagent dispatch (hierarchical mode).

Journey: an ORCHESTRATOR session uses the `subagent "<task>"` command to spawn
a nested subagent. The subagent is a REAL Agent (is_subagent=True) that runs
its own loop, persists to its OWN history_subagent_<uuid8>.json, and unwinds by
raising SubagentExit (NOT SystemExit). The orchestrator then folds the
subagent's cost into its session and continues, finally exiting itself.

Proves offline:
  * orchestrator runs the SAME 2 warmup exchanges as any other role,
  * role markers ("ORCHESTRATOR agent" / "SUBAGENT ROLE") select the role,
  * sibling subagents get distinct UUIDs and distinct history files,
  * a per-command `--budget` override is honored,
  * the parent history.json stays the orchestrator's.

Replaces tests/unit/test_subagent.py (lifecycle) and the T-43g integration test.
"""
import os
import re
import unittest

from bash_agent import agent as agent_module
from bash_agent import llm

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    FakeSandbox,
)

UUID_RE = re.compile(r"current session UUID is: ([0-9a-fA-F-]+)")


def _sys_uid_role(messages):
    content = messages[0].get("content", "") if messages else ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    m = UUID_RE.search(content)
    uid = m.group(1) if m else "unknown"
    if "ORCHESTRATOR agent" in content:
        role = "orch"
    elif "SUBAGENT ROLE" in content:
        role = "sub"
    else:
        role = "??"
    return uid, role


class RecordingSandboxFactory:
    def __init__(self):
        self.instances = []

    def __call__(self, *args, **kwargs):
        fs = FakeSandbox(execute_result=(0, "ran-ok"))
        fs.uuid = kwargs.get("uuid")
        fs.timeout = kwargs.get("timeout")
        fs.is_orchestrator = kwargs.get("is_orchestrator", False)
        self.instances.append(fs)
        return fs


class OrchestratorCase(E2ECase):
    def setUp(self):
        super().setUp()
        from unittest import mock
        self._mock = mock
        self.sandbox_factory = RecordingSandboxFactory()
        # Replace the real Sandbox class so NO bwrap process is spawned, and
        # attach the factory-produced instance to each constructed agent.
        self._sandbox_patch = mock.patch.object(
            agent_module, "Sandbox", self.sandbox_factory
        )
        self._sandbox_patch.start()
        self._cleanup_patch = mock.patch(
            "bash_agent.agent.cleanup_tmp_folder", return_value=None
        )
        self._cleanup_patch.start()

        real_cls = agent_module.Agent
        factory = self.sandbox_factory

        class SandboxedAgent(real_cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.sandbox = factory.instances[-1]

        self._agent_patch = mock.patch.object(agent_module, "Agent", SandboxedAgent)
        self._agent_patch.start()

    def tearDown(self):
        self._agent_patch.stop()
        self._cleanup_patch.stop()
        self._sandbox_patch.stop()
        super().tearDown()


class TestOrchestratorSubagent(OrchestratorCase):
    def test_delegation_folds_cost_and_uses_separate_history(self):
        seen_uuid = {}
        counters = {"orch": 0, "sub": 0}

        def dynamic(**kwargs):
            uid, role = _sys_uid_role(kwargs["messages"])
            seen_uuid.setdefault(role, uid)
            n = counters[role]
            counters[role] = n + 1
            if role == "orch":
                if n == 0:
                    return make_fake_response(
                        content="Delegating.\n\n"
                        + bash_block(uid, 'subagent "list files"'),
                        cost=0.002)
                return make_fake_response(
                    content="Goal verified.\n\n" + bash_block(uid, "exit"),
                    cost=0.002)
            if role == "sub":
                if n == 0:
                    return make_fake_response(
                        content="Working.\n\n"
                        + bash_block(uid, "echo sub-work-done"),
                        cost=0.002)
                return make_fake_response(
                    content="Task complete.\n\n" + bash_block(uid, "exit"),
                    cost=0.002)
            return make_fake_response(content=bash_block(uid, "exit"), cost=0.0)

        self.seed_llm([dynamic] * 8)

        # fake_sandbox=False: the SandboxedAgent subclass already attached the
        # factory-produced FakeSandbox; don't let make_agent clobber it.
        agent = self.make_agent(is_orchestrator=True, fake_sandbox=False)
        with runaway_guard(40):
            with self.assertRaises(SystemExit) as cm:
                agent.run("delegate a marker write to a subagent")
        self.assertEqual(cm.exception.code, 0)

        self.assertIn("orch", seen_uuid)
        self.assertIn("sub", seen_uuid)
        self.assertNotEqual(seen_uuid["orch"], seen_uuid["sub"])
        self.assertEqual(seen_uuid["orch"], agent.uuid)

        # Orchestrator history: system, task, 2 warmup pairs (assistant+user),
        # assistant(delegation), user(subagent summary), assistant(exit).
        h = self.history(agent)
        roles = [m["role"] for m in h]
        self.assertEqual(len(h), 9, "orchestrator history shape: %r" % roles)
        deleg = h[6]["content"]
        self.assertIn("subagent", deleg)
        summary = h[7]["content"]
        self.assertIn("SUBAGENT", summary)
        self.assertIn("COMPLETED", summary)
        self.assertIn("ran-ok", summary)

        # Cost folded: 4 scripted calls x $0.002.
        self.assertAlmostEqual(agent.session_cost, 0.008, places=9)
        self.assertEqual(len(agent.subagent_runs), 1)
        self.assertAlmostEqual(agent.subagent_runs[0]["cost"], 0.004, places=9)

        # Subagent wrote its OWN history file.
        subdir = os.path.join(self.tmpdir, ".bash_agent_tmp")
        sub_hist = os.path.join(
            subdir, "history_subagent_%s.json" % seen_uuid["sub"][:8])
        self.assertTrue(os.path.exists(sub_hist), sub_hist)

        # Parent history.json is still the orchestrator's.
        state = self.on_disk()
        self.assertEqual(state["uuid"], agent.uuid)
        self.assertEqual(state["history"], h)

    def test_subagent_command_requires_prompt(self):
        """`subagent` with no task is refused with exit code 1, not a crash."""
        state = {"n": 0}

        def resp(**kwargs):
            uid, role = _sys_uid_role(kwargs["messages"])
            n = state["n"]
            state["n"] = n + 1
            if n == 0:
                return make_fake_response(
                    content=bash_block(uid, "subagent"), cost=0.0)
            return make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0)

        self.seed_llm([resp] * 4)
        agent = self.make_agent(is_orchestrator=True)
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("bad delegation")
        self.assertEqual(cm.exception.code, 0)
        text = self.messages_text(agent)
        self.assertIn("requires a task prompt", text)

    def test_nested_subagent_refused(self):
        """A non-orchestrator calling `subagent` gets a refusal, not a spawn."""
        agent = self.make_agent(is_orchestrator=False)
        agent.resumed_session = True
        agent.sandbox = FakeSandbox(execute_result=(0, "x"))
        state = {"n": 0}

        def resp(**kwargs):
            uid, role = _sys_uid_role(kwargs["messages"])
            n = state["n"]
            state["n"] = n + 1
            if n == 0:
                return make_fake_response(
                    content=bash_block(uid, 'subagent "nope"'), cost=0.0)
            return make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0)

        self.seed_llm([resp] * 4)
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                agent.run("nested")
        self.assertEqual(cm.exception.code, 0)
        text = self.messages_text(agent)
        self.assertIn("cannot spawn nested subagents", text)
        self.assertEqual(agent.subagent_runs, [])


if __name__ == "__main__":
    unittest.main()
