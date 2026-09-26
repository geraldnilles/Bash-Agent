"""
Group 7 — Integration: full Agent.run() driven by a fake LLM (still offline).

T-31  Full offline agent loop (P0 — the centerpiece)
T-32  Resume flow end-to-end (P1)

T-31 drives a complete fresh session end-to-end:

  task -> scratchpad injection -> warmup exchanges (real sandbox) ->
  turn 1 (bash echo) -> turn 2 (python writes a marker file) -> turn 3 (exit)

and asserts across the whole stack:
  * the REAL bwrap sandbox ran the scripts (a file written by the python
    block exists on disk with the exact content),
  * OUTPUT blocks landed in history with correct EXIT_CODE headers,
  * history.json was written and is reloadable by a fresh ContextManager,
  * the scratchpad block was injected into the very first user message,
  * budget/stats accounting accumulated the scripted costs,
  * the session ended with a clean SystemExit(0) from the exit command.

T-32 continues from a persisted first session: a second Agent(resume=True)
re-binds the UUID from disk, keeps prior messages, skips warmup, and executes
a follow-up response that reads a file created by the first session.

Offline guarantees:
  * LLM boundary faked via llm._CLIENT_CACHE seeding (FakeLLMClient),
  * capability probe patched out (Agent._check_model_capabilities),
  * the only real subprocesses are local bwrap invocations.

A SIGALRM watchdog bounds every run() call so a protocol regression fails
fast instead of hanging the harness.

NOTE ON PROTOCOL HYGIENE IN THIS FILE: expected fence strings are assembled
with small concatenation helpers (_cmd_start/_cmd_end/_out_start/...) rather
than written as single literal tokens. Keeping the marker pieces apart makes
the source safe to transmit through any channel that itself parses the
UUID-fenced protocol (a bare literal marker in a transmitted block body can
be mistaken for a real fence).
"""
import contextlib
import io
import json
import os
import signal
import sys
import unittest
from unittest import mock

from bash_agent.agent import Agent
from bash_agent.context import ContextManager
from bash_agent import llm

from tests.helpers.fakes import (
    FakeLLMClient,
    make_fake_response,
    bash_block,
    python_block,
    chdir_repo_tmp,
    bwrap_available,
    _stub_model_context_info,
)


# ---------------------------------------------------------------------------
# Fence-string builders (kept piecewise — see module docstring)
# ---------------------------------------------------------------------------

def _cmd_start(kind: str, uid: str) -> str:
    """Opening command fence for kind in {'BASH','PYTHON'}."""
    return "---START_" + kind + "_COMMAND-" + uid + "---"


def _cmd_end(kind: str, uid: str) -> str:
    return "---END_" + kind + "_COMMAND-" + uid + "---"


def _out_start(kind: str, exit_code: int, uid: str, visible: int = 100) -> str:
    return (
        "---START_" + kind + "_OUTPUT-EXIT_CODE_"
        + str(exit_code) + "-VISIBLE_" + str(visible) + "%-" + uid + "---"
    )


def _out_end(kind: str, uid: str) -> str:
    return "---END_" + kind + "_OUTPUT-" + uid + "---"


def _scratch_start(uid: str) -> str:
    return "---START_" + "SCRATCHPAD" + ".md-VISIBLE_100%-" + uid + "---"


def _scratch_end(uid: str) -> str:
    return "---END_" + "SCRATCHPAD" + ".md-" + uid


@contextlib.contextmanager
def runaway_guard(seconds: float = 30):
    """
    Fail fast if the agent loop runs away instead of hanging the harness.
    Raises AssertionError from a SIGALRM handler if the body exceeds `seconds`.
    Must be used from the main thread (unittest guarantees this).
    """
    def _handler(signum, frame):
        raise AssertionError(
            "agent.run() exceeded %ss — runaway loop? Check that the fake "
            "LLM queue ends with an 'exit' turn." % seconds
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class FullLoopCase(unittest.TestCase):
    """Shared harness: repo-local CWD, captured stdout, patched seams."""

    def setUp(self):
        self._chdir_cm = chdir_repo_tmp(prefix="loop")
        self.tmpdir = self._chdir_cm.__enter__()

        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

        # config.HISTORY_FILE is bound at import time to <repo>/.bash_agent_tmp;
        # patch the name where context.py actually looks it up so persistence
        # lands inside this test's throwaway CWD.
        self.hist_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "history.json")
        self._hist_patch = mock.patch("bash_agent.context.HISTORY_FILE", self.hist_file)
        self._hist_patch.start()

        # Neutralize the network capability probe for every Agent constructed.
        self._cap_patch = mock.patch.object(
            Agent, "_check_model_capabilities", lambda self: None
        )
        self._cap_patch.start()

        # Neutralize the model context-length probe (kept offline).
        self._ctx_patch = mock.patch.object(
            Agent, "_fetch_model_context_limit", _stub_model_context_info
        )
        self._ctx_patch.start()

        # Fresh agent for the primary session; tests may construct more.
        self.agent = Agent(is_orchestrator=False)
        self.uid = self.agent.uuid

        # Remember prior openrouter client so tearDown restores global state.
        self._old_client = llm._CLIENT_CACHE.get("openrouter")

    def tearDown(self):
        self._ctx_patch.stop()
        self._cap_patch.stop()
        self._hist_patch.stop()
        if self._old_client is None:
            llm._CLIENT_CACHE.pop("openrouter", None)
        else:
            llm._CLIENT_CACHE["openrouter"] = self._old_client
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def seed_llm(self, responses):
        """Seed the module-level client cache with a scripted fake."""
        fake = FakeLLMClient(responses=responses)
        llm._CLIENT_CACHE["openrouter"] = fake
        return fake

    def history(self):
        return self.agent.context.history


MARKER_SCRIPT = (
    "with open('loop-marker.txt', 'w') as f:\n"
    "    f.write('written-by-real-sandbox')\n"
    "print('marker-written')\n"
)

FOLLOWUP_SCRIPT = "print(open('carried.txt').read().strip())\n"


# ---------------------------------------------------------------------------
# T-31 — Full offline agent loop
# ---------------------------------------------------------------------------

@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestFullOfflineAgentLoop(FullLoopCase):

    def test_full_loop_end_to_end(self):
        # Three scripted turns: bash echo -> python marker write -> exit.
        # Costs sum to exactly 0.0075 for the budget assertion below.
        self.seed_llm([
            make_fake_response(
                content="Step one:\n\n" + bash_block(self.uid, "echo step-one"),
                cost=0.0025, prompt_tokens=100, completion_tokens=20,
            ),
            make_fake_response(
                content="Writing the marker file now.\n\n"
                        + python_block(self.uid, MARKER_SCRIPT),
                cost=0.0040, prompt_tokens=120, completion_tokens=30,
            ),
            make_fake_response(
                content="All steps verified. Ending session.\n\n"
                        + bash_block(self.uid, "exit"),
                cost=0.0010, prompt_tokens=140, completion_tokens=10,
            ),
        ])

        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm:
                self.agent.run("integration test task")
        self.assertEqual(cm.exception.code, 0)

        h = self.history()

        # --- Message skeleton -------------------------------------------------
        # [0] system, [1] task, [2..5] warmup pairs, then 3 turns:
        # assistant+output, assistant+output, assistant(exit; no output commit).
        self.assertEqual(h[0]["role"], "system")
        roles = [m["role"] for m in h]
        self.assertEqual(len(h), 11, "unexpected history shape: %r" % roles)

        # --- Scratchpad injected into the very first message ------------------
        first_user = h[1]
        self.assertEqual(first_user["role"], "user")
        self.assertIn(_scratch_start(self.uid), first_user["content"])
        self.assertIn(_scratch_end(self.uid), first_user["content"])
        self.assertIn("integration test task", first_user["content"])

        # --- Warmup exchanges really executed ---------------------------------
        self.assertEqual(h[2]["role"], "assistant")
        self.assertIn(_cmd_start("PYTHON", self.uid), h[2]["content"])
        self.assertEqual(h[3]["role"], "user")
        self.assertIn(_out_start("PYTHON", 0, self.uid), h[3]["content"])
        expected_ver = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
        self.assertIn(expected_ver, h[3]["content"])

        self.assertEqual(h[4]["role"], "assistant")
        self.assertIn(_cmd_start("BASH", self.uid), h[4]["content"])
        self.assertEqual(h[5]["role"], "user")
        self.assertIn(_out_start("BASH", 0, self.uid), h[5]["content"])
        self.assertIn(".bash_agent_tmp", h[5]["content"])

        # --- Turn 1: bash echo round trip --------------------------------------
        self.assertIn("echo step-one", h[6]["content"])
        out1 = h[7]["content"]
        self.assertIn(_out_start("BASH", 0, self.uid), out1)
        self.assertIn("step-one", out1)
        self.assertIn(_out_end("BASH", self.uid), out1)

        # --- Turn 2: python wrote a REAL file through the real sandbox --------
        out2 = h[9]["content"]
        self.assertIn(_out_start("PYTHON", 0, self.uid), out2)
        self.assertIn("marker-written", out2)

        marker_path = os.path.join(self.tmpdir, "loop-marker.txt")
        self.assertTrue(
            os.path.exists(marker_path),
            "sandbox did not really execute the python block",
        )
        with open(marker_path) as f:
            self.assertEqual(f.read(), "written-by-real-sandbox")

        # --- Final assistant message is the exit turn --------------------------
        last = h[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn(_cmd_start("BASH", self.uid), last["content"])
        self.assertIn(_cmd_end("BASH", self.uid), last["content"])
        self.assertRegex(last["content"], r"\nexit\n")

        # --- Budget / stats accounting exercised -------------------------------
        self.assertAlmostEqual(self.agent.session_cost, 0.0075, places=9)

        # --- history.json written and reloadable --------------------------------
        self.assertTrue(os.path.exists(self.hist_file))
        with open(self.hist_file) as f:
            state = json.load(f)
        self.assertEqual(state["uuid"], self.uid)
        self.assertEqual(state["history"], h)

        fresh = ContextManager("ignored-new-uuid")
        self.assertTrue(fresh.load_history())
        self.assertEqual(fresh.uuid, self.uid)
        self.assertEqual(fresh.history, h)


# ---------------------------------------------------------------------------
# T-32 — Resume flow end-to-end
# ---------------------------------------------------------------------------

@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestResumeFlow(FullLoopCase):

    def test_resume_rebinds_uuid_and_continues(self):
        # ---- Session A: persist a small history with a carried-over file -----
        self.seed_llm([
            make_fake_response(
                content="Persisting state.\n\n"
                        + bash_block(self.uid, "echo persisted-value > carried.txt"),
                cost=0.002,
            ),
            make_fake_response(
                content="Done for now.\n\n" + bash_block(self.uid, "exit"),
                cost=0.001,
            ),
        ])
        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm_a:
                self.agent.run("first session task")
        self.assertEqual(cm_a.exception.code, 0)

        uuid_a = self.agent.uuid
        len_a = len(self.history())
        # system + task + 4 warmup + assistant/output + assistant(exit)
        self.assertEqual(len_a, 9)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "carried.txt")))

        # ---- Session B: resume from disk ---------------------------------------
        agent_b = Agent(resume=True, is_orchestrator=False)
        self.assertTrue(agent_b.resumed_session)
        self.assertEqual(agent_b.uuid, uuid_a, "UUID must be re-bound from disk")

        hb = agent_b.context.history
        self.assertEqual(len(hb), len_a, "resumed history must be intact")
        self.assertEqual(hb[0]["role"], "system")
        self.assertEqual(hb[-1]["role"], "assistant")  # exit turn of session A
        self.assertIn(uuid_a, hb[-1]["content"])

        # ---- Follow-up turns reference earlier work and execute ---------------
        fake_b = self.seed_llm([
            make_fake_response(
                content="Reading what the previous session left behind.\n\n"
                        + python_block(agent_b.uuid, FOLLOWUP_SCRIPT),
                cost=0.003,
            ),
            make_fake_response(
                content="Confirmed. Goodbye.\n\n"
                        + bash_block(agent_b.uuid, "exit"),
                cost=0.001,
            ),
        ])

        with runaway_guard(30):
            with self.assertRaises(SystemExit) as cm_b:
                agent_b.run("follow-up task")
        self.assertEqual(cm_b.exception.code, 0)

        # Follow-up task appended after all prior messages...
        self.assertGreater(len(agent_b.context.history), len_a)
        followup_task_msg = agent_b.context.history[len_a]
        self.assertEqual(followup_task_msg["role"], "user")
        self.assertIn("follow-up task", followup_task_msg["content"])

        # ...and its OUTPUT fence carries the RESTORED uuid, proving the whole
        # pipeline (parser, formatter, sandbox env) re-bound coherently.
        followup_out = agent_b.context.history[len_a + 2]["content"]
        self.assertIn(_out_start("PYTHON", 0, uuid_a), followup_out)
        self.assertIn("persisted-value", followup_out)

        # Disk state still consistent after the resumed session.
        with open(self.hist_file) as f:
            state = json.load(f)
        self.assertEqual(state["uuid"], uuid_a)
        self.assertEqual(state["history"], agent_b.context.history)

        # The fake saw exactly the two turns of session B.
        self.assertEqual(len(fake_b.calls), 2)




# ---------------------------------------------------------------------------
# Group 43 integration — Orchestrator -> Subagent end-to-end (offline)
# ---------------------------------------------------------------------------

@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestOrchestratorSubagentLoop(FullLoopCase):
    """
    T-43g: Drive a REAL orchestrator run whose only delegation is a `subagent`
    block, whose subagent runs REAL sandbox commands in the project root, and
    whose exit unwinds back into the orchestrator.

    Proves the whole hierarchy offline:
      * the orchestrator (home tree hidden) runs NO warmup and delegates via
        `subagent "<task>"`,
      * the spawned subagent constructs with is_subagent=True and executes a
        real bwrap bash command (a file appears on disk),
      * the subagent's `exit` raises SubagentExit (not SystemExit) so the
        orchestrator continues,
      * the orchestrator receives the subagent's LAST 2 messages + folded cost,
      * the subagent persisted to its own history_subagent_*.json, leaving the
        parent history.json intact,
      * the orchestrator's own `exit` ends the process with SystemExit(0).
    """

    def setUp(self):
        # Reuse FullLoopCase's seam-style setup, but the primary agent here is
        # an ORCHESTRATOR (the default role). FullLoopCase builds a
        # non-orchestrator; rebuild it in the orchestrator role.
        super().setUp()
        self.agent = Agent(is_orchestrator=True, keep_tmp=True)
        self.uid = self.agent.uuid

    def test_orchestrator_delegates_and_folds_cost(self):
        import re as _re

        subdir = os.path.join(self.tmpdir, ".bash_agent_tmp")
        sub_marker = os.path.join(self.tmpdir, "subagent-ran.txt")

        seen_uuid = {}
        counters = {"orch": 0, "sub": 0}

        def _sys_uid(messages):
            content = messages[0].get("content", "") if messages else ""
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            m = _re.search(r"current session UUID is: ([0-9a-fA-F-]+)", content)
            uid = m.group(1) if m else "unknown"
            role = "orch" if "ORCHESTRATOR agent" in content else (
                "sub" if "SUBAGENT ROLE" in content else "??"
            )
            return uid, role, content

        def dynamic(**kwargs):
            uid, role, content = _sys_uid(kwargs["messages"])
            seen_uuid.setdefault(role, uid)
            n = counters[role]
            counters[role] = n + 1
            if role == "orch":
                if n == 0:
                    # First (and only) intended delegation.
                    return make_fake_response(
                        content="Delegating now.\n\n"
                                + bash_block(uid, 'subagent "write a marker file"'),
                        cost=0.002,
                    )
                # After reviewing the subagent summary, finish.
                return make_fake_response(
                    content="Goal verified. Ending session.\n\n"
                            + bash_block(uid, "exit"),
                    cost=0.002,
                )
            if role == "sub":
                if n == 0:
                    # Subagent does REAL work in the project root.
                    script = (
                        "echo subagent-ran > subagent-ran.txt\n"
                        "echo done-work\n"
                    )
                    return make_fake_response(
                        content="Working.\n\n" + bash_block(uid, script),
                        cost=0.002,
                    )
                # Subagent finishes -> raises SubagentExit (never sys.exit).
                return make_fake_response(
                    content="Task complete.\n\n" + bash_block(uid, "exit"),
                    cost=0.002,
                )
            # Fallback: never leave the queue empty.
            return make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0
            )

        fake = FakeLLMClient(responses=[dynamic, dynamic, dynamic, dynamic])
        llm._CLIENT_CACHE["openrouter"] = fake

        with runaway_guard(40):
            with self.assertRaises(SystemExit) as cm:
                self.agent.run("delegate a marker write to a subagent")
        self.assertEqual(cm.exception.code, 0)

        # --- Roles actually exercised -----------------------------------------
        self.assertIn("orch", seen_uuid)
        self.assertIn("sub", seen_uuid)
        self.assertNotEqual(seen_uuid["orch"], seen_uuid["sub"])
        self.assertEqual(seen_uuid["orch"], self.uid)

        # --- Orchestrator history shape ---------------------------------------
        h = self.history()
        self.assertEqual(h[0]["role"], "system")
        roles = [m["role"] for m in h]
        # system, user(task), assistant(subagent cmd), user(summary), assistant(exit)
        self.assertEqual(len(h), 5, "unexpected orchestrator history: %r" % roles)

        deleg_cmd = h[2]["content"]
        self.assertIn(_cmd_start("BASH", self.uid), deleg_cmd)
        self.assertIn("subagent", deleg_cmd)

        summary = h[3]["content"]
        self.assertIn(_out_start("BASH", 0, self.uid), summary)
        self.assertIn("SUBAGENT", summary)
        self.assertIn("COMPLETED", summary)
        self.assertIn("done-work", summary, "summary must carry subagent output")

        # --- The subagent's real sandbox work landed on disk ------------------
        self.assertTrue(
            os.path.exists(sub_marker),
            "subagent's sandboxed command did not really run",
        )
        with open(sub_marker) as f:
            self.assertEqual(f.read().strip(), "subagent-ran")

        # --- Cost folded: 4 scripted calls x $0.002 ---------------------------
        self.assertAlmostEqual(self.agent.session_cost, 0.008, places=9)

        # --- Subagent used its OWN history file -------------------------------
        sub_hist = os.path.join(
            subdir, "history_subagent_%s.json" % seen_uuid["sub"][:8]
        )
        self.assertTrue(
            os.path.exists(sub_hist),
            "subagent history file missing: %s" % sub_hist,
        )
        # Parent history.json is still the orchestrator's and is consistent.
        with open(self.hist_file) as f:
            state = json.load(f)
        self.assertEqual(state["uuid"], self.uid)
        self.assertEqual(state["history"], h)


if __name__ == "__main__":
    unittest.main()
