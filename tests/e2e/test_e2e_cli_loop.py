"""
E2E — CLI entry point drives a REAL Agent loop.

Journey: `bash_agent.main.main()` is invoked with a patched `sys.argv`; the
REAL `Agent` is constructed (only the network probes are served offline by
`E2ECase`) and its `run()` loop is driven by a scripted fake LLM. This is the
end-to-end replacement for the unit-level argparse mapping tests
(tests/unit/test_main_cli.py): instead of asserting `main.Agent` was called
with a MagicMock, we let the real loop run to completion and assert on the
observable outcome.

Offline seams (all provided by tests/e2e/harness.E2ECase):
  * `llm._CLIENT_CACHE["openrouter"]` -> FakeLLMClient (chat completions),
  * `urllib.request.urlopen` -> empty /models catalog (probe fallbacks),
  * `bash_agent.search.requests.post` -> stubbed.
Additionally patched here:
  * `bash_agent.agent.Sandbox` -> FakeSandbox (no bwrap process needed),
  * `bash_agent.main.Agent` -> a recording subclass (captures ctor kwargs).

Protocol hygiene: expected fences are built piecewise via harness helpers.
"""
import unittest
from unittest import mock

from bash_agent import agent as agent_module
from bash_agent import main as main_module

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    FakeSandbox,
)

import re

UUID_RE = re.compile(r"current session UUID is: ([0-9a-fA-F-]+)")


def uuid_from_messages(messages):
    content = messages[0].get("content", "") if messages else ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    m = UUID_RE.search(content)
    return m.group(1) if m else "unknown"


class FakeSandboxFactory:
    """Callable stand-in for `bash_agent.agent.Sandbox` returning a FakeSandbox."""

    def __init__(self):
        self.instances = []

    def __call__(self, *args, **kwargs):
        fs = FakeSandbox()
        fs.uuid = kwargs.get("uuid")
        fs.timeout = kwargs.get("timeout")
        fs.is_orchestrator = kwargs.get("is_orchestrator", False)
        self.instances.append(fs)
        return fs


class CliLoopCase(E2ECase):
    """E2ECase + a recording Agent subclass + FakeSandbox factory."""

    def setUp(self):
        super().setUp()
        self.sandbox_factory = FakeSandboxFactory()
        self._sandbox_patch = mock.patch.object(
            agent_module, "Sandbox", self.sandbox_factory
        )
        self._sandbox_patch.start()
        # Do not let Agent.__init__ wipe the throwaway tmp dir mid-test.
        self._cleanup_patch = mock.patch(
            "bash_agent.agent.cleanup_tmp_folder", return_value=None
        )
        self._cleanup_patch.start()

        # Recording subclass wrapping the REAL Agent.
        self.captured = {}
        real_cls = agent_module.Agent
        captured = self.captured
        sandbox_factory = self.sandbox_factory

        class RecordingAgent(real_cls):
            def __init__(self, *a, **kw):
                captured.update(kw)
                super().__init__(*a, **kw)
                # Swap in the factory-produced FakeSandbox for this instance.
                self.sandbox = sandbox_factory.instances[-1]

        self.RecordingAgent = RecordingAgent
        self._main_agent_patch = mock.patch.object(
            main_module, "Agent", RecordingAgent
        )
        self._main_agent_patch.start()

    def tearDown(self):
        self._main_agent_patch.stop()
        self._cleanup_patch.stop()
        self._sandbox_patch.stop()
        super().tearDown()

    def run_main(self, argv):
        with mock.patch("sys.argv", ["bagent", *argv]):
            try:
                main_module.main()
                exc = None
            except SystemExit as e:
                exc = e
        return exc

    def exit_responder(self, greeting="Done.", cost=0.001):
        """Return a callable responder that ends the session with `exit`."""
        def responder(**kwargs):
            uid = uuid_from_messages(kwargs["messages"])
            return make_fake_response(
                content=greeting + "\n\n" + bash_block(uid, "exit"),
                cost=cost,
            )
        return responder


class TestCliHappyPath(CliLoopCase):
    def test_message_flag_runs_full_loop_and_exits_zero(self):
        """`-m <task>` drives a real Agent loop that ends with SystemExit(0)."""
        agent_cls = self.RecordingAgent
        self.seed_llm([self.exit_responder()])

        with runaway_guard(30):
            exc = self.run_main(["-m", "say hello"])

        self.assertIsInstance(exc, SystemExit)
        self.assertEqual(exc.code, 0)
        # Real Agent got the flags main forwards for a plain run.
        self.assertFalse(self.captured["is_orchestrator"])
        self.assertIsNone(self.captured["model"])
        # A FakeSandbox was constructed and used by the real loop.
        self.assertTrue(self.sandbox_factory.instances)

    def test_warmup_and_task_land_in_history(self):
        """The non-orchestrator path runs 2 warmup exchanges before the task."""
        self.seed_llm([self.exit_responder()])
        with runaway_guard(30):
            exc = self.run_main(["-m", "warmup check"])
        self.assertEqual(exc.code, 0)
        # Grab the agent instance from the factory-backed RecordingAgent is not
        # directly accessible; assert via the persisted history on disk instead.
        state = self.on_disk()
        roles = [m["role"] for m in state["history"]]
        self.assertEqual(roles[0], "system")
        # system, user(task), 4 warmup msgs, assistant(exit turn)
        self.assertEqual(len(roles), 7)
        self.assertIn("warmup check", state["history"][1]["content"])

    def test_flag_mapping_forwarded_to_agent(self):
        """CLI flags map onto the real Agent constructor kwargs."""
        self.seed_llm([self.exit_responder()])
        with runaway_guard(30):
            exc = self.run_main([
                "-m", "mapped",
                "--model", "vendor/model-x",
                "--reasoning-effort", "high",
                "--max-tokens", "2048",
                "--token-budget", "128K",
                "-t", "45",
                "-b", "2.50",
                "--subagent-budget", "0.05",
                "-k", "-d",
            ])
        self.assertEqual(exc.code, 0)
        self.assertEqual(self.captured["model"], "vendor/model-x")
        self.assertEqual(self.captured["reasoning_effort"], "high")
        self.assertEqual(self.captured["max_tokens"], 2048)
        self.assertEqual(self.captured["token_budget"], "128K")
        self.assertEqual(self.captured["timeout"], 45)
        self.assertEqual(self.captured["budget"], 2.50)
        self.assertEqual(self.captured["subagent_budget"], 0.05)
        self.assertTrue(self.captured["keep_tmp"])
        self.assertTrue(self.captured["debug"])

    def test_orchestrator_flag_and_model_alias(self):
        """`-o` enables orchestrator mode; `--orchestrator-model` sets model."""
        # Warmup runs for the orchestrator too; the responder always exits.
        def responder(**kwargs):
            uid = uuid_from_messages(kwargs["messages"])
            return make_fake_response(content=bash_block(uid, "exit"), cost=0.0)

        self.seed_llm([responder])
        with runaway_guard(30):
            exc = self.run_main([
                "-o",
                "--orchestrator-model", "vendor/orch-x",
                "-m", "orchestrate",
            ])
        self.assertEqual(exc.code, 0)
        self.assertTrue(self.captured["is_orchestrator"])
        self.assertEqual(self.captured["model"], "vendor/orch-x")

    def test_orchestrator_model_alone_does_not_enable_mode(self):
        """`--orchestrator-model` without `-o` keeps standalone mode."""
        self.seed_llm([self.exit_responder()])
        with runaway_guard(30):
            exc = self.run_main([
                "--orchestrator-model", "vendor/orch-x",
                "-m", "standalone",
            ])
        self.assertEqual(exc.code, 0)
        self.assertFalse(self.captured["is_orchestrator"])
        self.assertIsNone(self.captured["model"])


class TestCliArgValidation(CliLoopCase):
    def test_unknown_reasoning_effort_exits_2(self):
        exc = self.run_main(["-m", "x", "--reasoning-effort", "maximum"])
        self.assertIsInstance(exc, SystemExit)
        self.assertEqual(exc.code, 2)

    def test_invalid_token_budget_exits_2(self):
        exc = self.run_main(["-m", "x", "--token-budget", "not-a-number"])
        self.assertIsInstance(exc, SystemExit)
        self.assertEqual(exc.code, 2)


if __name__ == "__main__":
    unittest.main()
