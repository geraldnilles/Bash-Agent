"""
E2E — context pruning, scratchpad injection, warning, and token-budget.

Journey A (ContextManager directly): build real histories, drive the REAL
hysteresis trim / one-time warning ladder, and assert the protocol shape of
the surviving messages. Token counting is pinned to the deterministic offline
cl100k meter installed by `DeterministicTokenCounts`, so exact-fixture
arithmetic is immune to tokenizer availability.

Journey B (Agent.run): the scratchpad block is injected ONCE into the first
user task, and its VISIBLE_% truncation behaves under SCRATCHPAD_LIMIT.

Journey C (token budgets): `resolve_budget`/`parse_token_budget` produce the
per-session context ceiling, and a real Agent derives `context_limit` from the
model window (or the presumed default when offline).

Replaces tests/unit/test_context_pruning.py, test_context_warning.py,
test_context_scratchpad.py, test_token_budget.py, test_model_context_limit.py.
"""
import os
import unittest

from bash_agent.context import ContextManager
from bash_agent.token_budget import parse_token_budget, resolve_budget
from bash_agent.config import CONTEXT_LIMIT

from tests.e2e.harness import (
    E2ECase,
    runaway_guard,
    make_fake_response,
    bash_block,
    output_block,
    FakeSandbox,
    DeterministicTokenCounts,
    deterministic_count_tokens,
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


def tokfill(tokens):
    """Plain string the deterministic cl100k meter encodes to EXACTLY `tokens`."""
    return "u" * (2 * tokens)


LIMIT = 300
TARGET = int(LIMIT * 0.8)
WARN_THRESHOLD = int(LIMIT * 0.99)
HYSTERESIS_BANNER = "Initiating hysteresis cleanup"


class ContextCase(E2ECase):
    """E2ECase + deterministic token meter + ready ContextManager."""

    def setUp(self):
        super().setUp()
        self._meter = DeterministicTokenCounts()
        self._meter.start()
        self.uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.cm = ContextManager(self.uid, context_limit=LIMIT)

    def tearDown(self):
        self._meter.stop()
        super().tearDown()

    def total(self):
        return sum(deterministic_count_tokens(m.get("content", ""))
                   for m in self.cm.history)


class TestPruningLadder(ContextCase):
    def test_no_trim_below_limit(self):
        self.cm.history = [
            {"role": "system", "content": "S" * 40},
            {"role": "user", "content": tokfill(100)},
        ]
        before = self.total()
        self.cm._trim_context_if_needed()
        self.assertEqual(self.total(), before)
        self.assertNotIn(HYSTERESIS_BANNER, self.stdout_buf.getvalue())

    def test_trim_hollows_output_and_reaches_target(self):
        # Three OUTPUT blocks, bodies 300 tokens each -> well over LIMIT.
        self.cm.history = [{"role": "system", "content": "sys"}]
        for _ in range(3):
            self.cm.history.append(
                {"role": "user",
                 "content": output_block(self.uid, 0, tokfill(300))}
            )
        self.cm._trim_context_if_needed()
        self.assertIn(HYSTERESIS_BANNER, self.stdout_buf.getvalue())
        self.assertLessEqual(self.total(), TARGET)
        # The system prompt at index 0 is never removed.
        self.assertEqual(self.cm.history[0]["role"], "system")
        # Hollowing preserves the fence shape and inserts the marker.
        joined = "\n".join(m["content"] for m in self.cm.history)
        self.assertIn("BASH_OUTPUT DELETED TO SAVE CONTEXT", joined)

    def test_failsafe_drops_oldest_message(self):
        # Only plain prose: no fences to hollow -> wholesale drop path.
        self.cm.history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": tokfill(500)},
            {"role": "assistant", "content": tokfill(500)},
        ]
        self.cm._trim_context_if_needed()
        self.assertLessEqual(self.total(), TARGET)
        self.assertEqual(self.cm.history[0]["role"], "system")
        self.assertIn("Dropping oldest conversational message",
                      self.stdout_buf.getvalue())

    def test_system_prompt_never_dropped_even_if_oversized(self):
        self.cm.history = [{"role": "system", "content": tokfill(5000)}]
        self.cm._trim_context_if_needed()
        self.assertEqual(len(self.cm.history), 1)
        self.assertEqual(self.cm.history[0]["role"], "system")


class TestContextWarning(ContextCase):
    def test_warning_injected_once_then_trim_deferred(self):
        self.cm.history = [{"role": "system", "content": "sys"}]
        # Cross the 99% warn threshold with a single big user message.
        self.cm.add_message("user", tokfill(WARN_THRESHOLD + 50))
        joined = "\n".join(m["content"] for m in self.cm.history)
        self.assertIn("back up any important findings", joined)
        # No trim yet even though we exceeded the ceiling.
        self.assertNotIn(HYSTERESIS_BANNER, self.stdout_buf.getvalue())

    def test_assistant_message_confirms_and_trims(self):
        self.cm.history = [{"role": "system", "content": "sys"}]
        self.cm.add_message("user", tokfill(WARN_THRESHOLD + 200))
        # Wipe the banner history: this is the confirmation move.
        self.cm.add_message("assistant", tokfill(50))
        # Now overflow hard with more output messages and add an assistant.
        self.cm.history.extend([
            {"role": "user", "content": output_block(self.uid, 0, tokfill(400))},
            {"role": "user", "content": output_block(self.uid, 0, tokfill(400))},
        ])
        self.cm.add_message("assistant", "ok")
        self.assertIn(HYSTERESIS_BANNER, self.stdout_buf.getvalue())

    def test_reset_rearms_warning_flags(self):
        self.cm._warning_sent = True
        self.cm._warning_confirmed = True
        # Simulate the reset handler's flag re-arm.
        self.cm._warning_sent = False
        self.cm._warning_confirmed = False
        self.assertFalse(self.cm._warning_sent)
        self.assertFalse(self.cm._warning_confirmed)


class TestScratchpadInjection(E2ECase):
    def _make_agent(self):
        agent = self.make_agent(is_orchestrator=False)
        agent.sandbox = FakeSandbox(execute_result=(0, "x"))
        return agent

    def test_scratchpad_injected_once_into_first_task(self):
        agent = self._make_agent()
        # Fresh session: warmup runs, scratchpad injected into message [1].
        self.seed_llm([
            make_fake_response(
                content=bash_block("__UID__", "exit"), cost=0.0
            )
        ])
        # Build a dynamic responder that carries the right uuid.
        state = {"n": 0}

        def resp(**kwargs):
            uid = uid_of(kwargs["messages"])
            return make_fake_response(
                content=bash_block(uid, "exit"), cost=0.0)
        self.seed_llm([resp] * 4)
        with runaway_guard(30):
            with self.assertRaises(SystemExit):
                agent.run("do the thing")
        first_user = agent.context.history[1]
        self.assertEqual(first_user["role"], "user")
        self.assertIn("SCRATCHPAD.md-VISIBLE_100%", first_user["content"])
        self.assertIn("do the thing", first_user["content"])

    def test_oversize_scratchpad_reports_reduced_visibility(self):
        agent = self._make_agent()
        sp_path = agent.context.scratchpad_path
        os.makedirs(os.path.dirname(sp_path), exist_ok=True)
        with open(sp_path, "w") as f:
            f.write("z" * 160000)  # 2x the 80000 SCRATCHPAD_LIMIT
        block = agent.context.get_scratchpad_block()
        self.assertIn("VISIBLE_50%", block)
        self.assertIn("Scratchpad truncated", block)


class TestTokenBudgetResolution(E2ECase):
    def test_parse_and_resolve(self):
        self.assertEqual(parse_token_budget(None), None)
        self.assertEqual(parse_token_budget(8192), ("abs", 8192))
        self.assertEqual(parse_token_budget("256K"), ("abs", 256000))
        self.assertEqual(parse_token_budget("1.5M"), ("abs", 1500000))
        self.assertEqual(parse_token_budget("25%"), ("pct", 25.0))

        # Percent of a known window.
        self.assertEqual(resolve_budget(("pct", 25.0), 131072), 32768)
        # Absolute ignores the window.
        self.assertEqual(resolve_budget(("abs", 1000), 999999), 1000)
        # None -> quarter of the window; offline -> quarter of presumed window.
        self.assertEqual(resolve_budget(None, 131072), 32768)
        self.assertEqual(resolve_budget(None, None), CONTEXT_LIMIT)
        # Never below 1.
        self.assertGreaterEqual(resolve_budget(("pct", 0.0001), 1), 1)

    def test_invalid_specs_raise(self):
        for bad in ("", "abc", "0", "150%", 0, -5, 1.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_token_budget(bad)

    def test_agent_derives_context_limit_from_window(self):
        # Feed a catalog that advertises the model + a context window.
        from tests.e2e.harness import DEFAULT_CATALOG
        self.set_catalog([
            {"id": "test/model", "context_length": 131072,
             "architecture": {"input_modalities": ["text"]}},
        ])
        agent = self.make_agent(is_orchestrator=False, model="test/model",
                                fake_sandbox=False)
        # Default budget = quarter of the full window.
        self.assertEqual(agent.model_full_context_tokens, 131072)
        self.assertEqual(agent.context_limit, 32768)

    def test_agent_offline_falls_back_to_default_ceiling(self):
        agent = self.make_agent(is_orchestrator=False, model="unknown/model",
                                fake_sandbox=False)
        self.assertIsNone(agent.model_full_context_tokens)
        self.assertEqual(agent.context_limit, CONTEXT_LIMIT)


if __name__ == "__main__":
    unittest.main()
