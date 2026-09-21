"""
Group 44 — Reasoning-effort resolution (``Agent._get_lowest_reasoning_effort``).

T-44  "none" is implicitly available when reasoning is not mandatory (P0)
T-44a mandatory models never report "none" (P1)
T-44b catalog-reported "none" unchanged / fallback preserved (P1)
T-44c constructor wiring: requesting "none" on a non-mandatory model whose
      ``supported_efforts`` omits it resolves to "none" (P0)

Contract pinned here:

  OpenRouter's ``reasoning.supported_efforts`` enumerates the effort levels
  available *while reasoning is enabled*.  A model may therefore advertise
  ``mandatory: false`` while OMITTING "none" from that list — disabling
  reasoning is still a legal mode, it is just not a "level".  The bug was
  that ``_get_lowest_reasoning_effort()`` only scanned
  ``supported_efforts``, so such a model's lowest level came back as e.g.
  "low" and the reasoning-disabled mode was unreachable.  Fix: treat "none"
  as implicitly available whenever ``reasoning_mandatory`` is False.

Seams (offline only): Agent construction goes through
``helpers.fakes._make_agent`` (all three network probes stubbed, FakeSandbox
injected); the constructor-wiring test patches the same probes directly.
"""
import unittest

from tests.helpers.fakes import _make_agent


class LowestReasoningEffortCase(unittest.TestCase):
    """Direct unit coverage of Agent._get_lowest_reasoning_effort()."""

    def setUp(self):
        self.agent = _make_agent()

    def _resolve(self, supported, mandatory):
        self.agent.reasoning_supported_efforts = list(supported)
        self.agent.reasoning_mandatory = mandatory
        return self.agent._get_lowest_reasoning_effort()

    # -- T-44: the corner case --------------------------------------------

    def test_none_is_implicit_for_non_mandatory_model_omitting_none(self):
        # Model does not require reasoning but its catalog list omits "none".
        self.assertEqual(self._resolve(["high", "medium", "low"], False), "none")

    def test_none_is_implicit_even_when_only_high_listed(self):
        self.assertEqual(self._resolve(["high"], False), "none")

    def test_empty_supported_list_non_mandatory_yields_none(self):
        # No levels advertised + reasoning optional -> disabled is the floor.
        self.assertEqual(self._resolve([], False), "none")

    # -- T-44a: mandatory models must NOT disable reasoning ---------------

    def test_mandatory_model_never_returns_none(self):
        self.assertEqual(self._resolve(["high", "medium", "low"], True), "low")

    def test_mandatory_model_with_minimal_returns_minimal(self):
        self.assertEqual(self._resolve(["high", "minimal"], True), "minimal")

    def test_mandatory_model_that_advertises_none_still_respects_it(self):
        # Degenerate/contradictory catalog entry: mandatory but lists "none".
        # Honoring the explicit list keeps behavior deterministic.
        self.assertEqual(self._resolve(["none", "low"], True), "none")

    # -- T-44b: unchanged happy paths -------------------------------------

    def test_explicit_none_still_wins_when_listed(self):
        self.assertEqual(self._resolve(["none", "low", "high"], False), "none")

    def test_lowest_of_full_list_when_none_absent_but_not_mandatory(self):
        # "minimal" sits below the listed levels once "none" is implicit.
        self.assertEqual(
            self._resolve(["high", "medium", "low", "minimal"], False), "none"
        )

    def test_unknown_order_falls_back_to_medium(self):
        # No recognized effort in the list -> historical "medium" fallback.
        self.assertEqual(self._resolve(["extreme", "ultra"], True), "medium")


class ConstructorWiringCase(unittest.TestCase):
    """Agent.__init__ resolves a requested "none" to a usable effort."""

    def _build_agent(self, supported, mandatory, requested):
        """Construct Agent offline with a pinned reasoning capability set."""
        from unittest import mock
        from bash_agent.agent import Agent
        from tests.helpers.fakes import FakeSandbox

        def reasoning_stub(self):
            self.reasoning_supported_efforts = list(supported)
            self.reasoning_mandatory = mandatory
            self.reasoning_default_effort = "medium"

        def context_stub(self):
            self.model_context_limit_tokens = None
            self.model_full_context_tokens = None

        with mock.patch.object(Agent, "_check_model_capabilities", return_value=None):
            with mock.patch.object(Agent, "_fetch_model_reasoning_info", reasoning_stub):
                with mock.patch.object(Agent, "_fetch_model_context_limit", context_stub):
                    with mock.patch("bash_agent.agent.cleanup_tmp_folder", return_value=None):
                        agent = Agent(reasoning_effort=requested)
        agent.sandbox = FakeSandbox()
        return agent

    def test_requested_none_survives_on_non_mandatory_model(self):
        # T-44c: the regression the fix targets — requesting "none" no longer
        # gets bumped to "low" on a model whose list omits "none".
        agent = self._build_agent(["high", "medium", "low"], False, "none")
        self.assertEqual(agent.reasoning_effort, "none")

    def test_requested_none_bumped_for_mandatory_model(self):
        # Mandatory models cannot disable reasoning: "none" -> lowest level.
        agent = self._build_agent(["high", "medium", "low"], True, "none")
        self.assertEqual(agent.reasoning_effort, "low")

    def test_unsupported_requested_level_falls_to_lowest_none(self):
        # An unsupported NON-none level on a non-mandatory model should drop to
        # the true floor, which is now "none".
        agent = self._build_agent(["high", "medium"], False, "minimal")
        self.assertEqual(agent.reasoning_effort, "none")

    def test_supported_requested_level_untouched(self):
        agent = self._build_agent(["high", "medium", "low"], False, "medium")
        self.assertEqual(agent.reasoning_effort, "medium")


if __name__ == "__main__":
    unittest.main()
