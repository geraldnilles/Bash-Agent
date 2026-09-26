"""
Group 8b — System-prompt composition tests for the ORCHESTRATOR prompt.

T-36b  get_orchestrator_system_prompt — VISION advertising (P1)

The orchestrator prompt previously said nothing about ``vision`` even though
the orchestrator sandbox now re-binds ``venv/`` (which ships the ``vision``
CLI) and ``bash_agent/`` read-only.  This module pins that the orchestrator
prompt advertises VISION using the SAME two variants as the standard prompt
(shared ``_build_vision_section`` helper), selected from
``multimodal_capabilities``:

  * image-capable model -> ``vision <path>`` attaches into multimodal context
  * text-only model     -> ``vision [-p <prompt>] <path>`` fallback wording

Substring ("snapshot-style") assertions only, so prose can evolve.
"""

import unittest

from bash_agent.prompts import get_orchestrator_system_prompt

UUID = "0f0e1d2c-3b4a-4968-8778-99aabbccddee"
SCRATCHPAD = "/home/gerald/Code/demo-project/.bash_agent_tmp/SCRATCHPAD.md"

ATTACH_USAGE = "`vision <path_to_image>`"
ATTACH_NOTE = "Attached images will automatically be added to your multimodal context"
FALLBACK_USAGE = "`vision [-p <text prompt>] <path_to_image>`"


class OrchestratorPromptCase(unittest.TestCase):
    def build(self, **overrides):
        params = {"uuid": UUID, "scratchpad_path": SCRATCHPAD}
        params.update(overrides)
        return get_orchestrator_system_prompt(**params)


class TestOrchestratorVisionSection(OrchestratorPromptCase):
    def test_vision_header_present_exactly_once(self):
        # None / image / empty all carry exactly one header.
        self.assertEqual(self.build().count("## VISION CAPABILITIES"), 1)
        self.assertEqual(
            self.build(multimodal_capabilities=["image"]).count(
                "## VISION CAPABILITIES"), 1)
        self.assertEqual(
            self.build(multimodal_capabilities=[]).count(
                "## VISION CAPABILITIES"), 1)

    def test_none_capabilities_get_text_fallback_section(self):
        prompt = self.build(multimodal_capabilities=None)
        self.assertIn(FALLBACK_USAGE, prompt)
        self.assertNotIn(ATTACH_USAGE, prompt)
        self.assertNotIn(ATTACH_NOTE, prompt)

    def test_empty_capabilities_list_behaves_like_none(self):
        prompt = self.build(multimodal_capabilities=[])
        self.assertIn(FALLBACK_USAGE, prompt)
        self.assertNotIn(ATTACH_USAGE, prompt)

    def test_image_capable_model_gets_attach_mode_section(self):
        prompt = self.build(multimodal_capabilities=["image"])
        self.assertIn(ATTACH_USAGE, prompt)
        self.assertIn(ATTACH_NOTE, prompt)
        self.assertNotIn(FALLBACK_USAGE, prompt)

    def test_comma_style_capability_list_still_enables_attach_mode(self):
        prompt = self.build(multimodal_capabilities=["image", "audio"])
        self.assertIn(ATTACH_USAGE, prompt)
        self.assertNotIn(FALLBACK_USAGE, prompt)


class TestOrchestratorStructure(OrchestratorPromptCase):
    SECTIONS = [
        "## YOUR INTERFACE",
        "## ENVIRONMENT CONSTRAINTS (READ CAREFULLY)",
        "## VISION CAPABILITIES",
        "## THE `subagent` COMMAND",
        "## ORCHESTRATOR WORKFLOW",
        "## SCRATCHPAD MEMORY",
    ]

    def test_all_major_sections_present_exactly_once(self):
        prompt = self.build()
        for section in self.SECTIONS:
            with self.subTest(section=section):
                self.assertEqual(prompt.count(section), 1,
                                 f"{section!r} missing or duplicated")

    def test_scratchpad_path_is_interpolated(self):
        self.assertIn(SCRATCHPAD, self.build())


class TestOrchestratorBudgetLine(OrchestratorPromptCase):
    """The orchestrator prompt is budget-aware when a budget is supplied."""

    def test_budget_line_present_with_values(self):
        prompt = self.build(budget=0.20, session_cost=0.05,
                            subagent_budget=0.05)
        self.assertIn("total session budget of $0.20", prompt)
        self.assertIn("Used so far: $0.05", prompt)
        self.assertIn("Remaining: $0.15", prompt)
        self.assertIn("default subagent budget is $0.05", prompt)

    def test_budget_line_absent_without_budget(self):
        prompt = self.build()
        self.assertNotIn("total session budget", prompt)

    def test_budget_line_absent_for_zero_budget(self):
        prompt = self.build(budget=0.0)
        self.assertNotIn("total session budget", prompt)

    def test_subagent_command_documents_budget_flag(self):
        prompt = self.build(budget=0.20)
        self.assertIn("--budget", prompt)

    def test_workflow_has_budget_step(self):
        prompt = self.build(budget=0.20)
        self.assertIn("BUDGET: Before dispatching", prompt)


if __name__ == "__main__":
    unittest.main()
