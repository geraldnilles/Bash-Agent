"""
Group 8 — System-prompt composition tests for bash_agent.prompts.

T-36  get_system_prompt — composition & interpolation contract (P1)

get_system_prompt() builds the system prompt the agent LLM sees on turn 1.
It interpolates four volatile values (session UUID, CWD, scratchpad path,
today's date) and switches the VISION section between two variants:

  * image-capable model -> `vision <path>` attaches the image into the
    agent's multimodal context (no LLM call inside the sandbox)
  * text-only model     -> `vision [-p <prompt>] <path>` asks a fallback
    model to describe the image as text

A regression here is catastrophic but silent: the LLM would emit fences
with a wrong UUID (every block ignored) or lose the scratchpad contract.
These tests pin interpolation and both vision variants with
substring ("snapshot-style") assertions — deliberately NOT full-string
equality, so prompt prose can evolve without breaking the suite.

Seam notes:
  * get_system_prompt is a pure string function: no filesystem, no
    network, no chdir_tmp required.
  * Expected fences are assembled with the shared bash_block() /
    python_block() helpers (T-00b) rather than hand-concatenated
    literals, per the protocol-hygiene rule in tests/AGENTS.md.
"""

import datetime
import unittest

from bash_agent.prompts import get_system_prompt
from tests.helpers.fakes import bash_block, python_block

UUID = "0f0e1d2c-3b4a-4968-8778-99aabbccddee"
CWD = "/home/gerald/Code/demo-project"
SCRATCHPAD = "/home/gerald/Code/demo-project/.bash_agent_tmp/SCRATCHPAD.md"

DEFAULT_ROLE_LINE = "You are an expert, autonomous Linux Scripting Agent."
ATTACH_USAGE = "`vision <path_to_image>`"
ATTACH_NOTE = "Attached images will automatically be added to your multimodal context"
FALLBACK_USAGE = "`vision [-p <text prompt>] <path_to_image>`"


class GetSystemPromptCase(unittest.TestCase):
    """Base harness: builds prompts from stable fixture values."""

    def build(self, **overrides):
        params = {"uuid": UUID, "cwd": CWD, "scratchpad_path": SCRATCHPAD}
        params.update(overrides)
        return get_system_prompt(**params)


class TestInterpolation(GetSystemPromptCase):
    """The four volatile values must land in the rendered prompt."""

    def test_uuid_appears_in_session_line_and_fence_examples(self):
        prompt = self.build()
        self.assertIn(f"The current session UUID is: {UUID}", prompt)
        # The example blocks are what teach the LLM the fence format; they
        # must carry THIS session's UUID, byte-for-byte.
        expected_bash = bash_block(UUID, "[command goes here]")
        expected_python = python_block(UUID, "[python code goes here]")
        self.assertIn(expected_bash, prompt)
        self.assertIn(expected_python, prompt)
        # Exactly once each — duplicated examples would confuse parsing.
        self.assertEqual(prompt.count(expected_bash), 1)
        self.assertEqual(prompt.count(expected_python), 1)

    def test_cwd_is_interpolated(self):
        prompt = self.build()
        self.assertIn(f"current working directory: `{CWD}`", prompt)

    def test_scratchpad_path_is_interpolated_in_path_and_update_example(self):
        prompt = self.build()
        self.assertIn(f"- Path: {SCRATCHPAD}", prompt)
        # The update-strategy example appends via >> to the same path.
        self.assertIn(f">> {SCRATCHPAD}`", prompt)

    def test_today_date_is_rendered_isoformat(self):
        prompt = self.build()
        today = datetime.date.today().strftime("%Y-%m-%d")
        self.assertIn(f"Today's Date: {today}", prompt)


class TestRoleText(GetSystemPromptCase):
    """role_text replaces the default persona line; falsy falls back."""

    def test_default_role_used_when_role_text_is_none(self):
        prompt = self.build(role_text=None)
        self.assertIn(DEFAULT_ROLE_LINE, prompt)
        self.assertTrue(prompt.startswith(DEFAULT_ROLE_LINE))

    def test_custom_role_text_replaces_default(self):
        prompt = self.build(role_text="You are a grumpy but thorough sysadmin.")
        self.assertIn("You are a grumpy but thorough sysadmin.", prompt)
        self.assertNotIn(DEFAULT_ROLE_LINE, prompt)

    def test_empty_role_text_falls_back_to_default(self):
        # get_system_prompt treats falsy role_text as "no customization".
        prompt = self.build(role_text="")
        self.assertIn(DEFAULT_ROLE_LINE, prompt)


class TestVisionSection(GetSystemPromptCase):
    """Two mutually exclusive VISION variants keyed on 'image' capability."""

    def test_image_capable_model_gets_attach_mode_section(self):
        prompt = self.build(multimodal_capabilities=["image"])
        self.assertIn(ATTACH_USAGE, prompt)
        self.assertIn(ATTACH_NOTE, prompt)
        # The fallback-only flag usage must NOT leak into attach mode.
        self.assertNotIn(FALLBACK_USAGE, prompt)

    def test_comma_style_capability_list_still_enables_attach_mode(self):
        prompt = self.build(multimodal_capabilities=["image", "audio"])
        self.assertIn(ATTACH_USAGE, prompt)
        self.assertNotIn(FALLBACK_USAGE, prompt)

    def test_none_capabilities_get_text_fallback_section(self):
        prompt = self.build(multimodal_capabilities=None)
        self.assertIn(FALLBACK_USAGE, prompt)
        self.assertNotIn(ATTACH_USAGE, prompt)
        self.assertNotIn(ATTACH_NOTE, prompt)

    def test_empty_capabilities_list_behaves_like_none(self):
        prompt = self.build(multimodal_capabilities=[])
        self.assertIn(FALLBACK_USAGE, prompt)
        self.assertNotIn(ATTACH_USAGE, prompt)


class TestStructureStability(GetSystemPromptCase):
    """Loose skeleton check: sections survive prose edits."""

    SECTIONS = [
        "## EXECUTION BLOCK FORMATTING & UUID FENCING",
        "## OUTPUT METADATA",
        "## SPECIAL COMMANDS",
        "## SEMANTIC SEARCH",
        "## AGENTS.md FILES",
        "## VISION CAPABILITIES",
        "## TRANSCRIBE CAPABILITIES",
        "## PDF Processing",
        "## SCRATCHPAD MEMORY",
        "## WORKFLOW & ERROR RECOVERY",
    ]

    def test_all_major_sections_present_exactly_once(self):
        prompt = self.build()
        for section in self.SECTIONS:
            with self.subTest(section=section):
                self.assertEqual(prompt.count(section), 1,
                                 f"{section!r} missing or duplicated")

    def test_vision_header_present_in_both_variants(self):
        self.assertEqual(self.build().count("## VISION CAPABILITIES"), 1)
        self.assertEqual(
            self.build(multimodal_capabilities=["image"]).count(
                "## VISION CAPABILITIES"), 1)


if __name__ == "__main__":
    unittest.main()
