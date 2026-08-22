"""
Group 1 — Protocol Parsing tests for agent._extract_blocks.

T-01  Valid bash and python block extraction (P0)
T-02  Mixed prose and multiple blocks (P1)

The UUID-fenced protocol is the front door of the entire agent: the LLM's
response must be parsed into (cmd_type, script) pairs exactly. These tests
feed well-formed blocks built by the T-00b helpers through the production
regex parser and assert exact extraction semantics.

All tests are offline (pure regex over a string) and run through the
_make_agent() helper so no network probe or systemd round-trip occurs.
"""
import unittest
import uuid

from tests.helpers.fakes import (
    bash_block,
    python_block,
    _make_agent,
)


class TestExtractBlocksValid(unittest.TestCase):
    """T-01: A well-formed bash and python block are both extracted."""

    def _agent(self, uid):
        return _make_agent(uuid_str=uid)

    def test_extracts_single_bash_block(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        block = bash_block(uid, "echo hello")
        blocks, warning = agent._extract_blocks(block)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "echo hello")])

    def test_extracts_single_python_block(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        block = python_block(uid, "print('hello')")
        blocks, warning = agent._extract_blocks(block)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("PYTHON", "print('hello')")])

    def test_extracts_bash_then_python_in_document_order(self):
        """Feeds one well-formed bash block and one python block; both must
        come back in document order with exact scripts."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        bash_script = "ls -la | head"
        python_code = "import sys\nprint(sys.version_info[:2])"
        response = (
            bash_block(uid, bash_script) + "\n\n" + python_block(uid, python_code)
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(
            blocks,
            [("BASH", bash_script), ("PYTHON", python_code)],
        )

    def test_whitespace_around_script_is_stripped(self):
        """The parser must strip surrounding whitespace from the script body
        (the plan calls this out explicitly for T-01)."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        block = (
            f"---START_BASH_COMMAND-{uid}---\n"
            f"   \n"
            f"   echo padded   \n"
            f"   \n"
            f"---END_BASH_COMMAND-{uid}---"
        )
        blocks, warning = agent._extract_blocks(block)
        self.assertIsNone(warning)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], ("BASH", "echo padded"))

    def test_multiline_script_preserved_with_inner_newlines(self):
        """Inner whitespace/newlines are preserved; only the outer padding is
        stripped."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        script = "echo one\n  echo two\n echo three"
        block = bash_block(uid, script)
        blocks, warning = agent._extract_blocks(block)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", script)])

    def test_empty_script_extracts_with_empty_string(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        block = f"---START_BASH_COMMAND-{uid}---\n\n---END_BASH_COMMAND-{uid}---"
        blocks, warning = agent._extract_blocks(block)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "")])

    def test_wrong_uuid_is_not_extracted_by_strict_parser(self):
        """A block fenced with the wrong UUID must NOT match the strict
        pattern; this is the classic --resume failure mode covered by T-04."""
        uid = str(uuid.uuid4())
        wrong_uid = str(uuid.uuid4())
        agent = self._agent(uid)
        block = bash_block(wrong_uid, "echo hi")
        blocks, warning = agent._extract_blocks(block)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)



class TestExtractBlocksMixedProse(unittest.TestCase):
    """T-02: Blocks inside prose are found; foreign fences ignored; document
    order preserved; greedy-regex regressions caught."""

    def _agent(self, uid):
        return _make_agent(uuid_str=uid)

    def test_blocks_embedded_in_prose_are_found(self):
        """Markdown prose before/between/after blocks must not interfere."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        bash_script = "git status --short"
        python_code = "print(sum(range(10)))"
        response = (
            "## Plan\n"
            "I'll start with a quick status check, then a python sanity check.\n\n"
            + bash_block(uid, bash_script)
            + "\n\nIf that succeeds, I will verify the environment with python:\n\n"
            + python_block(uid, python_code)
            + "\n\nBoth steps are offline and safe to run."
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(
            blocks, [("BASH", bash_script), ("PYTHON", python_code)]
        )

    def test_two_blocks_of_same_type_with_prose_between(self):
        """Guards against greedy `.*` regressions that would swallow both
        blocks into a single match spanning first START to last END."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        first, second = "echo one", "echo two"
        response = (
            "Step 1:\n" + bash_block(uid, first)
            + "\n\nNow step 2, also bash:\n" + bash_block(uid, second)
            + "\nDone."
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", first), ("BASH", second)])

    def test_attached_image_fence_is_ignored(self):
        """A non-command fence (ATTACHED_IMAGE) must not produce a block nor
        corrupt extraction of neighboring command blocks."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        image_fence = (
            f"---START_ATTACHED_IMAGE-{uid}---\n"
            "data:image/png;base64,iVBORw0KGgo=\n"
            f"---END_ATTACHED_IMAGE-{uid}---"
        )
        response = (
            "Screenshot attached for reference:\n" + image_fence
            + "\n\nProceeding with the fix:\n" + bash_block(uid, "echo fixed")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "echo fixed")])
        self.assertNotIn("ATTACHED_IMAGE", blocks[0][1])

    def test_blocks_around_image_fence_keep_document_order(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        image_fence = (
            f"---START_ATTACHED_IMAGE-{uid}---\nX\n---END_ATTACHED_IMAGE-{uid}---"
        )
        response = (
            bash_block(uid, "echo A") + "\n" + image_fence + "\n"
            + python_block(uid, "print('B')")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(
            blocks, [("BASH", "echo A"), ("PYTHON", "print('B')")]
        )

    def test_valid_block_wins_over_stale_uuid_neighbor(self):
        """If at least one strict match exists, the relaxed fallback must not
        fire; the stale-UUID block (e.g. from a previous --resume session) is
        simply not returned."""
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            bash_block(uid, "echo good")
            + "\n\n(Stale block from the previous session:)\n"
            + bash_block(stale, "echo stale")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "echo good")])

    def test_prose_mentioning_fence_syntax_with_other_uuid_is_ignored(self):
        """Prose that merely mentions fence syntax bound to a different UUID
        must not be extracted (strict pattern is session-UUID-bound)."""
        uid = str(uuid.uuid4())
        other = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            f"To run a command, emit `---START_BASH_COMMAND-{other}---` ... "
            f"---END_BASH_COMMAND-{other}--- and wait for output.\n"
            + bash_block(uid, "echo real")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "echo real")])


if __name__ == "__main__":
    unittest.main()
