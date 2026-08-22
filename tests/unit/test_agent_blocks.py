"""
Group 1 — Protocol Parsing tests for agent._extract_blocks.

T-01  Valid bash and python block extraction (P0)
T-02  Mixed prose and multiple blocks (P1)
T-03  No-block response yields coaching warning (P1)
T-04  Malformed UUID triggers relaxed-pattern rescue (P0)
T-05  Cross-type fence mismatch rejected (P2)

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
    output_block,
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


class TestMalformedUuidRescue(unittest.TestCase):
    """T-04: A block with correct fence structure but a wrong/stale UUID
    (e.g. left over from a previous --resume session) triggers the relaxed
    fallback: nothing is parsed for execution, and a coaching warning embeds
    a CORRECTED block bound to the live session UUID carrying the model's
    original script. This path is the difference between a recoverable
    hiccup and a stuck session after --resume."""

    def _agent(self, uid):
        return _make_agent(uuid_str=uid)

    def test_stale_uuid_bash_block_is_rescued(self):
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks(bash_block(stale, "echo hi"))
        # Nothing is extracted for execution from a malformed block...
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        # ...but the model is handed a corrected fence bound to the LIVE uuid
        # carrying the exact script it wrote.
        self.assertIn(f"---START_BASH_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_BASH_COMMAND-{uid}---", warning)
        self.assertIn("echo hi", warning)
        # The stale UUID must not leak into the proposal: the model is told
        # to copy the corrected block verbatim, and a stale fence would never
        # match the strict parser.
        self.assertNotIn(stale, warning)

    def test_corrected_block_in_warning_round_trips_through_parser(self):
        """Self-consistency: copying the proposed block verbatim MUST parse
        under the strict pattern. If it did not, the rescue would coach the
        model straight into another dead end."""
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        script = "git status --short"
        _, warning = agent._extract_blocks(bash_block(stale, script))
        corrected = (
            f"---START_BASH_COMMAND-{uid}---\n{script}\n---END_BASH_COMMAND-{uid}---"
        )
        self.assertIn(corrected, warning)
        blocks, err = agent._extract_blocks(corrected)
        self.assertIsNone(err)
        self.assertEqual(blocks, [("BASH", script)])

    def test_stale_uuid_python_block_rescue_preserves_multiline_script(self):
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        code = "import sys\nprint(sys.version)\nfor i in range(3):\n    print(i)"
        blocks, warning = agent._extract_blocks(python_block(stale, code))
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self.assertIn(f"---START_PYTHON_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_PYTHON_COMMAND-{uid}---", warning)
        self.assertIn(code, warning)

    def test_missing_uuid_entirely_is_also_rescued(self):
        """A fence with NO uuid at all still trips the relaxed fallback."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = "---START_BASH_COMMAND---\necho bare\n---END_BASH_COMMAND---"
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self.assertIn(f"---START_BASH_COMMAND-{uid}---", warning)
        self.assertIn("echo bare", warning)

    def test_warning_names_the_problem_and_states_live_uuid(self):
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        _, warning = agent._extract_blocks(bash_block(stale, "ls"))
        self.assertIn("Malformed command block detected", warning)
        self.assertIn("UUID was missing or incorrect", warning)
        self.assertIn(f"The current session UUID is: {uid}", warning)
        self.assertIn("re-evaluate", warning)

    def test_only_first_malformed_block_is_proposed(self):
        """With several malformed blocks the rescue proposes only the first
        (relaxed_matches[0]). Pinning the single-proposal contract keeps a
        future multi-block proposal a deliberate change, not an accident."""
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        response = bash_block(stale, "echo one") + "\n\n" + bash_block(stale, "echo two")
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self.assertIn("echo one", warning)
        self.assertNotIn("echo two", warning)

    def test_inline_single_line_fence_falls_through_to_default_coaching(self):
        """The relaxed pattern requires the script body to start on its own
        line; a whole fence collapsed onto one line is NOT rescued and gets
        the generic no-block coaching instead."""
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            f"---START_BASH_COMMAND-{stale}--- echo inline "
            f"---END_BASH_COMMAND-{stale}---"
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self.assertIn("You did not provide any code block", warning)


class TestMalformedUuidRescuePipeline(unittest.TestCase):
    """T-04 (pipeline view): through parse_and_execute, a malformed-UUID
    response must be pure feedback — nothing executed, nothing committed."""

    def test_no_execution_and_no_commit_on_rescue_path(self):
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        history_len_before = len(agent.context.history)
        executed, feedback = agent.parse_and_execute(
            bash_block(stale, "echo should-not-run")
        )
        self.assertFalse(executed)
        self.assertIn(f"---START_BASH_COMMAND-{uid}---", feedback)
        self.assertEqual(agent.sandbox.executed_scripts, [])
        self.assertEqual(len(agent.context.history), history_len_before)

class TestNoBlockWarning(unittest.TestCase):
    """T-03: A response with no command fences returns ([], warning) where the
    warning embeds BOTH example fence templates bound to the LIVE session UUID.

    This warning is the LLM's only self-correction signal, so its exactness
    matters: a stale or missing UUID here silently breaks recovery — the model
    would copy a fence the parser can never match."""

    def _agent(self, uid):
        return _make_agent(uuid_str=uid)

    def _assert_coaching_warning(self, warning, uid):
        """Pin the coaching-warning contract shared by every no-block path."""
        self.assertIsInstance(warning, str)
        self.assertTrue(warning.strip(), "warning must not be blank")
        # Names the failure so the model understands what went wrong.
        self.assertIn("You did not provide any code block", warning)
        # Both example templates present, interpolated with the live UUID...
        self.assertIn(f"---START_BASH_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_BASH_COMMAND-{uid}---", warning)
        self.assertIn(f"---START_PYTHON_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_PYTHON_COMMAND-{uid}---", warning)
        # ...at least once each (the four fences above).
        self.assertGreaterEqual(warning.count(uid), 4)
        # Placeholder bodies intact so the model knows what goes inside.
        self.assertIn("[bash commands go here]", warning)
        self.assertIn("[python code goes here]", warning)
        # Coaching guidance: how to proceed and how to end the session.
        self.assertIn("exit", warning)

    def test_plain_prose_returns_empty_blocks_and_warning(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks(
            "I think I should start by listing the files, let me do that."
        )
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self._assert_coaching_warning(warning, uid)

    def test_empty_response_returns_warning(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks("")
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self._assert_coaching_warning(warning, uid)

    def test_whitespace_only_response_returns_warning(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks("\n\n   \t \n")
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self._assert_coaching_warning(warning, uid)

    def test_attached_image_only_response_returns_warning(self):
        """ATTACHED_IMAGE is a foreign fence type, not a command block; a
        response containing only image attachments still gets coached."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        image_fence = (
            f"---START_ATTACHED_IMAGE-{uid}---\n"
            "data:image/png;base64,iVBORw0KGgo=\n"
            f"---END_ATTACHED_IMAGE-{uid}---"
        )
        blocks, warning = agent._extract_blocks(image_fence)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self._assert_coaching_warning(warning, uid)

    def test_output_fence_echo_returns_warning(self):
        """A model echoing OUTPUT fences back (instead of emitting a command)
        must be coached, not parsed into executable blocks."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        echoed = output_block(uid, 0, "sandbox hello")
        blocks, warning = agent._extract_blocks(echoed)
        self.assertEqual(blocks, [])
        self.assertIsNotNone(warning)
        self._assert_coaching_warning(warning, uid)

    def test_warning_uuid_tracks_the_live_session_not_a_stale_one(self):
        """Two agents with different UUIDs receive differently-interpolated
        warnings — proving dynamic interpolation rather than a stale or
        hard-coded template (the --resume failure mode)."""
        uid_a = str(uuid.uuid4())
        uid_b = str(uuid.uuid4())
        self.assertNotEqual(uid_a, uid_b)
        warning_a = self._agent(uid_a)._extract_blocks("no fences here")[1]
        warning_b = self._agent(uid_b)._extract_blocks("no fences here")[1]
        self.assertIn(uid_a, warning_a)
        self.assertNotIn(uid_b, warning_a)
        self.assertIn(uid_b, warning_b)
        self.assertNotIn(uid_a, warning_b)

    def test_warning_contains_no_uninterpolated_placeholders(self):
        """The braces-style template variables must never leak through raw."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        _, warning = agent._extract_blocks("just chatting, no code today")
        self.assertNotIn("{self.uuid}", warning)
        self.assertNotIn("{uuid}", warning)


class TestCrossTypeFenceMismatch(unittest.TestCase):
    """T-05: A response where START says BASH but END says PYTHON must not
    match the strict pattern (backreference \\1) nor crash the relaxed
    fallback. Pins the regex backreference behavior so nobody "simplifies"
    it into accepting mismatched pairs.

    Two contracts are pinned here:

    1. REJECTION — an isolated cross-type pair matches neither pattern and
       falls through to the default coaching warning. The relaxed fallback
       must NOT fire for it: its whole purpose is rescuing *UUID* typos,
       not papering over structural garbage that would execute the wrong
       thing if "rescued".

    2. NO SPANNING — a mismatched pair must never glue onto a later
       same-type END fence to form one spanning match whose "script" is a
       chunk of prose laced with fence markers. Before the tempered body
       was added to both patterns, `START_BASH ... END_PYTHON ... prose ...
       START_BASH ... END_BASH` parsed as ONE BASH block containing the
       intervening fences and prose — i.e. the agent would have executed
       markdown as shell.
    """

    def _agent(self, uid):
        return _make_agent(uuid_str=uid)

    # -- helpers ----------------------------------------------------------

    def _assert_coaching_warning(self, warning, uid):
        """The shared no-block coaching contract (same wording T-03 pins)."""
        self.assertIsInstance(warning, str)
        self.assertTrue(warning.strip())
        self.assertIn("You did not provide any code block", warning)
        self.assertIn(f"---START_BASH_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_BASH_COMMAND-{uid}---", warning)
        self.assertIn(f"---START_PYTHON_COMMAND-{uid}---", warning)
        self.assertIn(f"---END_PYTHON_COMMAND-{uid}---", warning)

    def _assert_not_rescue_warning(self, warning):
        """The relaxed fallback must stay silent on cross-type mismatches:
        it exists to correct UUIDs only."""
        self.assertIsNotNone(warning)
        self.assertNotIn("Malformed command block detected", warning)
        self.assertNotIn("Did you mean to execute this?", warning)

    def _fence(self, start_type, end_type, uid, script):
        return (
            f"---START_{start_type}_COMMAND-{uid}---\n"
            f"{script}\n"
            f"---END_{end_type}_COMMAND-{uid}---"
        )

    # -- contract 1: isolated mismatch is rejected -------------------------

    def test_bash_start_python_end_is_rejected(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks(
            self._fence("BASH", "PYTHON", uid, "echo hi")
        )
        self.assertEqual(blocks, [])
        self._assert_not_rescue_warning(warning)
        self._assert_coaching_warning(warning, uid)

    def test_python_start_bash_end_is_rejected(self):
        """Symmetric case: the backreference must bind in both directions."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks(
            self._fence("PYTHON", "BASH", uid, "print(1)")
        )
        self.assertEqual(blocks, [])
        self._assert_not_rescue_warning(warning)
        self._assert_coaching_warning(warning, uid)

    def test_mismatch_with_stale_uuid_is_rejected_without_rescue(self):
        """A wrong UUID AND crossed END type must not be 'rescued' either —
        two independent malformations do not compose into a recovery path."""
        uid = str(uuid.uuid4())
        stale = str(uuid.uuid4())
        agent = self._agent(uid)
        blocks, warning = agent._extract_blocks(
            self._fence("BASH", "PYTHON", stale, "echo hi")
        )
        self.assertEqual(blocks, [])
        self._assert_not_rescue_warning(warning)
        self._assert_coaching_warning(warning, uid)

    def test_mismatch_script_never_leaks_into_any_warning(self):
        """Whatever the fallback says, it must NOT propose the mismatched
        script for execution — proposing it would coach the model into
        re-sending structurally broken fences forever."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        _, warning = agent._extract_blocks(
            self._fence("BASH", "PYTHON", uid, "echo poisoned")
        )
        self.assertNotIn("echo poisoned", warning)

    def test_mismatch_inside_prose_is_rejected(self):
        """Prose around the pair changes nothing: the strict pattern still
        refuses, and the model gets coached rather than executed."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            "Let me check the layout first:\n"
            + self._fence("BASH", "PYTHON", uid, "ls -la") + "\n"
            "That should show us what we need."
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self._assert_coaching_warning(warning, uid)

    # -- contract 2: no spanning across fences -----------------------------

    def test_mismatch_plus_valid_block_extracts_only_valid_block(self):
        """Regression guard: before the tempered-body fix, this input parsed
        as ONE spanning BASH block whose script contained the END_PYTHON
        fence, prose, and the second START fence. Only the well-formed block
        may be extracted; the mismatched one must vanish cleanly."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            self._fence("BASH", "PYTHON", uid, "echo bad") + "\n"
            "prose between\n"
            + bash_block(uid, "echo good")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("BASH", "echo good")])

    def test_compensating_pair_does_not_span_into_one_block(self):
        """START_BASH..END_PYTHON followed by START_PYTHON..END_BASH must not
        merge into a single 'BASH' block spanning both fences."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            self._fence("BASH", "PYTHON", uid, "echo a") + "\n"
            + self._fence("PYTHON", "BASH", uid, "print(1)")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self._assert_coaching_warning(warning, uid)

    def test_two_independent_mismatches_do_not_span(self):
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            self._fence("BASH", "PYTHON", uid, "echo one") + "\n"
            + self._fence("PYTHON", "BASH", uid, "print(2)")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertEqual(blocks, [])
        self._assert_coaching_warning(warning, uid)

    def test_valid_block_after_mismatch_pair_parses_normally(self):
        """The compensating-pair shape followed by a real block: only the
        real block survives, nothing spans, nothing executes from the junk."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        response = (
            self._fence("BASH", "PYTHON", uid, "echo a") + "\n"
            + self._fence("PYTHON", "BASH", uid, "print(1)") + "\n"
            + python_block(uid, "print('ok')")
        )
        blocks, warning = agent._extract_blocks(response)
        self.assertIsNone(warning)
        self.assertEqual(blocks, [("PYTHON", "print('ok')")])

    def test_backreference_is_load_bearing_for_both_types(self):
        """Directly pin WHY the backreference matters: replacing END_\\1_
        with a fixed literal would make every block parse as whichever type
        the START named. A same-type pair still parses; a crossed pair does
        not — in both directions, at any position in the document."""
        uid = str(uuid.uuid4())
        agent = self._agent(uid)
        ok_bash = bash_block(uid, "true")
        ok_py = python_block(uid, "pass")
        bad = self._fence("BASH", "PYTHON", uid, "false")
        for text, expected in (
            (ok_bash, [("BASH", "true")]),
            (ok_py, [("PYTHON", "pass")]),
            (bad, []),
        ):
            blocks, warning = agent._extract_blocks(text)
            self.assertEqual(blocks, expected)
            self.assertEqual(warning is None, bool(expected))

if __name__ == "__main__":
    unittest.main()
