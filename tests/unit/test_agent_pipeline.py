"""
Group 3 — Execution Pipeline tests for parse_and_execute / _execute_script.

T-12  Full happy-path turn (P0)
T-13  MAX_CODE_BLOCKS enforcement (P0)
T-14  Attached-image fence extraction (P0)
T-15  /tmp/ failure warning heuristic (P1)
T-17  Scratchpad co-commit ordering (P1)

These tests pin the smallest end-to-end proof that
parse -> dispatch -> format -> commit works as a unit:

  * A protocol-valid LLM response containing one fenced block is parsed,
  * dispatched to the right sandbox method (execute vs execute_python),
  * the raw result is wrapped in a protocol-valid OUTPUT fence carrying the
    exit code and VISIBLE_% header, and
  * exactly one user message is committed to the context containing that
    fence (plus whatever scratchpad co-commit logic adds).

All tests are offline:
  * Agent construction goes through helpers.fakes._make_agent (no network
    probe, FakeSandbox injected).
  * The sandbox is a FakeSandbox returning canned (exit_code, output) tuples;
    real-sandbox behavior is covered separately by Group 7 integration tests.

Every test runs inside a throwaway CWD (chdir_tmp) and captures stdout so the
production prints ([System] ..., colored output blocks) stay out of test runs.
"""
import contextlib
import io
import unittest
import uuid
from unittest import mock

from bash_agent.agent import _build_tmp_file_warning

from tests.helpers.fakes import (
    bash_block,
    python_block,
    output_block,
    attached_image_block,
    chdir_tmp,
    _make_agent,
    FakeSandbox,
)


class PipelineCase(unittest.TestCase):
    """Shared harness: temp CWD + captured stdout + ready-made agent."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.uid = str(uuid.uuid4())
        # Capture production prints; individual tests may assert on them.
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.agent = _make_agent(uuid_str=self.uid)

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def user_messages(self):
        """All committed user-role messages, in order."""
        return [m for m in self.agent.context.history if m["role"] == "user"]


# ---------------------------------------------------------------------------
# T-12 — Full happy-path turn
# ---------------------------------------------------------------------------

class TestFullHappyPathTurn(PipelineCase):
    def test_single_bash_block_round_trip(self):
        """One valid bash block -> sandbox executes it verbatim -> the commit
        carries an EXIT_CODE_0 OUTPUT fence and the call returns (True, "")."""
        fake_sb = FakeSandbox(execute_result=(0, "hello"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "echo hello")
        )

        # Return contract
        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        # Dispatch: the sandbox saw the script exactly as fenced
        self.assertEqual(fake_sb.executed_scripts, ["echo hello"])
        self.assertEqual(fake_sb.executed_python_scripts, [])

        # Commit: exactly ONE user message appended (system prompt is index 0)
        users = self.user_messages()
        self.assertEqual(len(users), 1)

        content = users[0]["content"]
        # The formatted output block must appear verbatim, exit code intact
        self.assertIn(output_block(self.uid, 0, "hello"), content)
        self.assertIn("EXIT_CODE_0", content)
        self.assertIn("VISIBLE_100%", content)
        # ...and the closing fence must match the session UUID
        self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", content)

    def test_single_python_block_dispatches_to_execute_python(self):
        """Same happy path, PYTHON flavor: proves cmd_type drives the choice
        between sandbox.execute and sandbox.execute_python."""
        fake_sb = FakeSandbox(execute_python_result=(0, "42"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            python_block(self.uid, "print(42)")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(fake_sb.executed_python_scripts, ["print(42)"])
        self.assertEqual(fake_sb.executed_scripts, [])

        users = self.user_messages()
        self.assertEqual(len(users), 1)
        self.assertIn(
            output_block(self.uid, 0, "42", cmd_type="PYTHON"),
            users[0]["content"],
        )
        self.assertIn("EXIT_CODE_0", users[0]["content"])

    def test_nonzero_exit_code_is_preserved_in_fence(self):
        """A failing command still commits feedback — the model needs the
        EXIT_CODE_N header to react — and returns (True, "")."""
        fake_sb = FakeSandbox(execute_result=(2, "ls: cannot access 'x'"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "ls x")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        users = self.user_messages()
        self.assertEqual(len(users), 1)
        self.assertIn(
            output_block(self.uid, 2, "ls: cannot access 'x'"),
            users[0]["content"],
        )
        self.assertIn("EXIT_CODE_2", users[0]["content"])

    def test_empty_output_still_commits_a_fenced_block(self):
        """Silent success (`touch file`) must still produce a well-formed
        empty OUTPUT fence so the model knows the command succeeded."""
        fake_sb = FakeSandbox(execute_result=(0, ""))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "touch file.txt")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        content = self.user_messages()[0]["content"]
        self.assertIn(
            f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---",
            content,
        )
        self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", content)


# ---------------------------------------------------------------------------
# T-13 — MAX_CODE_BLOCKS enforcement
# ---------------------------------------------------------------------------

class TestMaxCodeBlocksEnforcement(PipelineCase):
    """T-13: runaway multi-execution guard.

    parse_and_execute slices ``blocks[:MAX_CODE_BLOCKS]`` and appends a cutoff
    warning whenever ``len(blocks)`` exceeds the limit. The production module
    binds the constant with ``from bash_agent.config import MAX_CODE_BLOCKS``,
    so the default-limit tests below run UNPATCHED (config ships MAX=1) while
    the mirror tests patch ``bash_agent.agent.MAX_CODE_BLOCKS`` directly —
    patching ``bash_agent.config.MAX_CODE_BLOCKS`` would be a silent no-op.
    """

    def test_default_limit_one_executes_only_first_block(self):
        """Stock MAX_CODE_BLOCKS=1: two valid blocks -> only the first runs;
        the commit carries its OUTPUT fence plus the 'first 1 of 2' warning."""
        fake_sb = FakeSandbox(execute_result=(0, "first"))
        self.agent.sandbox = fake_sb

        response = "\n\n".join(
            [bash_block(self.uid, "echo first"), bash_block(self.uid, "echo second")]
        )
        executed, feedback = self.agent.parse_and_execute(response)

        # Return contract is unchanged by the cutoff
        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        # Exactly one sandbox call, and it was the FIRST fenced script
        self.assertEqual(fake_sb.executed_scripts, ["echo first"])

        # Commit: a single user message holding output + warning, in order
        users = self.user_messages()
        self.assertEqual(len(users), 1)
        content = users[0]["content"]

        self.assertIn(output_block(self.uid, 0, "first"), content)
        self.assertIn("Only the first 1 of 2", content)
        self.assertIn("The remaining 1 block(s) were skipped", content)
        self.assertIn("at most 1 code block(s)", content)

        out_idx = content.index(output_block(self.uid, 0, "first"))
        warn_idx = content.index("[SYSTEM WARNING]")
        self.assertLess(out_idx, warn_idx)

        # No OUTPUT fence may exist for the skipped block
        self.assertEqual(
            content.count(
                f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---"
            ),
            1,
        )

    def test_skipped_python_block_never_reaches_sandbox(self):
        """Mixed BASH+PYTHON response: enforcement happens BEFORE dispatch, so
        execute_python() must never be called for the dropped block."""
        fake_sb = FakeSandbox(execute_result=(0, "bash ran"))
        self.agent.sandbox = fake_sb

        response = "\n\n".join(
            [
                bash_block(self.uid, "echo bash ran"),
                python_block(self.uid, "print('py')"),
            ]
        )
        executed, feedback = self.agent.parse_and_execute(response)

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(fake_sb.executed_scripts, ["echo bash ran"])
        self.assertEqual(fake_sb.executed_python_scripts, [])

        content = self.user_messages()[0]["content"]
        self.assertIn("Only the first 1 of 2", content)
        # The skipped python source must not leak into the commit either
        self.assertNotIn("print('py')", content)

    def test_limit_is_read_dynamically_when_patched_to_two(self):
        """Mirror test: patch bash_agent.agent.MAX_CODE_BLOCKS to 2 -> BOTH
        blocks execute and no warning fires. Proves the limit is consulted at
        call time rather than baked into the slice."""
        fake_sb = FakeSandbox(execute_result=[(0, "one"), (0, "two")])
        self.agent.sandbox = fake_sb

        response = "\n\n".join(
            [bash_block(self.uid, "echo one"), bash_block(self.uid, "echo two")]
        )
        with mock.patch("bash_agent.agent.MAX_CODE_BLOCKS", 2):
            executed, feedback = self.agent.parse_and_execute(response)

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(fake_sb.executed_scripts, ["echo one", "echo two"])

        content = self.user_messages()[0]["content"]
        self.assertNotIn("SYSTEM WARNING", content)
        self.assertIn(output_block(self.uid, 0, "one"), content)
        self.assertIn(output_block(self.uid, 0, "two"), content)

    def test_limit_above_block_count_stays_warning_free(self):
        """Raising the limit PAST the block count must also suppress the
        warning — pins the ``len(blocks) > MAX`` comparison direction."""
        fake_sb = FakeSandbox(execute_result=[(0, "a"), (0, "b")])
        self.agent.sandbox = fake_sb

        response = "\n\n".join(
            [bash_block(self.uid, "echo a"), bash_block(self.uid, "echo b")]
        )
        with mock.patch("bash_agent.agent.MAX_CODE_BLOCKS", 5):
            executed, feedback = self.agent.parse_and_execute(response)

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(len(fake_sb.executed_scripts), 2)
        self.assertNotIn("SYSTEM WARNING", self.user_messages()[0]["content"])


# ---------------------------------------------------------------------------
# T-14 — Attached-image fence extraction
# ---------------------------------------------------------------------------

class TestAttachedImageFenceExtraction(PipelineCase):
    """T-14: exercises the exact injection path the ``vision`` tool relies on.

    When a sandboxed tool prints ``---START_ATTACHED_IMAGE-{uuid}---`` fences
    (vision.py multimodal mode), the pipeline must:

      * strip fences + payload from the DISPLAYED output,
      * append the ``[Image attached to conversation context.]`` note,
      * collect ``{"url": ...}`` dicts into ``agent._pending_multimodal_images``,
      * and, at commit time, convert them into OpenAI-style structured content
        (a leading text part followed by ``image_url`` parts) and clear pending.

    Fences bearing a DIFFERENT session UUID must be left completely untouched.
    """

    DATA_URL = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def _tool_output(self):
        """Simulate a tool mixing prose with a vision-style image fence."""
        return "\n".join(
            [
                "Captured screenshot.",
                attached_image_block(self.uid, self.DATA_URL),
                "Analysis queued.",
            ]
        )

    def test_payload_stripped_note_added_pending_populated(self):
        """_execute_script level: displayed output loses the base64 payload and
        gains the attachment note; pending queue receives the data URL."""
        fake_sb = FakeSandbox(execute_result=(0, self._tool_output()))
        self.agent.sandbox = fake_sb

        formatted = self.agent._execute_script("BASH", "vision shot.png")

        # Surrounding prose survives the strip
        self.assertIn("Captured screenshot.", formatted)
        self.assertIn("Analysis queued.", formatted)

        # Payload + fences are gone from the DISPLAYED output entirely
        self.assertNotIn(self.DATA_URL, formatted)
        self.assertNotIn("base64", formatted)
        self.assertNotIn("ATTACHED_IMAGE", formatted)

        # Attachment note added AFTER the trailing prose (inside OUTPUT fence)
        self.assertIn("[Image attached to conversation context.]", formatted)
        self.assertGreater(
            formatted.index("[Image attached"),
            formatted.index("Analysis queued."),
        )

        # Exit-code / VISIBLE headers intact
        self.assertIn(f"EXIT_CODE_0-VISIBLE_100%-{self.uid}", formatted)

        # Pending queue holds exactly the extracted URL object
        self.assertEqual(
            self.agent._pending_multimodal_images, [{"url": self.DATA_URL}]
        )

    def test_full_turn_commits_structured_multimodal_message(self):
        """End-to-end: parse_and_execute commits ONE user message whose content
        is a LIST: [{type: text, ...}, {type: image_url, ...}]; the pending
        queue is drained at commit time."""
        fake_sb = FakeSandbox(execute_result=(0, self._tool_output()))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "vision shot.png")
        )

        # Return contract unchanged by multimodal handling
        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        users = self.user_messages()
        self.assertEqual(len(users), 1)
        content = users[0]["content"]

        # Multimodal wire format: a LIST of typed parts
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        # Part 0 - text carrying the OUTPUT fence + note, sans any payload
        self.assertEqual(content[0]["type"], "text")
        text_part = content[0]["text"]
        self.assertIn(
            f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{self.uid}---",
            text_part,
        )
        self.assertIn(f"---END_BASH_OUTPUT-{self.uid}---", text_part)
        self.assertIn("[Image attached to conversation context.]", text_part)
        self.assertNotIn(self.DATA_URL, text_part)
        self.assertNotIn("ATTACHED_IMAGE", text_part)

        # Part 1 - image_url part wrapping the data URL verbatim
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"], {"url": self.DATA_URL})

        # Queue drained exactly once, at commit time
        self.assertEqual(self.agent._pending_multimodal_images, [])

    def test_python_flavor_attaches_identically(self):
        """A PYTHON block whose sandbox output embeds the fence takes the same
        strip -> note -> pending -> image_url-part path."""
        url = "data:image/png;base64,PYTHONPAYLOAD="
        fake_sb = FakeSandbox(
            execute_python_result=(0, attached_image_block(self.uid, url))
        )
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            python_block(self.uid, "print(open('shot.png', 'rb').read())")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        content = self.user_messages()[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": url}})
        self.assertNotIn(url, content[0]["text"])
        self.assertNotIn("ATTACHED_IMAGE", content[0]["text"])
        self.assertEqual(self.agent._pending_multimodal_images, [])

    def test_multiple_images_commit_in_document_order(self):
        """Two fences in one output -> two image_url parts, order preserved."""
        url_a = "data:image/png;base64,AAAA"
        url_b = "data:image/jpeg;base64,BBBB"
        out = "\n".join(
            [
                attached_image_block(self.uid, url_a),
                attached_image_block(self.uid, url_b),
            ]
        )
        fake_sb = FakeSandbox(execute_result=(0, out))
        self.agent.sandbox = fake_sb

        executed, _ = self.agent.parse_and_execute(bash_block(self.uid, "vision a b"))

        self.assertTrue(executed)
        content = self.user_messages()[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        self.assertEqual(
            image_parts,
            [
                {"type": "image_url", "image_url": {"url": url_a}},
                {"type": "image_url", "image_url": {"url": url_b}},
            ],
        )
        # Text part still present as part 0; payload never leaks anywhere
        self.assertEqual(content[0]["type"], "text")
        for p in content:
            blob = p["text"] if p["type"] == "text" else str(p)
            self.assertNotIn("ATTACHED_IMAGE", blob)

    def test_foreign_uuid_fence_is_never_consumed(self):
        """An ATTACHED_IMAGE fence bearing a DIFFERENT session UUID must not be
        stripped, must not populate pending, and leaves the commit a plain
        string message (no multimodal wrapping)."""
        foreign_uuid = str(uuid.uuid4())
        foreign_fence = attached_image_block(foreign_uuid, self.DATA_URL)
        fake_sb = FakeSandbox(execute_result=(0, f"before\n{foreign_fence}\nafter"))
        self.agent.sandbox = fake_sb

        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "cat other_session.png")
        )

        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        # No images pending -> classic string commit path
        self.assertEqual(self.agent._pending_multimodal_images, [])
        content = self.user_messages()[0]["content"]
        self.assertIsInstance(content, str)

        # Foreign fence passes through untouched, payload visible
        self.assertIn(foreign_fence, content)
        self.assertIn(self.DATA_URL, content)


# ---------------------------------------------------------------------------
# T-15 — /tmp/ failure warning heuristic
# ---------------------------------------------------------------------------

class TestTmpFileWarningHeuristic(unittest.TestCase):
    """Parametrized matrix over Agent._build_tmp_file_warning.

    Pure-function test per TEST_PLAN T-15: direct import, zero mocks, zero
    filesystem. The heuristic must fire ONLY when ALL three conditions hold:

      1. the command exited non-zero,
      2. the output references a "/tmp/" path, and
      3. the output contains a familiar file-not-found style error phrase.

    Each condition individually violated must yield None so successful runs,
    unrelated failures, and /tmp-free errors never get nagged.
    """

    # Phrasings drawn from every branch of agent._TMP_ERROR_PATTERN's
    # alternation (case-insensitive).
    ERROR_PHRASES = [
        "No such file or directory",
        "not found",
        "command not found",
        "does not exist",
        "cannot open",
        "cannot find",
        "cannot access",
        "cannot locate",
        "could not open",
        "could not find",
        "unable to open",
        "unable to locate",
        "failed to open",
        "failed to read",
        "failed to write",
    ]

    def test_all_conditions_met_returns_warning(self):
        """exit!=0 + '/tmp/' + error phrase -> a warning string."""
        for phrase in self.ERROR_PHRASES:
            with self.subTest(phrase=phrase):
                out = f"tool: /tmp/session_state.json: {phrase}"
                warning = _build_tmp_file_warning(exit_code=1, output=out)
                self.assertIsNotNone(warning)
                # The reminder must name the wiped directory AND the fix.
                self.assertIn("/tmp/", warning)
                self.assertIn(".bash_agent_tmp/", warning)

    def test_warning_fires_for_various_exit_codes(self):
        """Any non-zero exit code qualifies; 0 never does."""
        out = "cat: /tmp/notes.txt: No such file or directory"
        for code in (1, 2, 127, 255):
            with self.subTest(exit_code=code):
                self.assertIsNotNone(_build_tmp_file_warning(code, out))
        self.assertIsNone(_build_tmp_file_warning(0, out))

    def test_success_with_missing_file_is_silent(self):
        """Condition 1 violated: exit 0 + /tmp/ + error phrase -> None.

        A command may print 'No such file' while probing and still succeed;
        nagging then would be noise."""
        out = (
            "if [ -f /tmp/cache.bin ]; then cat /tmp/cache.bin; "
            "fi\ncat: /tmp/cache.bin: No such file or directory\ndone"
        )
        self.assertIsNone(_build_tmp_file_warning(0, out))

    def test_failure_without_tmp_path_is_silent(self):
        """Condition 2 violated: exit!=0 + error phrase but NO '/tmp/' -> None."""
        outputs = [
            "cat: config.yaml: No such file or directory",
            "ls: cannot access './missing': No such file or directory",
            "grep: pattern not found",
        ]
        for out in outputs:
            with self.subTest(output=out):
                self.assertIsNone(_build_tmp_file_warning(1, out))

    def test_bare_tmp_without_slash_is_silent(self):
        """/tmp without the trailing slash does not count as a path reference."""
        out = "see notes under /tmp for details: No such file"
        self.assertIsNone(_build_tmp_file_warning(1, out))

    def test_failure_without_error_phrase_is_silent(self):
        """Condition 3 violated: exit!=0 + '/tmp/' but no known phrase -> None."""
        outputs = [
            "/tmp/out.log: Permission denied",
            "head: error reading /tmp/big.log: Input/output error",
            "usage: tool [--flag] /tmp/input.dat",
            "",
        ]
        for out in outputs:
            with self.subTest(output=out):
                self.assertIsNone(_build_tmp_file_warning(1, out))

    def test_phrase_and_path_may_appear_on_different_lines(self):
        """The regex scans the whole output, so a multi-line traceback whose
        '/tmp/' reference and error phrase sit on separate lines still fires."""
        out = (
            "Traceback (most recent call last):\n"
            "  File \"script.py\", line 3, in <module>\n"
            "    data = open('/tmp/state.json').read()\n"
            "FileNotFoundError: [Errno 2] No such file or directory"
        )
        warning = _build_tmp_file_warning(1, out)
        self.assertIsNotNone(warning)
        self.assertIn(".bash_agent_tmp/", warning)

    def test_matching_is_case_insensitive(self):
        """_TMP_ERROR_PATTERN compiles with re.IGNORECASE; SHOUTING errors
        must trigger the same rescue as lowercase ones."""
        out = "TOOL: COULD NOT FIND /tmp/model.bin"
        self.assertIsNotNone(_build_tmp_file_warning(1, out))

    def test_warning_text_is_actionable(self):
        """The returned reminder tells the model exactly what happened and
        what to do instead — this is coaching, not just logging."""
        warning = _build_tmp_file_warning(
            1, "cat: /tmp/x: No such file or directory"
        )
        self.assertIn("SYSTEM WARNING", warning)
        # Must explain WHY it failed (non-persistent /tmp/) and WHERE to write
        # session-persistent files instead.
        self.assertIn("does not persist", warning)
        self.assertIn(".bash_agent_tmp/", warning)


# ---------------------------------------------------------------------------
# T-17 — Scratchpad co-commit ordering
# ---------------------------------------------------------------------------

class TestScratchpadCoCommitOrdering(PipelineCase):
    """
    When the scratchpad file changes during a turn, _commit_execution_feedback
    must prepend a fresh SCRATCHPAD block to the committed user message and
    strip older scratchpad fences from prior messages, so the model always
    sees exactly one, current copy of its notes.

    The "old-format" fence is produced by the real
    ContextManager.get_scratchpad_block(), so this test fails if the emitter
    and remove_old_scratchpads() ever drift apart.
    """

    def _write_scratchpad(self, text):
        with open(self.agent.context.scratchpad_path, "w") as f:
            f.write(text)

    def _seed_prior_turn(self, scratch_text, marker):
        """Seed a commit-shaped prior user message: scratchpad fence + output."""
        self._write_scratchpad(scratch_text)
        block = self.agent.context.get_scratchpad_block()
        self.assertTrue(block.startswith("\n---START_SCRATCHPAD.md-"))
        prior = block + "\n" + output_block(self.uid, 0, marker)
        self.agent.context.add_message("user", prior)
        return prior

    def test_changed_scratchpad_prepends_new_block_and_strips_old(self):
        """Main scenario: prior message carries an old fence; the file changes;
        one turn later the old fence is gone and the new fence appears exactly
        once, prepended to the committed output."""
        self._seed_prior_turn(
            "# Project Scratchpad\n\nOLD_STATE v1\n", "earlier result"
        )

        # The model edits the scratchpad mid-test...
        self._write_scratchpad("# Project Scratchpad\n\nFRESH_STATE v2\n")

        self.agent.sandbox = FakeSandbox(execute_result=(0, "ok"))
        executed, feedback = self.agent.parse_and_execute(
            bash_block(self.uid, "echo ok")
        )

        # Return contract
        self.assertTrue(executed)
        self.assertEqual(feedback, "")

        users = self.user_messages()
        self.assertEqual(len(users), 2)

        # Old fence (and its payload) stripped from the prior message; the
        # sibling output block must survive the strip untouched.
        self.assertNotIn("---START_SCRATCHPAD.md-", users[0]["content"])
        self.assertNotIn("OLD_STATE", users[0]["content"])
        self.assertIn(output_block(self.uid, 0, "earlier result"), users[0]["content"])

        # Exactly ONE scratchpad fence across the whole history, carrying the
        # fresh content and VISIBLE_100%, prepended ahead of this turn's output.
        total = sum(u["content"].count("---START_SCRATCHPAD.md-") for u in users)
        self.assertEqual(total, 1)
        expected_block = (
            f"\n---START_SCRATCHPAD.md-VISIBLE_100%-{self.uid}---\n"
            f"# Project Scratchpad\n\nFRESH_STATE v2\n"
            f"\n---END_SCRATCHPAD.md-{self.uid}---\n"
        )
        self.assertTrue(users[1]["content"].startswith(expected_block))
        self.assertIn("FRESH_STATE", users[1]["content"])
        self.assertIn(output_block(self.uid, 0, "ok"), users[1]["content"])

    def test_unchanged_scratchpad_is_not_recommitted(self):
        """Hash caching through the public pipeline: when the file did NOT
        change between turns, no SCRATCHPAD block is co-committed — and the
        previous turn's block is left alone (no strip without a replacement)."""
        self._write_scratchpad("# Project Scratchpad\n\nsession state v1\n")
        self.agent.sandbox = FakeSandbox(execute_result=(0, "one"))

        executed, _ = self.agent.parse_and_execute(bash_block(self.uid, "echo one"))
        self.assertTrue(executed)
        users = self.user_messages()
        self.assertEqual(len(users), 1)
        self.assertIn("---START_SCRATCHPAD.md-", users[0]["content"])

        # Turn 2 WITHOUT touching the file: hash cache must suppress co-commit.
        self.agent.sandbox.queue_execute(0, "two")
        executed, _ = self.agent.parse_and_execute(bash_block(self.uid, "echo two"))
        self.assertTrue(executed)
        users = self.user_messages()
        self.assertEqual(len(users), 2)
        self.assertNotIn("---START_SCRATCHPAD.md-", users[1]["content"])
        self.assertIn(output_block(self.uid, 0, "two"), users[1]["content"])
        # ...and the earlier block survives (no strip happened).
        self.assertIn("---START_SCRATCHPAD.md-", users[0]["content"])

    def test_all_prior_scratchpads_stripped_leaving_single_latest(self):
        """Multiple stale fences (legacy/buggy state) are ALL removed; only the
        newest copy survives after the turn."""
        for i, tag in enumerate(("STALE_A", "STALE_B")):
            self._seed_prior_turn(f"# Project Scratchpad\n\n{tag}\n", f"prior {i}")

        self._write_scratchpad("# Project Scratchpad\n\nLATEST\n")
        self.agent.sandbox = FakeSandbox(execute_result=(0, "done"))
        executed, _ = self.agent.parse_and_execute(bash_block(self.uid, "echo done"))
        self.assertTrue(executed)

        users = self.user_messages()
        self.assertEqual(len(users), 3)
        for m in users[:2]:
            self.assertNotIn("---START_SCRATCHPAD.md-", m["content"])
            self.assertNotIn("STALE_", m["content"])
        total = sum(u["content"].count("---START_SCRATCHPAD.md-") for u in users)
        self.assertEqual(total, 1)
        self.assertIn("LATEST", users[2]["content"])


if __name__ == "__main__":
    unittest.main()
