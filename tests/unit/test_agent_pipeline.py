"""
Group 3 — Execution Pipeline tests for parse_and_execute / _execute_script.

T-12  Full happy-path turn (P0)
T-13  MAX_CODE_BLOCKS enforcement (P0)
T-14  Attached-image fence extraction (P0)

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


if __name__ == "__main__":
    unittest.main()
