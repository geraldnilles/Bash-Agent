"""
Group 4 — Context management tests for ContextManager.

T-22  History persistence round-trip (P0)

ContextManager.save_history()/load_history() back the `--resume` flag: the
session UUID and the full conversation history survive a process restart via
.bash_agent_tmp/history.json. Contract pinned here:

  * save writes a JSON object with EXACTLY the keys {"uuid", "history"} —
    the on-disk format future versions must stay able to read,
  * a fresh manager loading that file ADOPTS THE SAVED UUID (not its own):
    Agent.__init__ constructs ContextManager with a brand-new random UUID
    and relies on load_history() to re-bind the persisted one; every fence
    a resumed session emits depends on this,
  * both content shapes survive the JSON round trip exactly: plain strings
    AND list-form multimodal content ([{"type": "text", ...},
    {"type": "image_url", ...}]) — pruning and _content_length depend on
    the list shape still being a list after reload,
  * load resets last_scratchpad_hash to None so the scratchpad re-injects
    on the first post-resume turn even though the FILE is unchanged (a new
    process has never seen it),
  * a missing file returns False and leaves state untouched (fresh session),
  * a corrupt or keyless file returns False, prints
    "[System Error] Failed to load history: ..." to STDERR (not stdout),
    and leaves the in-memory state untouched. Regression guard: this
    handler used to raise NameError because `sys` was not imported in
    context.py, crashing the whole resume path.

Seam notes:
  * context.py binds the path via `from bash_agent.config import
    HISTORY_FILE`, so tests patch bash_agent.context.HISTORY_FILE —
    patching bash_agent.config.HISTORY_FILE would NOT be seen (same
    subtlety as CONTEXT_LIMIT / SCRATCHPAD_LIMIT in the sibling test
    modules; TEST_PLAN.md's original wording said config and was fixed).
  * ContextManager.__init__ creates .bash_agent_tmp/SCRATCHPAD.md in the
    CWD, so every test runs inside chdir_tmp (T-00a).
"""

import contextlib
import io
import json
import os
import unittest
import uuid as uuid_module
from unittest import mock

from bash_agent.context import ContextManager
from tests.helpers.fakes import chdir_tmp, output_block

LOAD_ERROR_BANNER = "[System Error] Failed to load history:"

# Realistic multimodal content part pair (same shape agent.py builds for
# image-bearing turns; ~6400 chars are accounted per image_url part).
IMAGE_PARTS = [
    {"type": "text", "text": "What does this screenshot show?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
]

# Realistic audio-bearing content (same shape agent.py builds for
# transcribe-attached turns; flat-rated in _content_length).
AUDIO_PARTS = [
    {"type": "text", "text": "Transcribe this meeting."},
    {"type": "input_audio",
     "input_audio": {"data": "SUQzBAAAAA==", "format": "mp3"}},
]


def new_uid():
    return str(uuid_module.uuid4())


class PersistenceCase(unittest.TestCase):
    """
    Shared harness: throwaway CWD (ContextManager writes the scratchpad
    there), captured stdout AND stderr (load failures must land on stderr),
    and HISTORY_FILE patched in the *context* module namespace.
    """

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()
        self.stderr_buf = io.StringIO()
        self._stderr_cm = contextlib.redirect_stderr(self.stderr_buf)
        self._stderr_cm.__enter__()
        self._patches = []

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._stderr_cm.__exit__(None, None, None)
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- helpers ------------------------------------------------------------

    def patch_history_file(self, name="history.json"):
        """Point bash_agent.context.HISTORY_FILE at <tmpdir>/<name>."""
        path = os.path.abspath(os.path.join(self.tmpdir, name))
        p = mock.patch("bash_agent.context.HISTORY_FILE", path)
        p.start()
        self._patches.append(p)
        return path

    def history_path(self):
        # Only valid after patch_history_file(); kept as a tiny accessor so
        # tests read symmetrically.
        from bash_agent.context import HISTORY_FILE
        return HISTORY_FILE


# ---------------------------------------------------------------------------
# On-disk format contract
# ---------------------------------------------------------------------------

class TestSaveShape(PersistenceCase):
    """save_history() writes exactly {uuid, history} and creates parents."""

    def test_save_writes_exactly_uuid_and_history_keys(self):
        path = self.patch_history_file()
        uid = new_uid()
        cm = ContextManager(uid)
        cm.add_message("user", "list the files")
        cm.add_message("assistant", output_block(uid, 0, "file.txt"))

        cm.save_history()

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertIsInstance(state, dict)
        self.assertEqual(set(state.keys()), {"uuid", "history"})
        self.assertEqual(state["uuid"], uid)
        self.assertEqual(state["history"], cm.history)

    def test_save_creates_missing_parent_directories(self):
        # save_history() must makedirs(dirname(HISTORY_FILE)): the first
        # resume after a manual .bash_agent_tmp wipe cannot crash on a
        # missing directory.
        path = self.patch_history_file(os.path.join("nested", "deeper", "history.json"))
        cm = ContextManager(new_uid())
        cm.add_message("user", "hello")

        cm.save_history()
        self.assertTrue(os.path.exists(path))

        cm2 = ContextManager(new_uid())
        self.assertTrue(cm2.load_history())
        self.assertEqual(cm2.history, cm.history)


# ---------------------------------------------------------------------------
# Round-trip: uuid adoption, content shapes, hash reset
# ---------------------------------------------------------------------------

class TestRoundTrip(PersistenceCase):
    """A fresh manager restores everything it needs to continue a session."""

    def test_round_trip_adopts_uuid_and_restores_both_content_shapes(self):
        self.patch_history_file()
        uid_a, uid_b = new_uid(), new_uid()

        cm_a = ContextManager(uid_a)
        cm_a.add_message("system", "You are the agent's system prompt.")
        cm_a.add_message("user", "run ls please")
        cm_a.add_message("assistant", output_block(uid_a, 0, "file.txt"))
        cm_a.add_message("user", IMAGE_PARTS)  # list-form multimodal content
        # Sentinel: load_history() must clear whatever hash the old process
        # had cached (it refers to scratchpad state the new process never saw).
        cm_a.last_scratchpad_hash = "sentinel-from-previous-process"
        expected_history = list(cm_a.history)
        cm_a.save_history()

        cm_b = ContextManager(uid_b)  # different, freshly-generated UUID
        self.assertTrue(cm_b.load_history())

        # Resume re-binding: the saved UUID wins over the constructor's.
        self.assertEqual(cm_b.uuid, uid_a)
        self.assertNotEqual(cm_b.uuid, uid_b)
        # Both fields restored exactly — strings AND the list content.
        self.assertEqual(cm_b.history, expected_history)
        list_msgs = [m for m in cm_b.history if isinstance(m["content"], list)]
        self.assertEqual(len(list_msgs), 1)
        self.assertEqual(list_msgs[0]["content"], IMAGE_PARTS)
        # Hash cache reset on load.
        self.assertIsNone(cm_b.last_scratchpad_hash)

    def test_input_audio_parts_round_trip_exactly(self):
        """A message carrying an input_audio part survives the JSON round
        trip byte-for-byte (pruning/_content_length depend on the shape)."""
        self.patch_history_file()
        uid = new_uid()
        cm_a = ContextManager(uid)
        cm_a.add_message("user", AUDIO_PARTS)
        expected = list(cm_a.history)
        cm_a.save_history()

        cm_b = ContextManager(new_uid())
        self.assertTrue(cm_b.load_history())
        self.assertEqual(cm_b.history, expected)
        list_msgs = [m for m in cm_b.history if isinstance(m["content"], list)]
        self.assertEqual(len(list_msgs), 1)
        self.assertEqual(list_msgs[0]["content"], AUDIO_PARTS)

    def test_load_resets_hash_so_scratchpad_reinjects_after_resume(self):
        # Functional consequence of the hash reset: with the scratchpad file
        # UNCHANGED across the "restart", the pre-load manager is hash-cached
        # (second call -> "") while the post-load manager emits the block
        # again — otherwise the LLM would never see its memory after resume.
        self.patch_history_file()
        uid_a, uid_b = new_uid(), new_uid()

        cm_a = ContextManager(uid_a)
        first_block = cm_a.get_scratchpad_block()
        self.assertNotEqual(first_block, "")
        cm_a.save_history()

        cm_b = ContextManager(uid_b)
        self.assertNotEqual(cm_b.get_scratchpad_block(), "")
        self.assertEqual(cm_b.get_scratchpad_block(), "")  # hash-cached now

        self.assertTrue(cm_b.load_history())
        self.assertIsNone(cm_b.last_scratchpad_hash)
        reinjected = cm_b.get_scratchpad_block()
        self.assertNotEqual(reinjected, "")
        # The re-injected block is fenced with the ADOPTED uuid, proving the
        # restored UUID flows into subsequent protocol emissions.
        self.assertIn(f"---START_SCRATCHPAD.md-VISIBLE_100%-{uid_a}", reinjected)


# ---------------------------------------------------------------------------
# Failure modes: missing and corrupt files
# ---------------------------------------------------------------------------

class TestLoadFailureModes(PersistenceCase):
    """load_history() degrades gracefully instead of killing the session."""

    def test_missing_file_returns_false_and_leaves_state_untouched(self):
        path = self.patch_history_file("never-written.json")
        self.assertFalse(os.path.exists(path))
        uid = new_uid()
        cm = ContextManager(uid)
        cm.add_message("user", "fresh start")

        self.assertFalse(cm.load_history())
        self.assertEqual(cm.uuid, uid)
        self.assertEqual(
            cm.history, [{"role": "user", "content": "fresh start"}]
        )
        # No error banner for a plain fresh session.
        self.assertNotIn(LOAD_ERROR_BANNER, self.stderr_buf.getvalue())

    def test_corrupt_or_keyless_file_returns_false_with_stderr_banner(self):
        path = self.patch_history_file()
        uid = new_uid()
        cm = ContextManager(uid)
        cm.add_message("user", "keep me")
        snapshot = list(cm.history)

        # Two flavors of unreadable state: malformed JSON (the historical
        # NameError crash site) and valid JSON missing required keys.
        for payload in ["{this is not json", "{}"]:
            with self.subTest(payload=payload):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(payload)
                self.assertFalse(cm.load_history())
                stderr = self.stderr_buf.getvalue()
                # Must go to STDERR with the exact banner prefix...
                self.assertIn(LOAD_ERROR_BANNER, stderr)
                # ...and never to stdout.
                self.assertNotIn(LOAD_ERROR_BANNER, self.stdout_buf.getvalue())
                # In-memory state survives the failed load untouched.
                self.assertEqual(cm.uuid, uid)
                self.assertEqual(cm.history, snapshot)


if __name__ == "__main__":
    unittest.main()
