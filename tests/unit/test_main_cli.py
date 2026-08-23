"""
Group 8 — CLI entry-point tests for bash_agent.main.

T-37  main.parse_args / main() flag mapping (P1)

main.py translates CLI flags into an Agent construction + run(task) call,
with three early-exit branches that must fire BEFORE any LLM machinery:

  * --commit       -> rewrites itself into resume=True +
                      message="Commit the change."
  * -x/--execute   -> clipboard -> .bash_agent_tmp/SCRATCHPAD.md, then a
                      canned "execute the plan" objective
  * -s             -> wipes SCRATCHPAD.md (after task selection)
  * --copy-project -> copies to clipboard and sys.exit(0) without ever
                      constructing an Agent

These are regression guards for the human-facing contract: if --commit
stops implying --resume, a commit request would start a FRESH session and
lose history; if --copy-project ever reached Agent.__init__ it would wipe
.bash_agent_tmp/ (cleanup_tmp_folder) before copying.

Seam notes:
  * bash_agent.main.Agent is patched wholesale (class-level MagicMock), so
    no capability probe / sandbox / context is ever built — fully offline.
  * copy_project_to_clipboard and get_clipboard_content are patched at
    their bash_agent.main binding sites (imported names).
  * Filesystem effects (-x, -s) resolve paths against the process CWD at
    call time, so each such test runs inside chdir_tmp (T-00a).
  * sys.argv is patched per test; SystemExit from argparse/copy-project is
    caught with assertRaises.
"""

import contextlib
import io
import os
import unittest
from unittest import mock

from bash_agent import main as main_module
from tests.helpers.fakes import chdir_tmp


def run_cli(argv, clipboard_value=None, clipboard_error=None):
    """
    Invoke main_module.main() under patched sys.argv / collaborators.

    Returns (exit_exc_or_None, agent_cls_mock, stdout_text, stderr_text).
    """
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with mock.patch.object(main_module, "get_clipboard_content") as get_clip:
        if clipboard_error is not None:
            get_clip.side_effect = RuntimeError(clipboard_error)
        else:
            get_clip.return_value = clipboard_value
        with mock.patch.object(main_module, "copy_project_to_clipboard") as cp:
            with mock.patch.object(main_module, "Agent") as fake_agent_cls:
                with mock.patch("sys.argv", argv):
                    with contextlib.redirect_stdout(stdout_buf), \
                            contextlib.redirect_stderr(stderr_buf):
                        try:
                            main_module.main()
                            exc = None
                        except SystemExit as e:
                            exc = e
    return {
        "exit": exc,
        "agent_cls": fake_agent_cls,
        "copy_project": cp,
        "clipboard_getter": get_clip,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
    }


class TestParseArgDefaults(unittest.TestCase):
    """Pure argparse surface — no side effects."""

    def parse(self, *argv):
        with mock.patch("sys.argv", ["bagent", *argv]):
            return main_module.parse_args()

    def test_defaults(self):
        args = self.parse()
        self.assertIsNone(args.message)
        self.assertFalse(args.paste)
        self.assertFalse(args.execute)
        self.assertFalse(args.clear_scratchpad)
        self.assertFalse(args.keep_tmp)
        self.assertFalse(args.debug)
        self.assertIsNone(args.model)
        self.assertIsNone(args.reasoning_effort)
        self.assertIsNone(args.max_tokens)
        self.assertIsNone(args.timeout)
        self.assertEqual(args.budget, 0.10)
        self.assertFalse(args.commit)
        self.assertFalse(args.resume)
        self.assertFalse(args.copy_project)
        self.assertIsNone(args.files)

    def test_flag_mapping(self):
        args = self.parse("-m", "do things",
                          "-k", "-d",
                          "--model", "vendor/model-x",
                          "--reasoning-effort", "high",
                          "--max-tokens", "2048",
                          "-t", "45",
                          "-b", "2.50")
        self.assertEqual(args.message, "do things")
        self.assertTrue(args.keep_tmp)
        self.assertTrue(args.debug)
        self.assertEqual(args.model, "vendor/model-x")
        self.assertEqual(args.reasoning_effort, "high")
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.timeout, 45)
        self.assertEqual(args.budget, 2.50)

    def test_reasoning_effort_rejects_unknown_choice(self):
        # argparse errors exit(2); pins the choices whitelist.
        result = run_cli(["bagent", "--reasoning-effort", "maximum"])
        self.assertIsInstance(result["exit"], SystemExit)
        self.assertEqual(result["exit"].code, 2)


class TestPlainRun(unittest.TestCase):
    """Default path: -m message -> Agent(kwargs) -> run(message)."""

    def test_message_passed_and_default_agent_kwargs(self):
        res = run_cli(["bagent", "-m", "list files"])
        self.assertIsNone(res["exit"])
        res["agent_cls"].assert_called_once_with(
            keep_tmp=False, debug=False, model=None,
            reasoning_effort=None, max_tokens=None, timeout=None,
            resume=False, budget=0.10)
        res["agent_cls"].return_value.run.assert_called_once_with(
            "list files")

    def test_no_input_source_yields_run_none(self):
        # No -m/-p/-x: initial_task stays None; agent still runs.
        res = run_cli(["bagent"])
        res["agent_cls"].return_value.run.assert_called_once_with(None)

    def test_paste_reads_clipboard_into_initial_task(self):
        res = run_cli(["bagent", "--paste"],
                      clipboard_value="Pasted objective text")
        self.assertIsNone(res["exit"])
        res["agent_cls"].return_value.run.assert_called_once_with(
            "Pasted objective text")
        self.assertIn("[Reading from clipboard]", res["stdout"])

    def test_paste_clipboard_failure_degrades_to_none_task(self):
        # -p swallows RuntimeError and continues with initial_task=None.
        res = run_cli(["bagent", "-p"], clipboard_error="no clipboard tool")
        self.assertIsNone(res["exit"])
        res["agent_cls"].assert_called_once()
        res["agent_cls"].return_value.run.assert_called_once_with(None)
        self.assertIn("[Clipboard Error] no clipboard tool", res["stdout"])

    def test_budget_and_timeout_forwarded(self):
        res = run_cli(["bagent", "-m", "go", "-b", "1.25", "-t", "90"])
        _, kwargs = res["agent_cls"].call_args
        self.assertEqual(kwargs["budget"], 1.25)
        self.assertEqual(kwargs["timeout"], 90)

    def test_resume_flag_forwarded(self):
        res = run_cli(["bagent", "-m", "continue", "-r"])
        _, kwargs = res["agent_cls"].call_args
        self.assertTrue(kwargs["resume"])


class TestCommitFlag(unittest.TestCase):
    """--commit rewrites itself into a resume session with a canned task."""

    def test_commit_implies_resume_and_canned_message(self):
        res = run_cli(["bagent", "--commit"])
        self.assertIsNone(res["exit"])
        _, kwargs = res["agent_cls"].call_args
        self.assertTrue(kwargs["resume"])
        res["agent_cls"].return_value.run.assert_called_once_with(
            "Commit the change.")

    def test_explicit_resume_and_message_flags_still_accepted_alongside(self):
        # --commit overrides -m; pins the precedence order.
        res = run_cli(["bagent", "--commit", "-m", "unrelated"])
        res["agent_cls"].return_value.run.assert_called_once_with(
            "Commit the change.")


class TestExecuteFlag(unittest.TestCase):
    """-x: clipboard -> SCRATCHPAD.md + canned execute-the-plan objective."""

    OBJECTIVE = ("OBJECTIVE: Please read the plan detailed in the "
                 "SCRATCHPAD.md file and execute it step-by-step.")

    def test_clipboard_plan_written_to_scratchpad(self):
        with chdir_tmp():
            res = run_cli(["bagent", "-x"], clipboard_value="# Plan\nstep one")
            self.assertIsNone(res["exit"])
            scratch = os.path.join(os.getcwd(), ".bash_agent_tmp",
                                   "SCRATCHPAD.md")
            with open(scratch, encoding="utf-8") as f:
                self.assertEqual(f.read(), "# Plan\nstep one")
            res["agent_cls"].return_value.run.assert_called_once_with(
                self.OBJECTIVE)
            self.assertIn("[System] Clipboard content written to SCRATCHPAD.md",
                          res["stdout"])

    def test_overwrites_existing_scratchpad_content(self):
        with chdir_tmp():
            scratch_dir = os.path.join(os.getcwd(), ".bash_agent_tmp")
            os.makedirs(scratch_dir, exist_ok=True)
            scratch = os.path.join(scratch_dir, "SCRATCHPAD.md")
            with open(scratch, "w", encoding="utf-8") as f:
                f.write("stale previous plan")
            run_cli(["bagent", "-x"], clipboard_value="fresh plan")
            with open(scratch, encoding="utf-8") as f:
                self.assertEqual(f.read(), "fresh plan")

    def test_clipboard_failure_exits_1_without_building_agent(self):
        with chdir_tmp():
            res = run_cli(["bagent", "-x"], clipboard_error="wayland missing")
            self.assertIsInstance(res["exit"], SystemExit)
            self.assertEqual(res["exit"].code, 1)
            res["agent_cls"].assert_not_called()
            self.assertIn("[Clipboard Error] wayland missing", res["stdout"])
            self.assertFalse(
                os.path.exists(os.path.join(os.getcwd(), ".bash_agent_tmp")))


class TestClearScratchpad(unittest.TestCase):
    """-s wipes SCRATCHPAD.md before the Agent is constructed."""

    def test_clears_preexisting_scratchpad(self):
        with chdir_tmp():
            scratch_dir = os.path.join(os.getcwd(), ".bash_agent_tmp")
            os.makedirs(scratch_dir, exist_ok=True)
            scratch = os.path.join(scratch_dir, "SCRATCHPAD.md")
            with open(scratch, "w", encoding="utf-8") as f:
                f.write("old notes that must vanish")

            res = run_cli(["bagent", "-m", "hello", "-s"])

            self.assertIsNone(res["exit"])
            with open(scratch, encoding="utf-8") as f:
                self.assertEqual(f.read(), "")
            self.assertIn("[System] SCRATCHPAD.md cleared.", res["stdout"])
            res["agent_cls"].return_value.run.assert_called_once_with("hello")

    def test_creates_empty_scratchpad_when_absent(self):
        with chdir_tmp():
            res = run_cli(["bagent", "-m", "hello", "--clear-scratchpad"])
            self.assertIsNone(res["exit"])
            scratch = os.path.join(os.getcwd(), ".bash_agent_tmp",
                                   "SCRATCHPAD.md")
            self.assertTrue(os.path.exists(scratch))
            with open(scratch, encoding="utf-8") as f:
                self.assertEqual(f.read(), "")


class TestCopyProjectFlag(unittest.TestCase):
    """--copy-project short-circuits: clipboard then exit(0), no Agent."""

    def test_copy_project_exits_before_agent_construction(self):
        res = run_cli(["bagent", "--copy-project"])
        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 0)
        res["copy_project"].assert_called_once_with(None)
        # Guard: constructing an Agent here would be a real bug (it wipes
        # .bash_agent_tmp/ before the copy completes).
        res["agent_cls"].assert_not_called()
        self.assertIn("Project copied to clipboard. Exiting.", res["stdout"])

    def test_files_subset_forwarded(self):
        res = run_cli(["bagent", "--copy-project", "--files",
                       "README.md,src/app.py"])
        self.assertEqual(res["exit"].code, 0)
        res["copy_project"].assert_called_once_with("README.md,src/app.py")


if __name__ == "__main__":
    unittest.main()
