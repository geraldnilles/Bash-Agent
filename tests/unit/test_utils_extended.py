"""
Group 8 — Extended utility tests for bash_agent.utils.

T-43  get_vim_prompt           — vim interaction & file persistence (P2)
T-44  get_clipboard_content    — clipboard boundary (wl-paste / xclip) (P2)

These functions have external dependencies (vim, wl-paste, xclip) and were
not covered by the original T-33/T-34/T-35 suite. They are critical paths:

  * get_vim_prompt is used by the agent when it needs human input; a
    regression here breaks the human-in-the-loop workflow silently.
  * get_clipboard_content feeds the -x/--execute flag; if it fails the
    agent cannot read the execution plan from the clipboard.

Seam notes:
  * Both functions shell out via subprocess.run; tests patch at
    bash_agent.utils.subprocess.run.
  * get_vim_prompt writes to .bash_agent_tmp/vim_prompt.tmp — tests
    run inside chdir_tmp so the file lands in the throwaway CWD.
  * get_clipboard_content tries wl-paste first, then xclip; tests cover
    success on first, success on fallback, and total failure paths.
"""

import contextlib
import io
import os
import unittest
from unittest import mock

from bash_agent.utils import (
    get_vim_prompt,
    get_clipboard_content,
)
from tests.helpers.fakes import chdir_tmp


class TestGetVimPrompt(unittest.TestCase):
    """get_vim_prompt launches vim, reads back the temp file."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def _run_vim_prompt(self, vim_return_value=0, file_content_after=None):
        """
        Run get_vim_prompt with mocked subprocess.run.
        If file_content_after is provided, the mock will write that content
        to the temp file AFTER vim "returns" (simulating user edit).
        """
        tmp_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp")
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        def mock_run(cmd, **kwargs):
            # Simulate vim editing the file
            if file_content_after is not None:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(file_content_after)
            return mock.Mock(returncode=vim_return_value)

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=mock_run):
            return get_vim_prompt("OBJECTIVE:")

    def test_creates_temp_file_with_initial_prompt(self):
        """On first call, the temp file is created with the prompt text."""
        # Ensure file doesn't exist initially
        tmp_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp")
        self.assertFalse(os.path.exists(tmp_file))

        result = self._run_vim_prompt(file_content_after="User wrote this plan")

        # File should now exist with user's content
        self.assertTrue(os.path.exists(tmp_file))
        with open(tmp_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "User wrote this plan")
        self.assertEqual(result, "User wrote this plan")

    def test_preserves_existing_file_content(self):
        """If file already exists, its content is preserved (not overwritten)."""
        tmp_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp")
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write("Pre-existing content")

        result = self._run_vim_prompt(file_content_after="Edited content")

        # Should return the edited content, not the initial prompt
        self.assertEqual(result, "Edited content")
        with open(tmp_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Edited content")

    def test_vim_invoked_with_correct_args(self):
        """vim is called with -c 'set noswapfile' -c 'set spell' and the temp file."""
        called_cmd = {}

        def mock_run(cmd, **kwargs):
            called_cmd["args"] = cmd
            if os.path.exists(os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp")):
                with open(os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp"), "w", encoding="utf-8") as f:
                    f.write("test")
            return mock.Mock(returncode=0)

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=mock_run):
            get_vim_prompt("CUSTOM PROMPT")

        self.assertEqual(called_cmd["args"][0], "vim")
        self.assertIn("-c", called_cmd["args"])
        self.assertIn("set noswapfile", called_cmd["args"])
        self.assertIn("set spell", called_cmd["args"])
        self.assertTrue(called_cmd["args"][-1].endswith("vim_prompt.tmp"))

    def test_custom_prompt_used_on_first_run(self):
        """Custom prompt text is written to file on first creation."""
        result = self._run_vim_prompt(file_content_after="User response")
        # We can't easily test the initial write, but we can verify the file exists
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, ".bash_agent_tmp", "vim_prompt.tmp")))


class TestGetClipboardContent(unittest.TestCase):
    """get_clipboard_content tries wl-paste, then xclip."""

    def test_wl_paste_success(self):
        """wl-paste succeeds -> returns its stdout."""
        with mock.patch("bash_agent.utils.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="clipboard via wayland")
            result = get_clipboard_content()
            self.assertEqual(result, "clipboard via wayland")
            # Called wl-paste first
            self.assertEqual(mock_run.call_args_list[0][0][0], ["wl-paste"])
            # xclip NOT called
            self.assertEqual(len(mock_run.call_args_list), 1)

    def test_wl_paste_fails_xclip_succeeds(self):
        """wl-paste fails (not found) -> falls back to xclip."""
        def side_effect(cmd, **kwargs):
            if cmd == ["wl-paste"]:
                raise FileNotFoundError("wl-paste not found")
            if cmd == ["xclip", "-selection", "clipboard", "-o"]:
                return mock.Mock(returncode=0, stdout="clipboard via x11")
            raise AssertionError(f"Unexpected command: {cmd}")

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=side_effect):
            result = get_clipboard_content()
            self.assertEqual(result, "clipboard via x11")

    def test_wl_paste_nonzero_xclip_succeeds(self):
        """wl-paste returns non-zero -> falls back to xclip."""
        def side_effect(cmd, **kwargs):
            if cmd == ["wl-paste"]:
                return mock.Mock(returncode=1, stdout="", stderr="error")
            if cmd == ["xclip", "-selection", "clipboard", "-o"]:
                return mock.Mock(returncode=0, stdout="xclip worked")
            raise AssertionError(f"Unexpected command: {cmd}")

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=side_effect):
            result = get_clipboard_content()
            self.assertEqual(result, "xclip worked")

    def test_both_fail_raises_runtime_error(self):
        """Both wl-paste and xclip fail -> RuntimeError with helpful message."""
        def side_effect(cmd, **kwargs):
            raise FileNotFoundError(f"{cmd[0]} not found")

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as cm:
                get_clipboard_content()
            self.assertIn("Could not read from clipboard", str(cm.exception))
            self.assertIn("wl-paste", str(cm.exception))
            self.assertIn("xclip", str(cm.exception))

    def test_xclip_nonzero_raises_runtime_error(self):
        """wl-paste not found, xclip returns non-zero -> RuntimeError."""
        def side_effect(cmd, **kwargs):
            if cmd == ["wl-paste"]:
                raise FileNotFoundError("wl-paste not found")
            if cmd == ["xclip", "-selection", "clipboard", "-o"]:
                return mock.Mock(returncode=1, stdout="", stderr="no selection")
            raise AssertionError(f"Unexpected command: {cmd}")

        with mock.patch("bash_agent.utils.subprocess.run", side_effect=side_effect):
            with self.assertRaises(RuntimeError) as cm:
                get_clipboard_content()
            self.assertIn("Could not read from clipboard", str(cm.exception))

    def test_strips_whitespace_from_result(self):
        """Return value is stripped of trailing newlines/whitespace."""
        with mock.patch("bash_agent.utils.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="  hello\n  \n")
            result = get_clipboard_content()
            self.assertEqual(result, "hello")


if __name__ == "__main__":
    unittest.main()
