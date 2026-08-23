"""
Group 8 — Supporting-module tests for bash_agent.utils.

T-33  is_binary_file            – extension table + content sniffing (P1)
T-34  cleanup_tmp_folder        – whitelist survival contract (P0)
T-35  copy_project_to_clipboard – filtering & clipboard boundary (P1)

These three pin the utility layer that session startup and the
copy-to-clipboard / --copy-project features depend on:

  * is_binary_file gates which files ever reach the clipboard; a false
    negative leaks binary garbage into the LLM context, a false positive
    silently drops source files from the copy.
  * cleanup_tmp_folder runs on every Agent startup; an over-broad delete
    destroys user sessions (scratchpad/history), an under-broad one leaks
    sandbox debris into the project directory.
  * copy_project_to_clipboard's ignore/blacklist/binary filters and its
    <file path="..."> wrapping (with COPY_PROJECT_PREFIX/SUFFIX) are the
    contract consumed by main.py --copy-project and the agent special
    command of the same name.

Seam notes:
  * Every test runs inside chdir_tmp (T-00a): all three functions resolve
    paths against the process CWD at call time (.bash_agent_tmp/,
    .gitignore, clipboard blacklist).
  * The clipboard boundary is subprocess-based (tree / wl-copy / xclip), so
    T-35 patches subprocess.run with a recorder fake — fully offline — and
    asserts on the exact payload handed to the clipboard writer.
"""

import contextlib
import io
import os
import subprocess
import unittest
from unittest import mock

from bash_agent.prompts import COPY_PROJECT_PREFIX, COPY_PROJECT_SUFFIX
from bash_agent.utils import (
    cleanup_tmp_folder,
    copy_project_to_clipboard,
    is_binary_file,
)
from tests.helpers.fakes import chdir_tmp

# The survival set pinned by T-34 — must mirror cleanup_tmp_folder's inline
# whitelist. If the source list changes, this constant flags the drift.
WHITELIST = [
    "SCRATCHPAD.md",
    "vim_prompt.tmp",
    "ROLE.md",
    "embeddings.json",
    "search_disabled",
    "history.json",
    "clipboard_blacklist.txt",
]


def make_file(rel_path, data, binary=None):
    """Create <CWD>/rel_path (parents included).

    Mode defaults to binary for bytes payloads and utf-8 text otherwise;
    passing binary explicitly overrides (str + binary=True is encoded).
    """
    path = os.path.join(os.getcwd(), rel_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if binary is None:
        binary = isinstance(data, bytes)
    if binary:
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    return path


class UtilsCase(unittest.TestCase):
    """Shared harness: throwaway CWD + captured stdout."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    def stdout_text(self):
        return self.stdout_buf.getvalue()


# ---------------------------------------------------------------------------
# T-33 — is_binary_file
# ---------------------------------------------------------------------------

class TestIsBinaryFile(UtilsCase):
    """Extension table wins first; content sniffing covers the rest."""

    def test_known_binary_extensions_are_detected(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
                    ".zip", ".tar", ".gz", ".mp3", ".opus", ".mp4",
                    ".exe", ".dll", ".so", ".bin"):
            with self.subTest(ext=ext):
                path = make_file("artifact" + ext, b"whatever")
                self.assertTrue(is_binary_file(path))

    def test_extension_check_is_case_insensitive(self):
        path = make_file("photo.PNG", b"not really a png")
        self.assertTrue(is_binary_file(path))

    def test_null_byte_within_first_1024_bytes_is_binary(self):
        path = make_file("sneaky.txt", b"A" * 512 + b"\x00" + b"B" * 10,
                         binary=True)
        self.assertTrue(is_binary_file(path))

    def test_null_byte_after_first_1024_bytes_is_not_binary(self):
        # Only the first 1024 bytes are inspected by contract; a null byte
        # past that window must NOT flag the file.
        path = make_file("late_null.txt",
                         b"A" * 1024 + b"\x00" + b"B" * 16, binary=True)
        self.assertFalse(is_binary_file(path))

    def test_plain_text_file_is_not_binary(self):
        path = make_file("notes.txt", "just some plain text\n" * 20)
        self.assertFalse(is_binary_file(path))

    def test_empty_file_is_not_binary(self):
        path = make_file("empty.txt", "", binary=True)
        self.assertFalse(is_binary_file(path))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root can read chmod-000 files")
    def test_unreadable_file_is_treated_as_binary(self):
        # Unreadable -> treated cautiously -> True (fail closed).
        path = make_file("locked.txt", "harmless text")
        os.chmod(path, 0o000)
        try:
            self.assertTrue(is_binary_file(path))
        finally:
            os.chmod(path, 0o644)


# ---------------------------------------------------------------------------
# T-34 — cleanup_tmp_folder whitelist
# ---------------------------------------------------------------------------

class TestCleanupTmpFolder(UtilsCase):

    def test_whitelisted_files_survive_with_content_intact(self):
        contents = {}
        for name in WHITELIST:
            body = "keep-me: " + name
            contents[name] = body
            make_file(os.path.join(".bash_agent_tmp", name), body)

        cleanup_tmp_folder()

        for name in WHITELIST:
            with self.subTest(name=name):
                path = os.path.join(self.tmpdir, ".bash_agent_tmp", name)
                self.assertTrue(os.path.isfile(path), name)
                with open(path, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), contents[name])

    def test_junk_files_and_nested_dirs_are_removed(self):
        tmp = os.path.join(self.tmpdir, ".bash_agent_tmp")
        make_file(os.path.join(".bash_agent_tmp", "junk.txt"), "junk")
        make_file(os.path.join(".bash_agent_tmp", "old_run.log"), "log")
        make_file(os.path.join(".bash_agent_tmp", "stale/inner.txt"), "in")
        make_file(os.path.join(".bash_agent_tmp", "stale/deeper/x.txt"), "x")

        cleanup_tmp_folder()

        self.assertFalse(os.path.exists(os.path.join(tmp, "junk.txt")))
        self.assertFalse(os.path.exists(os.path.join(tmp, "old_run.log")))
        self.assertFalse(os.path.exists(os.path.join(tmp, "stale")))

    def test_missing_tmp_folder_is_a_safe_no_op(self):
        # No .bash_agent_tmp at all: must not raise and must not create it.
        cleanup_tmp_folder()
        self.assertFalse(
            os.path.exists(os.path.join(self.tmpdir, ".bash_agent_tmp")))


# ---------------------------------------------------------------------------
# T-35 — copy_project_to_clipboard filtering
# ---------------------------------------------------------------------------

class ClipboardRecorder:
    """
    subprocess.run stand-in recording every call.

    - "tree"    -> canned successful stdout (no real dependency)
    - "wl-copy" -> succeeds (or fails when fail_wl_copy=True), captures the
                   `input=` payload as the clipboard contents
    - "xclip"   -> succeeds, captures `input=` likewise (fallback path)
    """

    def __init__(self, tree_stdout="fake-tree-output\n", fail_wl_copy=False):
        self.calls = []
        self.clipboard = []
        self.tree_stdout = tree_stdout
        self.fail_wl_copy = fail_wl_copy

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        self.calls.append((cmd, kwargs))
        name = cmd[0]
        check = kwargs.get("check", False)
        rc, out, err = 0, "", ""
        if name == "tree":
            rc, out = 0, self.tree_stdout
        elif name == "wl-copy":
            if self.fail_wl_copy:
                rc, err = 1, "boom"
            else:
                self.clipboard.append(kwargs.get("input"))
        elif name == "xclip":
            self.clipboard.append(kwargs.get("input"))
        completed = subprocess.CompletedProcess(cmd, rc,
                                                stdout=out, stderr=err)
        if check and rc != 0:
            raise subprocess.CalledProcessError(rc, cmd, output=out,
                                                stderr=err)
        return completed

    def called_names(self):
        return [cmd[0] for cmd, _ in self.calls]

    def clipboard_text(self):
        return "\n\n".join(self.clipboard)


class CopyProjectCase(UtilsCase):
    """Builds a representative project tree in the throwaway CWD."""

    def setUp(self):
        super().setUp()
        make_file("a.txt", "alpha contents\n")
        make_file("nested/deep.txt", "deep contents\n")
        make_file("secret.txt", "top secret\n")          # gitignored exactly
        make_file("debug.log", "log line\n")             # gitignored *.log
        make_file("blob.bin", b"\x00\x01binary-ish", binary=True)
        make_file("photo.png", b"\x89PNG-fake", binary=True)
        make_file("blacklisted.txt", "nope\n")           # clipboard blacklist
        make_file(".gitignore", "secret.txt\n*.log\n")
        make_file(".bash_agent_tmp/internal.txt", "internal\n")
        make_file(".bash_agent_tmp/clipboard_blacklist.txt",
                  "# comment line\nblacklisted.txt\n")

    def run_copy(self, recorder, file_paths=None, ignore=None):
        with mock.patch("subprocess.run", recorder):
            copy_project_to_clipboard(file_paths=file_paths, ignore=ignore)

    def expected_block(self, rel_path):
        """The exact <file> element production must emit for rel_path."""
        with open(os.path.join(self.tmpdir, rel_path),
                  "r", encoding="utf-8") as f:
            content = f.read()
        return f'<file path="{rel_path}">\n{content}\n</file>'

    def assert_common_wrapping(self, text):
        self.assertTrue(text.startswith(COPY_PROJECT_PREFIX))
        self.assertTrue(text.endswith(COPY_PROJECT_SUFFIX))
        self.assertIn(self.expected_block("a.txt"), text)
        self.assertIn(self.expected_block("nested/deep.txt"), text)


class TestFullProjectCopy(CopyProjectCase):

    def test_wraps_files_with_prefix_suffix_and_tree_section(self):
        rec = ClipboardRecorder()
        self.run_copy(rec)

        self.assertEqual(len(rec.clipboard), 1)
        self.assertIn("tree", rec.called_names())
        self.assertIn("wl-copy", rec.called_names())
        text = rec.clipboard[0]
        self.assert_common_wrapping(text)
        self.assertIn("=== DIRECTORY TREE ===", text)
        self.assertIn("fake-tree-output", text)

    def test_excludes_gitignored_blacklisted_and_binary_files(self):
        rec = ClipboardRecorder()
        self.run_copy(rec)
        text = rec.clipboard[0]

        for absent in ("secret.txt", "debug.log", "blob.bin", "photo.png",
                       "blacklisted.txt", "internal.txt"):
            self.assertNotIn(f'<file path="{absent}">', text)
            self.assertNotIn(f'<file path="./{absent}">', text)

        # The warnings surface on stdout so the human sees why files dropped.
        out = self.stdout_text()
        self.assertIn("Warning: Binary file skipped: blob.bin", out)
        self.assertIn("Warning: Binary file skipped: photo.png", out)
        self.assertIn("Info: Ignored by clipboard blacklist: blacklisted.txt",
                      out)

    def test_gitignore_itself_is_copied(self):
        # fnmatch(".gitignore", ".git") is False: only the .git DIRECTORY is
        # ignored. Pin that the ignore file itself still reaches the LLM.
        rec = ClipboardRecorder()
        self.run_copy(rec)
        self.assertIn('<file path=".gitignore">', rec.clipboard[0])

    def test_falls_back_to_xclip_when_wl_copy_fails(self):
        rec = ClipboardRecorder(fail_wl_copy=True)
        self.run_copy(rec)

        self.assertIn("wl-copy", rec.called_names())
        self.assertIn("xclip", rec.called_names())
        self.assertEqual(len(rec.clipboard), 1)
        self.assert_common_wrapping(rec.clipboard[0])


class TestSubsetCopy(CopyProjectCase):
    """The comma-separated --files mode."""

    def test_subset_mode_copies_only_requested_files_and_skips_tree(self):
        rec = ClipboardRecorder()
        self.run_copy(rec, file_paths="a.txt, nested/deep.txt")

        self.assertNotIn("tree", rec.called_names())      # no tree in subset
        self.assertEqual(len(rec.clipboard), 1)
        text = rec.clipboard[0]
        self.assert_common_wrapping(text)
        self.assertNotIn("secret.txt", text)
        self.assertNotIn("blob.bin", text)
        self.assertNotIn("=== DIRECTORY TREE ===", text)

    def test_subset_mode_skips_missing_ignored_and_binary_with_warnings(self):
        rec = ClipboardRecorder()
        self.run_copy(rec,
                      file_paths="missing.txt, secret.txt, blob.bin, a.txt")

        text = rec.clipboard[0]
        self.assertIn('<file path="a.txt">', text)
        self.assertNotIn("missing.txt", text.split(COPY_PROJECT_PREFIX)[1])
        self.assertNotIn('<file path="secret.txt">', text)
        self.assertNotIn('<file path="blob.bin">', text)

        out = self.stdout_text()
        self.assertIn("Warning: File not found: missing.txt", out)
        self.assertIn("Warning: Ignored by .gitignore pattern: secret.txt",
                      out)
        self.assertIn("Warning: Binary file skipped: blob.bin", out)


class TestIgnorePatterns(CopyProjectCase):
    """--ignore patterns exclude files & directories from full-project and
    subset copies."""

    def test_ignore_excludes_files_and_dirs_in_full_copy(self):
        rec = ClipboardRecorder()
        self.run_copy(rec, ignore="secret.txt,*.log")

        text = rec.clipboard[0]
        self.assert_common_wrapping(text)
        # .gitignore'd + --ignore list both exclude secret.txt & debug.log
        self.assertNotIn('<file path="secret.txt">', text)
        self.assertNotIn('<file path="debug.log">', text)
        # Other files still copied
        self.assertIn('<file path="a.txt">', text)
        self.assertIn('<file path="nested/deep.txt">', text)

    def test_ignore_directory_in_full_copy(self):
        rec = ClipboardRecorder()
        self.run_copy(rec, ignore="nested")

        text = rec.clipboard[0]
        self.assertNotIn('<file path="nested/deep.txt">', text)
        self.assertIn('<file path="a.txt">', text)
        # Tree is invoked with -I for each user pattern + --prune so the
        # directory listing matches the file filter.
        tree_calls = [cmd for cmd, _ in rec.calls if cmd[0] == "tree"]
        self.assertEqual(tree_calls,
                         [["tree", "--gitignore", "-I", "nested", "--prune"]])

    def test_ignore_in_subset_copy(self):
        rec = ClipboardRecorder()
        self.run_copy(rec, file_paths="a.txt,debug.log,secret.txt",
                      ignore="*.log")

        text = rec.clipboard[0]
        self.assertIn('<file path="a.txt">', text)
        self.assertNotIn('<file path="debug.log">', text)
        self.assertNotIn('<file path="secret.txt">', text)

        out = self.stdout_text()
        self.assertIn("Warning: Ignored by --ignore pattern: debug.log", out)
        self.assertIn("Warning: Ignored by .gitignore pattern: secret.txt",
                      out)

    def test_ignore_with_spaces_and_multiple_commas(self):
        rec = ClipboardRecorder()
        self.run_copy(rec, ignore="*.log, ,debug.log ")

        text = rec.clipboard[0]
        self.assertNotIn('<file path="debug.log">', text)
        self.assertIn('<file path="a.txt">', text)


if __name__ == "__main__":
    unittest.main()
