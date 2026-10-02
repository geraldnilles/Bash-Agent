"""
Group 7 — Integration: real bubblewrap (``bwrap``) executions (still offline).

T-30  Real bwrap smoke tests (P1, skipUnless bwrap)

These tests drive a REAL Sandbox.execute()/execute_python() round-trip and
prove the isolation guarantees AGENTS.md promises:

  * `echo hello` -> exit 0 with stdout
  * `exit 3`     -> exit code 3 propagates back through subprocess.run
  * stderr is merged into stdout (`ls /nonexistent` shows the error inline)
  * reading /etc/hostname succeeds (the host root is bound read-only, so
    /etc stays visible); writing /etc/xyz fails
  * CWD visibility: project files are listable from inside the sandbox
  * BASH_AGENT_UUID is visible inside the sandbox via --setenv forwarding
  * private /tmp: host /tmp content is masked by a tmpfs
  * timeout case: `sleep 999` with timeout=2 -> exit 124 + banner, and the
    sandboxed process tree is torn down with it (--die-with-parent +
    --unshare-pid) so no runaway process leaks onto the host.
  * ORCHESTRATOR isolation: commands run (exit 0) at all, the whole project
    tree is READABLE (so `search`/`cat`/`ls` work), the scratch dir is the
    ONLY writable location (project-tree and /etc writes are refused), and
    python works. (Regression for the old isolation bug: masking the home tree
    wholesale also hid .bash_agent_tmp itself, so EVERY orchestrator command
    returned empty output. The orchestrator now shares the read-only root bind
    with every other role and differs only in its writable path.)

Leak hygiene: subprocess.run SIGKILLs ONLY the bwrap client on timeout. With
``--die-with-parent`` the container dies with it and, because the workload
runs in its own PID namespace (``--unshare-pid``), the kernel reaps every
process inside it. We still launch the leak probe with a unique marker in
argv[0] via ``exec -a <marker> sleep 999`` and verify afterwards that no
matching process survived.

Skipped automatically when the ``bwrap`` binary is absent or unprivileged
user namespaces are unavailable.
"""
import os
import subprocess
import unittest

from bash_agent.sandbox import Sandbox

from tests.helpers.fakes import chdir_repo_tmp, bwrap_available



@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestRealSandboxSmoke(unittest.TestCase):
    """T-30 — real Sandbox round-trips against the local bubblewrap install."""

    def setUp(self):
        self._chdir_cm = chdir_repo_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.sb = Sandbox("unused-scratchpad-path", uuid="test-uuid-123")

    def tearDown(self):
        self._chdir_cm.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Basic execution semantics
    # ------------------------------------------------------------------

    def test_echo_hello_exit_0_with_stdout(self):
        code, out = self.sb.execute("echo hello")
        self.assertEqual(code, 0)
        self.assertIn("hello", out)

    def test_nonzero_exit_code_propagates(self):
        code, out = self.sb.execute("exit 3")
        self.assertEqual(code, 3)

    def test_stderr_merged_into_stdout(self):
        code, out = self.sb.execute("ls /nonexistent-path-xyz")
        self.assertNotEqual(code, 0)
        self.assertIn("No such file or directory", out)

    # ------------------------------------------------------------------
    # Isolation guarantees
    # ------------------------------------------------------------------

    def test_read_only_etc_allowed_write_denied(self):
        # Reading protected paths is allowed (host root is bound read-only)...
        code, out = self.sb.execute("cat /etc/hostname")
        self.assertEqual(code, 0)
        self.assertTrue(out.strip())

        # ...but writes must fail.
        code, out = self.sb.execute("touch /etc/xyz-probe-file")
        self.assertNotEqual(code, 0)

    def test_cwd_visibility(self):
        marker = "sandbox-cwd-marker.txt"
        with open(marker, "w") as f:
            f.write("visible\n")

        code, out = self.sb.execute("ls -1")
        self.assertEqual(code, 0)
        self.assertIn(marker, out.splitlines())

    def test_uuid_env_forwarded_into_sandbox(self):
        code, out = self.sb.execute("echo $BASH_AGENT_UUID")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "test-uuid-123")

    def test_private_tmp_isolation(self):
        # bwrap mounts a throwaway tmpfs over /tmp, so host /tmp content
        # must not be visible inside the sandbox.
        sentinel = "/tmp/bash_agent_private_tmp_probe"
        with open(sentinel, "w") as f:
            f.write("host-side\n")
        try:
            code, out = self.sb.execute(
                f"if [ -f {sentinel} ]; then echo VISIBLE; else echo ISOLATED; fi"
            )
            self.assertEqual(code, 0)
            self.assertIn("ISOLATED", out)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)

    def test_python_round_trip(self):
        code, out = self.sb.execute_python("print('py-ok')")
        self.assertEqual(code, 0)
        self.assertIn("py-ok", out)

    # ------------------------------------------------------------------
    # Timeout semantics + leak hygiene
    # ------------------------------------------------------------------

    def test_timeout_maps_to_124(self):
        marker = f"bash-agent-t30-leak-{os.getpid()}"
        sb = Sandbox("unused", uuid="u", timeout=2)

        try:
            # exec -a puts the marker in argv[0]: a bash comment would
            # be stripped before exec, leaving the process unmatchable.
            code, out = sb.execute(f"exec -a {marker} sleep 999")
            # --die-with-parent + --unshare-pid must tear the whole sandbox
            # process tree down when bwrap is killed on the client timeout.
            survivors = subprocess.run(
                ["pkill", "-0", "-f", marker], capture_output=True
            )
        finally:
            # Belt-and-braces cleanup so the host stays clean regardless.
            subprocess.run(["pkill", "-9", "-f", marker], capture_output=True)

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Command timed out after 2 seconds.", out)
        # No runaway process survived the sandbox teardown.
        self.assertNotEqual(
            survivors.returncode, 0,
            msg="sandboxed process leaked past the timeout teardown",
        )


@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestOrchestratorSandboxIsolation(unittest.TestCase):
    """T-30b — real round-trips for is_orchestrator=True.

    The orchestrator shares the read-only root bind with every role (so it can
    READ the project tree / run ``search``) but its ONLY writable location is
    ``.bash_agent_tmp/``. Regression guard for the old bug where masking the
    home tree wholesale also hid the scratch dir and every orchestrator command
    returned empty output.
    """

    def setUp(self):
        # Run from the REPO ROOT (like production): the orchestrator's writable
        # path is the cwd-relative ``.bash_agent_tmp/``, so a cwd nested inside
        # ``.bash_agent_tmp/`` would resolve it incorrectly.
        self._orig = os.getcwd()
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        os.chdir(self.repo_root)
        self.scratch = os.path.join(self.repo_root, ".bash_agent_tmp")
        self.sb = Sandbox("unused-scratchpad-path", uuid="orch-uuid", is_orchestrator=True)

    def tearDown(self):
        os.chdir(self._orig)

    def test_orchestrator_commands_actually_run(self):
        # The headline regression: this used to return (200, "").
        code, out = self.sb.execute("echo orchestrator-alive")
        self.assertEqual(code, 0, msg=f"orchestrator sandbox failed: {out!r}")
        self.assertIn("orchestrator-alive", out)

    def test_orchestrator_can_write_and_read_scratch_dir(self):
        probe = os.path.join(self.scratch, "_orch_probe.txt")
        code, out = self.sb.execute(
            f"printf 'hello-from-orch' > {probe} && cat {probe} && rm -f {probe}"
        )
        self.assertEqual(code, 0, msg=f"scratch write/read failed: {out!r}")
        self.assertIn("hello-from-orch", out)

    def test_orchestrator_python_round_trip(self):
        code, out = self.sb.execute_python("print('orch-py-ok')")
        self.assertEqual(code, 0, msg=f"orchestrator python failed: {out!r}")
        self.assertIn("orch-py-ok", out)

    def test_orchestrator_can_read_project_tree(self):
        # The orchestrator shares the read-only root bind: project files are
        # READABLE (so `cat`/`ls`/`grep`/`search` all work), unlike the old
        # tmpfs-masked home tree.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        sentinel = os.path.join(repo_root, "orch_visible_probe.txt")
        with open(sentinel, "w") as f:
            f.write("project\n")
        try:
            code, out = self.sb.execute(
                f"if [ -e {sentinel} ]; then cat {sentinel}; else echo HIDDEN; fi"
            )
            self.assertEqual(code, 0)
            self.assertIn("project", out)
            self.assertNotIn("HIDDEN", out)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)

    def test_orchestrator_cannot_write_project_tree(self):
        # Writes outside .bash_agent_tmp/ must fail: only the scratch dir is
        # punched back in read/write.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        target = os.path.join(repo_root, "orch_write_probe.txt")
        code, out = self.sb.execute(
            f"touch {target} 2>&1 || echo WRITE_DENIED"
        )
        self.assertEqual(code, 0)
        self.assertIn("WRITE_DENIED", out)
        self.assertFalse(os.path.exists(target), "orchestrator wrote the repo tree")

    def test_orchestrator_run_from_project_root(self):
        # Workdir is the project root (not .bash_agent_tmp/), so relative paths
        # resolve against the real project like any other role.
        code, out = self.sb.execute("pwd")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), os.path.abspath("."))


@unittest.skipUnless(bwrap_available(), "bwrap unavailable")
class TestOrchestratorVisionAvailability(unittest.TestCase):
    """T-30c — the `vision` CLI and package source are reachable read-only.

    The orchestrator masks /home with a tmpfs, but the project ``venv/`` and
    ``bash_agent/`` package source are re-bound READ-ONLY for BOTH
    ``execute()`` (bash) and ``execute_python()``. These checks must run from
    the REPO ROOT (where ``./venv`` exists), unlike the scratch-dir-based T-30b
    tests, so we chdir there for the duration and restore afterwards.
    """

    def setUp(self):
        self._orig = os.getcwd()
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        os.chdir(self.repo_root)
        self.sb = Sandbox("unused", uuid="orch-vision", is_orchestrator=True)

    def tearDown(self):
        os.chdir(self._orig)

    def test_vision_cli_on_path_for_bash(self):
        code, out = self.sb.execute("command -v vision || echo NO_VISION")
        self.assertEqual(code, 0, msg=f"bash failed: {out!r}")
        self.assertIn("vision", out)
        self.assertNotIn("NO_VISION", out)

    def test_bash_agent_package_importable_for_bash(self):
        code, out = self.sb.execute(
            "python3 -c \'import bash_agent.sandbox; print(\"PKG_IMPORT_OK\")\'"
        )
        self.assertEqual(code, 0, msg=f"import failed: {out!r}")
        self.assertIn("PKG_IMPORT_OK", out)

    def test_venv_and_package_binds_are_read_only(self):
        # Writes into the re-bound paths must fail: isolation is preserved.
        code, out = self.sb.execute(
            "touch ./bash_agent/__probe__.txt 2>&1 || echo PKG_RO_OK; "
            "touch ./venv/__probe__.txt 2>&1 || echo VENV_RO_OK"
        )
        self.assertEqual(code, 0)
        self.assertIn("PKG_RO_OK", out)
        self.assertIn("VENV_RO_OK", out)


if __name__ == "__main__":
    unittest.main()
