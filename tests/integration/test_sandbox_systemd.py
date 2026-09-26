"""
Group 7 — Integration: real systemd-run executions (still offline).

T-30  Real systemd-run smoke tests (P1, skipUnless systemd-run)

These tests drive a REAL Sandbox.execute()/execute_python() round-trip and
prove the isolation guarantees AGENTS.md promises:

  * `echo hello` -> exit 0 with stdout
  * `exit 3`     -> exit code 3 propagates through --pipe
  * stderr is merged into stdout (`ls /nonexistent` shows the error inline)
  * reading /etc/hostname succeeds (ProtectSystem keeps /usr,/boot,/etc
    read-only, not invisible); writing /etc/xyz fails
  * CWD visibility: project files are listable from inside the sandbox
  * BASH_AGENT_UUID is visible inside the sandbox via env forwarding
  * timeout case: `sleep 999` with timeout=2 -> exit 124 + banner
  * ORCHESTRATOR isolation: commands run (exit 0) at all, the scratch
    dir is writable, python works, and the project tree is hidden.
    (Regression for the ProtectHome=yes bug: that masked .bash_agent_tmp
    itself, so EVERY orchestrator command returned exit 200 with empty
    output, and the orchestrator could never read back a file.)

Hazard handled here (verified empirically): when subprocess.run times out it
SIGKILLs the systemd-run CLIENT process. The transient service itself does
NOT die with it — without intervention a stray `sleep 999` would survive on
the host. We therefore launch the leak probe with a unique marker placed in
argv[0] via `exec -a <marker> sleep 999` (a shell comment would be stripped
before exec and leave the process unmatchable), and on timeout kill any
surviving service by marker before asserting.

Skipped automatically when no systemd user session / systemd-run binary.
"""
import os
import subprocess
import unittest

from bash_agent.sandbox import Sandbox

from tests.helpers.fakes import chdir_repo_tmp, systemd_user_bus_available



@unittest.skipUnless(systemd_user_bus_available(), "systemd-run user session unavailable")
class TestRealSandboxSmoke(unittest.TestCase):
    """T-30 — real Sandbox round-trips against the local systemd user bus."""

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
        # Reading protected paths is allowed under ProtectSystem=strict...
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
        # PrivateTmp=yes means host /tmp content must not be visible.
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
        finally:
            # subprocess.run SIGKILLs only the systemd-run client; the
            # transient unit survives. Kill anything still matching our
            # unique marker so the host stays clean regardless of outcome.
            subprocess.run(
                ["pkill", "-9", "-f", marker],
                capture_output=True,
            )

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Command timed out after 2 seconds.", out)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(systemd_user_bus_available(), "systemd-run user session unavailable")
class TestOrchestratorSandboxIsolation(unittest.TestCase):
    """T-30b — real round-trips for is_orchestrator=True.

    Regression guard for the ``ProtectHome=yes`` bug: the orchestrator's
    working directory (``.bash_agent_tmp/``) lives INSIDE the home tree, so
    ``ProtectHome=yes`` masked it and every orchestrator command died with
    ``status=200/CHDIR`` before running — returning exit 200 and empty output.
    The fix masks ``/home`` with a tmpfs and binds ``.bash_agent_tmp/`` back in.
    """

    def setUp(self):
        self._chdir_cm = chdir_repo_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.sb = Sandbox("unused-scratchpad-path", uuid="orch-uuid", is_orchestrator=True)

    def tearDown(self):
        self._chdir_cm.__exit__(None, None, None)

    def test_orchestrator_commands_actually_run(self):
        # The headline regression: this used to return (200, "").
        code, out = self.sb.execute("echo orchestrator-alive")
        self.assertEqual(code, 0, msg=f"orchestrator sandbox failed: {out!r}")
        self.assertIn("orchestrator-alive", out)

    def test_orchestrator_can_write_and_read_scratch_dir(self):
        code, out = self.sb.execute(
            "printf 'hello-from-orch' > hello.txt && cat hello.txt && rm -f hello.txt"
        )
        self.assertEqual(code, 0, msg=f"scratch write/read failed: {out!r}")
        self.assertIn("hello-from-orch", out)

    def test_orchestrator_python_round_trip(self):
        code, out = self.sb.execute_python("print('orch-py-ok')")
        self.assertEqual(code, 0, msg=f"orchestrator python failed: {out!r}")
        self.assertIn("orch-py-ok", out)

    def test_orchestrator_cannot_see_project_tree(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        sentinel = os.path.join(repo_root, "orch_hidden_probe.txt")
        with open(sentinel, "w") as f:
            f.write("project\n")
        try:
            code, out = self.sb.execute(
                f"if [ -e {sentinel} ]; then echo VISIBLE; else echo HIDDEN; fi"
            )
            self.assertEqual(code, 0)
            self.assertIn("HIDDEN", out)
        finally:
            if os.path.exists(sentinel):
                os.remove(sentinel)
