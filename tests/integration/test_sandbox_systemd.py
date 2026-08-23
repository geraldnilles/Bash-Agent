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
