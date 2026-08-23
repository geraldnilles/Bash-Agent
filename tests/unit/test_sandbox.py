"""
Group 8 — Unit tests for bash_agent.sandbox (offline, mock-driven).

T-45  Sandbox helper methods & request_write (P1)

Sandbox.py contains several pure-logic helpers that are fully testable
without a systemd user bus:

  * _unit_name()       — generates unique transient unit names
  * _reap_unit()       — best-effort systemctl stop (must never raise)
  * request_write()    — user prompt + approved_write_paths bookkeeping
  * execute() / execute_python() — temp file creation, cmd construction,
    timeout/reap logic (mocked at subprocess.run)

Seam notes:
  * All external calls go through subprocess.run / tempfile.mkstemp /
    os.chmod / os.remove / os.fdopen / input() / print() — all easily patched.
  * Systemd command assembly is deterministic given uuid/timeout/cwd;
    we assert the exact argv list to catch regressions in property
    ordering, env forwarding, and ReadWritePaths.
  * The timeout path (_reap_unit called, exit_code=124, banner present)
    is verified by raising TimeoutExpired from the mocked subprocess.run.
"""

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from bash_agent.sandbox import Sandbox


class SandboxTestCase(unittest.TestCase):
    """Shared harness: patch subprocess.run, tempfile, input, print."""

    def setUp(self):
        # Patch subprocess.run globally for this test class
        self.run_patcher = mock.patch("bash_agent.sandbox.subprocess.run")
        self.mock_run = self.run_patcher.start()
        self.addCleanup(self.run_patcher.stop)

        # Patch tempfile.mkstemp to return a controlled fd/path
        self.mkstemp_patcher = mock.patch("bash_agent.sandbox.tempfile.mkstemp")
        self.mock_mkstemp = self.mkstemp_patcher.start()
        self.addCleanup(self.mkstemp_patcher.stop)

        # Patch os.fdopen to return a StringIO (so we can capture writes)
        self.fdopen_patcher = mock.patch("bash_agent.sandbox.os.fdopen")
        self.mock_fdopen = self.fdopen_patcher.start()
        self.addCleanup(self.fdopen_patcher.stop)

        # Patch os.chmod, os.remove, os.path.exists
        self.chmod_patcher = mock.patch("bash_agent.sandbox.os.chmod")
        self.mock_chmod = self.chmod_patcher.start()
        self.addCleanup(self.chmod_patcher.stop)

        self.remove_patcher = mock.patch("bash_agent.sandbox.os.remove")
        self.mock_remove = self.remove_patcher.start()
        self.addCleanup(self.remove_patcher.stop)

        self.exists_patcher = mock.patch("bash_agent.sandbox.os.path.exists")
        self.mock_exists = self.exists_patcher.start()
        self.addCleanup(self.exists_patcher.stop)

        # Patch os.makedirs
        self.makedirs_patcher = mock.patch("bash_agent.sandbox.os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()
        self.addCleanup(self.makedirs_patcher.stop)

        # Patch os.path.abspath to return predictable, normalized paths.
        # Real abspath would resolve "." to the CWD with no trailing "/."
        # and collapse ".." segments, so we mirror that via normpath.
        self.abspath_patcher = mock.patch(
            "bash_agent.sandbox.os.path.abspath",
            side_effect=lambda x: os.path.normpath(x if x.startswith("/") else "/home/user/" + x),
        )
        self.mock_abspath = self.abspath_patcher.start()
        self.addCleanup(self.abspath_patcher.stop)

        # Patch os.getcwd so execute_python()'s venv lookup is deterministic
        self.getcwd_patcher = mock.patch("bash_agent.sandbox.os.getcwd", return_value="/home/user")
        self.mock_getcwd = self.getcwd_patcher.start()
        self.addCleanup(self.getcwd_patcher.stop)

        # Patch os.getpid for deterministic unit names
        self.getpid_patcher = mock.patch("bash_agent.sandbox.os.getpid", return_value=12345)
        self.mock_getpid = self.getpid_patcher.start()
        self.addCleanup(self.getpid_patcher.stop)

        # Patch uuid4 for deterministic unit names
        self.uuid_patcher = mock.patch("bash_agent.sandbox.uuid_lib.uuid4")
        self.mock_uuid = self.uuid_patcher.start()
        self.mock_uuid.return_value.hex = "abcdef0123456789"
        self.addCleanup(self.uuid_patcher.stop)

        # Patch os.environ.get for PATH and OPENROUTER_API_KEY
        self.environ_patcher = mock.patch("bash_agent.sandbox.os.environ.get", side_effect=lambda k, d="": {
            "PATH": "/usr/bin:/bin",
            "OPENROUTER_API_KEY": "sk-test-key",
        }.get(k, d))
        self.mock_environ = self.environ_patcher.start()
        self.addCleanup(self.environ_patcher.stop)

        # Suppress print output by default
        self.print_patcher = mock.patch("bash_agent.sandbox.print")
        self.mock_print = self.print_patcher.start()
        self.addCleanup(self.print_patcher.stop)

        # Create sandbox instance
        self.sb = Sandbox(scratchpad_path="/tmp/scratchpad.md", timeout=60, uuid="test-uuid-123", multimodal_capabilities=["image"])

    def _setup_mkstemp(self, path="/home/user/.bash_agent_tmp/script.sh", fd=42):
        """Configure mkstemp to return a specific (fd, path)."""
        self.mock_mkstemp.return_value = (fd, path)
        self.mock_exists.return_value = True  # file exists for cleanup
        # fdopen returns a mock file-like object
        self.mock_file = mock.MagicMock()
        self.mock_file.__enter__ = mock.Mock(return_value=self.mock_file)
        self.mock_file.__exit__ = mock.Mock(return_value=False)
        self.mock_fdopen.return_value = self.mock_file

    def _setup_success_run(self, stdout="output", returncode=0):
        """Configure subprocess.run to return success."""
        self.mock_run.return_value = mock.Mock(stdout=stdout, returncode=returncode)

    def _setup_timeout(self, stdout="partial output"):
        """Configure subprocess.run to raise TimeoutExpired."""
        exc = subprocess.TimeoutExpired(cmd=["systemd-run"], timeout=60)
        exc.stdout = stdout
        self.mock_run.side_effect = exc


class TestUnitName(SandboxTestCase):
    """_unit_name() generates unique, deterministic transient unit names."""

    def test_format_contains_pid_and_uuid_and_counter(self):
        name1 = self.sb._unit_name()
        name2 = self.sb._unit_name()

        # Format: bash-agent-{pid}-{counter}-{uuid8}.service
        self.assertTrue(name1.startswith("bash-agent-12345-0-"))
        self.assertTrue(name1.endswith(".service"))
        self.assertIn("abcdef01", name1)  # first 8 chars of mocked uuid

        # Counter increments
        self.assertTrue(name2.startswith("bash-agent-12345-1-"))


class TestReapUnit(SandboxTestCase):
    """_reap_unit() best-effort stops a transient systemd unit."""

    def test_calls_systemctl_user_stop_with_unit_name(self):
        self.sb._reap_unit("test-unit.service")
        self.mock_run.assert_called_once_with(
            ["systemctl", "--user", "stop", "test-unit.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )

    def test_never_raises_on_failure(self):
        self.mock_run.side_effect = Exception("systemctl failed")
        try:
            self.sb._reap_unit("test-unit.service")
        except Exception:
            self.fail("_reap_unit raised an exception")


class TestRequestWrite(SandboxTestCase):
    """request_write() prompts user and manages approved_write_paths."""

    def test_approve_y_adds_path_and_returns_true(self):
        with mock.patch("bash_agent.sandbox.input", return_value="y"):
            success, msg = self.sb.request_write("/home/user/project/file.txt")

        self.assertTrue(success)
        self.assertEqual(msg, "Write access granted.")
        self.assertIn("/home/user/project/file.txt", self.sb.approved_write_paths)

    def test_approve_n_returns_false(self):
        with mock.patch("bash_agent.sandbox.input", return_value="n"):
            success, msg = self.sb.request_write("/home/user/project/file.txt")

        self.assertFalse(success)
        self.assertEqual(msg, "Write access denied by user.")
        self.assertNotIn("/home/user/project/file.txt", self.sb.approved_write_paths)

    def test_free_text_returns_false_with_message(self):
        with mock.patch("bash_agent.sandbox.input", return_value="No, that's dangerous!"):
            success, msg = self.sb.request_write("/etc/passwd")

        self.assertFalse(success)
        self.assertIn("Write access denied. User message: No, that's dangerous!", msg)

    def test_prints_request_message(self):
        with mock.patch("bash_agent.sandbox.input", return_value="y"):
            self.sb.request_write("/home/user/file.txt")

        # Verify the approval prompt was printed
        self.mock_print.assert_any_call(
            "\n[AGENT REQUEST] The agent is requesting write access to: /home/user/file.txt"
        )

    def test_resolves_relative_paths(self):
        with mock.patch("bash_agent.sandbox.input", return_value="y"):
            with mock.patch("bash_agent.sandbox.os.path.abspath", side_effect=lambda x: "/abs/path/" + x):
                self.sb.request_write("relative/file.txt")

        self.assertIn("/abs/path/relative/file.txt", self.sb.approved_write_paths)


class TestExecuteBash(SandboxTestCase):
    """execute() builds and runs systemd-run command for bash scripts."""

    def test_creates_temp_script_in_bash_agent_tmp(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script123.sh")
        self._setup_success_run("hello")

        code, out = self.sb.execute("echo hello")

        # mkstemp called with dir=.bash_agent_tmp, suffix=.sh
        self.mock_mkstemp.assert_called_once()
        args, kwargs = self.mock_mkstemp.call_args
        self.assertEqual(kwargs["suffix"], ".sh")
        self.assertEqual(kwargs["dir"], "/home/user/.bash_agent_tmp")
        self.assertTrue(kwargs["text"])

        # Script written and chmod'd
        self.mock_file.write.assert_called_once_with("echo hello")
        self.mock_chmod.assert_called_once_with("/home/user/.bash_agent_tmp/script123.sh", 0o700)

        # Cleanup
        self.mock_remove.assert_called_once_with("/home/user/.bash_agent_tmp/script123.sh")

    def test_constructs_correct_systemd_run_command(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello")

        # Verify the command structure
        call_args = self.mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "systemd-run")
        self.assertIn("--user", call_args)
        self.assertIn("--quiet", call_args)
        self.assertIn("--wait", call_args)
        self.assertIn("--collect", call_args)
        self.assertIn("--pipe", call_args)
        self.assertTrue(any(arg.startswith("--unit=bash-agent-") for arg in call_args))
        self.assertTrue(any(arg == "--property=ProtectSystem=strict" for arg in call_args))
        self.assertTrue(any(arg == "--property=ProtectHome=read-only" for arg in call_args))
        self.assertTrue(any(arg == "--property=PrivateTmp=yes" for arg in call_args))
        # working-directory is the absolute path of current directory
        self.assertTrue(any(arg == "--working-directory=/home/user" for arg in call_args))
        self.assertTrue(any(arg == "--property=Environment=PATH=/usr/bin:/bin" for arg in call_args))
        self.assertTrue(any(arg == "--property=Environment=OPENROUTER_API_KEY=sk-test-key" for arg in call_args))
        self.assertTrue(any(arg == "--property=Environment=BASH_AGENT_UUID=test-uuid-123" for arg in call_args))
        self.assertTrue(any(arg == "--property=Environment=BASH_AGENT_MULTIMODAL=image" for arg in call_args))
        # approved_write_paths starts with [os.path.abspath(".")] which is "/home/user"
        self.assertTrue(any(arg == "--property=ReadWritePaths=/home/user" for arg in call_args))
        self.assertEqual(call_args[-2], "/bin/bash")
        self.assertEqual(call_args[-1], "/home/user/.bash_agent_tmp/script.sh")

    def test_includes_approved_write_paths(self):
        self.sb.approved_write_paths.append("/home/user/extra")
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello")

        call_args = self.mock_run.call_args[0][0]
        # Should have both the default cwd path and the extra path
        self.assertIn("--property=ReadWritePaths=/home/user", call_args)
        self.assertIn("--property=ReadWritePaths=/home/user/extra", call_args)

    def test_returns_exit_code_and_stdout_on_success(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("command output", returncode=0)

        code, out = self.sb.execute("echo hello")

        self.assertEqual(code, 0)
        self.assertEqual(out, "command output")

    def test_returns_nonzero_exit_code(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("error message", returncode=3)

        code, out = self.sb.execute("exit 3")

        self.assertEqual(code, 3)
        self.assertEqual(out, "error message")

    def test_timeout_returns_124_and_partial_output(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_timeout("partial output before timeout")

        code, out = self.sb.execute("sleep 999")

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Command timed out after 60 seconds.", out)
        self.assertIn("partial output before timeout", out)
        # _reap_unit should have been called
        self.mock_run.assert_any_call(
            ["systemctl", "--user", "stop", mock.ANY],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )

    def test_exception_returns_exit_1(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self.mock_run.side_effect = OSError("systemd-run not found")

        code, out = self.sb.execute("echo hello")

        self.assertEqual(code, 1)
        self.assertIn("systemd-run not found", out)

    def test_cleanup_removes_temp_script_even_on_exception(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self.mock_run.side_effect = OSError("boom")

        self.sb.execute("echo hello")

        self.mock_remove.assert_called_once_with("/home/user/.bash_agent_tmp/script.sh")


class TestExecutePython(SandboxTestCase):
    """execute_python() builds and runs systemd-run command for Python scripts."""

    def test_uses_python_suffix_and_python3_executable(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script123.py", fd=43)
        self._setup_success_run("py output")

        code, out = self.sb.execute_python("print('hello')")

        # mkstemp called with .py suffix
        args, kwargs = self.mock_mkstemp.call_args
        self.assertEqual(kwargs["suffix"], ".py")

        # Command ends with python3 and script path. os.getcwd() is mocked to
        # /home/user, and os.path.exists() is mocked to True, so the venv
        # interpreter is chosen deterministically.
        call_args = self.mock_run.call_args[0][0]
        self.assertEqual(call_args[-1], "/home/user/.bash_agent_tmp/script123.py")
        self.assertEqual(call_args[-2], "/home/user/venv/bin/python3")

    def test_uses_venv_python_when_available(self):
        """When venv/bin/python3 exists, it should be used."""
        # The mock for os.path.exists needs to return True for the venv path
        def exists_side_effect(path):
            if path == "/home/user/venv/bin/python3":
                return True
            return False
        
        with mock.patch("bash_agent.sandbox.os.path.exists", side_effect=exists_side_effect):
            self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
            self._setup_success_run("py output")

            self.sb.execute_python("print('hello')")

            call_args = self.mock_run.call_args[0][0]
            self.assertEqual(call_args[-2], "/home/user/venv/bin/python3")

    def test_includes_PYTHONPATH_in_environment(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        call_args = self.mock_run.call_args[0][0]
        # PYTHONPATH is set to the absolute path of the current directory
        self.assertTrue(any(arg == "--property=Environment=PYTHONPATH=/home/user" for arg in call_args))

    def test_timeout_handling_same_as_bash(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_timeout("partial py output")

        code, out = self.sb.execute_python("import time; time.sleep(999)")

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Python command timed out after 60 seconds.", out)
        self.assertIn("partial py output", out)


    def test_includes_approved_write_paths(self):
        self.sb.approved_write_paths.append("/home/user/extra")
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        call_args = self.mock_run.call_args[0][0]
        self.assertIn("--property=ReadWritePaths=/home/user", call_args)
        self.assertIn("--property=ReadWritePaths=/home/user/extra", call_args)

    def test_constructs_correct_security_posture(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        call_args = self.mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "systemd-run")
        self.assertIn("--user", call_args)
        self.assertIn("--quiet", call_args)
        self.assertIn("--wait", call_args)
        self.assertIn("--collect", call_args)
        self.assertIn("--pipe", call_args)
        self.assertTrue(any(arg.startswith("--unit=bash-agent-") for arg in call_args))
        self.assertIn("--property=ProtectSystem=strict", call_args)
        self.assertIn("--property=ProtectHome=read-only", call_args)
        self.assertIn("--property=PrivateTmp=yes", call_args)
        self.assertIn("--working-directory=/home/user", call_args)
        self.assertIn("--property=Environment=OPENROUTER_API_KEY=sk-test-key", call_args)
        self.assertIn("--property=Environment=BASH_AGENT_UUID=test-uuid-123", call_args)
        self.assertIn("--property=Environment=BASH_AGENT_MULTIMODAL=image", call_args)
        # NOTE: execute_python() does NOT forward PATH (unlike execute());
        # the interpreter is invoked by absolute path instead.
        self.assertFalse(any(arg.startswith("--property=Environment=PATH=") for arg in call_args))

    def test_returns_exit_code_and_stdout_on_success(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output", returncode=0)

        code, out = self.sb.execute_python("print('hello')")

        self.assertEqual(code, 0)
        self.assertEqual(out, "py output")

    def test_returns_nonzero_exit_code(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py error", returncode=2)

        code, out = self.sb.execute_python("raise SystemExit(2)")

        self.assertEqual(code, 2)
        self.assertEqual(out, "py error")

    def test_exception_returns_exit_1(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self.mock_run.side_effect = OSError("systemd-run not found")

        code, out = self.sb.execute_python("print('hello')")

        self.assertEqual(code, 1)
        self.assertIn("systemd-run not found", out)

    def test_cleanup_removes_temp_script_even_on_exception(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self.mock_run.side_effect = OSError("boom")

        self.sb.execute_python("print('hello')")

        self.mock_remove.assert_called_once_with("/home/user/.bash_agent_tmp/script.py")

    def test_uses_env_python_fallback_when_venv_missing(self):
        """Without ./venv/bin/python3, /usr/bin/env python3 is used."""
        def exists_side_effect(path):
            return False  # no venv interpreter anywhere

        with mock.patch("bash_agent.sandbox.os.path.exists", side_effect=exists_side_effect):
            self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
            self._setup_success_run("py output")

            self.sb.execute_python("print('hello')")

            call_args = self.mock_run.call_args[0][0]
            self.assertEqual(call_args[-3:-1], ["/usr/bin/env", "python3"])
            self.assertEqual(call_args[-1], "/home/user/.bash_agent_tmp/script.py")


class TestExecuteWithMultimodal(SandboxTestCase):
    """Multimodal capabilities are forwarded to the sandbox environment."""

    def test_multimodal_capabilities_forwarded_as_comma_separated(self):
        sb = Sandbox("/tmp/scratchpad.md", uuid="test-uuid", multimodal_capabilities=["image", "audio"])
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        sb.execute("echo hello")

        call_args = self.mock_run.call_args[0][0]
        self.assertIn("--property=Environment=BASH_AGENT_MULTIMODAL=image,audio", call_args)

    def test_empty_multimodal_capabilities_forwards_empty_string(self):
        sb = Sandbox("/tmp/scratchpad.md", uuid="test-uuid", multimodal_capabilities=[])
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        sb.execute("echo hello")

        call_args = self.mock_run.call_args[0][0]
        self.assertIn("--property=Environment=BASH_AGENT_MULTIMODAL=", call_args)


if __name__ == "__main__":
    unittest.main()
