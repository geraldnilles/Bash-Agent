"""
Group 8 — Unit tests for bash_agent.sandbox (offline, mock-driven).

T-45  Sandbox helper methods, argv assembly & request_write (P1)

Sandbox.py runs agent scripts inside a bubblewrap (``bwrap``) container. Its
pure-logic helpers are fully testable without a real sandbox:

  * request_write()    — user prompt + approved_write_paths bookkeeping
  * execute() / execute_python() — temp file creation, bwrap argv
    construction, timeout handling (mocked at subprocess.run)

Seam notes:
  * All external calls go through subprocess.run / tempfile.mkstemp /
    os.chmod / os.remove / os.fdopen / input() / print() — all easily patched.
  * bwrap command assembly is deterministic given uuid/timeout/cwd; we assert
    the exact argv shape to catch regressions in namespace ordering,
    filesystem posture (--ro-bind / /, --tmpfs /home for orchestrators),
    env forwarding (--setenv) and the interpreter tail.
  * The timeout path (exit_code=124, banner present, exactly one subprocess
    run) is verified by raising TimeoutExpired from the mocked subprocess.run.
  * There is NO transient unit to reap any more: --die-with-parent tears the
    container down on a client-side timeout, so no unit-reaping / cleanup
    subprocess is issued.
"""

import io
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from bash_agent.sandbox import (
    Sandbox,
    _PKG_DIR,
    _PROJECT_ROOT,
    _resolve_venv_dir,
)


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

        # Create sandbox instance (subagent posture: project tree writable)
        self.sb = Sandbox(scratchpad_path="/tmp/scratchpad.md", timeout=60, uuid="test-uuid-123", multimodal_capabilities=["image"])

    # -- harness helpers ----------------------------------------------------

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
        exc = subprocess.TimeoutExpired(cmd=["bwrap"], timeout=60)
        exc.stdout = stdout
        self.mock_run.side_effect = exc

    def _cmd(self):
        """The argv list handed to the (mocked) subprocess.run."""
        return self.mock_run.call_args[0][0]

    @staticmethod
    def _setenv(cmd):
        """Return the --setenv pairs of `cmd` as a {name: value} dict."""
        env = {}
        for i, tok in enumerate(cmd):
            if tok == "--setenv":
                env[cmd[i + 1]] = cmd[i + 2]
        return env

    @staticmethod
    def _bind_pairs(cmd):
        """Return the (src, dst) pairs of every --bind in `cmd`."""
        pairs = []
        for i, tok in enumerate(cmd):
            if tok == "--bind":
                pairs.append((cmd[i + 1], cmd[i + 2]))
        return pairs


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

    def test_orchestrator_refuses_request_write(self):
        """Orchestrators may never request write access outside scratch."""
        sb = Sandbox("sp", is_orchestrator=True)
        success, msg = sb.request_write("/etc/passwd")
        self.assertFalse(success)
        self.assertIn("Orchestrator agent cannot request write access", msg)


class TestExecuteBash(SandboxTestCase):
    """execute() builds and runs a bwrap command for bash scripts."""

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

    def test_constructs_correct_bwrap_command(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello")

        cmd = self._cmd()

        # 1. Launcher + common isolation prefix (order matters)
        self.assertEqual(cmd[0], "bwrap")
        for flag in (
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--tmpfs", "/var/tmp",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--die-with-parent",
        ):
            self.assertIn(flag, cmd)
        # The whole host root is bound read-only.
        self.assertIn(["--ro-bind", "/", "/"], [cmd[i:i + 3] for i in range(len(cmd) - 2)])

        # 2. Role posture: the approved write path is bind-mounted read/write.
        self.assertEqual(self._bind_pairs(cmd), [("/home/user", "/home/user")])
        # Non-orchestrator keeps /home visible (no tmpfs over /home).
        self.assertFalse(
            any(cmd[i:i + 2] == ["--tmpfs", "/home"] for i in range(len(cmd) - 1))
        )

        # 3. Env contract, forwarded via --setenv.
        env = self._setenv(cmd)
        # execute() prepends the package-location venv bin so CLI tools
        # (e.g. ``vision``) resolve regardless of the launch cwd.
        self.assertEqual(
            env["PATH"],
            os.path.join(_resolve_venv_dir(), "bin") + ":/usr/bin:/bin",
        )
        self.assertEqual(env["OPENROUTER_API_KEY"], "sk-test-key")
        self.assertEqual(env["BASH_AGENT_UUID"], "test-uuid-123")
        self.assertEqual(env["BASH_AGENT_MULTIMODAL"], "image")
        # execute() forwards PATH but not PYTHONPATH.
        self.assertNotIn("PYTHONPATH", env)

        # 4. Tail: --chdir <workdir> /bin/bash <script>
        self.assertEqual(cmd[-4:-2], ["--chdir", "/home/user"])
        self.assertEqual(cmd[-2], "/bin/bash")
        self.assertEqual(cmd[-1], "/home/user/.bash_agent_tmp/script.sh")

    def test_includes_approved_write_paths(self):
        self.sb.approved_write_paths.append("/home/user/extra")
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello")

        cmd = self._cmd()
        # Every approved path is bind-mounted read/write, in order.
        self.assertEqual(
            self._bind_pairs(cmd),
            [("/home/user", "/home/user"), ("/home/user/extra", "/home/user/extra")],
        )

    def test_orchestrator_masks_home_and_binds_scratch(self):
        """Orchestrator gets --tmpfs /home + rw bind of .bash_agent_tmp only."""
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        with mock.patch("bash_agent.sandbox.os.path.isdir", return_value=True):
            sb.execute("echo hello")

        cmd = self._cmd()
        # /home is masked with a tmpfs...
        self.assertIn(["--tmpfs", "/home"], [cmd[i:i + 2] for i in range(len(cmd) - 1)])
        # ...and only the scratch dir is bound back in read/write.
        self.assertEqual(
            self._bind_pairs(cmd),
            [("/home/user/.bash_agent_tmp", "/home/user/.bash_agent_tmp")],
        )
        # workdir for the orchestrator is .bash_agent_tmp.
        self.assertEqual(cmd[-4:-2], ["--chdir", "/home/user/.bash_agent_tmp"])
        # venv + package source are re-exposed READ-ONLY even for BASH, so
        # CLI tools like `vision` and package imports keep working.
        ro_binds = [cmd[i:i + 3] for i in range(len(cmd) - 2) if cmd[i] == "--ro-bind"]
        # Paths resolve from the PACKAGE location (not the launch cwd),
        # so they survive a foreign working directory.
        venv_dir = _resolve_venv_dir()
        self.assertIn(["--ro-bind", venv_dir, venv_dir], ro_binds)
        self.assertIn(["--ro-bind", _PKG_DIR, _PKG_DIR], ro_binds)

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
        # No transient unit to reap: bwrap is the only process ever spawned.
        self.assertEqual(self.mock_run.call_count, 1)

    def test_per_call_timeout_flows_into_subprocess_run(self):
        """A valid per-call timeout is forwarded to subprocess.run."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello", timeout=300)

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 300)

    def test_omitted_timeout_uses_sandbox_default(self):
        """Without a per-call timeout, self.timeout is used."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello")

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_non_int_timeout_falls_back_to_default(self):
        """Non-integer timeout falls back to self.timeout."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello", timeout="300")

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_zero_timeout_falls_back_to_default(self):
        """timeout < 1 falls back to self.timeout."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello", timeout=0)

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_bool_timeout_falls_back_to_default(self):
        """bool is not a valid timeout; falls back to self.timeout."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        self.sb.execute("echo hello", timeout=True)

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_timeout_banner_reflects_per_call_value(self):
        """On TimeoutExpired the banner shows the effective per-call value."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        exc = subprocess.TimeoutExpired(cmd=["bwrap"], timeout=300)
        exc.stdout = "partial output before timeout"
        self.mock_run.side_effect = exc

        code, out = self.sb.execute("sleep 999", timeout=300)

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Command timed out after 300 seconds.", out)
        self.assertIn("partial output before timeout", out)

    def test_exception_returns_exit_1(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self.mock_run.side_effect = OSError("bwrap not found")

        code, out = self.sb.execute("echo hello")

        self.assertEqual(code, 1)
        self.assertIn("bwrap not found", out)

    def test_cleanup_removes_temp_script_even_on_exception(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self.mock_run.side_effect = OSError("boom")

        self.sb.execute("echo hello")

        self.mock_remove.assert_called_once_with("/home/user/.bash_agent_tmp/script.sh")


class TestExecutePython(SandboxTestCase):
    """execute_python() builds and runs a bwrap command for Python scripts."""

    def test_uses_python_suffix_and_python3_executable(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script123.py", fd=43)
        self._setup_success_run("py output")

        code, out = self.sb.execute_python("print('hello')")

        # mkstemp called with .py suffix
        args, kwargs = self.mock_mkstemp.call_args
        self.assertEqual(kwargs["suffix"], ".py")

        # Command ends with python3 and script path. os.path.exists() is
        # mocked to True, and the interpreter is resolved from the
        # package-location venv, so it is chosen deterministically.
        cmd = self._cmd()
        self.assertEqual(cmd[-1], "/home/user/.bash_agent_tmp/script123.py")
        self.assertEqual(
            cmd[-2], os.path.join(_resolve_venv_dir(), "bin", "python3")
        )

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

            self.assertEqual(self._cmd()[-2], "/home/user/venv/bin/python3")

    def test_includes_PYTHONPATH_in_environment(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        env = self._setenv(self._cmd())
        # PYTHONPATH carries the workdir plus the project root so
        # ``bash_agent`` imports resolve from any launch cwd.
        self.assertEqual(
            env["PYTHONPATH"],
            os.pathsep.join(("/home/user", _PROJECT_ROOT)),
        )
        # execute_python() does NOT forward PATH.
        self.assertNotIn("PATH", env)

    def test_orchestrator_venv_is_bound_read_only(self):
        """Orchestrator python runs need the masked-back-in project venv."""
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        with mock.patch("bash_agent.sandbox.os.path.isdir", return_value=True):
            sb.execute_python("print('hello')")

        cmd = self._cmd()
        ro_binds = [cmd[i:i + 3] for i in range(len(cmd) - 2) if cmd[i] == "--ro-bind"]
        venv_dir = _resolve_venv_dir()
        self.assertIn(["--ro-bind", venv_dir, venv_dir], ro_binds)

    def test_timeout_handling_same_as_bash(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_timeout("partial py output")

        code, out = self.sb.execute_python("import time; time.sleep(999)")

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Python command timed out after 60 seconds.", out)
        self.assertIn("partial py output", out)

    def test_per_call_timeout_flows_into_subprocess_run(self):
        """A valid per-call timeout is forwarded to subprocess.run."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')", timeout=240)

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 240)

    def test_omitted_timeout_uses_sandbox_default_for_python(self):
        """Without a per-call timeout, self.timeout is used."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_non_int_timeout_falls_back_to_default_for_python(self):
        """Non-integer timeout falls back to self.timeout."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')", timeout="240")

        self.assertEqual(self.mock_run.call_args.kwargs["timeout"], 60)

    def test_timeout_banner_reflects_per_call_value_for_python(self):
        """On TimeoutExpired the banner shows the effective per-call value."""
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        exc = subprocess.TimeoutExpired(cmd=["bwrap"], timeout=240)
        exc.stdout = "partial py output"
        self.mock_run.side_effect = exc

        code, out = self.sb.execute_python("print('hi')", timeout=240)

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Python command timed out after 240 seconds.", out)
        self.assertIn("partial py output", out)

    def test_includes_approved_write_paths(self):
        self.sb.approved_write_paths.append("/home/user/extra")
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        self.assertEqual(
            self._bind_pairs(self._cmd()),
            [("/home/user", "/home/user"), ("/home/user/extra", "/home/user/extra")],
        )

    def test_constructs_correct_security_posture(self):
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.py")
        self._setup_success_run("py output")

        self.sb.execute_python("print('hello')")

        cmd = self._cmd()
        # Common isolation prefix.
        self.assertEqual(cmd[0], "bwrap")
        self.assertIn(["--ro-bind", "/", "/"], [cmd[i:i + 3] for i in range(len(cmd) - 2)])
        for flag in (
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--tmpfs", "/var/tmp", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup-try", "--die-with-parent",
        ):
            self.assertIn(flag, cmd)
        # Env contract (python path).
        env = self._setenv(cmd)
        self.assertEqual(env["OPENROUTER_API_KEY"], "sk-test-key")
        self.assertEqual(env["BASH_AGENT_UUID"], "test-uuid-123")
        self.assertEqual(env["BASH_AGENT_MULTIMODAL"], "image")
        # NOTE: execute_python() does NOT forward PATH (unlike execute());
        # the interpreter is invoked by absolute path instead.
        self.assertNotIn("PATH", env)
        self.assertEqual(cmd[-4:-2], ["--chdir", "/home/user"])

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
        self.mock_run.side_effect = OSError("bwrap not found")

        code, out = self.sb.execute_python("print('hello')")

        self.assertEqual(code, 1)
        self.assertIn("bwrap not found", out)

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

            cmd = self._cmd()
            self.assertEqual(cmd[-3:-1], ["/usr/bin/env", "python3"])
            self.assertEqual(cmd[-1], "/home/user/.bash_agent_tmp/script.py")


class TestExecuteWithMultimodal(SandboxTestCase):
    """Multimodal capabilities are forwarded to the sandbox environment."""

    def test_multimodal_capabilities_forwarded_as_comma_separated(self):
        sb = Sandbox("/tmp/scratchpad.md", uuid="test-uuid", multimodal_capabilities=["image", "audio"])
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        sb.execute("echo hello")

        self.assertEqual(self._setenv(self._cmd())["BASH_AGENT_MULTIMODAL"], "image,audio")

    def test_empty_multimodal_capabilities_forwards_empty_string(self):
        sb = Sandbox("/tmp/scratchpad.md", uuid="test-uuid", multimodal_capabilities=[])
        self._setup_mkstemp("/home/user/.bash_agent_tmp/script.sh")
        self._setup_success_run("output")

        sb.execute("echo hello")

        self.assertEqual(self._setenv(self._cmd())["BASH_AGENT_MULTIMODAL"], "")


if __name__ == "__main__":
    unittest.main()
