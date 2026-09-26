"""
Group 6 — Sandbox unit tests (no sandbox invocation involved).

T-27  Construction defaults & write-path bookkeeping (P1)
T-28  Command assembly inspection (P1)
T-29  Temp-script hygiene (P1)

These tests pin the Sandbox surface that the LLM-facing protocol depends on:

  * A fresh Sandbox falls back to config.BASH_TIMEOUT, starts with exactly
    one approved write path (the abspath of "."), and stores uuid/multimodal.
  * request_write() driven by patched builtins.input appends the *abspath*
    only on approval and always returns a (bool, message) tuple — the same
    contract agent.py unpacks as `success, msg = sandbox.request_write(path)`.
  * The argv assembled for bubblewrap carries the full isolation posture:
    the host root bound read-only (--ro-bind / /), a private /proc, /dev and
    throwaway /tmp + /var/tmp, namespace isolation (--unshare-*), robust
    teardown (--die-with-parent), the working directory via --chdir, env
    forwarding via --setenv (PATH for bash only; PYTHONPATH for python only;
    OPENROUTER_API_KEY only when set; BASH_AGENT_UUID / BASH_AGENT_MULTIMODAL),
    one read/write --bind per approved path, and a /bin/bash <script> or
    <venv-python|/usr/bin/env python3> <script>.py tail. If someone drops an
    isolation flag, these tests go red before users do.
  * Temp-script hygiene: the mkstemp'd script under .bash_agent_tmp/ has
    mode 0700 while it exists, is removed after every call (success or
    timeout), and TimeoutExpired maps to exit code 124 with the
    "[SYSTEM ERROR] ... timed out" banner plus partial stdout — the exit
    code the LLM reads to decide retries.

All tests are offline: subprocess.run is replaced with recorders/fakes, so
bwrap is never invoked. Real-sandbox behavior is covered separately by
Group 7 integration tests (T-30).

Every test runs inside a throwaway CWD (chdir_tmp) and captures stdout so
production prints stay out of test runs.
"""
import contextlib
import io
import os
import stat
import subprocess
import unittest
from unittest import mock

from bash_agent import config
from bash_agent.sandbox import Sandbox

from tests.helpers.fakes import chdir_tmp


class _FakeCompleted:
    """Minimal CompletedProcess stand-in for recorded runs."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


class SandboxCase(unittest.TestCase):
    """Shared harness: temp CWD + captured stdout."""

    def setUp(self):
        self._chdir_cm = chdir_tmp()
        self.tmpdir = self._chdir_cm.__enter__()
        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

    def tearDown(self):
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- recorder helpers ---------------------------------------------------

    def _record_run(self, result=None, exc=None):
        """
        Return (calls, fake_run). fake_run replaces subprocess.run inside
        bash_agent.sandbox; each invocation is appended to `calls` as
        {"cmd": [...], "kwargs": {...}} before returning `result` or raising
        `exc`.
        """
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append({"cmd": list(cmd), "kwargs": kwargs})
            if exc is not None:
                raise exc
            return result if result is not None else _FakeCompleted()

        return calls, fake_run

    def _run_execute(self, sandbox, script="true"):
        calls, fake_run = self._record_run()
        with mock.patch("bash_agent.sandbox.subprocess.run", new=fake_run):
            code, out = sandbox.execute(script)
        return calls, code, out

    def _run_execute_python(self, sandbox, code="print('x')"):
        calls, fake_run = self._record_run()
        with mock.patch("bash_agent.sandbox.subprocess.run", new=fake_run):
            rc, out = sandbox.execute_python(code)
        return calls, rc, out

    # -- assertion helpers --------------------------------------------------

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

    @staticmethod
    def _has_seq(cmd, seq):
        """True when the consecutive token sequence `seq` appears in `cmd`."""
        n = len(seq)
        return any(cmd[i:i + n] == seq for i in range(len(cmd) - n + 1))

    def _assert_isolation_posture(self, cmd):
        """The isolation flags AGENTS.md promises must be present."""
        self.assertEqual(cmd[0], "bwrap")
        # Host root read-only.
        self.assertTrue(self._has_seq(cmd, ["--ro-bind", "/", "/"]))
        # Private pseudo-filesystems + throwaway tmp.
        self.assertTrue(self._has_seq(cmd, ["--proc", "/proc"]))
        self.assertTrue(self._has_seq(cmd, ["--dev", "/dev"]))
        self.assertTrue(self._has_seq(cmd, ["--tmpfs", "/tmp"]))
        self.assertTrue(self._has_seq(cmd, ["--tmpfs", "/var/tmp"]))
        # Namespace isolation + teardown.
        for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts",
                     "--unshare-cgroup-try", "--die-with-parent"):
            self.assertIn(flag, cmd)


# ---------------------------------------------------------------------------
# T-27 — Construction defaults & write-path bookkeeping
# ---------------------------------------------------------------------------

class TestConstructionDefaults(SandboxCase):
    def test_defaults_fall_back_to_config(self):
        sb = Sandbox("scratchpad-path-unused")
        self.assertEqual(sb.timeout, config.BASH_TIMEOUT)
        self.assertEqual(sb.approved_write_paths, [os.path.abspath(".")])
        self.assertIsNone(sb.uuid)
        self.assertEqual(sb.multimodal_capabilities, [])
        self.assertFalse(sb.is_orchestrator)

    def test_explicit_values_are_stored(self):
        sb = Sandbox(
            "scratchpad-path-unused",
            timeout=7,
            uuid="uid-123",
            multimodal_capabilities=["image"],
        )
        self.assertEqual(sb.timeout, 7)
        self.assertEqual(sb.uuid, "uid-123")
        self.assertEqual(sb.multimodal_capabilities, ["image"])
        self.assertEqual(sb.approved_write_paths, [os.path.abspath(".")])

    def test_orchestrator_is_confined_to_scratch(self):
        sb = Sandbox("scratchpad-path-unused", is_orchestrator=True)
        self.assertTrue(sb.is_orchestrator)
        self.assertEqual(sb.approved_write_paths, [os.path.abspath(".bash_agent_tmp")])


class TestRequestWriteBookkeeping(SandboxCase):
    def _sandbox(self):
        return Sandbox("scratchpad-path-unused", uuid="u")

    def test_approval_appends_abspath_and_returns_tuple(self):
        sb = self._sandbox()
        with mock.patch("builtins.input", return_value="y"):
            ok, msg = sb.request_write("some/rel/path")
        self.assertTrue(ok)
        self.assertEqual(msg, "Write access granted.")
        self.assertIn(os.path.abspath("some/rel/path"), sb.approved_write_paths)
        # Exactly one path was added on top of the CWD default.
        self.assertEqual(len(sb.approved_write_paths), 2)

    def test_uppercase_y_also_approves(self):
        sb = self._sandbox()
        with mock.patch("builtins.input", return_value="Y"):
            ok, msg = sb.request_write("/tmp/probe")
        self.assertTrue(ok)
        self.assertIn("/tmp/probe", sb.approved_write_paths)

    def test_denial_does_not_append(self):
        sb = self._sandbox()
        with mock.patch("builtins.input", return_value="n"):
            ok, msg = sb.request_write("/etc")
        self.assertFalse(ok)
        self.assertNotIn("/etc", sb.approved_write_paths)
        self.assertEqual(sb.approved_write_paths, [os.path.abspath(".")])

    def test_free_text_reply_is_a_deny_with_user_message(self):
        sb = self._sandbox()
        with mock.patch("builtins.input", return_value="tell me why first"):
            ok, msg = sb.request_write("/etc")
        self.assertFalse(ok)
        self.assertEqual(msg, "Write access denied. User message: tell me why first")
        self.assertNotIn("/etc", sb.approved_write_paths)

    def test_orchestrator_never_gets_write_access(self):
        sb = Sandbox("scratchpad-path-unused", is_orchestrator=True)
        ok, msg = sb.request_write("/etc")
        self.assertFalse(ok)
        self.assertIn("Orchestrator agent cannot request write access", msg)


# ---------------------------------------------------------------------------
# T-28 — Command assembly inspection
# ---------------------------------------------------------------------------

class TestBashCommandAssembly(SandboxCase):
    def test_full_argv_shape(self):
        sb = Sandbox("sp", uuid="UID-1", multimodal_capabilities=["image"])
        calls, code, out = self._run_execute(sb, "echo hi")

        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertEqual(len(calls), 1)
        cmd = calls[0]["cmd"]

        # Isolation posture.
        self._assert_isolation_posture(cmd)

        # Non-orchestrator: the project tree is the one writable bind.
        self.assertEqual(self._bind_pairs(cmd), [(os.path.abspath("."), os.path.abspath("."))])
        # ...and /home is NOT masked for a subagent.
        self.assertFalse(self._has_seq(cmd, ["--tmpfs", "/home"]))

        # Working directory pinned to the project root.
        self.assertTrue(self._has_seq(cmd, ["--chdir", os.path.abspath(".")]))

        # Env forwarding: PATH always, UUID/multimodal from constructor state.
        env = self._setenv(cmd)
        self.assertEqual(env["PATH"], os.environ.get("PATH", ""))
        self.assertEqual(env["BASH_AGENT_UUID"], "UID-1")
        self.assertEqual(env["BASH_AGENT_MULTIMODAL"], "image")
        self.assertNotIn("PYTHONPATH", env)

        # Tail: /bin/bash <script under .bash_agent_tmp/*.sh>
        script = cmd[-1]
        self.assertEqual(cmd[-2], "/bin/bash")
        self.assertTrue(script.endswith(".sh"))
        self.assertTrue(
            script.startswith(os.path.join(os.path.abspath("."), ".bash_agent_tmp"))
        )

    def test_orchestrator_masks_home_and_binds_scratch(self):
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        calls, _, _ = self._run_execute(sb, "true")
        cmd = calls[0]["cmd"]

        self._assert_isolation_posture(cmd)
        # /home is masked, then only the scratch dir is punched back in rw.
        self.assertTrue(self._has_seq(cmd, ["--tmpfs", "/home"]))
        scratch = os.path.abspath(".bash_agent_tmp")
        self.assertEqual(self._bind_pairs(cmd), [(scratch, scratch)])
        # Orchestrator workdir is the scratch dir.
        self.assertTrue(self._has_seq(cmd, ["--chdir", scratch]))

    def test_orchestrator_bash_gets_venv_and_package_ro_binds(self):
        """The venv and package are ro-bound for BASH (not just Python)."""
        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        with mock.patch("bash_agent.sandbox.os.path.isdir", return_value=True):
            calls, _, _ = self._run_execute(sb, "true")
        cmd = calls[0]["cmd"]

        venv_dir = os.path.join(os.path.abspath("."), "venv")
        pkg_dir = os.path.join(os.path.abspath("."), "bash_agent")
        self.assertTrue(self._has_seq(cmd, ["--ro-bind", venv_dir, venv_dir]))
        self.assertTrue(self._has_seq(cmd, ["--ro-bind", pkg_dir, pkg_dir]))

    def test_run_kwargs_pin_timeout_and_merged_stderr(self):
        sb = Sandbox("sp", uuid="u", timeout=13)
        calls, _, _ = self._run_execute(sb)
        kwargs = calls[0]["kwargs"]
        self.assertEqual(kwargs["timeout"], 13)
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_api_key_forwarded_only_when_set(self):
        env_without_key = {
            k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"
        }

        # Unset -> no key forwarded at all
        with mock.patch.dict(os.environ, dict(env_without_key), clear=True):
            sb = Sandbox("sp", uuid="u")
            calls, _, _ = self._run_execute(sb)
            self.assertNotIn("OPENROUTER_API_KEY", self._setenv(calls[0]["cmd"]))

        # Set -> forwarded verbatim
        with mock.patch.dict(
            os.environ,
            {**env_without_key, "OPENROUTER_API_KEY": "sk-test"},
            clear=True,
        ):
            calls, _, _ = self._run_execute(sb)
            self.assertEqual(
                self._setenv(calls[0]["cmd"])["OPENROUTER_API_KEY"], "sk-test"
            )


class TestPythonCommandAssembly(SandboxCase):
    def test_fallback_interpreter_without_project_venv(self):
        # chdir_tmp gives us a bare dir: no ./venv/bin/python3 present.
        sb = Sandbox("sp", uuid="UID-2", multimodal_capabilities=[])
        calls, _, _ = self._run_execute_python(sb, "print('x')")
        cmd = calls[0]["cmd"]

        self._assert_isolation_posture(cmd)

        env = self._setenv(cmd)
        self.assertEqual(env["PYTHONPATH"], os.path.abspath("."))
        self.assertEqual(env["BASH_AGENT_UUID"], "UID-2")
        self.assertEqual(env["BASH_AGENT_MULTIMODAL"], "")
        # execute_python() does not forward PATH.
        self.assertNotIn("PATH", env)

        # Tail: /usr/bin/env python3 <script>.py
        script = cmd[-1]
        self.assertEqual(cmd[-3:-1], ["/usr/bin/env", "python3"])
        self.assertTrue(script.endswith(".py"))
        self.assertTrue(
            script.startswith(os.path.join(os.path.abspath("."), ".bash_agent_tmp"))
        )

    def test_prefers_project_venv_python_when_present(self):
        venv_bin = os.path.join(self.tmpdir, "venv", "bin")
        os.makedirs(venv_bin, exist_ok=True)
        venv_py = os.path.join(venv_bin, "python3")
        with open(venv_py, "w") as f:
            f.write("#!/bin/sh\n")  # never executed; argv inspection only

        sb = Sandbox("sp", uuid="u")
        calls, _, _ = self._run_execute_python(sb)
        cmd = calls[0]["cmd"]
        self.assertEqual(cmd[-2], venv_py)
        self.assertTrue(cmd[-1].endswith(".py"))

    def test_orchestrator_venv_bound_read_only(self):
        # A project venv is visible to the orchestrator read-only so the
        # interpreter keeps working even though /home is masked.
        venv_bin = os.path.join(self.tmpdir, "venv", "bin")
        os.makedirs(venv_bin, exist_ok=True)
        with open(os.path.join(venv_bin, "python3"), "w") as f:
            f.write("#!/bin/sh\n")

        sb = Sandbox("sp", uuid="u", is_orchestrator=True)
        calls, _, _ = self._run_execute_python(sb)
        cmd = calls[0]["cmd"]

        venv_dir = os.path.join(os.path.abspath("."), "venv")
        self.assertTrue(self._has_seq(cmd, ["--ro-bind", venv_dir, venv_dir]))
        self.assertTrue(self._has_seq(cmd, ["--tmpfs", "/home"]))


# ---------------------------------------------------------------------------
# T-29 — Temp-script hygiene
# ---------------------------------------------------------------------------

class TestTempScriptHygiene(SandboxCase):
    def test_script_has_mode_0700_while_running_then_removed(self):
        sb = Sandbox("sp", uuid="u")
        seen = {}

        def recorder(cmd, **kwargs):
            script = cmd[-1]
            seen["path"] = script
            seen["mode"] = stat.S_IMODE(os.stat(script).st_mode)
            with open(script) as f:
                seen["content"] = f.read()
            return _FakeCompleted()

        with mock.patch("bash_agent.sandbox.subprocess.run", new=recorder):
            code, out = sb.execute("echo marker-content")

        self.assertEqual(code, 0)
        # Mode was 0700 while the script existed at invocation time...
        self.assertEqual(seen["mode"], 0o700)
        # ...the exact script content was handed to the interpreter...
        self.assertEqual(seen["content"], "echo marker-content")
        # ...and the mkstemp'd file is gone afterwards.
        self.assertFalse(os.path.exists(seen["path"]))
        leftovers = [
            f for f in os.listdir(".bash_agent_tmp") if f.endswith((".sh", ".py"))
        ]
        self.assertEqual(leftovers, [])

    def test_bash_timeout_maps_to_124_with_banner_and_partial_output(self):
        sb = Sandbox("sp", uuid="u", timeout=5)
        seen = {}

        def boom(cmd, **kwargs):
            seen["path"] = cmd[-1]
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5, output="partial-line\n")

        with mock.patch("bash_agent.sandbox.subprocess.run", new=boom):
            code, out = sb.execute("sleep 999")

        # Load-bearing mapping: the LLM reads exit codes to decide retries.
        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Command timed out after 5 seconds.", out)
        self.assertIn("Partial Output:", out)
        self.assertIn("partial-line", out)
        # Cleanup happens even on the failure path.
        self.assertFalse(os.path.exists(seen["path"]))

    def test_python_timeout_maps_to_124_with_python_banner(self):
        sb = Sandbox("sp", uuid="u", timeout=3)

        def boom(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=3, output="traceback-ish\n")

        with mock.patch("bash_agent.sandbox.subprocess.run", new=boom):
            code, out = sb.execute_python("import time; time.sleep(999)")

        self.assertEqual(code, 124)
        self.assertIn("[SYSTEM ERROR] Python command timed out after 3 seconds.", out)
        self.assertIn("traceback-ish", out)


if __name__ == "__main__":
    unittest.main()
