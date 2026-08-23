"""
Group 6 — Sandbox unit tests (no systemd involved).

T-27  Construction defaults & write-path bookkeeping (P1)
T-28  Command assembly inspection (P1)
T-29  Temp-script hygiene (P1)

These tests pin the Sandbox surface that the LLM-facing protocol depends on:

  * A fresh Sandbox falls back to config.BASH_TIMEOUT, starts with exactly
    one approved write path (the abspath of "."), and stores uuid/multimodal.
  * request_write() driven by patched builtins.input appends the *abspath*
    only on approval and always returns a (bool, message) tuple — the same
    contract agent.py unpacks as `success, msg = sandbox.request_write(path)`.
  * The argv assembled for systemd-run carries the full security posture
    (ProtectSystem=strict, ProtectHome=read-only, PrivateTmp=yes), the
    working-directory flag, env forwarding (PATH always; OPENROUTER_API_KEY
    only when set; BASH_AGENT_UUID / BASH_AGENT_MULTIMODAL), one
    ReadWritePaths property per approved path, and a /bin/bash <script> or
    <venv-python|/usr/bin/env python3> <script>.py tail. If someone drops a
    security property, these tests go red before users do.
  * Temp-script hygiene: the mkstemp'd script under .bash_agent_tmp/ has
    mode 0700 while it exists, is removed after every call (success or
    timeout), and TimeoutExpired maps to exit code 124 with the
    "[SYSTEM ERROR] ... timed out" banner plus partial stdout — the exit
    code the LLM reads to decide retries.

All tests are offline: subprocess.run is replaced with recorders/fakes, so
systemd-run is never invoked. Real-sandbox behavior is covered separately by
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

    def _props(self, cmd):
        return [a for a in cmd if a.startswith("--property=")]

    def _assert_security_posture(self, cmd):
        """The three isolation properties AGENTS.md promises must be present."""
        props = self._props(cmd)
        self.assertIn("--property=ProtectSystem=strict", props)
        self.assertIn("--property=ProtectHome=read-only", props)
        self.assertIn("--property=PrivateTmp=yes", props)


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

        # Launcher flags
        self.assertEqual(cmd[0], "systemd-run")
        for flag in ("--user", "--quiet", "--wait", "--collect", "--pipe"):
            self.assertIn(flag, cmd)

        # Security posture
        self._assert_security_posture(cmd)

        # Working directory pinned to the project root
        self.assertIn(f"--working-directory={os.path.abspath('.')}", cmd)

        # Env forwarding: PATH always, UUID/multimodal from constructor state
        props = self._props(cmd)
        self.assertIn(f"--property=Environment=PATH={os.environ.get('PATH', '')}", props)
        self.assertIn("--property=Environment=BASH_AGENT_UUID=UID-1", props)
        self.assertIn("--property=Environment=BASH_AGENT_MULTIMODAL=image", props)

        # One ReadWritePaths per approved write path, in order
        rw = [p for p in props if p.startswith("--property=ReadWritePaths=")]
        self.assertEqual(
            rw,
            [f"--property=ReadWritePaths={p}" for p in sb.approved_write_paths],
        )

        # Tail: /bin/bash <script under .bash_agent_tmp/*.sh>
        script = cmd[-1]
        self.assertEqual(cmd[-2], "/bin/bash")
        self.assertTrue(script.endswith(".sh"))
        self.assertTrue(
            script.startswith(os.path.join(os.path.abspath("."), ".bash_agent_tmp"))
        )

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

        # Unset -> no key property at all
        with mock.patch.dict(os.environ, dict(env_without_key), clear=True):
            sb = Sandbox("sp", uuid="u")
            calls, _, _ = self._run_execute(sb)
            props = self._props(calls[0]["cmd"])
            self.assertFalse(any("OPENROUTER_API_KEY" in p for p in props))

        # Set -> forwarded verbatim
        with mock.patch.dict(
            os.environ,
            {**env_without_key, "OPENROUTER_API_KEY": "sk-test"},
            clear=True,
        ):
            calls, _, _ = self._run_execute(sb)
            props = self._props(calls[0]["cmd"])
            self.assertIn(
                "--property=Environment=OPENROUTER_API_KEY=sk-test", props
            )


class TestPythonCommandAssembly(SandboxCase):
    def test_fallback_interpreter_without_project_venv(self):
        # chdir_tmp gives us a bare dir: no ./venv/bin/python3 present.
        sb = Sandbox("sp", uuid="UID-2", multimodal_capabilities=[])
        calls, _, _ = self._run_execute_python(sb, "print('x')")
        cmd = calls[0]["cmd"]

        self._assert_security_posture(cmd)

        props = self._props(cmd)
        self.assertIn(f"--property=Environment=PYTHONPATH={os.path.abspath('.')}", props)
        self.assertIn("--property=Environment=BASH_AGENT_UUID=UID-2", props)
        self.assertIn("--property=Environment=BASH_AGENT_MULTIMODAL=", props)

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
