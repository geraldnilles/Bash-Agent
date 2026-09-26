"""Execution sandbox backed by bubblewrap (``bwrap``).

Agent-generated BASH/PYTHON scripts are executed inside a bubblewrap
container:

* ``--ro-bind / /`` makes the whole host filesystem read-only; the one
  approved write directory (the project tree for subagents, or
  ``.bash_agent_tmp/`` for the orchestrator) is then bind-mounted back in
  read/write with ``--bind``.
* A private ``/proc``, ``/dev`` and throwaway ``/tmp`` / ``/var/tmp`` keep
  the workload from observing or mutating host state.
* ``--unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try`` give
  the workload its own namespaces.

Unlike the previous ``systemd-run`` implementation, there is **no transient
unit to reap**: ``--die-with-parent`` ties the sandbox to the client process,
so when a client-side timeout kills ``bwrap`` the whole process tree (e.g. a
runaway ``sleep 999``) is torn down with it -- no ``systemctl`` cleanup
required.
"""

import os
import subprocess
import tempfile
from bash_agent.config import BASH_TIMEOUT


class Sandbox:
    def __init__(self, scratchpad_path: str = None, timeout: int = None, uuid: str = None, multimodal_capabilities: list = None, is_orchestrator: bool = False):
        self.is_orchestrator = is_orchestrator
        self._workdir = os.path.abspath(".bash_agent_tmp" if is_orchestrator else ".")
        if is_orchestrator:
            # Orchestrator is confined entirely to .bash_agent_tmp/: it may
            # write ONLY there and cannot read the user's home / project tree.
            # Tactical file work is delegated to subagents.
            self.approved_write_paths = [os.path.abspath(".bash_agent_tmp")]
        else:
            self.approved_write_paths = [os.path.abspath(".")]
        self.timeout = timeout if timeout is not None else BASH_TIMEOUT
        self.uuid = uuid
        self.multimodal_capabilities = multimodal_capabilities or []

    def _bwrap_common(self) -> list:
        """Common bubblewrap argv prefix shared by every invocation.

        The host root is bound read-only, then throwaway private
        pseudo-filesystems are mounted over ``/proc``, ``/dev``, ``/tmp`` and
        ``/var/tmp``.  Namespace isolation is applied and ``--die-with-parent``
        guarantees the container dies with its client process -- which is what
        provides clean process-tree teardown on a client-side timeout.
        """
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--tmpfs", "/var/tmp",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--die-with-parent",
        ]

    def _role_args(self, with_venv: bool = False) -> list:
        """Filesystem-isolation argv that depends on the agent's role.

        Non-orchestrators keep the user's home read-only (the ``--ro-bind / /``
        default): only the project tree is punched back in as writable, so the
        rest of ``/home`` stays visible but read-only.

        Orchestrators must see and write ONLY ``.bash_agent_tmp/`` -- but that
        directory lives *inside* the home tree.  A tmpfs is mounted over
        ``/home`` (masking the user's home / project tree) and the working
        directory is bind-mounted back in read/write.
        ``execute_python()`` additionally needs the project venv interpreter,
        which also lives under the masked tree, so it is bound back in
        read-only via ``with_venv=True``.
        """
        if not self.is_orchestrator:
            args = []
            for path in self.approved_write_paths:
                args += ["--bind", path, path]
            return args

        args = ["--tmpfs", "/home"]
        for path in self.approved_write_paths:
            args += ["--bind", path, path]
        if with_venv:
            venv_dir = os.path.join(os.getcwd(), "venv")
            if os.path.isdir(venv_dir):
                args += ["--ro-bind", venv_dir, venv_dir]
        return args

    def _resolve_timeout(self, timeout=None):
        """Resolve per-call timeout to a valid positive int (>=1),
        falling back to self.timeout for non-int / <1 / bool."""
        if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout >= 1:
            return timeout
        return self.timeout

    def request_write(self, path: str) -> bool:
        if self.is_orchestrator:
            return (
                False,
                "Orchestrator agent cannot request write access outside "
                ".bash_agent_tmp. Delegate file changes to a subagent.",
            )
        abs_path = os.path.abspath(path)
        print(f"\n[AGENT REQUEST] The agent is requesting write access to: {abs_path}")
        ans = input("Approve? (y/n/message): ")
        if ans.lower() == 'y':
            self.approved_write_paths.append(abs_path)
            return True, "Write access granted."
        elif ans.lower() == 'n':
            return False, "Write access denied by user."
        else:
            return False, f"Write access denied. User message: {ans}"

    def execute(self, script_content: str, timeout: int = None) -> tuple[int, str]:
        effective_timeout = self._resolve_timeout(timeout)
        # Create a local temp directory that the sandbox can see
        local_tmp = os.path.abspath(".bash_agent_tmp")
        os.makedirs(local_tmp, exist_ok=True)

        # Force tempfile to use our local directory instead of the host's /tmp
        fd, script_path = tempfile.mkstemp(suffix=".sh", dir=local_tmp, text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o700)

        host_path = os.environ.get("PATH", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        cmd = [
            *self._bwrap_common(),
            *self._role_args(),
            # Environment: bwrap inherits the parent env by default, so we only
            # (re)set the variables the previous implementation forwarded.
            "--setenv", "PATH", host_path,
        ]

        if openrouter_key:
            cmd += ["--setenv", "OPENROUTER_API_KEY", openrouter_key]

        if self.uuid:
            cmd += ["--setenv", "BASH_AGENT_UUID", self.uuid]
        cmd += ["--setenv", "BASH_AGENT_MULTIMODAL", ",".join(self.multimodal_capabilities)]

        # Start the bash session in the project directory (orchestrator:
        # confined to .bash_agent_tmp/).
        cmd += ["--chdir", self._workdir, "/bin/bash", script_path]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=effective_timeout
            )
            output = result.stdout
            exit_code = result.returncode

        except subprocess.TimeoutExpired as e:
            # --die-with-parent already tore down the sandboxed process tree
            # when bwrap was killed, so there is nothing left to reap.
            # Because we routed stderr to stdout, all partial output is in e.stdout
            partial_out = e.stdout if e.stdout else ""
            output = f"[SYSTEM ERROR] Command timed out after {effective_timeout} seconds.\nPartial Output:\n{partial_out}"
            exit_code = 124  # Standard timeout exit code

        except Exception as e:
            output = str(e)
            exit_code = 1

        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        return exit_code, output

    def execute_python(self, script_content: str, timeout: int = None) -> tuple[int, str]:
        effective_timeout = self._resolve_timeout(timeout)
        local_tmp = os.path.abspath(".bash_agent_tmp")
        os.makedirs(local_tmp, exist_ok=True)

        # Suffix is .py instead of .sh
        fd, script_path = tempfile.mkstemp(suffix=".py", dir=local_tmp, text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o700)

        cmd = [
            *self._bwrap_common(),
            *self._role_args(with_venv=True),
            # bwrap inherits the parent env; set exactly the variables the
            # previous implementation forwarded for the python path.
            "--setenv", "PYTHONPATH", self._workdir,
        ]

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            cmd += ["--setenv", "OPENROUTER_API_KEY", openrouter_key]

        if self.uuid:
            cmd += ["--setenv", "BASH_AGENT_UUID", self.uuid]
        cmd += ["--setenv", "BASH_AGENT_MULTIMODAL", ",".join(self.multimodal_capabilities)]

        # Execute python3 instead of bash.
        # Determine which python executable to use.
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python3")

        cmd += ["--chdir", self._workdir]
        if os.path.exists(venv_python):
            cmd += [venv_python, script_path]
        else:
            cmd += ["/usr/bin/env", "python3", script_path]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=effective_timeout
            )
            output = result.stdout
            exit_code = result.returncode

        except subprocess.TimeoutExpired as e:
            # --die-with-parent handles teardown (see execute()).
            partial_out = e.stdout if e.stdout else ""
            output = f"[SYSTEM ERROR] Python command timed out after {effective_timeout} seconds.\nPartial Output:\n{partial_out}"
            exit_code = 124

        except Exception as e:
            output = str(e)
            exit_code = 1

        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        return exit_code, output
