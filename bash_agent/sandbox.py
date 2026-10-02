"""Execution sandbox backed by bubblewrap (``bwrap``).

Agent-generated BASH/PYTHON scripts are executed inside a bubblewrap
container:

* ``--ro-bind / /`` makes the whole host filesystem READ-ONLY readable; the
  one approved write directory is then bind-mounted back in read/write with
  ``--bind``. The readable tree is identical for every role; only the
  approved-write path differs:

  - subagents / standalone agents write the project tree (the launch cwd);
  - the orchestrator writes ONLY ``.bash_agent_tmp/``.

  Because the whole root is readable, the orchestrator can inspect any file
  (e.g. run the semantic ``search`` tool, ``cat`` project source, list
  directories) while still being unable to modify anything outside its
  scratch directory.
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


# Resolve the project root from the PACKAGE LOCATION rather than the launch
# cwd.  ``bagent -o`` may be launched from any directory, and the venv's
# ``bin/`` directory is prepended to ``PATH`` so CLI tools such as ``vision``
# stay discoverable regardless of where the process was started.  Deriving the
# path from ``__file__`` keeps it correct even for a foreign launch cwd.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # <project>/bash_agent
_PROJECT_ROOT = os.path.dirname(_PKG_DIR)               # <project>


def _resolve_venv_dir() -> str:
    """Return the project ``venv/`` directory, resolved from the package
    location with a cwd-based fallback for unusual layouts.

    The venv's ``bin/`` is prepended to the sandbox ``PATH`` so CLI tools such
    as ``vision`` stay discoverable for every role.  Resolution must NOT depend
    on the launch cwd (``bagent -o`` can be started anywhere), which is why
    ``_PROJECT_ROOT`` is preferred.
    """
    candidate = os.path.join(_PROJECT_ROOT, "venv")
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(os.getcwd(), "venv")


class Sandbox:
    def __init__(self, scratchpad_path: str = None, timeout: int = None, uuid: str = None, multimodal_capabilities: list = None, is_orchestrator: bool = False):
        self.is_orchestrator = is_orchestrator
        # Every role reads the whole (read-only) host root and runs from the
        # project directory, so tools like ``search``/``ls``/``cat`` see the
        # real tree.  Only the approved WRITE path differs: the orchestrator may
        # write solely inside ``.bash_agent_tmp/`` (delegating code edits to
        # subagents), whereas subagents / standalone agents may write the
        # project tree.
        self._workdir = os.path.abspath(".")
        if is_orchestrator:
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

        The readable tree is IDENTICAL for every role: ``_bwrap_common()``
        already binds the whole host root read-only, so the orchestrator can
        inspect any file (project source, ``AGENTS.md``, the semantic ``search``
        index, ...) just like a subagent. The ONLY difference between roles is
        which directory is punched back in as writable via ``--bind``:

        * subagents / standalone agents: the project tree (the launch cwd);
        * orchestrators: ``.bash_agent_tmp/`` only.

        Nothing outside ``.bash_agent_tmp/`` ever becomes writable for an
        orchestrator, so it is still heavily incentivized to delegate code
        changes to a subagent -- but it no longer needs a subagent merely to
        READ a file or run ``search``.

        The ``with_venv`` argument is retained only for backward compatibility;
        it no longer affects the argv because the read-only root bind already
        exposes the project ``venv/`` (whose ``bin/`` is on ``PATH``).
        """
        args = []
        for path in self.approved_write_paths:
            args += ["--bind", path, path]
        return args

    def _resolve_timeout(self, timeout=None):
        """Resolve per-call timeout to a valid positive int (>=1),
        falling back to self.timeout for non-int / <1 / bool."""
        if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout >= 1:
            return timeout
        return self.timeout

    def request_write(self, path: str) -> bool:
        if self.is_orchestrator:
            # The orchestrator may read anything but write ONLY inside
            # .bash_agent_tmp/. Code/file edits must be delegated to a subagent.
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

        # Prepend the resolved venv ``bin`` so CLI tools such as ``vision``
        # are discoverable even when the orchestrator was launched from a
        # foreign directory; keep the inherited host PATH afterwards so system
        # tools still resolve.  ``--setenv PATH`` below remains the single
        # place PATH is set inside the sandbox.
        host_path = os.environ.get("PATH", "")
        venv_bin = os.path.join(_resolve_venv_dir(), "bin")
        if os.path.isdir(venv_bin):
            host_path = venv_bin + ":" + host_path
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

        # Start the bash session in the project directory (all roles).
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
            # Resolve ``bash_agent`` imports reliably even when launched from a
            # foreign cwd: the workdir alone is not the package location, so
            # additionally expose the project root (the editable finder maps
            # ``bash_agent`` -> _PKG_DIR, which lives under _PROJECT_ROOT).
            "--setenv", "PYTHONPATH", os.pathsep.join(
                p for p in (self._workdir, _PROJECT_ROOT) if p
            ),
        ]

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            cmd += ["--setenv", "OPENROUTER_API_KEY", openrouter_key]

        if self.uuid:
            cmd += ["--setenv", "BASH_AGENT_UUID", self.uuid]
        cmd += ["--setenv", "BASH_AGENT_MULTIMODAL", ",".join(self.multimodal_capabilities)]

        # Execute python3 instead of bash.
        # Determine which python executable to use.
        venv_python = os.path.join(_resolve_venv_dir(), "bin", "python3")
        if not os.path.exists(venv_python):
            # Fallback to a cwd-relative venv for unusual layouts.
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
