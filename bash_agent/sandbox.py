import os
import subprocess
import tempfile
from bash_agent.config import BASH_TIMEOUT

class Sandbox:
    def __init__(self, scratchpad_path: str, timeout: int = None, uuid: str = None, multimodal_capabilities: list = None):
        self.approved_write_paths = [os.path.abspath(".")]
        self.timeout = timeout if timeout is not None else BASH_TIMEOUT
        self.uuid = uuid
        self.multimodal_capabilities = multimodal_capabilities or []
        
    def request_write(self, path: str) -> bool:
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

    def execute(self, script_content: str) -> tuple[int, str]:
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
            "systemd-run", "--user", "--quiet", "--wait", "--collect", "--pipe",
            "--property=ProtectSystem=strict",
            "--property=ProtectHome=read-only",
            "--property=PrivateTmp=yes",
            # Optional but helpful: start the bash session in the project directory
            f"--working-directory={os.path.abspath('.')}",
            f"--property=Environment=PATH={host_path}"
        ]

        if openrouter_key:
            cmd.append(f"--property=Environment=OPENROUTER_API_KEY={openrouter_key}")

        if self.uuid:
            cmd.append(f"--property=Environment=BASH_AGENT_UUID={self.uuid}")
        cmd.append(f"--property=Environment=BASH_AGENT_MULTIMODAL={','.join(self.multimodal_capabilities)}")
        
        for path in self.approved_write_paths:
            cmd.append(f"--property=ReadWritePaths={path}")
            
        cmd.extend(["/bin/bash", script_path])


        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                timeout=self.timeout
            )
            output = result.stdout
            exit_code = result.returncode
            
            # --- DEBUG PRINT ---
            # Uncomment this if 127 persists so we can see Bash's exact complaint:
            # print(f"\n[DEBUG] Exit: {exit_code} | Output: {output.strip()}")

        except subprocess.TimeoutExpired as e:
            # Because we routed stderr to stdout, all partial output is in e.stdout
            partial_out = e.stdout if e.stdout else ""
            output = f"[SYSTEM ERROR] Command timed out after {self.timeout} seconds.\nPartial Output:\n{partial_out}"
            exit_code = 124 # Standard timeout exit code

        except Exception as e:
            output = str(e)
            exit_code = 1

        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        return exit_code, output

    def execute_python(self, script_content: str) -> tuple[int, str]:
        local_tmp = os.path.abspath(".bash_agent_tmp")
        os.makedirs(local_tmp, exist_ok=True)
        
        # Suffix is .py instead of .sh
        fd, script_path = tempfile.mkstemp(suffix=".py", dir=local_tmp, text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o700)

        cmd = [
            "systemd-run", "--user", "--quiet", "--wait", "--collect", "--pipe",
            "--property=ProtectSystem=strict",
            "--property=ProtectHome=read-only",
            "--property=PrivateTmp=yes",
            f"--working-directory={os.path.abspath('.')}",
            f"--property=Environment=PYTHONPATH={os.path.abspath(".")}",
        ]
        
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if openrouter_key:
            cmd.append(f"--property=Environment=OPENROUTER_API_KEY={openrouter_key}")

        if self.uuid:
            cmd.append(f"--property=Environment=BASH_AGENT_UUID={self.uuid}")
        cmd.append(f"--property=Environment=BASH_AGENT_MULTIMODAL={','.join(self.multimodal_capabilities)}")

        for path in self.approved_write_paths:
            cmd.append(f"--property=ReadWritePaths={path}")
            
        # Execute python3 instead of bash
        # Determine which python executable to use
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python3")

        if os.path.exists(venv_python):
            cmd.extend([venv_python, script_path])
        else:
            cmd.extend(["/usr/bin/env", "python3", script_path])

        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                timeout=self.timeout
            )
            output = result.stdout
            exit_code = result.returncode
            
        except subprocess.TimeoutExpired as e:
            partial_out = e.stdout if e.stdout else ""
            output = f"[SYSTEM ERROR] Python command timed out after {self.timeout} seconds.\nPartial Output:\n{partial_out}"
            exit_code = 124

        except Exception as e:
            output = str(e)
            exit_code = 1

        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        return exit_code, output

