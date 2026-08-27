#!/usr/bin/env python3

import os
import urllib.request
import json as json_module
import re
import sys
import time
import uuid
import select
from typing import List, Dict

from bash_agent.prompts import get_system_prompt
from bash_agent.config import DEFAULT_MODEL, OUTPUT_LIMIT, MAX_CODE_BLOCKS, COLOR_CMD, COLOR_OUT, COLOR_PY_CMD, COLOR_COST, COLOR_RESET, DEFAULT_REASONING_EFFORT, DEFAULT_MAX_TOKENS, CONTEXT_LIMIT, DEFAULT_BUDGET
from bash_agent.utils import cleanup_tmp_folder, copy_project_to_clipboard, get_clipboard_content, get_vim_prompt
from bash_agent.config_file import load_config
from bash_agent import llm
from bash_agent.context import ContextManager
from bash_agent.sandbox import Sandbox
try:
    from bash_agent import sfx as _sfx
    _SFX_AVAILABLE = True
except Exception:
    _sfx = None
    _SFX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Warmup exchanges
#
# Fresh sessions are pre-filled with two scripted assistant turns (each
# followed by its REAL sandbox output) so the model immediately sees concrete,
# correctly-formatted examples of the UUID-fenced protocol. This shortens the
# learning curve for models that struggle with the custom block format.
# Skipped entirely on --resume, since restored history already contains turns.
# Scripts are NOT duplicated here: _run_warmup_exchanges() parses each template
# with the production _extract_blocks() and executes whatever it extracts.
# ---------------------------------------------------------------------------
WARMUP_TURNS = [
    (
        "I'll start by getting oriented. First, let me check which Python "
        "version is available in the sandbox and print all 3rd-party PyPI "
        "packages currently installed here.\n\n"
        "---START_PYTHON_COMMAND-{uuid}---\n"
        "import sys\n"
        "from importlib import metadata\n"
        "\n"
        "print('Python', sys.version)\n"
        "stdlib = getattr(sys, 'stdlib_module_names', frozenset())\n"
        "seen = set()\n"
        "rows = []\n"
        "for dist in metadata.distributions():\n"
        "    name = dist.metadata.get('Name')\n"
        "    if not name or name.lower() in seen:\n"
        "        continue\n"
        "    seen.add(name.lower())\n"
        "    # Skip stdlib-only distributions (not real PyPI installs)\n"
        "    tops = set()\n"
        "    txt = dist.read_text('top_level.txt')\n"
        "    if txt:\n"
        "        tops.update(line.strip() for line in txt.splitlines() if line.strip())\n"
        "    if tops and tops <= stdlib:\n"
        "        continue\n"
        "    rows.append(name + '==' + dist.version)\n"
        "\n"
        "print()\n"
        "print('3rd-party PyPI packages installed in this sandbox:')\n"
        "for row in sorted(rows):\n"
        "    print(' ', row)\n"
        "print('total:', len(rows))\n"
        "---END_PYTHON_COMMAND-{uuid}---"
    ),
    (
        "Good. Now let me list the files in the current working directory to "
        "see what I'm working with.\n\n"
        "---START_BASH_COMMAND-{uuid}---\n"
        "ls -la\n"
        "---END_BASH_COMMAND-{uuid}---"
    ),
]


_TMP_ERROR_PATTERN = re.compile(
    r"("
    r"\bno such file\b"
    r"|\bnot found\b"
    r"|\bcannot (open|find|access|locate)\b"
    r"|\bcould not (open|find|access|locate)\b"
    r"|\bunable to (open|find|access|locate)\b"
    r"|\bfailed to (open|find|access|locate|read|write)\b"
    r"|\bdoes not exist\b"
    r")",
    re.IGNORECASE,
)

def _build_tmp_file_warning(exit_code: int, output: str) -> str | None:
    """
    Detect a failed command that referenced a /tmp/ path and return a reminder
    that /tmp/ is wiped after every turn, or None when no reminder is needed.

    The heuristic only fires when ALL of the following are true:
      * the command exited non-zero (failure)
      * the output references /tmp/
      * the output contains a familiar file-not-found style error phrase
    """
    if exit_code == 0:
        return None
    if "/tmp/" not in output:
        return None
    if not _TMP_ERROR_PATTERN.search(output):
        return None

    return (
        "\u26a0\ufe0f [SYSTEM WARNING] The command failed because it referenced a file under /tmp/\n"
        "The /tmp/ folder is refreshed after every turn on THIS host and does not persist between turns.\n"
        "If you need a temporary file that lasts the whole session, write it into the "
        ".bash_agent_tmp/ folder in the project directory instead."
    )


TRUNCATION_BANNER = "\n\n---⚠️⛔⚠️-OUTPUT_TRUNCATED_HERE-{uuid}-⚠️⛔⚠️---\n\n"


class Agent:
    def __init__(self, keep_tmp=False, debug=False, model=None, reasoning_effort=None, max_tokens=None, timeout=None, resume=False, budget=None):
        self.uuid = str(uuid.uuid4())
        self.debug = debug
        
        # Optional persistent settings file (.bash_agent_tmp/config.json).
        # Precedence: CLI args > config.json > env var > hard-coded defaults.
        try:
            self.file_config = load_config()
        except Exception as e:
            print(f"[Config] Failed to load .bash_agent_tmp/config.json ({e}); ignoring.", file=sys.stderr)
            self.file_config = {}

        # Model priority: CLI arg > config.json > OPENROUTER_MODEL env var > default
        if model:
            self.model = model
        elif "model" in self.file_config:
            self.model = self.file_config["model"]
        elif os.environ.get("OPENROUTER_MODEL"):
            self.model = os.environ.get("OPENROUTER_MODEL")
        else:
            self.model = DEFAULT_MODEL
        
        # Clean up tmp folder unless --keep-tmp was specified or --resume is active
        if not keep_tmp and not resume:
            cleanup_tmp_folder()
        
        self.context = ContextManager(self.uuid)
        
        # Multimodal detection (must happen before Sandbox instantiation so the
        # sandbox environment can expose BASH_AGENT_MULTIMODAL to vision.py)
        self.multimodal_capabilities = None
        self._check_model_capabilities()
        
        # Fetch reasoning capabilities from OpenRouter
        self._fetch_model_reasoning_info()

        # Handle Resume: attempt to restore previous session
        history_loaded = False
        if resume:
            history_loaded = self.context.load_history()
            if history_loaded:
                # Re-bind components to the restored UUID
                self.uuid = self.context.uuid
                self.sandbox = Sandbox(self.context.scratchpad_path, timeout=timeout, uuid=self.uuid, multimodal_capabilities=self.multimodal_capabilities)
                print(f"[System] Resumed previous session. Re-bound to UUID: {self.uuid}")
        
        if not history_loaded:
            self.sandbox = Sandbox(self.context.scratchpad_path, timeout=timeout, uuid=self.uuid, multimodal_capabilities=self.multimodal_capabilities)
        # Attach the session UUID to all OpenRouter calls (consistent routing
        # improves prompt-cache hits across requests in the same session).
        llm.set_session_id(self.uuid)
        # Remember whether this session was restored from history.json;
        # run() uses it to skip the protocol warmup exchanges on --resume.
        self.resumed_session = history_loaded
        
        # Reasoning effort and max tokens: CLI arg > config.json > hard-coded defaults.
        # Per-key fallback, so e.g. --max-tokens alone overrides only max_tokens.
        if reasoning_effort is not None:
            # Explicit CLI choice always wins over config.json; 'default'
            # maps to None (= defer to the model's built-in reasoning default).
            self.reasoning_effort = None if reasoning_effort == 'default' else reasoning_effort
        elif "reasoning_effort" in self.file_config:
            file_effort = self.file_config["reasoning_effort"]
            # 'default' in the file means: use the model's built-in default
            self.reasoning_effort = None if file_effort == 'default' else file_effort
        else:
            self.reasoning_effort = DEFAULT_REASONING_EFFORT

        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif "max_tokens" in self.file_config:
            self.max_tokens = self.file_config["max_tokens"]
        else:
            self.max_tokens = DEFAULT_MAX_TOKENS
        
        # Adjust reasoning_effort based on model capabilities
        if self.reasoning_effort == "none" and self.reasoning_mandatory:
            # Model requires reasoning, use the lowest supported level
            self.reasoning_effort = self._get_lowest_reasoning_effort()
        elif self.reasoning_effort is not None and self.reasoning_effort not in self.reasoning_supported_efforts:
            # Requested effort not supported, fall back to lowest supported
            self.reasoning_effort = self._get_lowest_reasoning_effort()

        if self.debug:
            # Consolidated model capabilities blob — shows the final
            # reasoning effort (after mandatory/supported adjustments) and
            # the full set of supported efforts.
            if self.reasoning_effort is None:
                _selected = f"default ({self.reasoning_default_effort})"
            else:
                _selected = self.reasoning_effort
            _blob = (
                f"[Debug] Model '{self.model}' capabilities:\n"
                f"[Debug]   reasoning supported={self.reasoning_supported_efforts}, "
                f"mandatory={self.reasoning_mandatory}, default={self.reasoning_default_effort}, selected={_selected}\n"
                f"[Debug]   multimodal={self.multimodal_capabilities}"
            )
            print(_blob)

        # Track attached images emitted by the sandbox (e.g., via the vision command)
        self._pending_multimodal_images = []
        # Track attached audio emitted by the sandbox (e.g., via transcribe)
        self._pending_multimodal_audio = []
        
        # Check for custom role definition in ROLE.md (only if not resuming)
        if not history_loaded:
            custom_role = None
            # First, check .bash_agent_tmp/ROLE.md
            role_file_path = os.path.join(os.path.abspath(".bash_agent_tmp"), "ROLE.md")
            if not os.path.exists(role_file_path):
                # Fallback: check ROLE.md in current working directory
                role_file_path = os.path.join(os.path.abspath("."), "ROLE.md")
            if os.path.exists(role_file_path):
                with open(role_file_path, "r", encoding="utf-8") as f:
                    custom_role = f.read().strip()
            system_prompt = get_system_prompt(self.uuid, os.path.abspath("."), self.context.scratchpad_path, role_text=custom_role, multimodal_capabilities=self.multimodal_capabilities)
            self.context.add_message("system", system_prompt)
        self.budget = budget if budget is not None else DEFAULT_BUDGET
        self.session_cost = 0.0
        self.last_step_cost = 0.0
        self.last_step_input_tokens = 0
        self.last_step_provider = None

    def _check_model_capabilities(self):
        """Query OpenRouter API to detect the active model's input modalities."""
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers=llm.get_attribution_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                models_data = json_module.loads(resp.read().decode("utf-8"))
            for m in models_data if isinstance(models_data, list) else models_data.get("data", []):
                if m.get("id") == self.model:
                    arch = m.get("architecture", {}) or {}
                    input_modalities = arch.get("input_modalities", [])
                    if input_modalities:
                        self.multimodal_capabilities = list(input_modalities)
                    return
        except Exception:
            self.multimodal_capabilities = None

    def _fetch_model_reasoning_info(self):
        """Query OpenRouter API to get the model's reasoning capabilities.

        Uses the /models list endpoint: there is no reliable per-model
        endpoint, and the list response nests each model under "data".
        """
        default_efforts = ["high", "medium", "low", "minimal", "none"]
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers=llm.get_attribution_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                models_data = json_module.loads(resp.read().decode("utf-8"))
            for m in models_data if isinstance(models_data, list) else models_data.get("data", []):
                if m.get("id") != self.model:
                    continue
                reasoning = m.get("reasoning", {}) or {}
                self.reasoning_supported_efforts = reasoning.get("supported_efforts") or default_efforts
                self.reasoning_mandatory = reasoning.get("mandatory", False)
                self.reasoning_default_effort = reasoning.get("default_effort", "medium")
                return
            # Model found in catalog without reasoning metadata; keep defaults.
        except Exception:
            pass
        # Network/API failure: fall back to permissive defaults.
        self.reasoning_supported_efforts = default_efforts
        self.reasoning_mandatory = False
        self.reasoning_default_effort = "medium"

    def _get_lowest_reasoning_effort(self):
        """Get the lowest supported reasoning effort for the current model."""
        # Order from lowest to highest
        effort_order = ["none", "minimal", "low", "medium", "high", "xhigh"]
        for effort in effort_order:
            if effort in self.reasoning_supported_efforts:
                return effort
        return "medium"  # fallback

    def _extract_blocks(self, response_text: str) -> tuple[list[tuple[str, str]], str | None]:
        """
        Extracts valid (cmd_type, script) blocks from the LLM response.
        Returns (blocks, None) on success, or ([], warning_message) if parsing fails.
        """
        # A fenced body must never contain another command-fence marker. This
        # prevents a mismatched START/END pair (e.g. START_BASH ... END_PYTHON,
        # see test T-05) from gluing onto a later same-type END fence and
        # yielding one spanning garbage match instead of a clean rejection.
        fence_marker = r"---(?:START|END)_(?:BASH|PYTHON)_COMMAND"
        body = rf"(?:(?!{fence_marker})[\s\S])*?"
        pattern = rf"---START_(BASH|PYTHON)_COMMAND-{self.uuid}---\s*({body})\s*---END_\1_COMMAND-{self.uuid}---"
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        if matches:
            return [(m.group(1), m.group(2).strip()) for m in matches], None

        # Relaxed pattern fallback for diagnosing malformed UUIDs
        relaxed_pattern = rf"---START_(BASH|PYTHON)_COMMAND[^\n]*\n({body})\n?---END_\1_COMMAND"
        relaxed_matches = list(re.finditer(relaxed_pattern, response_text, re.DOTALL))
        if relaxed_matches:
            cmd_type = relaxed_matches[0].group(1)
            script = relaxed_matches[0].group(2).strip()
            warning = (
                f"⚠️ [SYSTEM WARNING] Malformed command block detected. The UUID was missing or incorrect.\n"
                f"The current session UUID is: {self.uuid}\n\n"
                f"Did you mean to execute this?\n\n"
                f"---START_{cmd_type}_COMMAND-{self.uuid}---\n"
                f"{script}\n"
                f"---END_{cmd_type}_COMMAND-{self.uuid}---\n\n"
                f"Please re-evaluate and output the exact, correctly formatted block above to execute it."
            )
            return [], warning

        default_msg = (
            f"⚠️ [SYSTEM WARNING] You did not provide any code block.\n"
            f"Either provide a bash command:\n"
            f"---START_BASH_COMMAND-{self.uuid}---\n"
            f"[bash commands go here]\n"
            f"---END_BASH_COMMAND-{self.uuid}---\n\n"
            f"Or provide some python code: \n"
            f"---START_PYTHON_COMMAND-{self.uuid}---\n"
            f"[python code goes here]\n"
            f"---END_PYTHON_COMMAND-{self.uuid}---\n\n"
            f"Please provide a bash or python block to execute to proceed, or put 'exit' in a bash block to end the session."
        )
        return [], default_msg

    def _handle_special_command(self, cmd_type: str, script: str) -> tuple[bool, str]:
        """
        Checks if a script matches an intercepted agent command.
        Returns (handled, formatted_output).
        """
        if script.startswith("request-write "):
            path = script.split(" ", 1)[1]
            success, msg = self.sandbox.request_write(path)
            exit_code = 0 if success else 1
            return True, self._format_output(exit_code, msg, cmd_type)

        if script.startswith("ask-user "):
            question = script.split(" ", 1)[1]
            print(f"\n{COLOR_OUT}[Question to User] {question}{COLOR_RESET}")
            try:
                answer = input("[Human Response]: ")
            except EOFError:
                answer = "[User provided no response / EOF]"
            return True, self._format_output(0, answer, cmd_type)

        if script == "exit":
            print("\n[System] Agent has initiated exit.")
            self._log_debug_history()
            if _SFX_AVAILABLE:
                try:
                    _sfx.chime_sync()
                except Exception:
                    pass
            sys.exit(0)

        if script == "reset":
            print("\n[System] Agent resetting context...")
            # Preserve the system prompt (index 0); tolerate an empty history.
            self.context.history = [self.context.history[0]] if self.context.history else []
            return True, self._format_output(0, "Context history has been reset.", cmd_type)

        if script.startswith("copy-to-clipboard "):
            file_paths = script.split(" ", 1)[1]
            print(f"\n[System] Agent requested to copy files to clipboard: {file_paths}")
            copy_project_to_clipboard(file_paths)
            print("[System] Files copied to clipboard successfully. Exiting session.")
            self._log_debug_history()
            if _SFX_AVAILABLE:
                try:
                    _sfx.chime_sync()
                except Exception:
                    pass
            sys.exit(0)

        return False, ""

    def _execute_script(self, cmd_type: str, script: str) -> str:
        """
        Executes a bash or python script inside the sandbox, captures attached images,
        and returns formatted output.
        """
        if cmd_type == "BASH":
            exit_code, raw_output = self.sandbox.execute(script)
        else:
            exit_code, raw_output = self.sandbox.execute_python(script)

        # Process and strip any attached image payloads
        img_pattern = rf"---START_ATTACHED_IMAGE-{self.uuid}---\s*(.*?)\s*---END_ATTACHED_IMAGE-{self.uuid}---"
        aud_pattern = rf"---START_ATTACHED_AUDIO-{self.uuid}---\s*(.*?)\s*---END_ATTACHED_AUDIO-{self.uuid}---"
        image_matches = re.findall(img_pattern, raw_output, re.DOTALL)
        audio_matches = re.findall(aud_pattern, raw_output, re.DOTALL)
        for b64_url in image_matches:
            self._pending_multimodal_images.append({"url": b64_url.strip()})
        for b64_audio in audio_matches:
            self._pending_multimodal_audio.append({"data": b64_audio.strip(), "format": "mp3"})

        clean_output = re.sub(img_pattern, "", raw_output, flags=re.DOTALL)
        clean_output = re.sub(aud_pattern, "", clean_output, flags=re.DOTALL).strip()
        if image_matches:
            clean_output = f"{clean_output}\n[Image attached to conversation context.]".strip()
        if audio_matches:
            clean_output = f"{clean_output}\n[Audio attached to conversation context.]".strip()
        formatted_output = self._format_output(exit_code, clean_output, cmd_type)

        # Remind the model that /tmp/ is wiped every turn when a failed command
        # references a missing file under /tmp/ (see ROADMAP 'Add /tmp/ warning').
        tmp_warning = _build_tmp_file_warning(exit_code, clean_output)
        if tmp_warning:
            formatted_output = f"{formatted_output}\n\n{tmp_warning}"

        return formatted_output

    def _commit_execution_feedback(self, combined_outputs: list[str]) -> None:
        """
        Bundles output blocks and scratchpad updates into a message and appends to context.
        """
        if not combined_outputs:
            return

        scratchpad_block = self.context.get_scratchpad_block()
        if scratchpad_block:
            self.context.remove_old_scratchpads()
            final_text = f"{scratchpad_block}\n" + "\n".join(combined_outputs)
        else:
            final_text = "\n".join(combined_outputs)

        if self._pending_multimodal_images or self._pending_multimodal_audio:
            structured_content = [{"type": "text", "text": final_text}]
            for img_obj in self._pending_multimodal_images:
                structured_content.append({"type": "image_url", "image_url": img_obj})
            for aud_obj in self._pending_multimodal_audio:
                structured_content.append({"type": "input_audio", "input_audio": aud_obj})
            self.context.add_message("user", structured_content)
            self._pending_multimodal_images.clear()
            self._pending_multimodal_audio.clear()
        else:
            self.context.add_message("user", final_text)

    def parse_and_execute(self, response_text: str) -> tuple[bool, str]:
        """
        Parses LLM response, executes code or special commands, and records output.
        Returns (executed: bool, feedback: str).
        """
        blocks, error_feedback = self._extract_blocks(response_text)
        if error_feedback:
            return False, error_feedback

        combined_outputs = []
        blocks_to_execute = blocks[:MAX_CODE_BLOCKS]

        for cmd_type, script in blocks_to_execute:
            handled, special_output = self._handle_special_command(cmd_type, script)
            if handled:
                output = special_output
            else:
                output = self._execute_script(cmd_type, script)

            print(f"\n{COLOR_OUT}{output}{COLOR_RESET}")
            combined_outputs.append(output)

        if len(blocks) > MAX_CODE_BLOCKS:
            skipped = len(blocks) - MAX_CODE_BLOCKS
            cutoff_warning = (
                f"⚠️ [SYSTEM WARNING] Only the first {MAX_CODE_BLOCKS} of {len(blocks)} "
                f"code block(s) were executed. The remaining {skipped} block(s) were skipped. "
                f"Please limit responses to at most {MAX_CODE_BLOCKS} code block(s) per message."
            )
            print(f"\n{COLOR_OUT}{cutoff_warning}{COLOR_RESET}")
            combined_outputs.append(cutoff_warning)

        self._commit_execution_feedback(combined_outputs)
        return True, ""

    def _format_output(self, exit_code: int, output: str, cmd_type: str = "BASH") -> str:
        visible = 100
        if len(output) > OUTPUT_LIMIT:
            original_len = len(output)
            half_limit = OUTPUT_LIMIT // 2
            visible = int((OUTPUT_LIMIT / original_len) * 100)
            banner = TRUNCATION_BANNER.format(uuid=self.uuid)
            output = output[:half_limit] + banner + output[-half_limit:]

        result = f"---START_{cmd_type}_OUTPUT-EXIT_CODE_{exit_code}-VISIBLE_{visible}%-{self.uuid}---\n{output.rstrip("\n")}\n---END_{cmd_type}_OUTPUT-{self.uuid}---"
        if visible < 100:
            result += f"\n\n⚠️ WARNING: The {cmd_type} output was truncated. Only {visible}% of the output is shown above. Use other commands to target or filter the output (e.g., grep, head, tail, sed, awk, etc.) to avoid this."
        return result

    def _log_debug_history(self):
        if not self.debug:
            return

        log_path = os.path.abspath("/tmp/bash_agent_log.txt")
        tmp_path = log_path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("=== BASH AGENT CONVERSATION LOG ===\n")
            f.write(f"Agent UUID: {self.uuid}\n")
            f.write("===================================\n\n")

            for msg in self.context.history:
                role = msg.get("role", "UNKNOWN").upper()
                content = msg.get("content") or ""

                f.write(f"[{role}]\n")
                f.write("-" * 40 + "\n")

                if isinstance(content, str):
                    f.write(content + "\n")
                elif isinstance(content, list):
                    # Multimodal structured content (text + image_url blocks)
                    for block in content:
                        if not isinstance(block, dict):
                            f.write(str(block) + "\n")
                            continue
                        block_type = block.get("type")
                        if block_type == "text":
                            f.write(block.get("text", "") + "\n")
                        elif block_type == "image_url":
                            f.write("[Image payload]\n")
                        elif block_type == "input_audio":
                            f.write("[Audio payload]\n")
                        else:
                            f.write(f"[{block_type} payload]\n")
                else:
                    f.write(str(content) + "\n")

                f.write("=" * 80 + "\n\n")

        # Write atomically so a crash mid-write never corrupts the log
        os.replace(tmp_path, log_path)

    def _extract_reasoning_content(self, choice) -> str | None:
        """Extract reasoning text from response choice if available."""
        reasoning = getattr(choice.message, "reasoning", None)
        if not reasoning and hasattr(choice.message, "model_extra") and choice.message.model_extra:
            reasoning = choice.message.model_extra.get("reasoning")
        return reasoning.strip() if reasoning and isinstance(reasoning, str) else None

    def _extract_message_content(self, choice) -> str:
        """Extract and validate textual content or fallback reasoning from a response choice."""
        if choice.finish_reason == "tool_calls":
            raise ValueError(
                "Invalid finish_reason: tool_calls. The model attempted to use native tool_calls, "
                "which is not supported. Please use BASH and PYTHON code blocks instead."
            )
        if choice.finish_reason == "length":
            raise ValueError("Output truncated due to token limit (finish_reason: length).")

        agent_msg = choice.message.content
        if agent_msg is not None:
            return agent_msg

        # Safely attempt to extract reasoning text if content is None
        reasoning_text = self._extract_reasoning_content(choice)
        if choice.finish_reason == "stop" and reasoning_text:
            if self.debug:
                print("[Debug] API returned None for content. Falling back to reasoning text.")
            return reasoning_text

        raise ValueError("API returned None for message content (likely a safety filter or model glitch).")

    def _record_response_usage(self, response) -> None:
        """Extract cost, token count, and provider metadata from the API response."""
        try:
            resp_dict = response.model_dump()
            usage = resp_dict.get("usage", {}) or {}

            step_cost = usage.get("cost", 0.0)
            if step_cost is None:
                step_cost = 0.0

            self.last_step_input_tokens = usage.get("prompt_tokens", 0) or 0
            self.last_step_cost = step_cost
            self.session_cost += step_cost

            provider_info = resp_dict.get("provider")
            if isinstance(provider_info, dict):
                self.last_step_provider = provider_info.get("name")
            else:
                self.last_step_provider = provider_info
        except Exception as e:
            if self.debug:
                print(f"[Debug] Failed to parse cost metadata: {e}")

    def _handle_retry_backoff(self, retry_count: int) -> None:
        """Wait with exponential backoff while allowing interactive token limit doubling."""
        retry_delay = min(5 * (2 ** retry_count), 160)
        print(f"Retrying in {retry_delay} seconds... (Type '2x' and press Enter to double max_tokens to {self.max_tokens * 2})")

        start_wait = time.time()
        waited_fully = True
        while time.time() - start_wait < retry_delay:
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if ready:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == "2x":
                    self.max_tokens = int(self.max_tokens * 2)
                    print(f"\n[System] Max tokens doubled to {self.max_tokens}! Retrying immediately...")
                    waited_fully = False
                    break
        if waited_fully:
            print(f"Retry delay ({retry_delay}s) complete. Retrying...")

    def _get_llm_response(self) -> str:
        """Call the LLM with the current history, applying retry/backoff, thinking recovery, and extracting cost metadata."""
        retry_count = 0
        while True:
            response = None
            try:
                response = llm.create_chat_completion(
                    model=self.model,
                    messages=self.context.history,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort
                )

                if not response or not response.choices:
                    raise ValueError("Empty choices in response.")

                choice = response.choices[0]

                if choice.finish_reason == "tool_calls":
                    self._record_response_usage(response)
                    print(f"\n{COLOR_OUT}[System] Model returned finish_reason=tool_calls. Native tool_calls are not supported. Prompting for BASH/PYTHON blocks instead.{COLOR_RESET}")
                    tool_info = ""
                    try:
                        tc = getattr(choice.message, "tool_calls", None)
                        if tc:
                            tool_info = f"\nAttempted tool_calls: {tc}"
                    except Exception:
                        pass
                    # Record the failed attempt and warn with the desired syntax, then
                    # continue the conversation. Unlike the finish_reason == "length"
                    # recovery (which temporarily preserves thinking), the correction
                    # here is intentionally KEPT in the history so the model can see
                    # what went wrong and follow up with proper BASH/PYTHON blocks on
                    # the very next LLM call.
                    self.context.add_message("assistant", f"[Invalid tool_calls attempt]{tool_info}")
                    self.context.add_message(
                        "user",
                        "⚠️ [SYSTEM WARNING] Invalid finish_reason: tool_calls. The model attempted to use native tool_calls, "
                        "which is not supported in this environment. Please use BASH and PYTHON code blocks instead.\n"
                        f"---START_BASH_COMMAND-{self.uuid}---\n"
                        f"[bash commands go here]\n"
                        f"---END_BASH_COMMAND-{self.uuid}---\n"
                        f"or\n"
                        f"---START_PYTHON_COMMAND-{self.uuid}---\n"
                        f"[python code goes here]\n"
                        f"---END_PYTHON_COMMAND-{self.uuid}---"
                    )
                    # Continue the loop: the next iteration calls the LLM with the
                    # corrected history and returns the model's new response. The
                    # correction stays in the chat history permanently.
                    continue
                # Handle thinking token cutoff recovery
                if choice.finish_reason == "length":
                    reasoning_text = self._extract_reasoning_content(choice)
                    if reasoning_text:
                        self._record_response_usage(response)
                        print(f"\n{COLOR_OUT}[System] Model exceeded max tokens during reasoning. Ingesting thinking tokens ({len(reasoning_text)} chars) and requesting immediate answer...{COLOR_RESET}")

                        # 1. Inject temporary thinking and warning prompt
                        self.context.add_message("assistant", f"<thinking>\n{reasoning_text}\n</thinking>")
                        self.context.add_message(
                            "user",
                            "⚠️ [SYSTEM WARNING] You exceeded your max token budget during reasoning. "
                            "Review your previous thinking above and provide your immediate final response "
                            "and command execution blocks without restarting or repeating chain-of-thought reasoning."
                        )

                        try:
                            # 2. Call LLM for the direct response with lowest supported reasoning
                            # Use lowest effort to minimize token usage during recovery
                            recovery_reasoning_effort = self._get_lowest_reasoning_effort()
                            if self.debug:
                                print(f"[Debug] Max tokens recovery: using reasoning_effort='{recovery_reasoning_effort}'")
                            recovery_response = llm.create_chat_completion(
                                model=self.model,
                                messages=self.context.history,
                                max_tokens=self.max_tokens,
                                reasoning_effort=recovery_reasoning_effort
                            )

                            if not recovery_response or not recovery_response.choices:
                                raise ValueError("Empty choices in recovery response.")

                            recovery_choice = recovery_response.choices[0]
                            agent_msg = self._extract_message_content(recovery_choice)
                            self._record_response_usage(recovery_response)
                            return agent_msg

                        finally:
                            # 3. Always remove the temporary assistant and user messages
                            if len(self.context.history) >= 3:
                                self.context.history.pop()  # Remove user warning
                                self.context.history.pop()  # Remove assistant thinking
                    else:
                        raise ValueError("Output truncated due to token limit (finish_reason: length).")

                agent_msg = self._extract_message_content(choice)
                self._record_response_usage(response)
                return agent_msg

            except Exception as e:
                if response is not None:
                    print(f"\n[INVALID RESPONSE] Failed to extract message: {e}")
                    try:
                        print(f"[Response JSON]:\n{response.model_dump_json(indent=2)}")
                    except Exception:
                        print(f"[Response JSON]:\n{response}")
                else:
                    print(f"\n[API ERROR] {e}")

                self._handle_retry_backoff(retry_count)
                retry_count += 1

    def _colorize_commands(self, text: str) -> str:
        """Apply terminal ANSI colors to BASH and PYTHON command blocks for display."""
        display_msg = re.sub(
            rf"(---START_BASH_COMMAND-{self.uuid}---\n.*?\n---END_BASH_COMMAND-{self.uuid}---)",
            rf"{COLOR_CMD}\1{COLOR_RESET}",
            text,
            flags=re.DOTALL
        )
        display_msg = re.sub(
            rf"(---START_PYTHON_COMMAND-{self.uuid}---\n.*?\n---END_PYTHON_COMMAND-{self.uuid}---)",
            rf"{COLOR_PY_CMD}\1{COLOR_RESET}",
            display_msg,
            flags=re.DOTALL
        )
        return display_msg

    def _handle_turn_budget_and_stats(self) -> bool:
        """Report context/cost stats after a turn and enforce the session budget.
        Returns True if the session should continue, False if the budget was exceeded."""
        # Calculate current context size
        current_context_chars = sum(ContextManager._content_length(m.get("content", "")) for m in self.context.history)
        context_percent = (current_context_chars / CONTEXT_LIMIT) * 100

        if self.last_step_cost > 0.0:
            step_info = f"This request: ${self.last_step_cost:.3f} ({self.last_step_input_tokens} tokens)"
            total_info = f"Total: ${self.session_cost:.2f}"
        else:
            # Free models report no pricing data; still show token/context stats
            step_info = f"This request: {self.last_step_input_tokens} tokens"
            total_info = "Total: free"

        print(f"{COLOR_COST}[Session Stats] Context: {context_percent:.1f}% | {step_info} | {total_info} | Provider: {self.last_step_provider or 'N/A'}{COLOR_RESET}")

        if self.budget > 0 and self.session_cost >= self.budget:
            print(f"\n[Budget] Session cost ${self.session_cost:.2f} has reached the budget of ${self.budget:.2f}. Ending session.")
            return False

        self.last_step_cost = 0.0
        self.last_step_input_tokens = 0
        self.last_step_provider = None
        return True

    def _run_warmup_exchanges(self):
        """
        Pre-fill a FRESH session with two scripted assistant turns -- a PYTHON
        command printing the interpreter version plus every installed
        3rd-party PyPI package, then a BASH command listing the working
        directory -- executing each in the sandbox and recording
        the real formatted output as the following user message.

        Each template is parsed with the production _extract_blocks() and
        executed via _execute_script(), so the injected transcript is
        byte-for-byte identical in format to a live exchange. Never called for
        resumed sessions (see self.resumed_session).
        """
        print("[System] Running protocol warmup exchanges...")
        for template in WARMUP_TURNS:
            agent_msg = template.format(uuid=self.uuid)

            # Display exactly like a live turn
            print(f"\n[Agent]:\n{self._colorize_commands(agent_msg)}")
            self.context.add_message("assistant", agent_msg)
            self.context.save_history()

            # Dogfood the production parser: the fixed templates MUST parse.
            blocks, error_feedback = self._extract_blocks(agent_msg)
            if error_feedback or not blocks:
                raise RuntimeError(
                    f"Warmup template failed to parse: {error_feedback or 'no blocks found'}"
                )

            cmd_type, script = blocks[0]
            output = self._execute_script(cmd_type, script)
            print(f"\n{COLOR_OUT}{output}{COLOR_RESET}")
            self.context.add_message("user", output)
            self.context.save_history()
            self._log_debug_history()

    def run(self, initial_task: str = None):
        print(f"Agent initialized with UUID: {self.uuid}")
        print("Provide a task to begin.")

        task = initial_task if initial_task else get_vim_prompt()

        # Inject scratchpad into the very first message
        scratchpad_block = self.context.get_scratchpad_block()
        if scratchpad_block:
            task = scratchpad_block + "\n" + task

        self.context.add_message("user", task)
        self.context.save_history()
        self._log_debug_history()

        # Teach the protocol by example: two scripted turns with real sandbox
        # output, skipped when resuming a session that already has history.
        if not self.resumed_session:
            self._run_warmup_exchanges()

        try:
            while True:
                print("[Agent is thinking...]")
                agent_msg = self._get_llm_response()
                if _SFX_AVAILABLE:
                    try:
                        _sfx.click()
                    except Exception:
                        pass

                print(f"\n[Agent]:\n{self._colorize_commands(agent_msg)}")
                self.context.add_message("assistant", agent_msg)
                self.context.save_history()

                executed, feedback_msg = self.parse_and_execute(agent_msg)
                if not executed:
                    self.context.add_message("user", feedback_msg)
                    self.context.save_history()

                self._log_debug_history()

                if not self._handle_turn_budget_and_stats():
                    if _SFX_AVAILABLE:
                        try:
                            _sfx.chime_sync()
                        except Exception:
                            pass
                    break

        except KeyboardInterrupt:
            self._log_debug_history()
            print("\n[System] Session terminated by user (Ctrl+C). Exiting.")
            sys.exit(0)
