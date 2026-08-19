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
from bash_agent import llm
from bash_agent.context import ContextManager
from bash_agent.sandbox import Sandbox

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


class Agent:
    def __init__(self, keep_tmp=False, debug=False, model=None, reasoning_effort=None, max_tokens=None, timeout=None, resume=False, budget=None):
        self.uuid = str(uuid.uuid4())
        self.debug = debug
        
        # Model priority: CLI arg > env var > default
        if model:
            self.model = model
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
        
        # Reasoning effort and max tokens from config
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None and reasoning_effort != 'default' else None
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        # Track attached images emitted by the sandbox (e.g., via the vision command)
        self._pending_multimodal_images = []
        
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
        # All Gemini models are multimodal, no API call needed
        if llm.get_backend(self.model) == "gemini":
            self.multimodal_capabilities = ["image"]
            if self.debug:
                print(f"[Debug] Gemini backend: multimodal assumed for model '{self.model}'.")
            return
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                models_data = json_module.loads(resp.read().decode("utf-8"))
            for m in models_data if isinstance(models_data, list) else models_data.get("data", []):
                if m.get("id") == self.model:
                    arch = m.get("architecture", {}) or {}
                    input_modalities = arch.get("input_modalities", [])
                    if input_modalities:
                        self.multimodal_capabilities = list(input_modalities)
                        if self.debug:
                            print(f"[Debug] Model '{self.model}' supports input modalities: {self.multimodal_capabilities}. Vision interception enabled.")
                    return
        except Exception as e:
            if self.debug:
                print(f"[Debug] Model capability check failed: {e}. Defaulting to text-only.")
            self.multimodal_capabilities = None
    def _extract_blocks(self, response_text: str) -> tuple[list[tuple[str, str]], str | None]:
        """
        Extracts valid (cmd_type, script) blocks from the LLM response.
        Returns (blocks, None) on success, or ([], warning_message) if parsing fails.
        """
        pattern = rf"---START_(BASH|PYTHON)_COMMAND-{self.uuid}---\s*(.*?)\s*---END_\1_COMMAND-{self.uuid}---"
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        if matches:
            return [(m.group(1), m.group(2).strip()) for m in matches], None

        # Relaxed pattern fallback for diagnosing malformed UUIDs
        relaxed_pattern = r"---START_(BASH|PYTHON)_COMMAND[^\n]*\n(.*?)\n?---END_\1_COMMAND"
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
            sys.exit(0)

        if script == "reset":
            print("\n[System] Agent resetting context...")
            self.context.history = [self.context.history[0]]
            return True, self._format_output(0, "Context history has been reset.", cmd_type)

        if script.startswith("copy-to-clipboard "):
            file_paths = script.split(" ", 1)[1]
            print(f"\n[System] Agent requested to copy files to clipboard: {file_paths}")
            copy_project_to_clipboard(file_paths)
            print("[System] Files copied to clipboard successfully. Exiting session.")
            self._log_debug_history()
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
        image_matches = re.findall(img_pattern, raw_output, re.DOTALL)
        for b64_url in image_matches:
            self._pending_multimodal_images.append({"url": b64_url.strip()})

        clean_output = re.sub(img_pattern, "", raw_output, flags=re.DOTALL).strip()
        if image_matches:
            clean_output = f"{clean_output}\n[Image attached to conversation context.]".strip()

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

        if self._pending_multimodal_images:
            structured_content = [{"type": "text", "text": final_text}]
            for img_obj in self._pending_multimodal_images:
                structured_content.append({"type": "image_url", "image_url": img_obj})
            self.context.add_message("user", structured_content)
            self._pending_multimodal_images.clear()
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
            output = output[:half_limit] + "\n...[Output Truncated]...\n" + output[-half_limit:]
            visible = int((OUTPUT_LIMIT / original_len) * 100)

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
                        else:
                            f.write(f"[{block_type} payload]\n")
                else:
                    f.write(str(content) + "\n")

                f.write("=" * 80 + "\n\n")

        # Write atomically so a crash mid-write never corrupts the log
        os.replace(tmp_path, log_path)

    def _extract_message_content(self, choice) -> str:
        """Extract and validate textual content or fallback reasoning from a response choice."""
        if choice.finish_reason == "length":
            raise ValueError("Output truncated due to token limit (finish_reason: length).")

        agent_msg = choice.message.content
        if agent_msg is not None:
            return agent_msg

        # Safely attempt to extract reasoning text if content is None
        reasoning_text = getattr(choice.message, "reasoning", None)
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
        retry_delay = 2 ** (retry_count + 1)
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
        """Call the LLM with the current history, applying retry/backoff and extracting cost metadata."""
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

                agent_msg = self._extract_message_content(response.choices[0])
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
        if self.last_step_cost <= 0.0:
            return True

        # Calculate current context size
        current_context_chars = sum(ContextManager._content_length(m.get("content", "")) for m in self.context.history)
        context_percent = (current_context_chars / CONTEXT_LIMIT) * 100

        print(f"{COLOR_COST}[Session Stats] Context: {context_percent:.1f}% | This request: ${self.last_step_cost:.3f} ({self.last_step_input_tokens} tokens) | Total: ${self.session_cost:.2f} | Provider: {self.last_step_provider or 'N/A'}{COLOR_RESET}")

        if self.budget > 0 and self.session_cost >= self.budget:
            print(f"\n[Budget] Session cost ${self.session_cost:.2f} has reached the budget of ${self.budget:.2f}. Ending session.")
            return False

        self.last_step_cost = 0.0
        self.last_step_input_tokens = 0
        self.last_step_provider = None
        return True

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

        try:
            while True:
                print("[Agent is thinking...]")
                agent_msg = self._get_llm_response()

                print(f"\n[Agent]:\n{self._colorize_commands(agent_msg)}")
                self.context.add_message("assistant", agent_msg)
                self.context.save_history()

                executed, feedback_msg = self.parse_and_execute(agent_msg)
                if not executed:
                    self.context.add_message("user", feedback_msg)
                    self.context.save_history()

                self._log_debug_history()

                if not self._handle_turn_budget_and_stats():
                    break

        except KeyboardInterrupt:
            self._log_debug_history()
            print("\n[System] Session terminated by user (Ctrl+C). Exiting.")
            sys.exit(0)
