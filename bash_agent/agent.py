#!/usr/bin/env python3

import os
import urllib.request
import json as json_module
import re
import sys
import time
import uuid
import select
import shlex
import base64
import io
from PIL import Image
from typing import List, Dict

from bash_agent.prompts import get_system_prompt
from bash_agent.config import DEFAULT_MODEL, OUTPUT_LIMIT, COLOR_CMD, COLOR_OUT, COLOR_PY_CMD, COLOR_COST, COLOR_RESET, DEFAULT_REASONING_EFFORT, DEFAULT_MAX_TOKENS, CONTEXT_LIMIT, MAX_PIXELS, DEFAULT_BUDGET
from bash_agent.utils import cleanup_tmp_folder, copy_project_to_clipboard, get_clipboard_content, get_vim_prompt
from bash_agent import llm
from bash_agent.context import ContextManager
from bash_agent.sandbox import Sandbox

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
        
        # Handle Resume: attempt to restore previous session
        history_loaded = False
        if resume:
            history_loaded = self.context.load_history()
            if history_loaded:
                # Re-bind components to the restored UUID
                self.uuid = self.context.uuid
                self.sandbox = Sandbox(self.context.scratchpad_path, timeout=timeout)
                print(f"[System] Resumed previous session. Re-bound to UUID: {self.uuid}")
        
        if not history_loaded:
            self.sandbox = Sandbox(self.context.scratchpad_path, timeout=timeout)
        
        # Reasoning effort and max tokens from config
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None and reasoning_effort != 'default' else None
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        # Multimodal detection for native vision command interception
        self.is_multimodal = False
        self._check_model_capabilities()
        
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
            system_prompt = get_system_prompt(self.uuid, os.path.abspath("."), self.context.scratchpad_path, role_text=custom_role, is_multimodal=self.is_multimodal)
            self.context.add_message("system", system_prompt)
        self.budget = budget if budget is not None else DEFAULT_BUDGET
        self.session_cost = 0.0
        self.last_step_cost = 0.0
        self.last_step_input_tokens = 0
        self.last_step_provider = None

    def _check_model_capabilities(self):
        """Query OpenRouter API to detect if the active model supports image input natively."""
        # All Gemini models are multimodal, no API call needed
        if llm.get_backend(self.model) == "gemini":
            self.is_multimodal = True
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
                    if "image" in input_modalities:
                        self.is_multimodal = True
                        if self.debug:
                            print(f"[Debug] Model '{self.model}' supports native image input. Vision interception enabled.")
                    return
        except Exception as e:
            if self.debug:
                print(f"[Debug] Model capability check failed: {e}. Defaulting to text-only.")
            self.is_multimodal = False
    def parse_and_execute(self, response_text: str) -> tuple[bool, str]:
        # Extract both BASH and PYTHON blocks using a capture group
        pattern = rf"---START_(BASH|PYTHON)_COMMAND-{self.uuid}---\s*(.*?)\s*---END_\1_COMMAND-{self.uuid}---"
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        
        if not matches:
            # Fallback: Check for mangled blocks using a relaxed regex
            relaxed_pattern = r"---START_(BASH|PYTHON)_COMMAND[^\n]*\n(.*?)\n?---END_\1_COMMAND"
            relaxed_matches = list(re.finditer(relaxed_pattern, response_text, re.DOTALL))
            
            if relaxed_matches:
                cmd_type = relaxed_matches[0].group(1)
                script = relaxed_matches[0].group(2).strip()
                
                warning_msg = (
                    f"⚠️ [SYSTEM WARNING] Malformed command block detected. The UUID was missing or incorrect.\n"
                    f"The current session UUID is: {self.uuid}\n\n"
                    f"Did you mean to execute this?\n\n"
                    f"---START_{cmd_type}_COMMAND-{self.uuid}---\n"
                    f"{script}\n"
                    f"---END_{cmd_type}_COMMAND-{self.uuid}---\n\n"
                    f"Please re-evaluate and output the exact, correctly formatted block above to execute it."
                )
                return False, warning_msg
            
            # No blocks found at all
            default_msg = (
                    f"⚠️ [SYSTEM WARNING] You did not provide any code block.\n"
                    "Either provide a bash command:\n"
                    f"---START_BASH_COMMAND-{self.uuid}---\n"
                    f"[bash commands go here]\n"
                    f"---END_BASH_COMMAND-{self.uuid}---\n\n"
                    "Or provide some python code: \n"
                    f"---START_BASH_COMMAND-{self.uuid}---\n"
                    f"[bash commands go here]\n"
                    f"---END_BASH_COMMAND-{self.uuid}---\n\n"
                    "Please provide a bash or python block to execute to procced, or put 'exit' in a bash block to end the session."
                    )
            return False, default_msg
            
        combined_outputs = []
        
        for i, match in enumerate(matches):
            if i > 0:
                # Only execute the first block, skip any additional ones
                continue
            cmd_type = match.group(1) # "BASH" or "PYTHON"
            script = match.group(2).strip()
            
            # 1. Check for Special Commands (Must be the only thing in the block)
            if script.startswith("request-write "):
                path = script.split(" ", 1)[1]
                success, msg = self.sandbox.request_write(path)
                exit_code = 0 if success else 1
                formatted_out = self._format_output(exit_code, msg, cmd_type)
                print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                combined_outputs.append(formatted_out)
                continue
                
            elif script.startswith("ask-user "):
                question = script.split(" ", 1)[1]
                print(f"\n{COLOR_OUT}[Question to User] {question}{COLOR_RESET}")
                try:
                    answer = input("[Human Response]: ")
                except EOFError:
                    answer = "[User provided no response / EOF]"
                formatted_out = self._format_output(0, answer, cmd_type)
                print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                combined_outputs.append(formatted_out)
                continue
                
            elif script == "exit":
                print("\n[System] Agent has initiated exit.")
                sys.exit(0)
                
            elif script == "reset":
                print("\n[System] Agent resetting context...")
                self.context.history = [self.context.history[0]]
                formatted_out = self._format_output(0, "Context history has been reset.", cmd_type)
                print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                combined_outputs.append(formatted_out)
                continue

            elif script.startswith("copy-to-clipboard "):
                # Extract the comma-separated file paths
                file_paths = script.split(" ", 1)[1]
                
                print(f"\n[System] Agent requested to copy files to clipboard: {file_paths}")
                
                # Call the existing function
                copy_project_to_clipboard(file_paths)
                
                print("[System] Files copied to clipboard successfully. Exiting session.")
                
                # Exit the agent loop as requested
                sys.exit(0)
                
            elif (script.startswith("vision ") or script == "vision") and self.is_multimodal:
                # --- NATIVE MULTIMODAL INTERCEPTION ---
                # Parse arguments safely
                try:
                    args = shlex.split(script)
                except ValueError:
                    args = script.split()
                
                # Extract prompt and image path
                extracted_prompt = "Convert the image into Markdown text."
                image_path = None
                
                if len(args) > 1:
                    # Look for -p or --prompt flag
                    i = 1
                    while i < len(args):
                        if args[i] in ("-p", "--prompt"):
                            if i + 1 < len(args):
                                extracted_prompt = args[i + 1]
                                i += 2
                            else:
                                i += 1
                        elif not args[i].startswith("-"):
                            image_path = args[i]
                            i += 1
                        else:
                            i += 1
                
                if image_path and os.path.exists(image_path):
                    # Validate image size before encoding
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            total_pixels = width * height
                            if total_pixels > MAX_PIXELS:
                                error_msg = f"Error: Image is too large ({width}x{height} = {total_pixels} pixels). Maximum allowed is {MAX_PIXELS} pixels."
                                formatted_out = self._format_output(1, error_msg, cmd_type)
                                print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                                combined_outputs.append(formatted_out)
                                continue
                            # Encode the image
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
                        
                        # Build multimodal payload
                        combined_outputs.append(
                                self._format_output(0, "The image you requested has been attached.", cmd_type)
                        )
                        
                        # Store the multimodal payload for injection into the final user message
                        self._pending_multimodal = {
                            "url": f"data:image/png;base64,{b64_image}"
                        }
                        continue
                        
                    except Exception as e:
                        formatted_out = self._format_output(1, f"[SYSTEM ERROR] Image encoding failed: {e}", cmd_type)
                        print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                        combined_outputs.append(formatted_out)
                        continue
                else:
                    error_msg = f"[SYSTEM ERROR] File not found: {image_path or 'No path provided'}"
                    formatted_out = self._format_output(1, error_msg, cmd_type)
                    print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
                    combined_outputs.append(formatted_out)
                    continue
                    
            # 2. Execute the script (bash or python)
            if cmd_type == "BASH":
                exit_code, output = self.sandbox.execute(script)
            elif cmd_type == "PYTHON":
                exit_code, output = self.sandbox.execute_python(script)
                
            formatted_out = self._format_output(exit_code, output, cmd_type)
            print(f"\n{COLOR_OUT}{formatted_out}{COLOR_RESET}")
            combined_outputs.append(formatted_out)

        # 3. Compile the final response to feed back to the LLM
        if combined_outputs:
            scratchpad_block = self.context.get_scratchpad_block()
            
            if scratchpad_block:
                # A change was detected! Scrub the old ones, then append the new one.
                self.context.remove_old_scratchpads()
                final_user_message = scratchpad_block + "\n" + "\n".join(combined_outputs)
            else:
                # No changes to the scratchpad, just send the outputs.
                final_user_message = "\n".join(combined_outputs)
            
            # If a native multimodal payload was intercepted, convert final_user_message to a structured content array
            pending_mm = getattr(self, "_pending_multimodal", None)
            if pending_mm is not None:
                structured_content = [
                    {"type": "text", "text": final_user_message},
                    {"type": "image_url", "image_url": pending_mm}
                ]
                self.context.add_message("user", structured_content)
                self._pending_multimodal = None
            else:
                self.context.add_message("user", final_user_message)
            
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
            
        with open(os.path.abspath("/tmp/bash_agent_log.txt"), "w") as f:
            f.write("=== BASH AGENT CONVERSATION LOG ===\n")
            f.write(f"Agent UUID: {self.uuid}\n")
            f.write("===================================\n\n")
            
            for msg in self.context.history:
                role = msg.get("role", "UNKNOWN").upper()
                content_text = msg.get("content") or ""
                
                f.write(f"[{role}]\n")
                f.write("-" * 40 + "\n")
                f.write(f"{content_text}\n")
                f.write("=" * 80 + "\n\n")

    def run(self, initial_task: str = None):
        print(f"Agent initialized with UUID: {self.uuid}")
        print("Provide a task to begin.")
        
        task = initial_task if initial_task else get_vim_prompt()
        
        # Inject scratchpad into the very first message
        scratchpad_block = self.context.get_scratchpad_block()
        if scratchpad_block:
            task = scratchpad_block + "\n" + task
        
        self.context.add_message("user", task)
        try:
            retry_count = 0
            while True:
                print("[Agent is thinking...]")
                self._log_debug_history()
                response = None
                try:
                    response = llm.create_chat_completion(
                        model=self.model,
                        messages=self.context.history,
                        max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort
                    )
                    
                    if not response.choices:
                        raise ValueError("Empty choices in response.")
                    choice = response.choices[0]
                    agent_msg = choice.message.content
                    if choice.finish_reason == "length":
                        raise ValueError(f"Output truncated due to token limit (finish_reason: length).")

                    if agent_msg is None:
                        # Safely attempt to extract reasoning text if it exists
                        reasoning_text = getattr(choice.message, "reasoning", None)

                        # Check if the model naturally finished AND has reasoning text
                        if choice.finish_reason == "stop" and reasoning_text:
                            if self.debug:
                                print("[Debug] API returned None for content. Falling back to reasoning text.")
                            agent_msg = reasoning_text
                        else:
                            raise ValueError("API returned None for message content (likely a safety filter or model glitch).")
                    # Safely extract cost, accumulate, and print
                    try:
                        resp_dict = response.model_dump()
                        step_cost = resp_dict.get("usage", {}).get("cost", 0.0)
                        
                        # OpenRouter sometimes returns None if the cost isn't calculated yet
                        if step_cost is None:
                            step_cost = 0.0
                        input_tokens = resp_dict.get("usage", {}).get("prompt_tokens", 0)
                        self.last_step_input_tokens = input_tokens
                            
                        self.last_step_cost = step_cost
                        self.session_cost += step_cost
                        self.last_step_provider = resp_dict.get("provider", {}).get("name") if isinstance(resp_dict.get("provider"), dict) else resp_dict.get("provider")
                    except Exception as e:
                        if self.debug:
                            print(f"[Debug] Failed to parse cost metadata: {e}")
                except Exception as e:
                    if response is not None:
                        print(f"\n[INVALID RESPONSE] Failed to extract message: {e}")
                        try:
                            print(f"[Response JSON]:\n{response.model_dump_json(indent=2)}")
                        except Exception:
                            print(f"[Response JSON]:\n{response}")
                    else:
                        print(f"\n[API ERROR] {e}")

                        
                    retry_delay = 2 ** (retry_count + 1)
                    print(f"Retrying in {retry_delay} seconds... (Type '2x' and press Enter to double max_tokens to {self.max_tokens * 2})")
                    
                    start_wait = time.time()
                    waited_fully = True
                    while time.time() - start_wait < retry_delay:
                        # Check stdin for data for 0.5 seconds without blocking
                        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                        if ready:
                            user_input = sys.stdin.readline().strip().lower()
                            if user_input == "2x":
                                self.max_tokens = int(self.max_tokens * 2)
                                print(f"\n[System] Max tokens doubled to {self.max_tokens}! Retrying immediately...")
                                waited_fully = False
                                break # Exit the wait loop and retry immediately
                    if waited_fully:
                        print(f"Retry delay ({retry_delay}s) complete. Retrying...")
                    retry_count += 1
                    continue
                
                # Apply color to the BASH_COMMAND blocks for the console print
                display_msg = re.sub(
                    rf"(---START_BASH_COMMAND-{self.uuid}---\n.*?\n---END_BASH_COMMAND-{self.uuid}---)",
                    rf"{COLOR_CMD}\1{COLOR_RESET}",
                    agent_msg,
                    flags=re.DOTALL
                )
                # Apply color to the PYTHON_COMMAND blocks for the console print
                display_msg = re.sub(
                    rf"(---START_PYTHON_COMMAND-{self.uuid}---\n.*?\n---END_PYTHON_COMMAND-{self.uuid}---)",
                    rf"{COLOR_PY_CMD}\1{COLOR_RESET}",
                    display_msg,
                    flags=re.DOTALL
                )
                print(f"\n[Agent]:\n{display_msg}")
                
                self.context.add_message("assistant", agent_msg)
                self.context.save_history()
                
                executed, feedback_msg = self.parse_and_execute(agent_msg)
                if self.last_step_cost > 0.0:
                    # Calculate current context size
                    current_context_chars = sum(ContextManager._content_length(m.get("content", "")) for m in self.context.history)
                    context_percent = (current_context_chars / CONTEXT_LIMIT) * 100
                    
                    print(f"{COLOR_COST}[Session Stats] Context: {context_percent:.1f}% | This request: ${self.last_step_cost:.3f} ({self.last_step_input_tokens} tokens) | Total: ${self.session_cost:.2f} | Provider: {self.last_step_provider or "N/A"}{COLOR_RESET}")
                    if self.budget > 0 and self.session_cost >= self.budget:
                        print(f"\n[Budget] Session cost ${self.session_cost:.2f} has reached the budget of ${self.budget:.2f}. Ending session.")
                        break
                    self.last_step_cost = 0.0
                    self.last_step_input_tokens = 0
                    self.last_step_provider = None
                if not executed:
                    self.context.add_message("user", feedback_msg)
                    self.context.save_history()
        except KeyboardInterrupt:
            print("\n[System] Session terminated by user (Ctrl+C). Exiting.")
            sys.exit(0)

