import os
import re
import hashlib
from typing import List, Dict, Union
import json
import sys
from bash_agent.config import CONTEXT_LIMIT, SCRATCHPAD_LIMIT, HISTORY_FILE

class ContextManager:
    def __init__(self, uuid_str: str):
        self.history: List[Dict[str, str]] = []
        self.uuid = uuid_str
        self.last_scratchpad_hash = None
        
        # Create the new temp directory and set the scratchpad path inside it
        tmp_dir = os.path.abspath(".bash_agent_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        self.scratchpad_path = os.path.join(tmp_dir, "SCRATCHPAD.md")

        if not os.path.exists(self.scratchpad_path):
            with open(self.scratchpad_path, "w") as f:
                f.write("# Project Scratchpad\n\n")
    @staticmethod
    def _content_length(msg_content):
        """Calculate character length of content, supporting both string and list (multimodal) formats."""
        if isinstance(msg_content, str):
            return len(msg_content)
        elif isinstance(msg_content, list):
            total = 0
            for item in msg_content:
                if isinstance(item, dict):
                    # Text type items
                    if item.get("type") == "text":
                        total += len(item.get("text", ""))
                    # Lets assume each Image is roughly 800 tokens.  Or 6400 characters
                    elif item.get("type") == "image_url":
                        total += 6400
                elif isinstance(item, str):
                    total += len(item)
            return total
        return 0

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._trim_context_if_needed()

    def _trim_context_if_needed(self):
        total_chars = sum(ContextManager._content_length(m.get("content", "")) for m in self.history)

        # Guard: Do not trigger cleanup until the strict CONTEXT_LIMIT is reached/exceeded
        if total_chars <= CONTEXT_LIMIT:
            return

        # Calculate the 80% hysteresis target limit
        target_limit = int(CONTEXT_LIMIT * 0.8)
        print(f"[System] Context limit exceeded ({total_chars} chars). Initiating hysteresis cleanup down to 80% ({target_limit} chars)...")

        # Incrementally trim the oldest messages until under the hysteresis target limit
        while True:
            total_chars = sum(ContextManager._content_length(m.get("content", "")) for m in self.history)
            if total_chars <= target_limit:
                break

            trimmed_something = False

            # Iterate from oldest to newest (skipping the system prompt at index 0)
            for i in range(1, len(self.history)):
                msg = self.history[i]
                original_content = msg["content"]

                # Multimodal (image-containing) messages can't be block-trimmed because
                # the regex operations below require a string (a list would raise
                # TypeError). Drop the whole message to free the image data; the
                # surrounding conversation makes it obvious what happened.
                if not isinstance(msg["content"], str):
                    self.history.pop(i)
                    trimmed_something = True
                    print("[System] Dropped an old image-bearing message to save context.")
                    break

                # 1. Try to delete the oldest BASH_OUTPUT first (biggest space savings)
                pattern_out = rf"(---START_(?:BASH|PYTHON)_OUTPUT-[^-]+-[^-]+-{self.uuid}---\s*)(.*?)(\s*---END_(?:BASH|PYTHON)_OUTPUT-{self.uuid}---)"
                if re.search(pattern_out, msg["content"], re.DOTALL):
                    msg["content"] = re.sub(
                        pattern_out,
                        r"\1[BASH_OUTPUT DELETED TO SAVE CONTEXT]\3",
                        msg["content"],
                        flags=re.DOTALL
                    )

                # 2. If no output to delete, try to truncate the oldest BASH_COMMAND
                pattern_cmd = rf"(---START_(?:BASH|PYTHON)_COMMAND-{self.uuid}---\s*)(.*?)(\s*---END_(?:BASH|PYTHON)_COMMAND-{self.uuid}---)"
                if msg["content"] == original_content and re.search(pattern_cmd, msg["content"], re.DOTALL):
                    if "...[TRUNCATED]" not in msg["content"]:
                        msg["content"] = re.sub(
                            pattern_cmd,
                            lambda m: f"{m.group(1)}{m.group(2)[:80]}...[TRUNCATED]{m.group(3)}",
                            msg["content"],
                            flags=re.DOTALL
                        )

                # If we successfully reduced the size of this message, break and recalculate total size
                if msg["content"] != original_content:
                    trimmed_something = True
                    print("[System] Context trimmed an old block to save space.")
                    break

            # Failsafe: if we couldn't trim any bash blocks but are still over the limit,
            # we must aggressively drop the oldest message entirely to prevent an infinite loop.
            if not trimmed_something:
                if len(self.history) > 1:
                    print("[System] Dropping oldest conversational message to save space.")
                    self.history.pop(1)
                else:
                    break

    def remove_old_scratchpads(self):
        # Match the scratchpad block including optional error message and surrounding newlines
        pattern = rf"\n?---START_SCRATCHPAD\.md-VISIBLE_\d+%-{self.uuid}---\n.*?\n---END_SCRATCHPAD\.md-{self.uuid}---(\n\[ERROR\]: Scratchpad truncated\. Please clean it up using bash commands\.)?\n?"
        for msg in self.history:
            if isinstance(msg["content"], str):
                msg["content"] = re.sub(pattern, "", msg["content"], flags=re.DOTALL).strip()


    def save_history(self):
        """Persist UUID and conversation history to disk."""
        state = {
            "uuid": self.uuid,
            "history": self.history
        }
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_history(self) -> bool:
        """Restore UUID and history from disk. Returns True if successful."""
        if not os.path.exists(HISTORY_FILE):
            return False
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.uuid = state["uuid"]
            self.history = state["history"]
            self.last_scratchpad_hash = None
            return True
        except Exception as e:
            print(f"[System Error] Failed to load history: {e}", file=sys.stderr)
            return False

    def get_scratchpad_block(self) -> str:
        with open(self.scratchpad_path, "r") as f:
            content = f.read()
            
        current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if current_hash == self.last_scratchpad_hash:
            return ""
            
        self.last_scratchpad_hash = current_hash
        
        visible = 100
        error_msg = ""
        original_len = len(content)
        if original_len > SCRATCHPAD_LIMIT:
            content = content[:SCRATCHPAD_LIMIT]
            visible = int((SCRATCHPAD_LIMIT / original_len) * 100)
            error_msg = "\n[ERROR]: Scratchpad truncated. Please clean it up using bash commands."
            
        return f"\n---START_SCRATCHPAD.md-VISIBLE_{visible}%-{self.uuid}---\n{content}\n---END_SCRATCHPAD.md-{self.uuid}---{error_msg}\n"

