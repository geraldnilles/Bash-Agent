import os
import re
import io
import base64
import hashlib
from typing import List, Dict, Union
import json
import sys
from bash_agent.config import CONTEXT_LIMIT, SCRATCHPAD_LIMIT, HISTORY_FILE

# Token → character conversion rate used throughout the context-accounting
# code. Tokens are converted to characters at 8 chars/token (the historically
# documented exchange rate).
CHARS_PER_TOKEN = 8

# Multimodal content accounting rates (in tokens per unit):
IMAGE_TOKENS_PER_MEGAPIXEL = 1000
AUDIO_TOKENS_PER_MINUTE = 400

# Approximate MP3 bitrate (kbps) used by transcribe.py's convert_to_mp3()
# for the payload-size fallback estimate of clip length.
_MP3_FALLBACK_KBPS = 128

# Fallback flat estimates (in characters) used when a payload cannot be
# decoded / parsed to derive the true size. Mirrors the legacy flat rates.
IMAGE_FALLBACK_CHARS = 800 * CHARS_PER_TOKEN      # 6400
AUDIO_FALLBACK_CHARS = 50000

# Bounded cache: payload string -> estimated MP3 duration (or None).
# _content_length() re-measures the same history over and over during the
# hysteresis pruning loop; decoding a multi-megabyte base64 payload each
# time would dominate that loop. The payload strings are held by the
# history anyway, so the cache only stores references + a float.
_MP3_DURATION_CACHE: Dict[str, float] = {}
_MP3_DURATION_CACHE_MAX = 128
# Bounded cache: data-URL -> megapixels (None when undecodable). Same
# rationale as the audio cache below.
_IMAGE_MP_CACHE: Dict[str, float] = {}
_IMAGE_MP_CACHE_MAX = 128
# Sentinel distinguishing "not cached" from "cached as None (unparseable)".
_MISSING = object()


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
                    # Images are charged by resolution: 1000 tokens per megapixel.
                    elif item.get("type") == "image_url":
                        total += ContextManager._image_content_length(item)
                    # Audio is charged by clip length: 400 tokens per minute.
                    elif item.get("type") == "input_audio":
                        total += ContextManager._audio_content_length(item)
                elif isinstance(item, str):
                    total += len(item)
            return total
        return 0

    @staticmethod
    def _image_content_length(item) -> int:
        """Character cost of one image_url part, scaled by resolution.

        Uses 1000 tokens per megapixel (converted at CHARS_PER_TOKEN).  Falls
        back to the legacy flat estimate (≈800 tokens) when the resolution
        cannot be determined (non-data URL, undecodable payload, etc.).
        """
        # Normalize the part: {"image_url": {"url": ...}} is the shape
        # agent.py emits, but tolerate the legacy bare-string shape.
        url = item.get("image_url")
        if isinstance(url, dict):
            url = url.get("url", "")
        if not isinstance(url, str) or not url.startswith("data:"):
            return IMAGE_FALLBACK_CHARS
        mp = _IMAGE_MP_CACHE.get(url, _MISSING)
        if mp is _MISSING:
            b64 = url.split(",", 1)[1] if "," in url else ""
            try:
                from PIL import Image
                raw = base64.b64decode(b64, validate=True)
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size
                mp = (width * height) / 1_000_000
            except Exception:
                mp = None
            if len(_IMAGE_MP_CACHE) >= _IMAGE_MP_CACHE_MAX:
                _IMAGE_MP_CACHE.clear()
            _IMAGE_MP_CACHE[url] = mp
        if mp is None:
            return IMAGE_FALLBACK_CHARS
        return round(mp * IMAGE_TOKENS_PER_MEGAPIXEL * CHARS_PER_TOKEN)

    @staticmethod
    def _audio_content_length(item) -> int:
        """Character cost of one input_audio part, scaled by clip length.

        Uses 400 tokens per minute (converted at CHARS_PER_TOKEN).  Falls
        back to the legacy flat estimate (≈6k tokens) when the duration
        cannot be determined.
        """
        audio = item.get("input_audio")
        # Tolerate the legacy bare-string shape; dict shape is what
        # agent.py emits ({"data": ..., "format": ...}).
        data = audio.get("data", "") if isinstance(audio, dict) else audio
        # Anything but a non-empty base64 string is unparseable junk; use
        # the flat historical estimate rather than a misleading 0.
        if not isinstance(data, str) or not data:
            return AUDIO_FALLBACK_CHARS
        # Cache hit avoids re-decoding the (often multi-megabyte) base64
        # payload on every pruning pass over the same history.
        duration = _MP3_DURATION_CACHE.get(data, _MISSING)
        if duration is _MISSING:
            try:
                raw = base64.b64decode(data, validate=True)
            except Exception:
                raw = b""
            if not raw:
                duration = None
            else:
                try:
                    duration = ContextManager._estimate_mp3_duration(raw)
                    if duration is None:
                        # Header parse failed BUT genuine MP3 bytes are
                        # present: proxy the length from a 128kbps CBR
                        # assumption — the exact profile transcribe.py's
                        # convert_to_mp3 emits — so long clips still count
                        # far above the short-junk flat rate.
                        header_pos = ContextManager._skip_id3(raw)
                        if ContextManager._parse_first_frame_header(raw, header_pos) is not None:
                            duration = (len(raw) * 8) / (_MP3_FALLBACK_KBPS * 1000)
                except Exception:
                    duration = None
            if len(_MP3_DURATION_CACHE) >= _MP3_DURATION_CACHE_MAX:
                _MP3_DURATION_CACHE.clear()
            _MP3_DURATION_CACHE[data] = duration
        if duration is None:
            return AUDIO_FALLBACK_CHARS
        tokens = duration * (AUDIO_TOKENS_PER_MINUTE / 60.0)
        return round(tokens * CHARS_PER_TOKEN)

    @staticmethod
    def _estimate_mp3_duration(raw: bytes):
        """Return approximate MP3 duration in seconds, or None if unparseable.

        O(1) estimation: skip any ID3v2 tag, decode the FIRST MPEG frame
        header for bitrate/sample-rate, then divide the remaining payload
        bytes by the byte rate. This is exact for the constant-bitrate mono
        128kbps MP3s transcribe.py emits (convert_to_mp3), and within a few
        percent for ordinary CBR files. VBR files are approximated by their
        average bitrate, which is adequate for context accounting.
        """
        if not raw:
            return None
        pos = ContextManager._skip_id3(raw)
        header = ContextManager._parse_first_frame_header(raw, pos)
        if header is None:
            return None
        bitrate, _sample_rate, _frame_samples = header
        audio_bytes = len(raw) - pos
        if audio_bytes <= 0:
            return None
        # bitrate is in bits/second; MP3 audio bytes are ≈ total bytes.
        # (The first frame's header bytes are negligible at this scale.)
        return audio_bytes * 8 / bitrate

    @staticmethod
    def _skip_id3(raw: bytes):
        """Return the byte offset just past an optional ID3v2 tag, or 0."""
        if raw[:3] != b"ID3":
            return 0
        try:
            size = (raw[6] << 21) | (raw[7] << 14) | (raw[8] << 7) | raw[9]
            flags = raw[5]
            pos = 10 + size
            if flags & 0x10:  # footer present
                pos += 10
            return pos
        except IndexError:
            return 0

    @staticmethod
    def _parse_first_frame_header(raw: bytes, pos: int):
        """Parse the first MPEG Layer III frame header at pos.

        Returns (bitrate_bps, sample_rate_hz, samples_per_frame) or None if
        the bytes do not form a valid MPEG-1/2/2.5 Layer III frame header.
        """
        if pos + 4 > len(raw):
            return None
        b = raw[pos:pos + 4]
        if b[0] != 0xFF or (b[1] & 0xE0) != 0xE0:
            return None
        version_bits = (b[1] >> 3) & 0x3
        layer_bits = (b[1] >> 1) & 0x3
        bitrate_idx = (b[2] >> 4) & 0xF
        sample_idx = (b[2] >> 2) & 0x3
        if layer_bits == 0 or bitrate_idx == 0 or bitrate_idx == 15 or sample_idx == 3:
            return None
        if version_bits == 1:  # reserved
            return None
        if layer_bits != 1:  # only Layer III supported by callers
            return None
        if version_bits == 3:  # MPEG1
            bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                        160, 192, 224, 256, 320, 0]
            rates = [44100, 48000, 32000]
        elif version_bits == 2:  # MPEG2
            bitrates = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80,
                        96, 112, 128, 144, 160, 0]
            rates = [22050, 24000, 16000]
        else:  # MPEG2.5
            bitrates = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80,
                        96, 112, 128, 144, 160, 0]
            rates = [11025, 12000, 8000]
        bitrate_kbps = bitrates[bitrate_idx]
        sample_rate = rates[sample_idx]
        frame_samples = 1152 if version_bits == 3 else 576
        return (bitrate_kbps * 1000, sample_rate, frame_samples)

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

