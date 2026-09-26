#!/usr/bin/env python3
"""
History viewer: print the text contents of agent history JSON files in a
readable, colorized form, with compact placeholders for any image/audio
payloads (base64 data is never dumped to the terminal).

History files live in .bash_agent_tmp/ and share a common shape:

    {
      "uuid": "<session-uuid>",
      "history": [
        {"role": "system"|"user"|"assistant", "content": <str|list[block]>},
        ...
      ]
    }

Each ``content`` is either a plain string or a list of multimodal blocks
(``text``, ``image_url`` and ``input_audio``). Text blocks are printed as-is;
image/audio blocks are summarized as ``<Image payload (image/png, N bytes)>``
style placeholders.

Usage examples:
    bhistory                         # main orchestrator history
    bhistory -a                      # main + all subagent histories
    bhistory .bash_agent_tmp/history_subagent_*.json
    bhistory --no-color | less       # pipe-friendly (no ANSI escapes)
    bhistory -t 2000                 # truncate each message to 2000 chars
"""
import os
import sys
import glob
import json
import base64
import argparse

from bash_agent.config import (
    HISTORY_FILE,
    SUBAGENT_HISTORY_PREFIX,
    COLOR_CMD,
    COLOR_COST,
    COLOR_PY_CMD,
    COLOR_RESET,
)

# Unique console color per role. Reuse the palette already defined in config.py
# (cyan / bright-green / magenta) and COLOR_RESET so output matches house style.
ROLE_COLORS = {
    "system": COLOR_CMD,
    "user": COLOR_COST,
    "assistant": COLOR_PY_CMD,
}
UNKNOWN_ROLE_COLOR = COLOR_RESET

# Fallback when a role (or block type) has no dedicated color.
DEFAULT_MIME_IMAGE = "image/*"
DEFAULT_MIME_AUDIO = "audio/*"
HEADER_WIDTH = 80
SEP_WIDTH = 80


def _colorize(text: str, color: str, use_color: bool) -> str:
    """Wrap *text* in *color* when color output is enabled."""
    if not use_color or not color:
        return text
    return f"{color}{text}{COLOR_RESET}"


def _parse_data_url(url: str):
    """Parse a ``data:<mime>;base64,<payload>`` URL.

    Returns ``(mime, nbytes)``; either value may be ``None`` when it cannot be
    determined. Never raises.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    header, _, payload = url.partition(",")
    mime = header[len("data:"):].split(";")[0] or None
    nbytes = None
    if payload:
        try:
            nbytes = len(base64.b64decode(payload, validate=False))
        except Exception:
            nbytes = None
    return mime, nbytes


def _decode_b64_len(data) -> int | None:
    """Best-effort byte length of a base64 string, or ``None`` on failure."""
    if not isinstance(data, str) or not data:
        return None
    try:
        return len(base64.b64decode(data, validate=False))
    except Exception:
        return None


def _format_image_block(block: dict) -> str:
    """Build a compact placeholder for an ``image_url`` block."""
    image_url = block.get("image_url")
    # Legacy: image_url may be a bare string.
    url = image_url.get("url") if isinstance(image_url, dict) else image_url

    if isinstance(url, str) and url.startswith("data:"):
        mime, nbytes = _parse_data_url(url)
        mime = mime or DEFAULT_MIME_IMAGE
        if nbytes is not None:
            return f"<Image payload ({mime}, {nbytes} bytes)>"
        return f"<Image payload ({mime})>"
    if isinstance(url, str) and url:
        return f"<Image payload (URL: {url})>"
    return "<Image payload>"


def _format_audio_block(block: dict) -> str:
    """Build a compact placeholder for an ``input_audio`` block."""
    audio = block.get("input_audio")
    # Legacy: input_audio may be a bare base64 string.
    if isinstance(audio, dict):
        fmt = audio.get("format") or DEFAULT_MIME_AUDIO
        nbytes = _decode_b64_len(audio.get("data"))
    else:
        fmt = DEFAULT_MIME_AUDIO
        nbytes = _decode_b64_len(audio)

    if nbytes is not None:
        return f"<Audio payload ({fmt}, {nbytes} bytes)>"
    return f"<Audio payload ({fmt})>"


def _render_block(block) -> str:
    """Render a single multimodal content block to text."""
    if not isinstance(block, dict):
        return str(block)

    block_type = block.get("type")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "image_url":
        return _format_image_block(block)
    if block_type == "input_audio":
        return _format_audio_block(block)
    return f"<{block_type} payload>"


def _render_content(content, use_color: bool, truncate: int = 0) -> str:
    """Flatten message content (str or list of blocks) into printable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [_render_block(b) for b in content]
        text = "\n".join(parts)
    else:
        text = str(content)

    if truncate and len(text) > truncate:
        omitted = len(text) - truncate
        text = text[:truncate] + f"\n... [{omitted} more characters truncated]"
    return text


def load_history(path: str):
    """Load and minimally validate a history JSON file.

    Returns the parsed dict, or ``None`` (with a warning on stderr) when the
    file is missing, unreadable, invalid JSON, or not shaped like a history.
    """
    if not os.path.isfile(path):
        print(f"warning: file not found: {path}", file=sys.stderr)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not read {path}: {e}", file=sys.stderr)
        return None

    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        print(f"warning: {path} is not a valid history file (missing 'history' list)", file=sys.stderr)
        return None
    return data


def print_history(path: str, data: dict, use_color: bool, truncate: int = 0) -> None:
    """Print a single history file: header followed by each message."""
    history = data.get("history", [])
    uuid = data.get("uuid", "unknown")
    display_name = os.path.basename(path) or path

    header = f"{display_name}  |  session {uuid}  |  {len(history)} message(s)"
    print()
    print(_colorize("=" * HEADER_WIDTH, COLOR_CMD, use_color))
    print(_colorize(header, COLOR_CMD, use_color))
    print(_colorize("=" * HEADER_WIDTH, COLOR_CMD, use_color))

    for msg in history:
        if not isinstance(msg, dict):
            print(_colorize(str(msg), UNKNOWN_ROLE_COLOR, use_color))
            continue

        role = msg.get("role") or "unknown"
        color = ROLE_COLORS.get(role, UNKNOWN_ROLE_COLOR)
        label = f"[{role.upper()}]"

        print()
        print(_colorize(label, color, use_color))
        print(_colorize("-" * SEP_WIDTH, color, use_color))
        print(_render_content(msg.get("content"), use_color, truncate))


def resolve_paths(files: list[str], include_all: bool) -> list[str]:
    """Compute the ordered, de-duplicated list of history files to display.

    Explicit path arguments (globs allowed) are expanded; with ``--all`` the
    default orchestrator history and every subagent history are appended.
    """
    paths: list[str] = []

    def add(p: str):
        ap = os.path.abspath(p)
        if ap not in paths:
            paths.append(ap)

    for pattern in files:
        matches = glob.glob(pattern)
        if matches:
            for m in sorted(matches):
                add(m)
        else:
            # Keep the literal path so load_history can warn about it.
            add(pattern)

    if include_all or not files:
        add(HISTORY_FILE)
    if include_all:
        tmp_dir = os.path.dirname(HISTORY_FILE)
        subagent_glob = os.path.join(tmp_dir, f"{SUBAGENT_HISTORY_PREFIX}*.json")
        for m in sorted(glob.glob(subagent_glob)):
            add(m)

    return paths


def main() -> None:
    use_color_default = not os.environ.get("NO_COLOR")
    parser = argparse.ArgumentParser(
        prog="bhistory",
        description="Print the text contents of agent history JSON files in a readable, colorized form "
                    "(with placeholders for image/audio payloads).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="History JSON file(s) to display (globs allowed). "
             f"Defaults to the orchestrator history ({HISTORY_FILE}).",
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Also include every subagent history "
             f"({SUBAGENT_HISTORY_PREFIX}*.json) alongside the default file.",
    )
    parser.add_argument(
        "-t", "--truncate",
        type=int,
        default=0,
        metavar="N",
        help="Truncate each message to N characters (0 = no truncation; default).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color (also honored via the NO_COLOR environment variable).",
    )
    args = parser.parse_args()

    use_color = use_color_default and not args.no_color

    paths = resolve_paths(args.files, args.all)

    loaded = 0
    for path in paths:
        data = load_history(path)
        if data is None:
            continue
        print_history(path, data, use_color, truncate=args.truncate)
        loaded += 1

    if loaded == 0:
        print("error: no readable history files found", file=sys.stderr)
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
