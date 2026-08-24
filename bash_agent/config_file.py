"""Optional persistent user configuration: .bash_agent_tmp/config.json

Lets the user pin per-project LLM settings without editing source code or
repeating CLI flags every session:

    {
      "model": "deepseek/deepseek-v4-pro",
      "max_tokens": 16384,
      "reasoning_effort": "medium"
    }

Precedence (highest to lowest):
    1. Command-line arguments (--model / --max-tokens / --reasoning-effort)
    2. This config file
    3. OPENROUTER_MODEL environment variable (model only)
    4. Hard-coded defaults in bash_agent/config.py

The file is intentionally NOT deleted by cleanup_tmp_folder() — like
SCRATCHPAD.md, ROLE.md and history.json it survives between sessions.

NOTE: Unlike config.HISTORY_FILE, the config path is resolved against the
CWD *at call time* (not import time) so tooling/tests can chdir freely.
"""

import json
import os
import sys

CONFIG_FILENAME = "config.json"

# Keys we understand; anything else triggers a warning and is ignored.
KNOWN_KEYS = ("model", "max_tokens", "reasoning_effort")

# Mirrors the argparse choices for --reasoning-effort in main.py.
VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "default")


def get_config_path(path=None):
    """Resolve the config file path (explicit arg > CWD-relative default)."""
    if path is not None:
        return path
    return os.path.join(os.path.abspath(".bash_agent_tmp"), CONFIG_FILENAME)


def _warn(msg):
    print(f"[Config] {msg}", file=sys.stderr)


def load_config(path=None):
    """Load and validate the optional config.json.

    Returns a dict containing only known keys with sane values, e.g.
    {"model": "...", "max_tokens": 4096, "reasoning_effort": "low"}.
    Missing file, unreadable file, malformed JSON, or a non-object root all
    yield {} (feature silently disabled apart from a stderr warning).
    """
    config_path = get_config_path(path)
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        _warn(f"Ignoring {config_path}: invalid JSON ({e})")
        return {}

    if not isinstance(raw, dict):
        _warn(f"Ignoring {config_path}: expected a JSON object at top level")
        return {}

    settings = {}

    model = raw.get("model")
    if model is not None:
        if isinstance(model, str) and model.strip():
            settings["model"] = model.strip()
        else:
            _warn(f"Ignoring key 'model' in {CONFIG_FILENAME}: expected a non-empty string")

    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
        # isinstance(True, int) is True, so exclude bools explicitly.
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens >= 1:
            settings["max_tokens"] = max_tokens
        else:
            _warn(f"Ignoring key 'max_tokens' in {CONFIG_FILENAME}: expected a positive integer")

    effort = raw.get("reasoning_effort")
    if effort is not None:
        if isinstance(effort, str) and effort in VALID_REASONING_EFFORTS:
            settings["reasoning_effort"] = effort
        else:
            _warn(
                f"Ignoring key 'reasoning_effort' in {CONFIG_FILENAME}: "
                f"expected one of {', '.join(VALID_REASONING_EFFORTS)}"
            )

    unknown = sorted(set(raw.keys()) - set(KNOWN_KEYS))
    if unknown:
        _warn(f"Ignoring unknown key(s) in {CONFIG_FILENAME}: {', '.join(unknown)}")

    return settings
