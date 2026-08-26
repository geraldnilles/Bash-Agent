"""
Group 42 — Optional persistent configuration file (.bash_agent_tmp/config.json).

T-42  config_file.load_config + Agent precedence wiring (P1)

The file lets users pin per-project settings without repeating CLI flags:

    {
      "model": "deepseek/deepseek-v4-pro",
      "max_tokens": 16384,
      "reasoning_effort": "medium"
    }

Contract under test:

  * Resolution precedence per key: CLI arg > config.json > env var > default.
    Per-key fallback means `--max-tokens` alone overrides ONLY max_tokens;
    model/reasoning_effort still come from the file.
  * The model slot sits BETWEEN the CLI flag and OPENROUTER_MODEL:
    config.json beats the env var, but never the flag.
  * Malformed/unreadable/non-object files degrade gracefully to {} with a
    stderr warning — never crash the agent.
  * Invalid values for known keys are dropped individually (with a warning);
    unknown keys warn and are ignored.
  * cleanup_tmp_folder() must PRESERVE config.json across fresh sessions
    (it is persistent user state, like SCRATCHPAD.md / history.json).

Seams (offline only, per tests/AGENTS.md rules):
  * load_config resolves its path against the CWD at CALL time, so chdir_tmp
    isolates every filesystem effect — no import-time constants involved.
  * Agent precedence tests construct via helpers._make_agent, which patches
    the capability probe and reasoning-info fetch (no network) and swaps in
    FakeSandbox afterwards (no systemd).
  * OPENROUTER_MODEL is removed from os.environ for determinism — the host
    may legitimately have it exported.

Value validation mirrors production constraints:
  * reasoning_effort choices match argparse in main.py exactly
    (none/minimal/low/medium/high/default, case-SENSITIVE).
  * max_tokens must be a positive int (bools excluded — bool subclasses int).
"""

import contextlib
import io
import json
import os
import unittest
from unittest import mock

from bash_agent import utils as utils_module
from bash_agent.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from bash_agent.config_file import (
    CONFIG_FILENAME,
    KNOWN_KEYS,
    VALID_REASONING_EFFORTS,
    get_config_path,
    load_config,
)
from tests.helpers.fakes import _make_agent, _stub_model_reasoning_info, chdir_tmp


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def without_env_vars(*names):
    """Temporarily remove environment variables, restoring them afterwards."""
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ConfigFileTestBase(unittest.TestCase):
    """Runs each test inside an isolated CWD containing .bash_agent_tmp/."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.tmpdir = stack.enter_context(chdir_tmp())
        os.makedirs(".bash_agent_tmp", exist_ok=True)
        self.config_path = os.path.join(".bash_agent_tmp", CONFIG_FILENAME)

    def write_config(self, data):
        """Serialize `data` as JSON to .bash_agent_tmp/config.json."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def write_raw_config(self, text):
        """Write literal text (for malformed-JSON cases)."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(text)


# ---------------------------------------------------------------------------
# T-42a — load_config basics
# ---------------------------------------------------------------------------

class TestLoadConfigBasics(ConfigFileTestBase):

    def test_missing_file_returns_empty_dict(self):
        # No config.json written — feature silently disabled.
        self.assertEqual(load_config(), {})

    def test_full_valid_config_round_trips(self):
        data = {
            "model": "deepseek/deepseek-v4-pro",
            "max_tokens": 16384,
            "reasoning_effort": "medium",
        }
        self.write_config(data)
        self.assertEqual(load_config(), data)

    def test_partial_config_only_contains_present_keys(self):
        self.write_config({"max_tokens": 2048})
        self.assertEqual(load_config(), {"max_tokens": 2048})

    def test_explicit_path_argument_is_used_verbatim(self):
        custom_path = os.path.join(self.tmpdir, "elsewhere.json")
        with open(custom_path, "w", encoding="utf-8") as f:
            json.dump({"model": "explicit/path"}, f)
        self.assertEqual(load_config(custom_path), {"model": "explicit/path"})

    def test_get_config_path_resolution(self):
        self.assertEqual(get_config_path("/x/y.json"), "/x/y.json")
        expected_tail = os.path.join(".bash_agent_tmp", "config.json")
        self.assertTrue(get_config_path().endswith(expected_tail))
        self.assertTrue(os.path.isabs(get_config_path()))

    def test_known_keys_constant_matches_validation_surface(self):
        self.assertEqual(KNOWN_KEYS, ("model", "max_tokens", "reasoning_effort"))
        self.assertIn("default", VALID_REASONING_EFFORTS)


# ---------------------------------------------------------------------------
# T-42b — graceful degradation & per-key validation
# ---------------------------------------------------------------------------

class TestLoadConfigGracefulDegradation(ConfigFileTestBase):

    def _load_capturing_stderr(self):
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            result = load_config()
        return result, stderr_buf.getvalue()

    def test_malformed_json_yields_empty_dict_with_warning(self):
        self.write_raw_config("{not valid json!!!")
        result, err = self._load_capturing_stderr()
        self.assertEqual(result, {})
        self.assertIn("[Config]", err)
        self.assertIn(CONFIG_FILENAME, err)

    def test_top_level_array_yields_empty_dict_with_warning(self):
        self.write_raw_config('["model", "max_tokens"]')
        result, err = self._load_capturing_stderr()
        self.assertEqual(result, {})
        self.assertIn("[Config]", err)

    def test_top_level_scalar_yields_empty_dict_with_warning(self):
        self.write_raw_config('"just a string"')
        result, _ = self._load_capturing_stderr()
        self.assertEqual(result, {})


class TestLoadConfigPerKeyValidation(ConfigFileTestBase):

    def _dropped(self, payload, key):
        self.write_config(payload)
        self.assertNotIn(key, load_config())

    def test_model_rejects_non_string(self):
        self._dropped({"model": 42}, "model")

    def test_model_rejects_empty_string(self):
        self._dropped({"model": ""}, "model")

    def test_model_strips_surrounding_whitespace(self):
        self.write_config({"model": "  openai/gpt-oss-120b \n"})
        self.assertEqual(load_config()["model"], "openai/gpt-oss-120b")

    def test_max_tokens_rejects_zero_and_negative(self):
        self._dropped({"max_tokens": 0}, "max_tokens")
        self._dropped({"max_tokens": -100}, "max_tokens")

    def test_max_tokens_rejects_bool_even_though_bool_subclasses_int(self):
        self._dropped({"max_tokens": True}, "max_tokens")
        self._dropped({"max_tokens": False}, "max_tokens")

    def test_max_tokens_rejects_strings_and_floats(self):
        self._dropped({"max_tokens": "8192"}, "max_tokens")
        self._dropped({"max_tokens": 12.5}, "max_tokens")

    def test_max_tokens_accepts_positive_int(self):
        self.write_config({"max_tokens": 32768})
        self.assertEqual(load_config()["max_tokens"], 32768)

    def test_reasoning_effort_accepts_every_documented_choice(self):
        for effort in ("none", "minimal", "low", "medium", "high", "default"):
            with self.subTest(effort=effort):
                self.write_config({"reasoning_effort": effort})
                self.assertEqual(load_config()["reasoning_effort"], effort)

    def test_reasoning_effort_is_case_sensitive_like_argparse_choices(self):
        # argparse choices=['none', ..., 'high'] would reject 'HIGH'; the
        # file loader must behave identically rather than silently normalizing.
        self._dropped({"reasoning_effort": "HIGH"}, "reasoning_effort")

    def test_reasoning_effort_rejects_unknown_values(self):
        self._dropped({"reasoning_effort": "extreme"}, "reasoning_effort")

    def test_invalid_value_for_one_key_does_not_poison_other_keys(self):
        self.write_config({
            "model": "valid/model",
            "max_tokens": "oops",
            "reasoning_effort": "high",
        })
        result = load_config()
        self.assertEqual(result, {"model": "valid/model", "reasoning_effort": "high"})

    def test_unknown_keys_warn_and_are_ignored(self):
        self.write_config({"temperature": 0.7, "model": "ok/model"})
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            result = load_config()
        self.assertEqual(result, {"model": "ok/model"})
        self.assertIn("temperature", stderr_buf.getvalue())


# ---------------------------------------------------------------------------
# T-42c — cleanup_tmp_folder preservation
# ---------------------------------------------------------------------------

class TestCleanupPreservesConfigFile(ConfigFileTestBase):

    def test_cleanup_keeps_config_json_but_removes_junk(self):
        self.write_config({"model": "keep/me"})
        junk_path = os.path.join(".bash_agent_tmp", "session_junk.txt")
        with open(junk_path, "w") as f:
            f.write("ephemeral")
        scratchpad_path = os.path.join(".bash_agent_tmp", "SCRATCHPAD.md")
        with open(scratchpad_path, "w") as f:
            f.write("# pad")

        utils_module.cleanup_tmp_folder()

        self.assertTrue(os.path.exists(self.config_path),
                        "config.json must survive cleanup (persistent user setting)")
        with open(self.config_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"model": "keep/me"})
        self.assertFalse(os.path.exists(junk_path))
        self.assertTrue(os.path.exists(scratchpad_path))  # regression guard

    def test_cleanup_still_protects_legacy_persistent_files(self):
        for name, content in [("ROLE.md", "# role"), ("history.json", "{}"),
                              ("clipboard_blacklist.txt", "")]:
            with open(os.path.join(".bash_agent_tmp", name), "w") as f:
                f.write(content)
        utils_module.cleanup_tmp_folder()
        for name, _ in [("ROLE.md", None), ("history.json", None),
                        ("clipboard_blacklist.txt", None)]:
            self.assertTrue(os.path.exists(os.path.join(".bash_agent_tmp", name)),
                            f"{name} must remain protected")


# ---------------------------------------------------------------------------
# T-42d — Agent precedence wiring
# ---------------------------------------------------------------------------

class TestAgentConfigPrecedence(ConfigFileTestBase):
    """CLI arg > config.json > OPENROUTER_MODEL env var > hard-coded default."""

    def _build_agent(self, **kwargs):
        return _make_agent(**kwargs)

    def test_defaults_apply_without_file_or_cli_or_env(self):
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent()
        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertEqual(agent.reasoning_effort, DEFAULT_REASONING_EFFORT)
        self.assertEqual(agent.max_tokens, DEFAULT_MAX_TOKENS)

    def test_config_file_overrides_hardcoded_defaults(self):
        self.write_config({
            "model": "config-file/model",
            "max_tokens": 9999,
            "reasoning_effort": "high",
        })
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent()
        self.assertEqual(agent.model, "config-file/model")
        self.assertEqual(agent.max_tokens, 9999)
        self.assertEqual(agent.reasoning_effort, "high")

    def test_config_file_beats_openrouter_model_env_var(self):
        # Documented priority: config.json outranks the legacy env var.
        self.write_config({"model": "config-file/model"})
        with mock.patch.dict(os.environ, {"OPENROUTER_MODEL": "env-var/model"}):
            agent = self._build_agent()
        self.assertEqual(agent.model, "config-file/model")

    def test_env_var_still_applies_when_file_absent_or_lacks_model(self):
        self.write_config({"max_tokens": 512})  # no "model" key
        with mock.patch.dict(os.environ, {"OPENROUTER_MODEL": "env-var/model"}):
            agent = self._build_agent()
        self.assertEqual(agent.model, "env-var/model")
        self.assertEqual(agent.max_tokens, 512)

    def test_cli_args_beat_config_file_for_every_key(self):
        self.write_config({
            "model": "config-file/model",
            "max_tokens": 9999,
            "reasoning_effort": "high",
        })
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent(
                model="cli-flag/model",
                reasoning_effort="minimal",
                max_tokens=111,
            )
        self.assertEqual(agent.model, "cli-flag/model")
        self.assertEqual(agent.max_tokens, 111)
        self.assertEqual(agent.reasoning_effort, "minimal")

    def test_per_key_fallback_cli_max_tokens_only(self):
        # A lone --max-tokens must NOT discard file-based model/effort.
        self.write_config({
            "model": "config-file/model",
            "max_tokens": 9999,
            "reasoning_effort": "medium",
        })
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent(max_tokens=42)
        self.assertEqual(agent.max_tokens, 42)          # CLI won
        self.assertEqual(agent.model, "config-file/model")   # file retained
        self.assertEqual(agent.reasoning_effort, "medium")   # file retained

    def test_reasoning_effort_default_in_file_means_none(self):
        # 'default' asks for the model's built-in behavior -> internal None.
        self.write_config({"reasoning_effort": "default"})
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent()
        self.assertIsNone(agent.reasoning_effort)

    def test_cli_reasoning_effort_default_disables_file_setting_too(self):
        # --reasoning-effort default is an explicit user choice; it must win
        # over the file rather than being treated as "flag absent".
        self.write_config({"reasoning_effort": "high"})
        with without_env_vars("OPENROUTER_MODEL"):
            agent = self._build_agent(reasoning_effort="default")
        self.assertIsNone(agent.reasoning_effort)

    def test_broken_config_file_falls_back_to_defaults_not_crash(self):
        self.write_raw_config("{ truncated json")
        stderr_buf = io.StringIO()
        with without_env_vars("OPENROUTER_MODEL"):
            with contextlib.redirect_stderr(stderr_buf):
                agent = self._build_agent()
        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertEqual(agent.reasoning_effort, DEFAULT_REASONING_EFFORT)
        self.assertEqual(agent.max_tokens, DEFAULT_MAX_TOKENS)
        self.assertIn("[Config]", stderr_buf.getvalue())


class TestFreshSessionKeepsConfigOnDisk(ConfigFileTestBase):
    """End-to-end guard: constructing a FRESH Agent (no keep-tmp) runs
    cleanup_tmp_folder(); config.json must still be readable afterwards."""

    def test_real_agent_construction_preserves_config(self):
        self.write_config({"model": "survivor/model", "max_tokens": 256})
        with open(os.path.join(".bash_agent_tmp", "junk.bin"), "wb") as f:
            f.write(b"x")

        from bash_agent.agent import Agent
        with mock.patch.object(Agent, "_check_model_capabilities", return_value=None):
            with mock.patch.object(Agent, "_fetch_model_reasoning_info", _stub_model_reasoning_info):
                with without_env_vars("OPENROUTER_MODEL"):
                    agent = Agent()  # fresh session -> cleanup runs inside __init__

        self.assertEqual(agent.model, "survivor/model")
        self.assertEqual(agent.max_tokens, 256)
        self.assertFalse(os.path.exists(os.path.join(".bash_agent_tmp", "junk.bin")))
        with open(self.config_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["model"], "survivor/model")


if __name__ == "__main__":
    unittest.main()
