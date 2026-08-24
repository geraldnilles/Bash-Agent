"""
Group 8 — Configuration tests for bash_agent.config.

T-42  config module constants and environment variable handling (P1)

The config module is the single source of truth for all runtime constants.
While mostly declarative, regressions here cascade silently:

  * DEFAULT_MODEL change -> wrong provider, wrong capabilities
  * HISTORY_FILE path resolution -> resume loses session or writes to /tmp
  * CONTEXT_LIMIT / SCRATCHPAD_LIMIT -> pruning fires at wrong boundaries
  * MAX_PIXELS -> vision tool accepts oversized images
  * OPENROUTER_API_KEY -> auth failures with cryptic errors

These tests pin the import-time bindings and env-var fallbacks so a future
refactor cannot silently shift behavior.
"""

import os
import unittest
from unittest import mock


class TestConfigConstants(unittest.TestCase):
    """Import-time constant values (with env unset)."""

    def _import_config_with_empty_env(self):
        """Import config with a clean environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import bash_agent.config as config_module
            importlib.reload(config_module)
            return config_module

    def test_default_model_is_set(self):
        config = self._import_config_with_empty_env()
        self.assertIsInstance(config.DEFAULT_MODEL, str)
        self.assertTrue(len(config.DEFAULT_MODEL) > 0)

    def test_api_keys_default_to_none(self):
        config = self._import_config_with_empty_env()
        # Module reads os.environ.get -> None when unset
        self.assertIsNone(config.OPENROUTER_API_KEY)

    def test_gemini_settings_removed(self):
        """Gemini direct-API support was removed; config must not define it."""
        config = self._import_config_with_empty_env()
        self.assertFalse(hasattr(config, "GEMINI_API_KEY"))
        self.assertFalse(hasattr(config, "GEMINI_BASE_URL"))

    def test_openrouter_base_url_is_fixed(self):
        config = self._import_config_with_empty_env()
        self.assertEqual(config.OPENROUTER_BASE_URL, "https://openrouter.ai/api/v1")

    def test_limits_are_positive_integers(self):
        config = self._import_config_with_empty_env()
        self.assertIsInstance(config.CONTEXT_LIMIT, int)
        self.assertGreater(config.CONTEXT_LIMIT, 0)
        self.assertIsInstance(config.SCRATCHPAD_LIMIT, int)
        self.assertGreater(config.SCRATCHPAD_LIMIT, 0)
        self.assertIsInstance(config.OUTPUT_LIMIT, int)
        self.assertGreater(config.OUTPUT_LIMIT, 0)
        self.assertIsInstance(config.MAX_PIXELS, int)
        self.assertGreater(config.MAX_PIXELS, 0)
        self.assertIsInstance(config.MAX_CODE_BLOCKS, int)
        self.assertGreater(config.MAX_CODE_BLOCKS, 0)
        self.assertIsInstance(config.BASH_TIMEOUT, int)
        self.assertGreater(config.BASH_TIMEOUT, 0)

    def test_default_budget_is_float(self):
        config = self._import_config_with_empty_env()
        self.assertIsInstance(config.DEFAULT_BUDGET, float)
        self.assertGreater(config.DEFAULT_BUDGET, 0.0)

    def test_console_colors_are_ansi_sequences(self):
        config = self._import_config_with_empty_env()
        for name in ["COLOR_CMD", "COLOR_OUT", "COLOR_PY_CMD", "COLOR_COST", "COLOR_RESET"]:
            val = getattr(config, name)
            self.assertIsInstance(val, str)
            self.assertTrue(val.startswith("\033["), f"{name} not ANSI: {val!r}")

    def test_openrouter_reasoning_effort_valid(self):
        config = self._import_config_with_empty_env()
        self.assertIn(config.DEFAULT_REASONING_EFFORT, ["none", "minimal", "low", "medium", "high"])

    def test_default_max_tokens(self):
        config = self._import_config_with_empty_env()
        self.assertIsInstance(config.DEFAULT_MAX_TOKENS, int)
        self.assertGreater(config.DEFAULT_MAX_TOKENS, 0)

    def test_model_providers_is_dict(self):
        config = self._import_config_with_empty_env()
        self.assertIsInstance(config.MODEL_PROVIDERS, dict)
        for model, providers in config.MODEL_PROVIDERS.items():
            self.assertIsInstance(model, str)
            self.assertIsInstance(providers, list)
            for p in providers:
                self.assertIsInstance(p, str)


class TestConfigEnvOverrides(unittest.TestCase):
    """Environment variables override module constants at import time."""

    def _fresh_import(self, env_vars):
        """Import config under a specific environment, return the module."""
        with mock.patch.dict(os.environ, env_vars, clear=True):
            # Force reload to re-evaluate os.environ.get calls
            import importlib
            import bash_agent.config as config_module
            importlib.reload(config_module)
            return config_module

    def test_openrouter_api_key_from_env(self):
        cfg = self._fresh_import({"OPENROUTER_API_KEY": "openrouter-key-456"})
        self.assertEqual(cfg.OPENROUTER_API_KEY, "openrouter-key-456")


class TestHistoryFilePathResolution(unittest.TestCase):
    """HISTORY_FILE uses os.path.abspath('.bash_agent_tmp/history.json') —
    this resolves against the CWD at IMPORT TIME, not at call time. Tests
    must patch CWD or the module constant directly."""

    def test_history_file_is_absolute_path(self):
        config = self._import_config_with_empty_env()
        self.assertTrue(os.path.isabs(config.HISTORY_FILE))
        self.assertTrue(config.HISTORY_FILE.endswith(".bash_agent_tmp/history.json"))

    def _import_config_with_empty_env(self):
        """Import config with a clean environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import bash_agent.config as config_module
            importlib.reload(config_module)
            return config_module

    def test_history_file_can_be_patched_for_tests(self):
        # This is how the test suite isolates history per-test:
        # patch bash_agent.context.HISTORY_FILE (where it's USED) or
        # patch bash_agent.config.HISTORY_FILE (where it's DEFINED).
        # Both work; we verify the latter here.
        with mock.patch("bash_agent.config.HISTORY_FILE", "/custom/path/history.json"):
            import bash_agent.config as cfg
            self.assertEqual(cfg.HISTORY_FILE, "/custom/path/history.json")


class TestConfigExportedSymbols(unittest.TestCase):
    """Verify the public surface that other modules depend on."""

    def _import_config_with_empty_env(self):
        """Import config with a clean environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import bash_agent.config as config_module
            importlib.reload(config_module)
            return config_module

    def test_all_expected_symbols_exist(self):
        cfg = self._import_config_with_empty_env()
        expected = [
            "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
            "DEFAULT_MODEL",
            "HISTORY_FILE", "CONTEXT_LIMIT", "SCRATCHPAD_LIMIT",
            "OUTPUT_LIMIT", "MAX_PIXELS", "MAX_CODE_BLOCKS",
            "BASH_TIMEOUT", "DEFAULT_BUDGET",
            "COLOR_CMD", "COLOR_OUT", "COLOR_PY_CMD", "COLOR_COST", "COLOR_RESET",
            "DEFAULT_REASONING_EFFORT", "DEFAULT_MAX_TOKENS",
            "MODEL_PROVIDERS",
            "APP_URL", "APP_TITLE", "APP_CATEGORIES",
        ]
        for name in expected:
            self.assertTrue(hasattr(cfg, name), f"Missing symbol: {name}")


if __name__ == "__main__":
    unittest.main()
