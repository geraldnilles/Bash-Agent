"""
Unit tests for the CLI context-window token-budget grammar / resolution
(bash_agent.token_budget).

The per-session context ceiling can be overridden on the command line with:

    * an absolute TOKEN count      -- "327680", "256K", "1.5M", plain ints
    * a PERCENTAGE of the model's full context window  -- "25%", "12.5%"

Suffixes K / M are metric (10^3 / 10^6). Bare floats are rejected; a decimal
mantissa is allowed only with a K/M suffix ("1.5M"). Percentages must fall in
(0, 100].

parse_token_budget() normalizes user input into a spec tuple:

    None                no override
    ("pct", float)      percentage of the full model window
    ("abs", int)        exact token ceiling

resolve_budget() folds a spec (+ the model's full context window, when known)
into the concrete integer ceiling that ContextManager prunes against.
"""
import unittest

from bash_agent.config import CONTEXT_LIMIT as MODULE_CONTEXT_LIMIT
from bash_agent.token_budget import parse_token_budget, resolve_budget


class ParseTokenBudgetCase(unittest.TestCase):
    """Grammar: absolute counts (int / K / M) and percentages."""

    def test_none_returns_none(self):
        self.assertIsNone(parse_token_budget(None))

    def test_bare_int_absolute(self):
        self.assertEqual(parse_token_budget(8192), ("abs", 8192))
        self.assertEqual(parse_token_budget(1), ("abs", 1))

    def test_plain_string_absolute(self):
        self.assertEqual(parse_token_budget("327680"), ("abs", 327680))
        self.assertEqual(parse_token_budget("  4096 "), ("abs", 4096))

    def test_k_suffix_decimal(self):
        # 'K' = 10^3 (metric), matching how context windows are displayed.
        self.assertEqual(parse_token_budget("256K"), ("abs", 256_000))
        self.assertEqual(parse_token_budget("128k"), ("abs", 128_000))

    def test_m_suffix_decimal(self):
        self.assertEqual(parse_token_budget("1M"), ("abs", 1_000_000))
        self.assertEqual(parse_token_budget("1.5M"), ("abs", 1_500_000))
        self.assertEqual(parse_token_budget("0.5M"), ("abs", 500_000))
        self.assertEqual(parse_token_budget("2m"), ("abs", 2_000_000))

    def test_percent(self):
        self.assertEqual(parse_token_budget("25%"), ("pct", 25.0))
        self.assertEqual(parse_token_budget("12.5%"), ("pct", 12.5))
        self.assertEqual(parse_token_budget(" 50 % "), ("pct", 50.0))
        self.assertEqual(parse_token_budget("100%"), ("pct", 100.0))

    def test_bare_float_rejected(self):
        with self.assertRaises(ValueError):
            parse_token_budget(1.5)
        # decimal without a K/M suffix is not supported
        with self.assertRaises(ValueError):
            parse_token_budget("1.5")

    def test_bool_rejected_like_int_underflow(self):
        # bools are ints in Python but must not be accepted as budgets.
        with self.assertRaises(ValueError):
            parse_token_budget(True)

    def test_invalid_percent_ranges(self):
        for bad in ("0%", "0.0%", "-5%", "101%", "100.1%"):
            with self.assertRaises(ValueError):
                parse_token_budget(bad)

    def test_zero_and_negative_absolute(self):
        with self.assertRaises(ValueError):
            parse_token_budget(0)
        with self.assertRaises(ValueError):
            parse_token_budget("0K")
        with self.assertRaises(ValueError):
            parse_token_budget("0M")

    def test_garbage(self):
        for bad in ("", "   ", "%", "abc", "12x", "1.2.3%"):
            with self.assertRaises(ValueError):
                parse_token_budget(bad)

    def test_non_string_non_int_rejected(self):
        with self.assertRaises(ValueError):
            parse_token_budget(["25%"])


class ResolveBudgetCase(unittest.TestCase):
    """Concrete ceiling derivation from spec + known full context window."""

    def test_none_default_quarter_of_window(self):
        self.assertEqual(resolve_budget(None, 4096), int(4096 / 4))

    def test_none_default_presumed_window_offline(self):
        # Unknown window -> CONTEXT_LIMIT is treated as the default 25% base.
        self.assertEqual(resolve_budget(None, None), MODULE_CONTEXT_LIMIT)

    def test_percent_uses_real_window(self):
        self.assertEqual(resolve_budget(("pct", 50.0), 4096), 2048)
        self.assertEqual(resolve_budget(("pct", 100.0), 4096), 4096)

    def test_percent_offline_uses_presumed_default_window(self):
        # CONTEXT_LIMIT = quarter of 1,310,720-token default model window.
        presumed_full = int(MODULE_CONTEXT_LIMIT * 4)
        self.assertEqual(
            resolve_budget(("pct", 25.0), None),
            int(presumed_full * 0.25),
        )
        # so the historical default is reproduced exactly even offline
        self.assertEqual(resolve_budget(("pct", 25.0), None), MODULE_CONTEXT_LIMIT)
        self.assertEqual(resolve_budget(("pct", 50.0), None), int(presumed_full / 2))

    def test_absolute_ignores_window(self):
        self.assertEqual(resolve_budget(("abs", 131_072), 4096), 131_072)
        self.assertEqual(resolve_budget(("abs", 131_072), None), 131_072)

    def test_never_below_one(self):
        # Defensive floor even if callers manufacture tiny specs.
        self.assertEqual(resolve_budget(("abs", 0), None), 1)


if __name__ == "__main__":
    unittest.main()
