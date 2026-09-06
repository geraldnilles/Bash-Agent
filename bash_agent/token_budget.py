"""Context-window token-budget parsing & resolution.

The per-session context ceiling ("token budget") may be set from the CLI in
two ways:

  * an absolute number of TOKENS  ->  "327680", "256K", "1.5M", plain ints
  * a PERCENTAGE of the model's full context window -> "25%", "12.5%"

Suffixes K / M are metric (10^3 / 10^6), matching how model context windows
are typically marketed/displayed (e.g. 128K = 131,072 tokens). A bare number
is a positive integer count of tokens. Percentages must be in (0, 100].

parse_token_budget() returns a normalized SPEC tuple:

    None                                   no override (default behavior)
    ("pct", float)                         fraction not applied yet
    ("abs", int)                           exact token ceiling

resolve_budget() turns a spec + the model's reported full context window
into the concrete integer ceiling that ContextManager will prune against.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import re
from bash_agent.config import CONTEXT_LIMIT

# ---------------------------------------------------------------------------
# Spec representation
# ---------------------------------------------------------------------------

# Canonical token-budget spec, produced by parse_token_budget().
#   None          -> no override; caller keeps its default behavior
#   ("pct", 25.0) ->  25% of the model's full context window
#   ("abs", 8192) -> exactly 8,192 tokens
TokenBudgetSpec = Optional[Tuple[str, Union[float, int]]]

_PERCENT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")          # "25%" / "12 %"
_ABS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM])\s*$")    # "128K" / "2m" / "1.5M"
_INT_ABS_RE = re.compile(r"^\s*(\d+)\s*$")                        # "327680"

_K = 1_000
_M = 1_000_000


def parse_token_budget(value: Union[str, int, None]) -> TokenBudgetSpec:
    """Normalize a user-supplied token budget into a SPEC.

    Accepts:
      * None                                   -> None
      * int (>= 1), bare float token counts are NOT accepted
      * "327680" | "256K" | "128k" | "1M" | "1.5M"   -> absolute token count
      * "25%"  | "12.5%" | "50 %"                     -> percentage (0, 100]

    Raises ValueError with a human-readable message when the value cannot be
    parsed or is out-of-range. Whitespace is tolerated around the number.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1:
            raise ValueError(
                f"token budget must be at least 1 token, got {value}"
            )
        return ("abs", value)
    if isinstance(value, float):
        raise ValueError(
            "a bare token budget cannot be a float; use an integer "
            "token count (e.g. 327680 or 256K) or a percentage (e.g. 25%)"
        )
    if not isinstance(value, str):
        raise ValueError(
            f"cannot interpret token budget {value!r}: expected a token "
            "count (e.g. 327680, 256K) or a percentage (e.g. 25%)"
        )

    s = value.strip()
    if not s:
        raise ValueError("token budget cannot be empty")

    pct = _PERCENT_RE.match(s)
    if pct:
        p = float(pct.group(1))
        if not 0.0 < p <= 100.0:
            raise ValueError(
                f"token-budget percentage must be in (0, 100], got '{value}'"
            )
        return ("pct", p)

    # Decimal mantissa is allowed ONLY with an explicit K/M suffix
    # ("1.5M"); a bare decimal token count is not a supported input.
    m = _ABS_RE.match(s)
    if m:
        n = float(m.group(1))
        suffix = m.group(2).lower()
        multiplier = _M if suffix == "m" else _K
        n *= multiplier
        n = int(round(n))
        if n < 1:
            raise ValueError("token budget must be at least 1 token")
        return ("abs", n)

    m2 = _INT_ABS_RE.match(s)
    if m2:
        n = int(m2.group(1))
        if n < 1:
            raise ValueError("token budget must be at least 1 token")
        return ("abs", n)

    raise ValueError(
        f"cannot interpret token budget {value!r}: expected an absolute "
        "token count (e.g. 327680, 256K, 1.5M) or a percentage ending in "
        "'%' (e.g. 25%)"
    )


def resolve_budget(spec: TokenBudgetSpec, full_window_tokens: Optional[int]) -> int:
    """Resolve a parsed SPEC into a concrete integer token ceiling.

    * ("abs", N)      ->  N (independent of the model window)
    * ("pct", p)      ->  int(base * p / 100), where base is the model's FULL
                          context window when known; when the window is
                          unavailable (offline / model not catalogued) it
                          falls back to CONTEXT_LIMIT * 4, i.e. the presumed
                          full window of the default model (so the default
                          25% reproduces CONTEXT_LIMIT itself).
    * None            ->  Quarter of the FULL window when known (historical
                          default) ; CONTEXT_LIMIT when window unknown.

    The returned ceiling is always >= 1.
    """
    if full_window_tokens is None:
        full_window_tokens = int(CONTEXT_LIMIT * 4)  # presumed full default window

    if spec is None:
        ceiling = int(full_window_tokens / 4)
    else:
        kind, val = spec
        if kind == "abs":
            ceiling = int(val)
        else:  # "pct"
            ceiling = int(full_window_tokens * float(val) / 100.0)

    return max(1, ceiling)
