"""Local token estimation for context accounting.

The ContextManager must decide WHEN to warn / trim based on how full the
conversation is. That decision is made from text on hand, before any provider
round-trip returns a real ``usage.prompt_tokens`` count, so we need a fast
LOCAL estimate of how many tokens a message string will consume.

DeepSeek's BPE averages only ~3.5 characters per token (not the old
rule-of-thumb of 8), so the historical ``len(chars) / 8`` shortcut undercounted
token pressure by more than 2x. That made the context meter read ~2.3x too
low AND let real sessions drift well past the intended safety quarter. This
module fixes it by tokenizing with the model's ACTUAL tokenizer when the
package is installed, otherwise falling back to a 3.5 chars/token estimate.

The ``deepseek-tokenizer`` package is pure Python with ZERO runtime
dependencies and bundles its own ``tokenizer.json`` (the same vocabulary the
DeepSeek API applies server-side, vocab 129283), so no network access is
needed at runtime. If it is missing, we degrade gracefully to the measured
3.5 chars/token ratio (still ~2.3x more accurate than the old /8 heuristic).
"""

# Fallback rate used only when the real tokenizer package is unavailable.
# ~3.5 chars/token measured across real DeepSeek session transcripts.
FALLBACK_CHARS_PER_TOKEN = 3.5

# Bounded cache so the hysteresis trim loop (which re-measures the same
# strings repeatedly) does not re-encode text every pass.
_TEXT_TOKEN_CACHE: dict = {}
_TEXT_TOKEN_CACHE_MAX = 1024
_MISSING = object()

_tokenizer = None


def _load_tokenizer():
    """Import (once) and cache the real DeepSeek tokenizer instance.

    ``deepseek-tokenizer`` is offline pure-Python: importing it loads the
    bundled vocab immediately, nothing is fetched from the network.
    """
    global _tokenizer
    if _tokenizer is None:
        import deepseek_tokenizer  # type: ignore[import-not-found]
        _tokenizer = deepseek_tokenizer.deepseek_tokenizer
    return _tokenizer


def clear_token_cache():
    """Drop the token-count cache (mostly useful for tests)."""
    _TEXT_TOKEN_CACHE.clear()


def count_tokens(text: str) -> int:
    """Return an accurate estimate of the model's token count for ``text``.

    Uses the real DeepSeek tokenizer when the package is installed, otherwise
    falls back to ``round(len(text) / FALLBACK_CHARS_PER_TOKEN)``. Always
    non-negative. Results are cached by exact string value.
    """
    if not text:
        return 0
    cached = _TEXT_TOKEN_CACHE.get(text, _MISSING)
    if cached is not _MISSING:
        return cached
    try:
        enc = _load_tokenizer()
        n = len(enc.encode(text))
    except Exception:
        # Package missing / import failed / tokenizer raised. Fall back to a
        # measured ~3.5 chars/token density estimate.
        n = max(1, round(len(text) / FALLBACK_CHARS_PER_TOKEN))
    if _TEXT_TOKEN_CACHE_MAX and len(_TEXT_TOKEN_CACHE) >= _TEXT_TOKEN_CACHE_MAX:
        _TEXT_TOKEN_CACHE.clear()
    _TEXT_TOKEN_CACHE[text] = n
    return n
