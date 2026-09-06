"""Vendor-neutral local token estimation for context accounting.

The ContextManager must decide WHEN to warn / trim based on how full the
conversation is. That decision is made from text on hand, before any provider
round-trip returns a real ``usage.prompt_tokens`` count, so we need a fast
LOCAL estimate of how many tokens a message string will consume.

Historical shortcut (``len(chars) / 8``) undercounted token pressure by a
large margin once real BPE tokenizers (~3.5 chars/token on average for the
DeepSeek family; similar for many others) replaced naive char heuristics.
That made the context meter read too low AND let real sessions drift well
past the intended safety quarter.

This module therefore estimates with the model's ACTUAL tokenizer whenever
possible, in rough priority:

1. **Known family tokenizers** (used when the model slug belongs to a family
   we can tokenize precisely and OFFLINE):
   * ``deepseek/...``            -> bundled ``deepseek_tokenizer`` (an exact
     BPE tokenizer for the DeepSeek family, packaged with the project's venv).
   * OpenAI-family slugs maps to tiktoken's model-aware encodings via
     ``tiktoken.encoding_for_model`` (e.g. ``gpt-4o`` -> ``o200k_base``,
     ``gpt-oss-120b`` -> ``o200k_harmony``). We also strip a leading
     ``openai/`` provider prefix before the lookup, which fixes slugs like
     ``openai/gpt-4o`` that LiteLLM's ``open_ai_chat_completion_models`` set
     does NOT contain (LiteLLM would otherwise remap them to ``gpt-3.5-turbo``
     / ``cl100k_base``, losing the 4o family's encoding).
2. **Provider-neutral LiteLLM path** (``litellm.token_counter``) for any
   other slug -- understands OpenRouter slugs and degrades to a bundled
   tiktoken BPE (``cl100k_base`` / ``o200k_base``) for unknown ones.
3. **Measured chars/token heuristic** (~3.5 chars/token) as a last resort when
   no tokenizer library is available / raises, so the system never
   hard-fails or requires a network round-trip just to estimate.

LiteLLM, tiktoken and the DeepSeek tokenizer are all imported **LAZILY** (each
inside its own counting path) because ``import litellm`` alone pulls in dozens
of optional third-party packages even for token-only usage, which would break
offline unit runs that do not have litellm installed.

To stay deterministic offline we force
``litellm.disable_hf_tokenizer_download = True`` immediately after the lazy
import -- LiteLLM will otherwise reach out to the Hugging Face hub for
provider tokenizers it cannot resolve locally, and instead degrades to the
bundled generic tiktoken encoding.
"""

from bash_agent.config import DEFAULT_MODEL


# Fallback rate used only when no tokenizer library is available / fails.
# ~3.5 chars/token measured across real session transcripts.
FALLBACK_CHARS_PER_TOKEN = 3.5

# Bounded cache so the hysteresis trim loop (which re-measures the same
# strings repeatedly) does not re-encode text every pass. Keyed by
# ``(model, text)`` so different model slugs never share wrong counts.
_TEXT_TOKEN_CACHE: dict = {}
_TEXT_TOKEN_CACHE_MAX = 1024
_MISSING = object()

# --- Lazy LiteLLM loader state ---------------------------------------------
# Imported ONLY on first counting attempt (never at module import). We cache
# the imported module and record availability so repeated calls don't re-try,
# and expose it as a boolean for tests / offline instrumentation.
_litellm = None
_litellm_import_attempted = False
_litellm_available = False

# --- Lazy tiktoken loader state ---------------------------------------------
_tiktoken = None
_tiktoken_import_attempted = False

# --- Lazy DeepSeek tokenizer loader state ------------------------------------
_deepseek_tok = None
_deepseek_import_attempted = False


def _load_litellm():
    """Import (once) and cache the LiteLLM module, or return ``None``.

    The import happens lazily here (NOT at module top) so that merely
    importing this tokenizer module stays light and works offline without
    litellm installed. Immediately after a successful import we force
    ``litellm.disable_hf_tokenizer_download = True`` so tokenizers are
    resolved locally (bundled tiktoken), never fetched from Hugging Face.
    """
    global _litellm, _litellm_import_attempted, _litellm_available

    if _litellm_import_attempted:
        return _litellm if _litellm_available else None

    _litellm_import_attempted = True
    try:
        import litellm  # type: ignore[import-not-found]
        # Keep estimation deterministic/offline: force the generic tiktoken
        # path instead of attempting HF hub downloads for known provider slugs.
        litellm.disable_hf_tokenizer_download = True
        _litellm = litellm
        _litellm_available = True
        return litellm
    except Exception:
        # Import failed (offline / not installed / broken env). Signal
        # unavailability; calling code falls back to the chars heuristic.
        _litellm_available = False
        return None


def _load_tiktoken():
    """Import (once) and cache the tiktoken module, or return ``None``."""
    global _tiktoken, _tiktoken_import_attempted

    if _tiktoken_import_attempted:
        return _tiktoken

    _tiktoken_import_attempted = True
    try:
        import tiktoken  # type: ignore[import-not-found]
        _tiktoken = tiktoken
        return tiktoken
    except Exception:
        _tiktoken = None
        return None


def _load_deepseek_tokenizer():
    """Import (once) and cache the bundled DeepSeek tokenizer instance.

    Returns the module-level ``ds_token`` instance (built from the tokenizer
    files shipped inside the ``deepseek_tokenizer`` package), or ``None`` if
    the package is not installed / fails to load.
    """
    global _deepseek_tok, _deepseek_import_attempted

    if _deepseek_import_attempted:
        return _deepseek_tok

    _deepseek_import_attempted = True
    try:
        import deepseek_tokenizer  # type: ignore[import-not-found]
        # module-level ready-to-use instance (DeepSeekTokenizer.from_pretrained)
        _deepseek_tok = deepseek_tokenizer.ds_token
        return _deepseek_tok
    except Exception:
        _deepseek_tok = None
        return None


def _encode_with_litellm(model: str, text: str) -> int:
    """Return the token count for ``text`` under ``model`` via LiteLLM.

    Raises (``RuntimeError`` / whatever ``litellm.token_counter`` raises)
    when LiteLLM is unavailable or the estimation itself fails -- callers are
    expected to catch and fall back to the chars/token heuristic.
    """
    lm = _load_litellm()
    if lm is None:
        raise RuntimeError("LiteLLM is not available")
    # text-only form: litellm.token_counter(model=..., text=..., ...) -> int
    return int(lm.token_counter(model=model, text=text))


def _route_deepseek(text: str) -> int | None:
    """Token count via the bundled DeepSeek tokenizer, or ``None``.

    ``None`` means "deepseek_tokenizer is not available"; returning a count
    means the model slug routed here (DeepSeek family) was tokenized exactly.
    """
    ds = _load_deepseek_tokenizer()
    if ds is None:
        return None
    try:
        ids = ds.encode(text, add_special_tokens=False)
    except Exception:
        return None
    return len(ids)


def _route_openai_family(model: str, text: str) -> int | None:
    """Token count via tiktoken's model-aware encoding for OpenAI-family.

    ``None`` means the model slug is not one tiktoken knows (or tiktoken is
    unavailable). We first strip an ``openai/`` provider prefix so
    ``openai/gpt-4o`` resolves exactly like bare ``gpt-4o`` (LiteLLM's own
    openai catalog does NOT include provider-prefixed names, silently losing
    the 4o family's ``o200k_base`` encoding).
    """
    tk = _load_tiktoken()
    if tk is None:
        return None
    bare = model
    if bare.lower().startswith("openai/"):
        bare = bare[len("openai/"):]
    try:
        encoding = tk.encoding_for_model(bare)
    except Exception:
        return None
    try:
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        return None


def _route_model_specific(model: str, text: str) -> int | None:
    """Return a model-specific token count or ``None`` to fall through.

    The returned count is produced by the best OFFLINE tokenizer available
    for the given model slug family. ``None`` means "no family tokenizer
    applied" so the caller continues down the LiteLLM -> chars/token chain.
    """
    low = model.lower()
    if "deepseek/" in low:
        n = _route_deepseek(text)
        if n is not None:
            return n
    # OpenAI-family (bare names or openai/-prefixed): o200k-family encodings
    # are far more accurate than LiteLLM's remap-to-cl100k for these slugs.
    openai_prefix = low.startswith("openai/")
    if openai_prefix or (
        "/" not in low
        and any(
            k in low
            for k in (
                "gpt-4o", "gpt-4.1", "gpt-4", "gpt-3.5", "gpt-oss",
                "o1", "o3", "gpt-5", "chatgpt-4o", "text-embedding",
                "text-davinci", "code-davinci", "instruction",
            )
        )
    ):
        n = _route_openai_family(model, text)
        if n is not None:
            return n
    return None


def clear_token_cache():
    """Drop the token-count cache (mostly useful for tests)."""
    _TEXT_TOKEN_CACHE.clear()


def count_tokens(text: str, model: str | None = None) -> int:
    """Return an accurate estimate of ``model``'s token count for ``text``.

    Parameters
    ----------
    text :
        The string whose token count is requested. Empty / falsy text
        short-circuits to 0.
    model :
        Model slug to estimate against. ``None`` (the default) means
        "use the configured DEFAULT_MODEL" so existing no-model callers keep
        working; unit tests may pass an explicit model.

    Estimation strategy (first applicable wins):
      1. family-specific offline tokenizer (DeepSeek via ``deepseek_tokenizer``,
         OpenAI-family via tiktoken ``encoding_for_model``, with an ``openai/``
         prefix stripped);
      2. vendor-neutral LiteLLM ``token_counter`` (resolves provider-slugs and
         degrades to a generic tiktoken BPE for unregistered slugs);
      3. measured ``round(len(text) / FALLBACK_CHARS_PER_TOKEN)`` heuristic.

    Returns
    -------
    int
        Always non-negative. Results are cached by ``(model, text)``.
    """
    if model is None:
        model = DEFAULT_MODEL
    if not text:
        return 0

    key = (model, text)
    cached = _TEXT_TOKEN_CACHE.get(key, _MISSING)
    if cached is not _MISSING:
        return cached

    # 1) Family-specific OFFLINE tokenizer (exact when available).
    n = _route_model_specific(model, text)

    # 2) Provider-neutral LiteLLM path.
    if n is None:
        try:
            n = _encode_with_litellm(model=model, text=text)
        except Exception:
            # LiteLLM unavailable / import failed / token_counter raised.
            n = None

    # 3) Measured chars/token heuristic last-resort.
    if n is None:
        n = max(1, round(len(text) / FALLBACK_CHARS_PER_TOKEN))

    if _TEXT_TOKEN_CACHE_MAX and len(_TEXT_TOKEN_CACHE) >= _TEXT_TOKEN_CACHE_MAX:
        _TEXT_TOKEN_CACHE.clear()
    _TEXT_TOKEN_CACHE[key] = n
    return n
