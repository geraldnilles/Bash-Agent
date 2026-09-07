# 🔮 Future Roadmap

A scratchpad of planned features and ideas for Bash Agent. Items are roughly ordered by priority/enthusiasm, not commitment.

---

## In Progress / High Priority

- **Improved error recovery:** Better handling of API failures, model glitches, and safety filters with more graceful fallback strategies.

---

## Medium Priority

- **Sub Agents:** Allow the top-level agent to spawn sub agents
  - There will be a `subagent` bash tool where the prompt and role can be passed in as command line arguments
  - Only the top level agent will have access to this rule
  - The subagent will operate in the same sandbox
  - The main benefit is that context can be controlled and kept in check

---

## Low Priority / Nice-to-Have


---

## ✅ Completed
- **Dynamic Command Timeout:** a single BASH/PYTHON block can now raise its own
  grace period up to 600s via an optional first-line directive `# timeout: N`
  (N in [60, 600]). The default remains 60s (CLI `-t` or `config.BASH_TIMEOUT`),
  and malformed/misplaced directives degrade safely to the default with a
  teaching note — never a silent change. Implemented across `config.py`
  (`MAX_COMMAND_TIMEOUT = 600`), `agent.py` (`extract_timeout_directive` pure
  helper + `_execute_script` wiring), and `sandbox.py` (optional per-call
  `timeout=` kwarg; TimeoutExpired banner reports the ACTUAL applied seconds).
  Covered by `tests/unit/test_timeout_directive.py` (T-46/T-47), new sandbox
  cases, and shared `FakeSandbox.timeouts_used`.
- **Vendor-Neutral Token Counting:** `bash_agent.tokenizer.count_tokens`
  now estimates through **LiteLLM's** `token_counter` (lazy import, forcing
  `disable_hf_tokenizer_download = True` so the bundled tiktoken BPE —
  `cl100k_base` / `o200k_base` — resolves unknown/arbitrary OpenRouter
  slugs deterministically offline), with a measured ~3.5 chars/token heuristic
  as the last-resort fallback. The `deepseek-tokenizer` dependency was removed
  from `pyproject.toml`/`requirements.txt`. ContextManager is now model-aware
  (`ContextManager(... , model=...)`, `_content_tokens(content, model=None)`,
  `_history_tokens()`); Agent threads its resolved model slug through and
  reuses the same counting path for pruning, warning thresholds, and per-turn
  stats.
- **Model Specific Context Limit:** the per-session context ceiling is now derived from the OpenRouter model's `context_length` in TRUE TOKENS instead of a characters/8 heuristic. `Agent` queries `/api/v1/models` once at startup, and `--token-budget` overrides the ceiling (default: a quarter of the model window) with an exact token count (`256K`, `1.5M`) or a percentage (`50%`). The resulting `context_limit` is passed into `ContextManager`, which honors it for the SCRATCHPAD warning threshold and the 80% hysteresis trim. On API failure or model missing from the catalog it gracefully falls back to `config.CONTEXT_LIMIT` (327,680 tokens).
- **Context limit warning:** when the conversation crosses `CONTEXT_WARN_PERCENT`% (99%) of `CONTEXT_LIMIT`, `add_message()` injects a one-time user-role message telling the LLM to back up important findings/notes to the SCRATCHPAD before the oldest ~20% of history is trimmed. Trimming is deferred until an assistant turn confirms the warning was read, so the backup commands+outputs at the tail survive the trim; it is acceptable to briefly exceed `CONTEXT_LIMIT` to deliver the warning. `reset` re-arms the flags.
- **Scratchpad injected once at session start (not on every change)** — `Agent.run()` reads `SCRATCHPAD.md` once and prepends it to the first user message of a fresh session. Later edits are NOT auto-injected; the model re-reads via `cat` when it needs a refresh. `SCRATCHPAD_LIMIT` (80k) truncation with `VISIBLE_%` reporting is preserved. Simplifies context accounting and avoids cache bloat from repeated re-injection.
- **Thinking token recovery on length termination** — captures reasoning tokens on `finish_reason: length`, injects `<thinking>` block and immediate answer prompt, then cleans up temporary messages from history upon completion.

- **Add a budget for a particular session in $USD** — implemented via `--budget` / `-b` flag and `DEFAULT_BUDGET` config
- **Add dict of providers for a given model slug in config.py** — implemented via `MODEL_PROVIDERS` dict in `config.py`
