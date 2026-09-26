# 🔮 Future Roadmap

A scratchpad of planned features and ideas for Bash Agent. Items are roughly ordered by priority/enthusiasm, not commitment.

---

## In Progress / High Priority

- **Improved error recovery:** Better handling of API failures, model glitches, and safety filters with more graceful fallback strategies.

---

## Medium Priority


---

## Low Priority / Nice-to-Have


---

## ✅ Completed
- **Sub Agents / Hierarchical Orchestrator Mode:** with the opt-in `-o` /
  `--orchestrator` flag, the top-level agent runs as a planning **Orchestrator**
  that delegates tactical work to **Subagents** via the built-in
  `subagent "<task>"` command (sole content of a bash block). Without the flag
  the agent runs in the default standalone single-agent (direct) mode.
  - Only the top-level orchestrator may spawn; a subagent trying to spawn a
    nested subagent is refused.
  - The orchestrator runs in a STRICT, delegated sandbox (home/project tree
    hidden, working dir anchored at `.bash_agent_tmp/`, writes locked there, and
    `request-write` disabled) so it cannot touch project code directly; the
    subagent runs with full project read/write in the standard sandbox.
  - Each subagent is a nested in-process `Agent(is_subagent=True)` whose `exit`
    raises `SubagentExit` (instead of `sys.exit`) so the orchestrator regains
    control; its cumulative cost is folded into the parent session and its last
    2 messages are returned as the command's output.
  - Subagents persist to an isolated `history_subagent_<uuid8>.json`; they can
    use a different (cheaper) model via `--subagent-model` / `config.json`
    `subagent_model`, while the orchestrator uses `--orchestrator-model`.
  - The main benefit is context control: tactical noise stays in the subagent's
    context, and only a compact summary flows back to the orchestrator.
  - Single-agent / direct mode is the **DEFAULT**, and the orchestrator
    hierarchy is opt-in via `-o` / `--orchestrator`.
- **Dynamic Command Timeout:** a single BASH/PYTHON block can now raise its own
  grace period up to 600s via an optional first-line directive `# timeout: N`
  (N in [60, 600]). The default remains 60s (CLI `-t` or `config.BASH_TIMEOUT`).
  Malformed/misplaced directives (line 2+, space-before-colon, non-integer
  value) are treated as ordinary script content — they run verbatim with the
  default timeout and produce no warning, since a mistyped directive is
  low-stakes. When a command actually times out (sandbox exit 124), the
  harness appends a retry nudge teaching the model to re-run with a first-line
  `# timeout: N` directive to extend the limit. Implemented across `config.py`
  (`MAX_COMMAND_TIMEOUT = 600`), `agent.py` (`extract_timeout_directive` pure
  helper + `_build_timeout_nudge` + `_execute_script` wiring), and `sandbox.py`
  (optional per-call `timeout=` kwarg; TimeoutExpired banner reports the ACTUAL
  applied seconds). Covered by `tests/unit/test_timeout_directive.py`
  (T-46/T-47 + nudge cases), new sandbox cases, and shared
  `FakeSandbox.timeouts_used`.
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
