# 🔮 Future Roadmap

A scratchpad of planned features and ideas for Bash Agent. Items are roughly ordered by priority/enthusiasm, not commitment.

---

## In Progress / High Priority

- **Improved error recovery:** Better handling of API failures, model glitches, and safety filters with more graceful fallback strategies.
- **Use Tokens for Context Limit:** DONE — conversation accounting now uses real tokens via bash_agent.tokenizer.count_tokens. Marking complete; left as historical marker. 


---

## Medium Priority

- **Sub Agents:** Allow the top-level agent to spawn sub agents
  - There will be a `subagent` bash tool where the prompt and role can be passed in as command line arguments
  - Only the top level agent will have access to this rule
  - The subagent will operate in the same sandbox
  - The main benefit is that context can be controlled and kept in check

- **Dynamic Command Timeout:**
  - Currently, all commands have a fixed timeout
  - Sometimes, an agent might need a longer timeout (i.e. downloading a big file from a server)
  - Globally increasing the timeout might slow the agent down a lot
  - Let the Agent dynamically bump up the timeout for the bash command. Add a new field in the BASH_COMMAND block

---

## Low Priority / Nice-to-Have


---

## ✅ Completed
- **Model Specific Context Limit:** the per-session context ceiling is now derived from the OpenRouter model's `context_length` in TRUE TOKENS instead of a characters/8 heuristic. `bash_agent.tokenizer.count_tokens` uses the model's real DeepSeek tokenizer for all local accounting; `Agent` queries `/api/v1/models` once at startup, and `--token-budget` overrides the ceiling (default: a quarter of the model window) with an exact token count (`256K`, `1.5M`) or a percentage (`50%`). The resulting `context_limit` is passed into `ContextManager`, which honors it for the SCRATCHPAD warning threshold and the 80% hysteresis trim. On API failure or model missing from the catalog it gracefully falls back to `config.CONTEXT_LIMIT` (327,680 tokens).
- **Context limit warning:** when the conversation crosses `CONTEXT_WARN_PERCENT`% (99%) of `CONTEXT_LIMIT`, `add_message()` injects a one-time user-role message telling the LLM to back up important findings/notes to the SCRATCHPAD before the oldest ~20% of history is trimmed. Trimming is deferred until an assistant turn confirms the warning was read, so the backup commands+outputs at the tail survive the trim; it is acceptable to briefly exceed `CONTEXT_LIMIT` to deliver the warning. `reset` re-arms the flags.
- **Scratchpad injected once at session start (not on every change)** — `Agent.run()` reads `SCRATCHPAD.md` once and prepends it to the first user message of a fresh session. Later edits are NOT auto-injected; the model re-reads via `cat` when it needs a refresh. `SCRATCHPAD_LIMIT` (80k) truncation with `VISIBLE_%` reporting is preserved. Simplifies context accounting and avoids cache bloat from repeated re-injection.
- **Thinking token recovery on length termination** — captures reasoning tokens on `finish_reason: length`, injects `<thinking>` block and immediate answer prompt, then cleans up temporary messages from history upon completion.

- **Add a budget for a particular session in $USD** — implemented via `--budget` / `-b` flag and `DEFAULT_BUDGET` config
- **Add dict of providers for a given model slug in config.py** — implemented via `MODEL_PROVIDERS` dict in `config.py`
