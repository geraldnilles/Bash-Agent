# 🔮 Future Roadmap

A scratchpad of planned features and ideas for Bash Agent. Items are roughly ordered by priority/enthusiasm, not commitment.

---

## In Progress / High Priority

- **Context limit warning:** Add a warning message to the LLM when approaching the context limit, suggesting the agent store important notes in the SCRATCHPAD.md before history is pruned.
- **Improved error recovery:** Better handling of API failures, model glitches, and safety filters with more graceful fallback strategies.
- **Expand Scratchpad limit and ONLY include once during initialization:** Refreshing the scratch pad every time it is written is blowing up the cache. and presumably, the agent that wrote the change understands it
- **Model Specific Context Limit:** Look at model properties and set a context limit based on what the model can actually handle.
- **Use Tokens for Context Limit:** Right now, we are using characters as a proxy for tokens. THat is helpful for when decided how many message to remove, but 


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
- **Thinking token recovery on length termination** — captures reasoning tokens on `finish_reason: length`, injects `<thinking>` block and immediate answer prompt, then cleans up temporary messages from history upon completion.

- **Add a budget for a particular session in $USD** — implemented via `--budget` / `-b` flag and `DEFAULT_BUDGET` config
- **Add dict of providers for a given model slug in config.py** — implemented via `MODEL_PROVIDERS` dict in `config.py`
