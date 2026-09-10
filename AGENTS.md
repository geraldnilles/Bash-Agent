# AGENTS.md — Bash Agent Technical Architecture

> **For LLM agents editing this project.** This document describes the internal architecture, module responsibilities, data flow, and conventions. Read this before making changes.

---

## Project Overview

Bash Agent is a Python CLI that orchestrates an autonomous LLM-in-the-loop agent. The LLM communicates via a custom text-based protocol (UUID-fenced blocks) and executes commands in an ephemeral `systemd-run` sandbox. The project is a single Python package (`bash_agent/`) with no external service dependencies beyond an LLM API (OpenRouter).

**Entry point:** `python3 -m bash_agent.main` or the `bagent` console script (defined in `pyproject.toml`).

---

## Module Map

```
bash_agent/
├── main.py          # CLI argument parsing, entry point, glue
├── agent.py         # Core Agent class — the main run loop
├── config.py        # All constants, defaults, env var names
├── config_file.py   # Optional .bash_agent_tmp/config.json loader (model/max_tokens/reasoning_effort)
├── tokenizer.py     # Model-aware local token counting (DeepSeek BPE, tiktoken o200k family for OpenAI, o200k proxy for Gemini, LiteLLM, ~3.5 chars/token last resort)
├── context.py       # ContextManager — conversation history, pruning
├── sandbox.py       # Sandbox — systemd-run execution wrapper
├── llm.py           # LLM provider adapter layer (OpenRouter)
├── prompts.py       # System prompt template generation
├── utils.py         # Misc helpers (clipboard, cleanup, vim prompt)
├── search.py        # Semantic code search (embeddings + reranking)
├── vision.py        # Image analysis via LLM (native multimodal or fallback)
├── transcribe.py    # Audio transcription via LLM (native multimodal or fallback)
├── token_budget.py  # --token-budget parsing/derivation (absolute tokens or % of model window)
├── ignore.py        # gitignore-style glob matcher shared by --copy-project/--include/--ignore
├── sfx.py           # Subtle sound FX — programmatic wav synthesis (PipeWire: pw-play/paplay)
└── memo.py          # Voice memo recording (PipeWire + ffmpeg)
```

### Entry Point: `main.py`

Parses all CLI flags (`-m`, `-p`, `--resume`, `--commit`, etc.), resolves the initial task (from args, clipboard, or stdin), instantiates `Agent`, and calls `agent.run(initial_task)`. The `--copy-project` path short-circuits: it copies the project to the clipboard (via `copy_project_to_clipboard`), then prints the token count of the copied text as counted by the selected model's tokenizer (`bash_agent.tokenizer.count_tokens`, which the `_resolve_model` helper derives with the same CLI > config.json > `OPENROUTER_MODEL` > (`DEFAULT_AUDIO_MODEL` when `--audio` is set) > `DEFAULT_MODEL` precedence as `Agent.__init__`) before exiting 0.

**Key responsibility:** Translate CLI flags into `Agent` constructor kwargs. Does NOT contain agent logic.

**If adding a new CLI flag:**
1. Add the argument in `main.py:parse_args()`
2. Pass it through to `Agent.__init__()` if it affects the loop (e.g., `--budget`)
3. Handle it in `Agent` or pass it further to `Sandbox`/`ContextManager`

---

### Core Loop: `agent.py` → `class Agent`

This is the heart of the project. The `Agent` class:

1. **Initializes** the UUID, sandbox, context manager, and system prompt.
2. **Enters a loop** (`agent.run()`):
   - Send the conversation history to the LLM
   - Parse the LLM's response for fenced execution blocks
   - Execute them in the sandbox
   - Feed the output back into the conversation
   - Repeat until `exit` command or budget exhausted

**Key methods (in `agent.py`):**

| Method | Purpose |
|--------|---------|
| `__init__()` | Sets up UUID, model, sandbox, context, budget. Handles `--resume`. Resolves the model's context ceiling — by default a quarter of the model's `context_length`, overridable via the `token_budget`/`--token-budget` arg — and passes it to `ContextManager`, threading the model slug as well so content accounting uses that model's tokenizer (see `_fetch_model_context_limit` / `token_budget.py`). |
| `run(initial_task)` | The main loop. Sends prompts, parses responses, executes commands. |
| `_run_warmup_exchanges()` | Protocol warmup: on FRESH sessions only (`--resume` skips it), pre-fills history with two scripted assistant turns from the `WARMUP_TURNS` constant (a PYTHON version check plus a listing of all installed 3rd-party PyPI packages, then a BASH `ls -la`). Each template is parsed via the production `_extract_blocks()` (raises if it ever fails to parse) and executed via `_execute_script()`, so the injected transcript is byte-for-byte identical in format to a live exchange. Called from `run()` right after the initial task is committed. |
| `parse_and_execute(agent_msg)` | Coordination pipeline: extracts blocks via `_extract_blocks()`, dispatches each to `_handle_special_command()` or `_execute_script()`, enforces `MAX_CODE_BLOCKS` limit, and commits results via `_commit_execution_feedback()`. Returns `(executed: bool, feedback: str)`. |
| `_extract_blocks(response_text)` | Regex-parses the LLM response for `---START_BASH_COMMAND-{uuid}---` and `---START_PYTHON_COMMAND-{uuid}---` blocks. Returns `(blocks, None)` on success, or `([], warning_message)` when no blocks or malformed UUID fences are found. |
| `_handle_special_command(cmd_type, script)` | Intercepts built-in agent commands (`exit`, `reset`, `request-write`, `ask-user`, `copy-to-clipboard`). Returns `(handled: bool, formatted_output: str)`. Commands that terminate the session (`exit`, `copy-to-clipboard`) call `sys.exit()` in-process. |
| `_execute_script(cmd_type, script)` | Executes a bash or python script via `Sandbox.execute()` / `Sandbox.execute_python()`. First calls the pure helper `extract_timeout_directive(script)` which parses an optional first-line `# timeout: N` directive (N in [60, 600]) — a valid directive is scrubbed from the script and forwarded as the per-call `timeout=` kwarg; malformed/misplaced directives leave the ORIGINAL script intact (no data loss) and run verbatim with the default timeout (no teaching note). When the command actually times out (sandbox exit 124), appends a retry nudge (see `_build_timeout_nudge()`) reminding the model to use a first-line `# timeout: N` directive to extend the limit. Clamping of a valid out-of-range directive produces an advisory note. Scans sandbox output for `---START_ATTACHED_IMAGE-{uuid}---` and `---START_ATTACHED_AUDIO-{uuid}---` fences, strips base64 payloads, collects them in `self._pending_multimodal_images` / `self._pending_multimodal_audio`, and returns formatted output. On non-zero exit whose output references a `/tmp/` file with a file-not-found style error, appends a reminder that `/tmp/` is wiped each turn and `.bash_agent_tmp/` should be used instead (see `_build_tmp_file_warning()`). |
| `extract_timeout_directive(script)` | Pure string helper (no Agent/sandbox needed): parses an optional first-line `# timeout: N` directive from a BASH/PYTHON block body. Returns `(scrubbed_script, effective_seconds_or_None, note)`. Valid directives (case-insensitive key, positive base-10 int, first-line-only) are scrubbed; values are clamped to [BASH_TIMEOUT=60, MAX_COMMAND_TIMEOUT=600] with an advisory note when clamped. Malformed/misplaced directives (line 2+, space-before-colon, non-integer value, etc.) return the ORIGINAL script (no data loss) and are ignored silently — a mistyped directive is low-stakes and runs verbatim with the default timeout. Absent directives return `(script, None, None)`. |
| `_commit_execution_feedback(outputs)` | Bundles output blocks into a user message and appends to context. Builds structured multimodal content when `self._pending_multimodal_images` or `self._pending_multimodal_audio` is non-empty. |
| `_get_models_catalog()` | Fetches & caches the OpenRouter `/api/v1/models` catalog for ~1h. Returns a list of model dicts, or `[]` on API failure so callers fall back to safe defaults. Sharing one HTTP request across the multimodal/reasoning/context probes avoids three API calls at startup. |
| `_check_model_capabilities()` | Queries the OpenRouter models API to determine the model's supported input modalities. Sets `self.multimodal_capabilities` to a list like `["image"]`, or `None` for text-only models (or if the probe fails). |
| `_fetch_model_reasoning_info()` | Queries the OpenRouter models API for the model's reasoning support and sets `reasoning_supported_efforts`, `reasoning_mandatory`, `reasoning_default_effort`. Falls back to permissive defaults on network failure. |
| `_fetch_model_context_limit()` | Queries the OpenRouter models API for the model's `context_length` (tokens) and stores it on `model_full_context_tokens`. `__init__` then feeds that full window into `resolve_budget()` (plus the optional `--token-budget` override) to compute `self.context_limit`; with no override the budget defaults to a QUARTER of the window. On any failure/model miss the window stays `None` so `resolve_budget()` falls back to `config.CONTEXT_LIMIT` (or the equivalent percentage of the presumed default window). |

**The fenced-block regex pattern** (used in `_extract_blocks`):
- Bash: `---START_BASH_COMMAND-{uuid}---\n(.*?)\n---END_BASH_COMMAND-{uuid}---`
- Python: `---START_PYTHON_COMMAND-{uuid}---\n(.*?)\n---END_PYTHON_COMMAND-{uuid}---`

The agent strips leading/trailing whitespace from the captured code before execution.

**Output formatting:** Results are wrapped in:
```
---START_BASH_OUTPUT-EXIT_CODE_{n}-VISIBLE_{pct}%-{uuid}---
[output]
---END_BASH_OUTPUT-{uuid}---
```

**Special commands** must be the SOLE content of a bash block. They are intercepted in `_handle_special_command()` BEFORE sandbox execution. See the system prompt in `prompts.py` for the complete list.

**Error handling:** On API failure, the agent uses exponential backoff starting at 5s and doubling, capped at 160s max (`min(5 * 2^n, 160)` seconds). During the wait, the user can type `2x` + Enter to double `max_tokens` and retry immediately. This is handled via `select.select()` on stdin.

When `finish_reason == "length"` occurs and `choice.message.reasoning` contains partial thoughts, the agent performs a three-step recovery:
1. Temporarily appends `<thinking>{reasoning}</thinking>` as an assistant message and an instruction prompt as a user message.
2. Issues a follow-up completion request with `reasoning_effort="none"`.
3. Uses a `try...finally` block to pop both temporary recovery messages from `self.context.history` before returning the final response to the main loop.

When `finish_reason == "tool_calls"` occurs, the behavior is intentionally different:
1. Records the failed attempt as an assistant message (`[Invalid tool_calls attempt]`, including the tool call payload when available) and appends a user `[SYSTEM WARNING]` with the exact desired BASH/PYTHON fence syntax (live UUID interpolated).
2. `continue`s the `_get_llm_response()` loop with NO follow-up request and NO history cleanup — the correction is **permanently kept** in the conversation so the next LLM call sees what went wrong and replies with proper BASH/PYTHON blocks.
Unlike the `length` recovery, nothing here is temporary: the failed attempt and the warning remain in `history.json`.

**Budget tracking:** After each LLM call, cost is extracted from the API response (OpenRouter provides it natively). When `session_cost >= budget`, the loop exits.

---

### Configuration: `config.py`

All tunable constants. **Modify this file to change defaults.**

| Constant | Default | Where Used |
|----------|---------|------------|
| `DEFAULT_MODEL` | `"deepseek/deepseek-v4-pro"` | `agent.py` — fallback model |
| `DEFAULT_AUDIO_MODEL` | `"xiaomi/mimo-v2.5"` | `agent.py` / `main.py` — fallback model when `--audio` is passed (audio-capable input) |
| `CONTEXT_LIMIT` | 327,680 tokens (fallback) | `context.py` / `token_budget.py` — *fallback* TOKEN ceiling (¼ of the 1,310,720-token DeepSeek v4 flash window). The runtime ceiling is normally model-derived by `agent.py` (`resolve_budget()`), optionally overridden via `--token-budget`. This constant (×4, the presumed default full window) is only used when the probe fails or the model isn't catalogued. |
| `CONTEXT_WARN_PERCENT` | 99% | `context.py` — % of the *instance* `context_limit` at which the one-time SCRATCHPAD-backup warning is injected |
| `SCRATCHPAD_LIMIT` | 80,000 chars | `context.py` — scratchpad truncation warning |
| `OUTPUT_LIMIT` | 10,000 chars | `agent.py` — output block truncation |
| `MAX_CODE_BLOCKS` | 1 | `agent.py` — max code blocks executed per LLM response |
| `BASH_TIMEOUT` | 60 seconds | `sandbox.py` — default session/block timeout. A single block may extend this up to `MAX_COMMAND_TIMEOUT` via a first-line `# timeout: N` directive in the block body. |
| `MAX_COMMAND_TIMEOUT` | 600 seconds | `agent.py` / `sandbox.py` — hard ceiling for the optional per-command `# timeout: N` directive. Values above this are clamped down (with an advisory note). |
| `DEFAULT_BUDGET` | 0.10 USD | `agent.py` — session cost limit |
| `DEFAULT_REASONING_EFFORT` | `"low"` | `agent.py` — reasoning effort for OpenRouter |
| `DEFAULT_MAX_TOKENS` | 8192 | `agent.py` — max output tokens |
| `MAX_PIXELS` | 2,000,000 | `vision.py` — max image resolution |
| `MODEL_PROVIDERS` | `{}` | `llm.py` — provider whitelist per model |
| `APP_URL` | `"https://github.com/geraldnilles/Bash-Agent"` | `llm.py`, `agent.py`, `search.py` — OpenRouter App Attribution (`HTTP-Referer`) |
| `APP_TITLE` | `"Bash Agent"` | `llm.py`, `agent.py`, `search.py` — OpenRouter App Attribution (`X-OpenRouter-Title`) |
| `APP_CATEGORIES` | `"cli-agent"` | `llm.py`, `agent.py`, `search.py` — OpenRouter App Attribution (`X-OpenRouter-Categories`) |

**Color constants** (`COLOR_CMD`, `COLOR_OUT`, etc.) are ANSI escape codes for terminal output. Only used in `agent.py`'s console display.

---

### Conversation Management: `context.py` → `class ContextManager`

Manages the message list (`self.history: List[Dict[str, str]]`), context pruning, scratchpad one-shot injection, and session persistence.

Constructor: `ContextManager(uuid_str, context_limit=None, model=None)` — `model`
(an OpenRouter slug, e.g. `"deepseek/deepseek-v4-flash-0731"`) selects which
tokenizer `count_tokens` estimates against; when `None` it falls back to
`config.DEFAULT_MODEL`. `Agent` always passes its resolved model so pruning /
warning thresholds and the per-turn token stats share the same code path.

**Key responsibilities:**

1. **Message storage:** `add_message(role, content)` appends. The effective ceiling comes from the instance attribute `self.context_limit` (set by `Agent` from the model's `context_length`; defaults to the module constant `CONTEXT_LIMIT` when constructed bare).
   - **Context-limit warning:** once the conversation crosses `CONTEXT_WARN_PERCENT`% of the instance `context_limit` (99%), a one-time user-role message is injected telling the LLM to back up important notes to the SCRATCHPAD before the oldest ~20% of history is trimmed. Trimming is DEFERRED until the warning is confirmed — an ASSISTANT message must be added afterward (proving the model read the warning and issued its backup commands). It is acceptable to briefly exceed the ceiling to deliver the warning. `reset` re-arms the flags.
2. **Context pruning** (`_trim_context_if_needed()`): When TOTAL TOKENS (measured via `_history_tokens()`, which runs `bash_agent.tokenizer.count_tokens` per message with `self.model`) exceed the instance `context_limit`, incrementally trims the oldest messages down to 80% of that limit:
   - Multimodal messages (list content, e.g. `image_url` blocks) cannot be block-trimmed because the regex operations require strings (a list would raise `TypeError`). They are dropped entirely with no breadcrumb marker; surrounding context makes it obvious what happened.
   - Step 1: Delete the content of old `BASH_OUTPUT`/`PYTHON_OUTPUT` blocks entirely (replaced with `[BASH_OUTPUT DELETED TO SAVE CONTEXT]`)
   - Step 2: Truncate old `BASH_COMMAND`/`PYTHON_COMMAND` blocks to 80 chars
   - Step 3 (failsafe): Drop the oldest message entirely
   - Uses hysteresis (targets 80%) to avoid thrashing on every message
3. **Scratchpad one-shot injection** (`get_scratchpad_block()`): Reads `SCRATCHPAD.md` and returns a fenced block with a `VISIBLE_{pct}%` header (truncated at `SCRATCHPAD_LIMIT` with an `[ERROR]` suffix when oversized). `Agent.run()` calls it once for a fresh session's first user message — NOT re-injected on later changes.
4. **Persistence** (`save_history()` / `load_history()`): Serializes `{uuid, history}` to `.bash_agent_tmp/history.json`. Called after every message. Loaded on `--resume`.

**Content TOKEN calculation** (`ContextManager._content_tokens(content, model=None)`): Text (strings and `{"type": "text"}` parts) is tokenized with the model's vendor-neutral local tokenizer (`bash_agent.tokenizer.count_tokens(text, model=model)` — no character heuristics). The `_history_tokens()` instance method sums every message through that same static helper using `self.model`, and `add_message` / `_trim_context_if_needed` / the per-turn stats all call `_history_tokens()` so a single code path does the measurement. `image_url` parts are charged 1000 tokens/MEGAPIXEL from the decoded data URL; `input_audio` parts 400 tokens/MINUTE from parsed MP3 frames (both cached). Undecodable payloads fall back to flat TOKEN estimates (800 image / 6000 audio). NEVER scales with the raw base64 payload size.

---

### Tokenizer: `tokenizer.py` → `count_tokens`

Provides the **fast LOCAL** token estimate the ContextManager needs *before* any
provider round-trip returns a real `usage.prompt_tokens` count. Estimation is
**model-aware**: it tries the most accurate OFFLINE tokenizer for the model's
family first, then falls back through a provider-neutral path to a character
heuristic:

1. **Known family tokenizers** (offline, exact):
   * `deepseek/...` → the bundled **`deepseek_tokenizer`** BPE (same tokenizer
     files shipped with the model) so DeepSeek contexts are measured exactly.
   * OpenAI-family slugs (bare `gpt-4o`, `openai/gpt-4o`, `gpt-oss-120b`,
     `o1`, `gpt-5`, …) → **tiktoken `encoding_for_model`** (e.g. `gpt-4o` →
     `o200k_base`, `gpt-oss-120b` → `o200k_harmony`). A leading `openai/`
     provider prefix is stripped *before* the lookup because LiteLLM's own
     `open_ai_chat_completion_models` catalog contains only bare names — it
     would otherwise remap `openai/gpt-4o` to `gpt-3.5-turbo`/`cl100k_base`,
     silently losing the 4o family's encoding.
   * `gemini/*` and `google/gemini*` → **tiktoken `o200k_base`** as the best
     OFFLINE proxy for Gemini's real (256k-vocab) SentencePiece tokenizer,
     which is not open/downloadable. `o200k_base` tracks Gemini's density on
     multilingual/emoji text far better than LiteLLM's generic `cl100k_base`
     (the encoding every unknown slug otherwise collapses onto).
2. **Provider-neutral LiteLLM path** (`litellm.token_counter`) for any other
   slug — understands OpenRouter slugs and degrades to a bundled tiktoken BPE
   for unregistered ones.
3. **Measured chars/token heuristic** (~3.5 chars/token) as a last resort when
   no tokenizer library is available, so the system never hard-fails or
   requires a network round-trip just to estimate.

- **`count_tokens(text, model=None) -> int`** — `model` (an OpenRouter slug)
  selects the tokenizer; `None` falls back to `config.DEFAULT_MODEL`. Empty
  text returns 0; results are cached by `(model, text)` in a bounded dict so
  hysteresis pruning (which re-measures the same strings repeatedly) does not
  re-encode each pass.
- **Lazy imports** — `litellm`, `tiktoken`, and `deepseek_tokenizer` are each
  imported lazily (never at module import) so merely importing this module
  stays light and works offline when extra deps are absent. Immediately after
  a successful `litellm` import the module forces
  `litellm.disable_hf_tokenizer_download = True` so the LiteLLM path stays
  deterministic/offline (bundled tiktoken, never a Hugging Face download).
- **Graceful degradation** — any step that is unavailable or raises falls
  through to the next; the last resort never hard-fails.

**Historical note:** earlier code divided character counts by a nominal 8
(undercounting real BPE token pressure > 2×), then pinned to the DeepSeek-only
`deepseek-tokenizer`, then went model-agnostic through LiteLLM (which collapses
almost every non-bare slug onto the generic `cl100k_base`). The current
implementation restores true model-family fidelity offline while remaining
vendor-neutral for everything else.

### Sandbox Execution: `sandbox.py` → `class Sandbox`

Wraps `systemd-run` for isolated command execution.

**Constructor:** Takes `scratchpad_path`, an optional session-default `timeout` override (CLI `-t`; falls back to `config.BASH_TIMEOUT` = 60), and optional `uuid`/`multimodal_capabilities` flags. Individual `execute` / `execute_python` calls may override this per-call via their `timeout=` kwarg (up to the caller's discretion; `agent.py` clamps it to `MAX_COMMAND_TIMEOUT` = 600). Initializes `approved_write_paths` with at least the current working directory.

**`execute(script_content: str, timeout: int = None) -> (exit_code, output)`**: Writes the script to a temp file in `.bash_agent_tmp/`, then runs:
```
systemd-run --user --quiet --wait --collect --pipe \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=PrivateTmp=yes \
  --working-directory={cwd} \
  --property=ReadWritePaths={approved_paths} \
  /bin/bash {script_path}
```

**`execute_python(script_content: str, timeout: int = None) -> (exit_code, output)`**: Same as `execute()` but runs `python3` (preferring the venv's python3 if it exists) and sets `PYTHONPATH`.

**`request_write(path: str) -> (bool, str)`**: Interactive prompt for expanding `approved_write_paths`.

**Key details:**
- Temp scripts are created in `.bash_agent_tmp/` (NOT host `/tmp`) so the sandbox can access them
- `stderr` is merged into `stdout` via `subprocess.STDOUT` — output is always a single string
- Both `execute` and `execute_python` accept an optional per-call `timeout` kwarg.
  When provided (a positive int), it overrides `self.timeout` for that single
  invocation; otherwise `self.timeout` (from CLI `-t` or `config.BASH_TIMEOUT`)
  applies. On `TimeoutExpired`, the banner reports the ACTUAL applied seconds
  (the per-call override when used). Non-int / `< 1` values fall back to
  `self.timeout`.
- Timeout produces exit code 124 (matching `timeout` command convention)
- The host `PATH` and `OPENROUTER_API_KEY` are forwarded into the sandbox environment
- When `uuid` is set, `BASH_AGENT_UUID` is forwarded. `BASH_AGENT_MULTIMODAL` is set to a comma-separated list of the model's input modalities (e.g. `image` or `image,audio`, empty string when text-only) so tools like `vision.py` and `transcribe.py` can emit attached-image / attached-audio payloads

---

### LLM Adapter: `llm.py`

Abstraction layer over the OpenRouter backend. Normalizes payloads and request shaping.

**`get_llm_client()`**: Returns a cached `openai.OpenAI` client instance configured with `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`.

**`get_attribution_headers()`**: Returns the OpenRouter App Attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`, `X-OpenRouter-Categories`) built from the `APP_URL`, `APP_TITLE`, and `APP_CATEGORIES` config constants. These identify this application on OpenRouter's analytics dashboards and public model rankings. Used by every OpenRouter-backed request (chat, embedding, model-metadata probes in `agent.py`, and the rerank call in `search.py`).

**Session id (cache routing)**: `set_session_id()` / `get_session_id()` manage a module-level session identifier. `Agent.__init__()` calls `llm.set_session_id(self.uuid)` once the session UUID is resolved (fresh or resumed), so ALL OpenRouter calls in one session (chat, embedding, and the rerank call in `search.py` via `llm.get_session_id()`) carry the same `session_id`. OpenRouter uses this for consistent request routing, improving prompt-cache hits. When unset, no `session_id` key is added to any payload.

**`create_chat_completion(model, messages, max_tokens, extra_body, reasoning_effort)`**: The main LLM call:
- Injects `extra_headers = get_attribution_headers()` on every request
- Injects `session_id` into `extra_body` (when a session id is set)
- Injects `reasoning.effort` into `extra_body`
- Injects `provider.only` whitelist from `MODEL_PROVIDERS` config

**`create_embedding(model, input_texts)`**: Thin wrapper for embedding generation (used by `search.py`). Attaches `extra_headers = get_attribution_headers()` and `session_id` (when set) on every request.

---

### System Prompt: `prompts.py`

Generates the massive system prompt that defines the agent's behavior. Key function:

**`get_system_prompt(uuid, cwd, scratchpad_path, role_text, multimodal_capabilities)`**:
- Returns a formatted string with all protocol rules
- Injects the session UUID, working directory, current date
- Conditionally includes native-image attach instructions when `"image"` is in `multimodal_capabilities` (e.g. `["image"]`); otherwise includes the text-only vision fallback instructions
- Conditionally includes `role_text` if a `ROLE.md` file exists
- Includes rules for: execution blocks, output metadata, special commands, file editing, semantic search, vision, transcription, PDF processing, scratchpad usage, workflow & error recovery

**If modifying agent behavior, this is the most impactful file.** The prompt is ~4000 tokens and defines the entire protocol.

---

### Utilities: `utils.py`

| Function | Purpose |
|----------|---------|
| `cleanup_tmp_folder()` | Removes contents of `.bash_agent_tmp/` except protected files (SCRATCHPAD.md, ROLE.md, vim_prompt.tmp, embeddings.json, search_disabled, history.json, clipboard_blacklist.txt, config.json) |
| `copy_project_to_clipboard(file_paths=None, ignore=None, include=None)` | Copies project files to system clipboard as XML-like tagged format. `file_paths`/`include` and `ignore` accept **gitignore-syntax globs** (implemented by the shared `bash_agent.ignore.GitIgnoreMatcher`); `.gitignore`, clipboard blacklist, and user ignores are flattened into one last-match-wins rule list. `--files` is a deprecated alias of the `include` param. Returns the full composed clipboard text (prefix + file blocks + directory tree + suffix) whether or not the clipboard write succeeded, so callers can count its tokens without re-reading the clipboard. |
| `get_clipboard_content()` | Reads from system clipboard (supports xclip, wl-paste, pbpaste) |
| `get_vim_prompt()` | Reads user input from a temporary vim file |
| `is_binary_file(file_path)` | Checks if a file is binary (by extension or null byte detection) |

---

### Semantic Search: `search.py`

Standalone CLI (`search` command). Indexes the project directory using OpenAI-compatible embeddings with re-ranking.

**Architecture:**
1. Embeddings are cached in `.bash_agent_tmp/embeddings.json` keyed by file path
2. File hashes detect changes — only changed/new files are re-embedded
3. On query: embed the query, compute cosine similarity against all files, take top N×5 candidates, re-rank using the LLM, return top N
4. Files excluded by `.gitignore` patterns and common binary directories (`venv/`, `node_modules/`, `.git/`, etc.) are skipped
5. A sentinel file `.bash_agent_tmp/search_disabled` in the project directory disables search entirely

**Entry point:** `search.main()` — parses args, orchestrates indexing + query.

**Key functions:**
- `load_embeddings_db()` / `save_embeddings_db()` — JSON persistence
- `get_all_files(root_dir)` — walks directory tree, applies exclusion rules
- `get_file_hash(path)` — MD5 hash for change detection
- `get_file_content(file_path, root_dir, max_chars)` — reads file content (truncated for display)
- `fetch_embedding(client, texts)` — batches embedding API calls (batch size 10)
- `cosine_similarity(a, b)` — numpy dot product of normalized vectors
- `rerank_documents(client, query, docs, top_k)` — LLM-based re-ranking of candidates

---

### Vision: `vision.py`

Standalone CLI (`vision` command). Sends images to an LLM for description/analysis.

**Flow:**
1. The sandbox is launched with `BASH_AGENT_UUID` and `BASH_AGENT_MULTIMODAL` environment variables (set by `Sandbox` from `Agent` state).
2. When `BASH_AGENT_MULTIMODAL` includes `image` and a session UUID is present, `vision.py` emits a fenced base64 payload (`---START_ATTACHED_IMAGE-{uuid}---`) on stdout instead of calling the API.
3. `agent.py:_execute_script()` scans all sandbox output for these fences, strips the base64 from the visible output/context, and collects them in `self._pending_multimodal_images`.
4. If images were collected, the user message is built as a structured content array with `image_url` blocks; otherwise it stays plain text.
5. When the env vars are absent (standalone CLI or text-only model), `vision.py` runs its original OpenRouter call to a hosted vision endpoint.

**Key functions:**
- `encode_image(path)` — opens image, saves as PNG, base64-encodes
- `check_image_size(path)` — validates total pixels <= `MAX_PIXELS`

---

### Transcription: `transcribe.py`

Standalone CLI (`transcribe` command). Sends audio files to an LLM for transcription.

**Flow:**
1. Reads the audio file, checks size (default max 50 MB)
2. Auto-converts to mono MP3 via ffmpeg for broader backend compatibility
3. When `BASH_AGENT_MULTIMODAL` includes `audio` and a session UUID is present, emits a fenced base64 MP3 payload (`---START_ATTACHED_AUDIO-{uuid}---`) on stdout instead of calling the API.
4. `agent.py:_execute_script()` scans all sandbox output for these fences, strips the base64 from the visible output/context, and collects them in `self._pending_multimodal_audio`.
5. If audio was collected, the user message is built as a structured content array with `input_audio` blocks; otherwise it stays plain text.
6. When the env vars are absent (standalone CLI or text-only model), base64-encodes and sends to the LLM with a transcription prompt (or custom prompt via `-p`)
7. Optional: includes context files (`-c file1.md file2.md -- audio.opus`)

---

### Voice Memos: `memo.py`

Standalone CLI (`memo` command). Records audio from PipeWire microphones.

**Flow:**
1. Lists available audio sources via `pactl list sources short`
2. Records WAV via `pw-record` to a temp file
3. Converts to Opus (default) or MP3 via `ffmpeg`
4. Outputs timestamped file: `YYYY-MM-DDTHHMMSS_DURATION.opus`

**Supports:** source selection (`-s`), duration limit (`-d`), format choice (`-f opus|mp3`).

---

## Data Flow Diagram

```
User CLI (main.py)
    │
    ▼
Agent.run() loop
    │
    ├─► ContextManager.add_message("user", task)
    ├─► llm.create_chat_completion(history)     ──► OpenRouter API
    │       │
    │       ▼ (LLM response with fenced blocks)
    │
    ├─► Agent.parse_and_execute(response)
    │       │
    │       ├─► _extract_blocks()               ──► Parse UUID-fenced blocks
    │       ├─► _handle_special_command()       ──► exit, reset, request-write, ask-user, copy-to-clipboard
    │       ├─► _execute_script()               ──► extract_timeout_directive ──► Sandbox.execute / execute_python(timeout=?)
    │       │       ├─► image fences extracted  ──► _pending_multimodal_images
    │       │       ├─► audio fences extracted  ──► _pending_multimodal_audio
    │       │       ├─► clamp note appended     ──► trailing [SYSTEM WARNING]
    │       │       └─► timeout nudge (exit 124) ──► retry with `# timeout: N`
    │       ├─► _commit_execution_feedback()    ──► ContextManager.add_message (text or multimodal)
    │       │
    │       ▼ (output blocks injected into conversation)
    │
    ├─► ContextManager.add_message("assistant", response)  ── confirms warning
    ├─► ContextManager.add_message("user", output)           ── may inject warning
    ├─► ContextManager._trim_context_if_needed()  (hysteresis pruning, deferred until warning confirmed)
    ├─► ContextManager.save_history()            ──► history.json
    │
    └─► Loop continues until exit or budget exhausted
```

---

## Key Conventions

### 1. UUID-Fenced Protocol
- The session UUID is generated once at `Agent.__init__()` and embedded in ALL fenced block markers
- Fresh sessions are pre-filled with two scripted example exchanges (`WARMUP_TURNS` in `agent.py`) to teach new models the block format; edit those templates carefully — they are validated by the production parser at runtime
- Regex patterns in `_extract_blocks()` must match the exact UUID
- The UUID is persisted in `history.json` for session resumption

### 2. File Paths
- The agent's working directory is always the project root (where `main.py` was invoked)
- `.bash_agent_tmp/` is the ONLY writable scratch space (besides the project root)
- Temp files for sandbox execution MUST be created inside `.bash_agent_tmp/` (not host `/tmp`) because the sandbox's `PrivateTmp=yes` gives it an isolated `/tmp`

### 3. Output Truncation
- `OUTPUT_LIMIT` (10,000 chars) caps any single output block
- Truncation preserves first and last 5,000 characters, joining them with a single-line `TRUNCATION_BANNER` fence (`--⚠️⛔⚠️-OUTPUT_TRUNCATED_HERE-{uuid}-⚠️⛔⚠️--`) in the same dash-fence style as the START/END markers, so the break point is unmistakable yet costs almost no tokens
- The `VISIBLE_{pct}%` header tells the LLM how much was shown

### 4. Context Pruning
- Uses 80% hysteresis: only prunes when over the instance `context_limit` (model-derived; falls back to the `CONTEXT_LIMIT` constant), prunes down to 80% of that ceiling
- Token pressure is measured by `ContextManager._history_tokens()`, which runs `count_tokens(content, model=self.model)` over every message, so local accounting stays consistent with the selected model's tokenizer
- System prompt (index 0) is NEVER pruned
- Old OUTPUT blocks are deleted first (biggest savings), then COMMAND blocks are truncated, then entire messages are dropped

### 5. Sandbox Properties
- `ProtectSystem=strict`: /usr, /boot, /etc are read-only
- `ProtectHome=read-only`: /home is read-only
- `PrivateTmp=yes`: isolated /tmp and /var/tmp
- `ReadWritePaths`: dynamically expanded via `request-write`
- Unique `--unit=` per invocation (`bash-agent-<pid>-<n>-<hex>.service`); on
  client-side timeout the unit is stopped via `systemctl --user stop`
  (`Sandbox._reap_unit`) so timed-out workloads don't leak onto the host.
  NOTE: `--working-directory=`/`ReadWritePaths=` under `/tmp` break namespace
  setup when combined with `PrivateTmp=yes` (exit 226) — production always
  runs from a real project directory, so this only matters for tooling/tests.

### 6. Model / Token / Reasoning Priority (per key)

Settings resolve independently per key: CLI flag > `.bash_agent_tmp/config.json` > environment variable > hard-coded default.

1. CLI flags (`--model`, `--max-tokens`, `--reasoning-effort`, `--token-budget`) — always win
2. Optional persistent file `.bash_agent_tmp/config.json` (loaded by `config_file.py`; survives tmp-folder cleanup)
3. `OPENROUTER_MODEL` environment variable (model key only)
4. `--audio` flag — selects `DEFAULT_AUDIO_MODEL` when no model was chosen above (model key only)
5. Hard-coded defaults in `config.py` (`DEFAULT_MODEL` / `DEFAULT_AUDIO_MODEL`, `DEFAULT_MAX_TOKENS`; reasoning defaults to off)

Note: an explicit CLI `--reasoning-effort default` is a real choice that overrides the file; inside the file it means "defer to the model's built-in default".

---

## Adding a New Built-in Tool

To add a new tool (like `vision` or `search`):

1. **Create a new module** in `bash_agent/` (e.g., `bash_agent/foo.py`) with a `main()` function
2. **Register it** in `pyproject.toml` under `[project.scripts]`:
   ```toml
   foo = "bash_agent.foo:main"
   ```
3. **Update the system prompt** in `prompts.py` to document the new tool for the LLM
4. **Pass agent state via environment variables** (e.g., `Sandbox` already exposes `BASH_AGENT_UUID` and `BASH_AGENT_MULTIMODAL`); have your tool emit `---START_ATTACHED_IMAGE-{uuid}---` (image) or `---START_ATTACHED_AUDIO-{uuid}---` (audio) fenced payloads on stdout if it needs to inject multimodal content, and `agent.py` will parse and attach them automatically.
5. **Add test/example** if applicable

---

## Common Pitfalls When Editing

- **Regex escaping in `_extract_blocks()`**: The fenced block patterns use raw strings (`r"..."`). Be careful with the UUID interpolation — it's a literal string, not a regex group.
- **`systemd-run` permissions**: Adding `--property=` flags can break isolation. Always test with a command that tries to write to `/etc` to confirm sandboxing.
- **Context pruning off-by-one**: The system prompt is at index 0. Pruning iterates from index 1. Don't change this without understanding the trimming loop.
- **Scratchpad one-shot injection**: `Agent.run()` calls `ContextManager.get_scratchpad_block()` once at the start of a FRESH session (skipped on `--resume` since history already carries it). The block is prepended to the first user message. It is NOT re-injected on later changes — the LLM re-reads via `cat` when it needs a refresh. `get_scratchpad_block()` applies the `SCRATCHPAD_LIMIT` truncation (80k chars) with an honest `VISIBLE_%` header.
- **ContextManager uses a BOUND `count_tokens`**: `bash_agent/context.py` does
  `from bash_agent.tokenizer import count_tokens`, binding the name into its own
  module namespace. To fake/patch token counting for ContextManager tests it is
  NOT enough to patch `bash_agent.tokenizer.count_tokens` — you must patch
  `bash_agent.context.count_tokens` too. (`tests/helpers/fakes.py` exposes the
  reusable `DeterministicTokenCounts` patcher that covers both seams.)
- **Multimodal content format**: When `multimodal_capabilities` includes `"image"` and/or `"audio"`, images/audio attached via `vision.py`/`transcribe.py` cause the agent to construct content as a list of content blocks `[{"type": "text", ...}, {"type": "image_url", ...}, {"type": "input_audio", ...}]` instead of a plain string. The `ContextManager._content_tokens()` method and pruning logic must handle both formats.

---

## Dependencies

```
openai           # LLM client (OpenRouter)
litellm          # Vendor-neutral token counting (token_counter; lazy import)
numpy            # Embedding similarity calculations
Pillow           # Image resizing for vision
requests         # HTTP calls (search reranking via OpenRouter API)
```

All are declared in `pyproject.toml`. Note: `litellm` is imported lazily by
`tokenizer.py` so merely importing the tokenizer module stays light and works
offline when litellm is absent. The project uses `setuptools` as the build
backend.

---

## Testing

A formal offline test suite lives in `tests/` (stdlib `unittest`, run via the
project venv). See `tests/AGENTS.md` for the full inventory and implementation status:

```bash
./venv/bin/python -m unittest discover -s tests -v          # everything
./venv/bin/python -m unittest discover -s tests/unit -v     # fast unit tests only
```

When making changes:
1. Run the unit suite — it must stay green; add/extend tests for new behavior
   following the mocking seams documented in `tests/AGENTS.md`
2. Test with `bagent -m "Run ls and tell you what you see"` for live protocol validation
3. Test sandbox changes with a command that tries to write to `/etc` (should fail)
4. Test context pruning by artificially lowering `CONTEXT_LIMIT` and running a long session
5. Test resume with `bagent --resume` after a short session

---

## Environment

- **Required**: `OPENROUTER_API_KEY`
- **Required**: Linux with `systemd` and `systemd-run` (user mode)
- **Optional**: `OPENROUTER_MODEL`
- **Build tools**: `pipewire-utils` (for `memo`), `ffmpeg` (for `memo`), `poppler-utils` (for PDF processing in sandbox), `xclip` or `wl-paste` (for clipboard features)
