# AGENTS.md — Bash Agent Technical Architecture

> **For LLM agents editing this project.** This document describes the internal architecture, module responsibilities, data flow, and conventions. Read this before making changes.

---

## Project Overview

Bash Agent is a Python CLI that orchestrates an autonomous LLM-in-the-loop agent. The LLM communicates via a custom text-based protocol (UUID-fenced blocks) and executes commands in an ephemeral `systemd-run` sandbox. The project is a single Python package (`bash_agent/`) with no external service dependencies beyond an LLM API (OpenRouter or Gemini).

**Entry point:** `python3 -m bash_agent.main` or the `bagent` console script (defined in `pyproject.toml`).

---

## Module Map

```
bash_agent/
├── main.py          # CLI argument parsing, entry point, glue
├── agent.py         # Core Agent class — the main run loop
├── config.py        # All constants, defaults, env var names
├── context.py       # ContextManager — conversation history, pruning, scratchpad
├── sandbox.py       # Sandbox — systemd-run execution wrapper
├── llm.py           # LLM provider adapter layer (OpenRouter / Gemini)
├── prompts.py       # System prompt template generation
├── utils.py         # Misc helpers (clipboard, cleanup, vim prompt)
├── search.py        # Semantic code search (embeddings + reranking)
├── vision.py        # Image analysis via LLM (native multimodal or fallback)
├── transcribe.py    # Audio transcription via LLM
└── memo.py          # Voice memo recording (PipeWire + ffmpeg)
```

### Entry Point: `main.py`

Parses all CLI flags (`-m`, `-p`, `--resume`, `--commit`, etc.), resolves the initial task (from args, clipboard, or stdin), instantiates `Agent`, and calls `agent.run(initial_task)`.

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
| `__init__()` | Sets up UUID, model, sandbox, context, budget. Handles `--resume`. Checks multimodal capabilities. |
| `run(initial_task)` | The main loop. Sends prompts, parses responses, executes commands. |
| `_run_warmup_exchanges()` | Protocol warmup: on FRESH sessions only (`--resume` skips it), pre-fills history with two scripted assistant turns from the `WARMUP_TURNS` constant (a PYTHON version check, then a BASH `ls -la`). Each template is parsed via the production `_extract_blocks()` (raises if it ever fails to parse) and executed via `_execute_script()`, so the injected transcript is byte-for-byte identical in format to a live exchange. Called from `run()` right after the initial task is committed. |
| `parse_and_execute(agent_msg)` | Coordination pipeline: extracts blocks via `_extract_blocks()`, dispatches each to `_handle_special_command()` or `_execute_script()`, enforces `MAX_CODE_BLOCKS` limit, and commits results via `_commit_execution_feedback()`. Returns `(executed: bool, feedback: str)`. |
| `_extract_blocks(response_text)` | Regex-parses the LLM response for `---START_BASH_COMMAND-{uuid}---` and `---START_PYTHON_COMMAND-{uuid}---` blocks. Returns `(blocks, None)` on success, or `([], warning_message)` when no blocks or malformed UUID fences are found. |
| `_handle_special_command(cmd_type, script)` | Intercepts built-in agent commands (`exit`, `reset`, `request-write`, `ask-user`, `copy-to-clipboard`). Returns `(handled: bool, formatted_output: str)`. Commands that terminate the session (`exit`, `copy-to-clipboard`) call `sys.exit()` in-process. |
| `_execute_script(cmd_type, script)` | Executes a bash or python script via `Sandbox.execute()` / `Sandbox.execute_python()`. Scans sandbox output for `---START_ATTACHED_IMAGE-{uuid}---` fences, strips base64 payloads, collects them in `self._pending_multimodal_images`, and returns formatted output. On non-zero exit whose output references a `/tmp/` file with a file-not-found style error, appends a reminder that `/tmp/` is wiped each turn and `.bash_agent_tmp/` should be used instead (see `_build_tmp_file_warning()`). |
| `_commit_execution_feedback(outputs)` | Bundles output blocks and scratchpad updates into a user message and appends to context. Builds structured multimodal content when `self._pending_multimodal_images` is non-empty. |
| `_check_model_capabilities()` | Queries OpenRouter API or checks for Gemini prefix to determine the model's supported input modalities. Sets `self.multimodal_capabilities` to a list like `["image"]`, or `None` for text-only models. |

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

**Error handling:** On API failure, the agent uses exponential backoff (2^n seconds). During the wait, the user can type `2x` + Enter to double `max_tokens` and retry immediately. This is handled via `select.select()` on stdin.

When `finish_reason == "length"` occurs and `choice.message.reasoning` contains partial thoughts, the agent performs a three-step recovery:
1. Temporarily appends `<thinking>{reasoning}</thinking>` as an assistant message and an instruction prompt as a user message.
2. Issues a follow-up completion request with `reasoning_effort="none"`.
3. Uses a `try...finally` block to pop both temporary recovery messages from `self.context.history` before returning the final response to the main loop.

**Budget tracking:** After each LLM call, cost is extracted from the API response (OpenRouter provides it natively; Gemini cost is calculated manually in `llm.py`). When `session_cost >= budget`, the loop exits.

---

### Configuration: `config.py`

All tunable constants. **Modify this file to change defaults.**

| Constant | Default | Where Used |
|----------|---------|------------|
| `DEFAULT_MODEL` | `"deepseek/deepseek-v4-pro"` | `agent.py` — fallback model |
| `CONTEXT_LIMIT` | 256,000 chars | `context.py` — triggers pruning |
| `SCRATCHPAD_LIMIT` | 80,000 chars | `context.py` — scratchpad truncation warning |
| `OUTPUT_LIMIT` | 10,000 chars | `agent.py` — output block truncation |
| `MAX_CODE_BLOCKS` | 1 | `agent.py` — max code blocks executed per LLM response |
| `BASH_TIMEOUT` | 60 seconds | `sandbox.py` — subprocess timeout |
| `DEFAULT_BUDGET` | 0.10 USD | `agent.py` — session cost limit |
| `DEFAULT_REASONING_EFFORT` | `"low"` | `agent.py` — reasoning effort for OpenRouter |
| `DEFAULT_MAX_TOKENS` | 8192 | `agent.py` — max output tokens |
| `MAX_PIXELS` | 2,000,000 | `vision.py` — max image resolution |
| `MODEL_PROVIDERS` | `{}` | `llm.py` — provider whitelist per model |

**Color constants** (`COLOR_CMD`, `COLOR_OUT`, etc.) are ANSI escape codes for terminal output. Only used in `agent.py`'s console display.

---

### Conversation Management: `context.py` → `class ContextManager`

Manages the message list (`self.history: List[Dict[str, str]]`), context pruning, scratchpad injection, and session persistence.

**Key responsibilities:**

1. **Message storage:** `add_message(role, content)` appends and triggers pruning.
2. **Context pruning** (`_trim_context_if_needed()`): When total characters exceed `CONTEXT_LIMIT`, incrementally trims the oldest messages down to 80% of the limit:
   - Multimodal messages (list content, e.g. `image_url` blocks) cannot be block-trimmed because the regex operations require strings (a list would raise `TypeError`). They are dropped entirely with no breadcrumb marker; surrounding context makes it obvious what happened.
   - Step 1: Delete the content of old `BASH_OUTPUT`/`PYTHON_OUTPUT` blocks entirely (replaced with `[BASH_OUTPUT DELETED TO SAVE CONTEXT]`)
   - Step 2: Truncate old `BASH_COMMAND`/`PYTHON_COMMAND` blocks to 80 chars
   - Step 3 (failsafe): Drop the oldest message entirely
   - Uses hysteresis (targets 80%) to avoid thrashing on every message
3. **Scratchpad injection** (`get_scratchpad_block()`): Reads `SCRATCHPAD.md`, hashes it, and only injects it into the conversation if it changed. Injected as:
   ```
   ---START_SCRATCHPAD.md-VISIBLE_{pct}%-{uuid}---
   [content]
   ---END_SCRATCHPAD.md-{uuid}---
   ```
4. **Scratchpad cleanup** (`remove_old_scratchpads()`): Strips old scratchpad blocks from history when a new one is injected.
5. **Persistence** (`save_history()` / `load_history()`): Serializes `{uuid, history}` to `.bash_agent_tmp/history.json`. Called after every message. Loaded on `--resume`.

**Content length calculation** (`_content_length()`): Handles both string content (legacy) and list-of-dicts content (multimodal format). Image blocks are estimated at ~800 tokens (~6400 chars).

---

### Sandbox Execution: `sandbox.py` → `class Sandbox`

Wraps `systemd-run` for isolated command execution.

**Constructor:** Takes `scratchpad_path`, an optional `timeout` override, and optional `uuid`/`multimodal_capabilities` flags. Initializes `approved_write_paths` with at least the current working directory.

**`execute(script_content: str) -> (exit_code, output)`**: Writes the script to a temp file in `.bash_agent_tmp/`, then runs:
```
systemd-run --user --quiet --wait --collect --pipe \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=PrivateTmp=yes \
  --working-directory={cwd} \
  --property=ReadWritePaths={approved_paths} \
  /bin/bash {script_path}
```

**`execute_python(script_content: str) -> (exit_code, output)`**: Same as `execute()` but runs `python3` (preferring the venv's python3 if it exists) and sets `PYTHONPATH`.

**`request_write(path: str) -> (bool, str)`**: Interactive prompt for expanding `approved_write_paths`.

**Key details:**
- Temp scripts are created in `.bash_agent_tmp/` (NOT host `/tmp`) so the sandbox can access them
- `stderr` is merged into `stdout` via `subprocess.STDOUT` — output is always a single string
- Timeout produces exit code 124 (matching `timeout` command convention)
- The host `PATH` and `OPENROUTER_API_KEY` are forwarded into the sandbox environment
- When `uuid` is set, `BASH_AGENT_UUID` is forwarded. `BASH_AGENT_MULTIMODAL` is set to a comma-separated list of the model's input modalities (e.g. `image` or `image,audio`, empty string when text-only) so tools like `vision.py` can emit attached-image payloads

---

### LLM Adapter: `llm.py`

Abstraction layer over OpenRouter and Gemini backends. Normalizes payloads and cost tracking.

**`get_backend(model_name)`**: Returns `"gemini"` if model starts with `"google"` AND `GEMINI_API_KEY` is set; otherwise `"openrouter"`.

**`get_llm_client(backend)`**: Returns a cached `openai.OpenAI` client instance with appropriate `base_url` and `api_key`.

**`create_chat_completion(model, messages, max_tokens, extra_body, reasoning_effort)`**: The main LLM call:
- Strips `google/` prefix for Gemini direct API
- Injects `reasoning.effort` into `extra_body` for OpenRouter
- Injects `provider.only` whitelist from `MODEL_PROVIDERS` config
- For Gemini: monkey-patches `response.model_dump()` to inject a calculated `cost` field

**`create_embedding(model, input_texts)`**: Thin wrapper for embedding generation (used by `search.py`).

**`calculate_gemini_cost(model_name, prompt_tokens, completion_tokens)`**: Manual pricing tier lookup since Gemini doesn't return cost natively.

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
| `cleanup_tmp_folder()` | Removes contents of `.bash_agent_tmp/` except protected files (SCRATCHPAD.md, ROLE.md, vim_prompt.tmp, embeddings.json, search_disabled, history.json, clipboard_blacklist.txt) |
| `copy_project_to_clipboard(files, ignore=None)` | Copies project files to system clipboard as XML-like tagged format; `ignore` is a comma-separated list of glob patterns (files/dirs) to exclude |
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
3. Base64-encodes and sends to the LLM with a transcription prompt (or custom prompt via `-p`)
4. Optional: includes context files (`-c file1.md file2.md -- audio.opus`)

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
    ├─► ContextManager.get_scratchpad_block()  ──► Reads SCRATCHPAD.md
    ├─► ContextManager.add_message("user", task)
    ├─► llm.create_chat_completion(history)     ──► OpenRouter / Gemini API
    │       │
    │       ▼ (LLM response with fenced blocks)
    │
    ├─► Agent.parse_and_execute(response)
    │       │
    │       ├─► _extract_blocks()               ──► Parse UUID-fenced blocks
    │       ├─► _handle_special_command()       ──► exit, reset, request-write, ask-user, copy-to-clipboard
    │       ├─► _execute_script()               ──► Sandbox.execute / execute_python
    │       │       └─► image fences extracted  ──► _pending_multimodal_images
    │       ├─► _commit_execution_feedback()    ──► ContextManager.add_message (text or multimodal)
    │       │
    │       ▼ (output blocks injected into conversation)
    │
    ├─► ContextManager.add_message("assistant", response)
    ├─► ContextManager.add_message("user", output)
    ├─► ContextManager._trim_context_if_needed()  (hysteresis pruning)
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
- Truncation preserves first and last 5,000 characters, inserting `...[Output Truncated]...` in the middle
- The `VISIBLE_{pct}%` header tells the LLM how much was shown

### 4. Context Pruning
- Uses 80% hysteresis: only prunes when over `CONTEXT_LIMIT`, prunes down to 80% of `CONTEXT_LIMIT`
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

### 6. Model Selection Priority
1. CLI `--model` flag
2. `OPENROUTER_MODEL` environment variable
3. `DEFAULT_MODEL` in `config.py`

---

## Adding a New Built-in Tool

To add a new tool (like `vision` or `search`):

1. **Create a new module** in `bash_agent/` (e.g., `bash_agent/foo.py`) with a `main()` function
2. **Register it** in `pyproject.toml` under `[project.scripts]`:
   ```toml
   foo = "bash_agent.foo:main"
   ```
3. **Update the system prompt** in `prompts.py` to document the new tool for the LLM
4. **Pass agent state via environment variables** (e.g., `Sandbox` already exposes `BASH_AGENT_UUID` and `BASH_AGENT_MULTIMODAL`); have your tool emit `---START_ATTACHED_IMAGE-{uuid}---` fenced payloads on stdout if it needs to inject multimodal content, and `agent.py` will parse and attach them automatically.
5. **Add test/example** if applicable

---

## Common Pitfalls When Editing

- **Regex escaping in `_extract_blocks()`**: The fenced block patterns use raw strings (`r"..."`). Be careful with the UUID interpolation — it's a literal string, not a regex group.
- **`systemd-run` permissions**: Adding `--property=` flags can break isolation. Always test with a command that tries to write to `/etc` to confirm sandboxing.
- **Context pruning off-by-one**: The system prompt is at index 0. Pruning iterates from index 1. Don't change this without understanding the trimming loop.
- **Gemini cost patching**: `llm.py` monkey-patches `response.model_dump`. If the OpenAI library changes its response object structure, this will break silently (cost will be None).
- **Scratchpad hash caching**: `ContextManager.last_scratchpad_hash` prevents re-injecting unchanged scratchpad. If you modify scratchpad injection logic, reset this hash or you'll get stale behavior.
- **Multimodal content format**: When `multimodal_capabilities` includes `"image"`, images attached via `vision.py` cause the agent to construct content as a list of content blocks `[{"type": "text", ...}, {"type": "image_url", ...}]` instead of a plain string. The `_content_length()` method and pruning logic must handle both formats.

---

## Dependencies

```
openai           # LLM client (OpenRouter and Gemini)
numpy            # Embedding similarity calculations
Pillow           # Image resizing for vision
requests         # HTTP calls (search reranking via OpenRouter API)
```

All are declared in `pyproject.toml`. The project uses `setuptools` as the build backend.

---

## Testing

A formal offline test suite lives in `tests/` (stdlib `unittest`, run via the
project venv). See `tests/AGENTS.md` and `tests/TEST_PLAN.md` for the full
inventory and implementation status:

```bash
./venv/bin/python -m unittest discover -s tests -v          # everything
./venv/bin/python -m unittest discover -s tests/unit -v     # fast unit tests only
```

When making changes:
1. Run the unit suite — it must stay green; add/extend tests for new behavior
   following the seams documented in `tests/TEST_PLAN.md`
2. Test with `bagent -m "Run ls and tell you what you see"` for live protocol validation
3. Test sandbox changes with a command that tries to write to `/etc` (should fail)
4. Test context pruning by artificially lowering `CONTEXT_LIMIT` and running a long session
5. Test resume with `bagent --resume` after a short session

---

## Environment

- **Required**: `OPENROUTER_API_KEY` (or `GEMINI_API_KEY` for direct Gemini)
- **Required**: Linux with `systemd` and `systemd-run` (user mode)
- **Optional**: `OPENROUTER_MODEL`, `GEMINI_BASE_URL`
- **Build tools**: `pipewire-utils` (for `memo`), `ffmpeg` (for `memo`), `poppler-utils` (for PDF processing in sandbox), `xclip` or `wl-paste` (for clipboard features)
