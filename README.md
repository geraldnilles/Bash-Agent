# Bash Agent

> **Give the LLM a terminal, a few safety rules, and let it rip.**

Bash Agent is an autonomous LLM agent that does real work on your Linux machine using **only Bash and Python** as its tools. No sprawling plugin ecosystems, no fragile JSON tool-calling schemas, no Docker overhead — just a language model, a sandboxed shell, and a simple text-based protocol.

---

## Why This Exists

Most AI coding agents suffer from the same problems:

| Problem | Bash Agent's Solution |
|---------|----------------------|
| **JSON tool-calling is brittle** — escaping multi-line scripts with quotes, variables, and special characters into a single JSON string is painful for LLMs | **Fenced Markdown blocks** — the LLM writes raw, unescaped Bash or Python inside UUID-fenced blocks. No escaping, no JSON parser failures. |
| **Docker sandboxing is heavy** — spinning up containers per command is slow and resource-intensive | **Native `systemd-run` sandbox** — transient systemd services with `ProtectSystem=strict` provide genuine isolation with near-zero overhead |
| **Context windows overflow** — long agent sessions burn tokens and degrade performance | **Intelligent context pruning** — mimics a terminal scrollback buffer, aggressively trimming old output while preserving recent context |
| **The agent goes rogue** — unrestricted file writes are dangerous | **Dynamic permission gating** — the agent must explicitly `request-write` for any file outside its working directory, and the human approves or denies |

---

## What It Can Do

The agent operates entirely through the command line. Since it has access to a real Linux environment, it can:

- **Read, write, edit, and search files** using standard tools (`grep`, `sed`, `awk`, `find`, `git`)
- **Write and execute Python scripts** for complex logic
- **Use tools** like `curl`, `wget`, `jq`, `pandoc` (and install packages into the project venv)
- **Browse the web** via `curl` and parse HTML
- **Analyze images** using built-in vision capabilities
- **Transcribe audio** using built-in transcription
- **Record voice memos** from your microphone (PipeWire)
- **Semantically search** your codebase with embedding-based retrieval
- **Persist memory** between turns via a `SCRATCHPAD.md` file
- **Resume sessions** — stop and pick up exactly where you left off

---

## Quick Start

### Prerequisites

- **Linux** with `systemd` (the sandbox depends on `systemd-run`)
- **Python 3.8+**
- **An OpenRouter API key** ([get one here](https://openrouter.ai/keys)) — or a Gemini API key for direct Gemini access

### Installation

```bash
git clone https://github.com/geraldnilles/bash-agent.git  # REPLACE WITH YOUR FORK URL
cd bash-agent
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Configuration

Set your API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Optionally, set a default model:

```bash
export OPENROUTER_MODEL="deepseek/deepseek-v4-pro"
```

### Run It

```bash
# Interactive — prompts for your task
bagent

# One-shot — pass a task directly
bagent -m "Fix the bug in auth.py where users can bypass login"

# From clipboard — copy a task, then run
bagent --paste

# Execute a plan from clipboard (writes it to SCRATCHPAD.md first)
bagent --execute

# Resume your last session
bagent --resume

# Generate a git commit message for recent changes
bagent --commit
```

---

## How It Works (In 60 Seconds)

1. **You give it a task.** The agent receives your objective and a detailed system prompt explaining the protocol.
2. **The LLM responds** with a fenced code block:

   ```
   ---START_BASH_COMMAND-{uuid}---
   ls -la
   ---END_BASH_COMMAND-{uuid}---
   ```

3. **The agent executes it** in an ephemeral `systemd-run` sandbox and returns the output:

   ```
   ---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{uuid}---
   total 108
   -rw-r--r--  1 user user 16676 Mar 26 14:15 agent.py
   ...
   ---END_BASH_OUTPUT-{uuid}---
   ```

4. **The LLM sees the output** and decides on the next command. This loop continues until the task is complete.

**Each command runs in a fresh shell** — there's no persistent bash process. If the agent needs to `cd` or set variables, it includes those in each block.

### Safety by Default

The agent's sandbox has **read-only access** to your entire system. It can only write to:
- The project directory (`/home/gerald/Code/Bash-Agent/`)
- The `.bash_agent_tmp/` directory

If it needs to modify anything else, it must ask:

```
---START_BASH_COMMAND-{uuid}---
request-write /etc/nginx/nginx.conf
---END_BASH_COMMAND-{uuid}---
```

You approve or deny interactively.

---

## Key Features in Depth

### Vision (Image Analysis)

If your chosen model supports native image input (Gemini models, GPT-4V, etc.), the agent can analyze images directly:

```bash
vision architecture_diagram.png
```

For text-only models, the `vision` command falls back to a multi-modal model on OpenRouter and prints a text-based representation of the image. You can add a specific prompt if you want more than just a basic image-to-Markdown OCR task.

```bash
vision -p "Extract all SQL table names from this ERD" schema.png
```

### Audio Transcription

This tool uses an OpenRouter model that accesses audio input to generate a text transcription.

```bash
transcribe meeting_recording.opus
transcribe -p "List all action items from this meeting" status_call.mp3
```

### Voice Memo Recording

This is a simple tool i wrote for quickly recording audio which can be used by
your agent.

```bash
memo                      # Interactive — pick mic, record until Ctrl+C
memo -d 30               # Record exactly 30 seconds
memo -l                  # List available microphones
memo -s "Yeti Stereo"    # Use a specific mic
```

### Semantic Code Search

```bash
search "where is the authentication logic" -n 5
search "database connection pooling" -n 10
```

Uses OpenAI-compatible embeddings with re-ranking for accuracy. Indexes your entire project tree and updates incrementally.

### Session Resume

Crash your terminal? Hit Ctrl+C? No problem:

```bash
bagent --resume
```

The entire conversation history is persisted to `.bash_agent_tmp/history.json`.

### Budget Control

```bash
bagent -m "Refactor the entire codebase" --budget 0.50   # $0.50 USD max
```

When the budget is exhausted, the agent stops gracefully.

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `-m "task"` | Provide the task as a command-line argument |
| `-p, --paste` | Read task from clipboard |
| `-x, --execute` | Write clipboard to SCRATCHPAD.md and execute it |
| `-s, --clear-scratchpad` | Clear SCRATCHPAD.md before starting |
| `-k, --keep-tmp` | Preserve `.bash_agent_tmp/` between runs |
| `-d, --debug` | Dump full conversation to `/tmp/bash_agent_log.txt` |
| `--model <name>` | Override the default model (e.g., `openai/gpt-4o`) |
| `--reasoning-effort <level>` | Set reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `default` |
| `--max-tokens <n>` | Override max output tokens (default: 8192) |
| `-t, --timeout <n>` | Command timeout in seconds (default: 60) |
| `-b, --budget <n>` | Session cost budget in USD (default: $0.10) |
| `-r, --resume` | Restore previous session and continue |
| `--commit` | Resume and auto-generate a git commit message |
| `-c, --copy-project` | Copy project files to clipboard (for sharing with other AIs) |
| `--files "a.py,b.py"` | Specific files for `--copy-project` |

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `deepseek/deepseek-v4-pro` | Default model for OpenRouter |
| `GEMINI_API_KEY` | (optional) | Direct Gemini API key (bypasses OpenRouter) |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini API endpoint |

---

## Philosophy

Bash Agent was built on a contrarian bet: **generalization beats specialization.** Rather than building a fragile web of purpose-built tools (ReadFile, WriteFile, SearchCode, RunTest, etc.), it gives the LLM exactly two tools — Bash and Python — and trusts it to figure out the rest. This mirrors how human developers actually work: we don't have a "ReadFile tool," we just use a terminal.

The UUID-fenced block protocol is opinionated by design. JSON tool calling forces LLMs to contort natural code into escaped strings. Fenced blocks let the model output exactly what a developer would type — no translation layer, no impedance mismatch.

---

## License

MIT
