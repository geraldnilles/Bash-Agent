# Bash Agent — Test Suite (Planning Documents)

> **STATUS: IMPLEMENTATION IN PROGRESS.** Tests are being implemented group by
> group following [`TEST_PLAN.md`](TEST_PLAN.md) — see its Progress Summary for
> what is done. Helpers (`helpers/fakes.py`), the protocol-parsing group
> (`unit/test_agent_blocks.py`) and the special-commands group
> (`unit/test_agent_special_commands.py`) are live; run them with the command below.

---

## Purpose

This folder will hold the automated test suite for the `bash_agent` package.
The suite is designed around four hard constraints:

### 1. Fully offline
Tests must **never** contact remote servers. Every network boundary in the
codebase is mocked or faked:

| Network boundary | Production call site | Test strategy |
|---|---|---|
| OpenRouter chat completions | `llm.create_chat_completion()` | Fake `OpenAI` client injected into `llm._CLIENT_CACHE` |
| OpenRouter `/api/v1/models` (capability probe) | `Agent._check_model_capabilities()` | Patch `urllib.request.urlopen` |
| OpenRouter embeddings | `llm.create_embedding()` | Fake client returning fixed vectors |
| OpenRouter rerank endpoint | `search.rerank_documents()` | Patch `requests.post` |
| Gemini direct API | `llm.get_backend()` routing | Route through the same fake client |

The only *real* external process the suite may invoke is `systemd-run`
(integration tests), which is local and requires no network.

### 2. Standard library only
`pytest` is not installed in the project venv. The suite uses `unittest`,
`unittest.mock`, and `tempfile` exclusively. No new dependencies are added to
`pyproject.toml`.

### 3. Filesystem isolation
Nearly every module resolves paths against the **current working directory**
(`.bash_agent_tmp/`, `.gitignore`, `config.HISTORY_FILE`, embeddings DB). A
shared helper (`chdir_tmp`) will give each test a throwaway working directory
so tests never pollute the repo and never see each other's state.

### 4. Source code stays untouched
If a test exposes a bug in `bash_agent/`, the test pins the *current*
behavior (or is marked `@unittest.expectedFailure`) and the bug is recorded in
the table below. Fixes happen in separate, deliberate commits — never as a
side effect of adding tests.

---

## Planned Layout

```
tests/
├── README.md            # This file — how to run, principles, known bugs
├── AGENTS.md            # Context for AI coding agents working in this folder
├── TEST_PLAN.md         # The proposed test inventory (the design doc)
├── helpers/
│   ├── __init__.py
│   └── fakes.py         # Shared fakes: FakeSandbox, FakeLLMClient,
│                        #   fenced-block builders, chdir_tmp context manager
├── unit/                # Fast, pure-logic tests (no subprocesses)
│   ├── __init__.py
│   ├── test_agent_blocks.py
│   ├── test_agent_special_commands.py
│   ├── test_agent_pipeline.py
│   ├── test_agent_output_format.py
│   ├── test_agent_usage_budget.py
│   ├── test_context_pruning.py
│   ├── test_context_scratchpad.py
│   ├── test_context_persistence.py
│   ├── test_llm_adapter.py
│   ├── test_sandbox_construction.py
│   ├── test_utils.py
│   ├── test_prompts.py
│   ├── test_search_helpers.py
│   ├── test_vision.py
│   ├── test_transcribe.py
│   ├── test_memo.py
│   └── test_main_cli.py
└── integration/         # Real processes, still offline
    ├── __init__.py
    ├── test_sandbox_systemd.py     # Real systemd-run executions
    └── test_full_agent_loop.py     # Full Agent.run() driven by a fake LLM
```

---

## Known-Bug Register

Bugs found by this suite are recorded here; fixes happen in separate,
deliberate commits — never as a side effect of adding tests.

| # | Status | Location | Description | Pinned by |
|---|--------|----------|-------------|-----------|
| 1 | **FIXED** | `agent._extract_blocks` | Cross-type fence pairs (e.g. `START_BASH`/`END_PYTHON`) could glue onto a later same-type END fence, producing one spanning garbage "script" containing fences and prose that would be executed as code. Both strict and relaxed patterns now use a tempered body forbidding command-fence markers inside a block body. | `unit/test_agent_blocks.py::TestCrossTypeFenceMismatch` (T-05) |

---

## Running the Tests

```bash
# Everything
./venv/bin/python -m unittest discover -s tests -v

# Unit tests only (fast, no systemd dependency)
./venv/bin/python -m unittest discover -s tests/unit -v

# Integration tests only (requires a systemd user session)
./venv/bin/python -m unittest discover -s tests/integration -v

# Single module
./venv/bin/python -m unittest tests.unit.test_context_pruning -v
```

Environment requirements:
- Python ≥ 3.12 (declared in `pyproject.toml`; the source uses PEP 701 nested f-strings)
- `systemd-run` available for integration tests (auto-skipped otherwise via
  `@unittest.skipUnless`)
- No API keys required; tests must pass with `OPENROUTER_API_KEY` and
  `GEMINI_API_KEY` **unset**

---

## Design Notes

- **Seams already exist.** The codebase is unusually testable: `Sandbox` is
  injectable, `llm._CLIENT_CACHE` is a module-level dict that can be seeded
  with fakes, and `ContextManager` takes its UUID as a parameter. The plan
  exploits these seams rather than requiring refactors.
- **The fake LLM loop is the centerpiece.** `tests/integration/test_full_agent_loop.py`
  drives a complete `Agent.run()` session — task → fenced bash block → output
  feedback → `exit` — with zero network. This validates the UUID protocol,
  output formatting, context persistence, and budget enforcement end-to-end.
- **Protocol fixtures are generated, not hardcoded.** Helpers build
  `---START_BASH_COMMAND-{uuid}---` fences from a live UUID so tests survive
  any future change to marker names in exactly one place.
