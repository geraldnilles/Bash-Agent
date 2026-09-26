# Bash Agent — Test Suite

> The suite is now **E2E-first**. A handful of end-to-end journeys drive the
> REAL `bash_agent` harness code (CLI → `Agent.run()` → LLM seam → parser →
> sandbox seam → context/budget → disk) with every network/OS boundary faked.
> The old `tests/unit/` micro-suite has been removed; its coverage is now
> subsumed by the journeys in `tests/e2e/` (see the mapping table below).

Run everything with:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

Run just the offline E2E suite (no `bwrap`, fastest, no network):

```bash
./venv/bin/python -m unittest discover -s tests/e2e -v
```

---

## Purpose

This folder holds the automated test suite for the `bash_agent` package.
The suite is designed around four hard constraints:

### 1. Fully offline
Tests must **never** contact remote servers. Every network boundary in the
codebase is mocked or faked at a narrow seam:

| Network boundary | Production call site | Test strategy |
|---|---|---|
| OpenRouter chat completions | `llm.create_chat_completion()` | Fake `OpenAI` client injected into `llm._CLIENT_CACHE["openrouter"]` |
| OpenRouter `/api/v1/models` (capability probe) | `Agent._get_models_catalog()` / `_check_model_capabilities()` | Patch `urllib.request.urlopen` with a canned catalog (or a raiser) |
| OpenRouter embeddings | `llm.create_embedding()` | Fake client returning fixed vectors |
| OpenRouter rerank endpoint | `search.rerank_documents()` | Patch `bash_agent.search.requests.post` |

The only *real* external process the suite may invoke is `bwrap`
(`tests/integration/`), which is local and requires no network.

### 2. Standard library only
`pytest` is not installed in the project venv. The suite uses `unittest`,
`unittest.mock`, `tempfile`, and `contextlib` exclusively. No new
dependencies are added to `pyproject.toml`.

### 3. Filesystem isolation
Nearly every module resolves paths against the **current working directory**
(`.bash_agent_tmp/`, `.gitignore`, `config.HISTORY_FILE`, embeddings DB).
The harness gives each test a throwaway working directory so tests never
pollute the repo and never see each other's state.

### 4. Source code stays untouched
If a test exposes a bug in `bash_agent/`, the suite pins the *current*
behavior (or marks it `@unittest.expectedFailure`) and the bug is recorded in
the table below. Fixes happen in separate, deliberate commits — never as a
side effect of adding tests.

---

## Layout

```
tests/
├── README.md            # This file — how to run, principles, known bugs
├── AGENTS.md            # Context for AI coding agents working in this folder
├── __init__.py
├── helpers/
│   ├── __init__.py
│   └── fakes.py         # Shared offline fakes (used by BOTH e2e and integration):
│                        #   FakeLLMClient, FakeSandbox, make_fake_response,
│                        #   fenced-block builders, chdir_tmp / chdir_repo_tmp,
│                        #   DeterministicTokenCounts, probe stubs, bwrap_available
├── e2e/                 # End-to-end journeys over the REAL harness (no bwrap)
│   ├── __init__.py
│   ├── harness.py               # E2ECase: patches the seams + runaway_guard()
│   ├── test_e2e_cli_loop.py     # main.main() -> real Agent.run() happy path + arg validation
│   ├── test_e2e_recovery.py     # finish_reason=tool_calls / length recovery mid-loop
│   ├── test_e2e_special_commands.py  # request-write, ask-user, exit, reset, copy-to-clipboard
│   ├── test_e2e_context.py      # pruning ladder, warning, scratchpad, token-budget resolution
│   ├── test_e2e_resume.py       # --resume: restore history + re-bind UUID
│   ├── test_e2e_budget.py       # budget accounting, exhaustion, save-before-break
│   ├── test_e2e_orchestrator.py # orchestrator -> subagent dispatch (hierarchical mode)
│   └── test_e2e_supporting.py   # tokenizer routing, config.json, ignore matcher,
│                                #   vision/transcribe/memo/history helpers, multimodal attach
└── integration/         # Real processes, still offline
    ├── __init__.py
    ├── test_sandbox_bwrap.py       # Real bwrap executions (smoke, timeout, isolation)
    └── test_full_agent_loop.py     # Full Agent.run() driven by a fake LLM
```

`tests/e2e/harness.py` is the single place that knows how to keep the suite
offline. `E2ECase` patches `urllib.request.urlopen`, seeds
`llm._CLIENT_CACHE`, stubs `bash_agent.search.requests.post`, and points
`bash_agent.context.HISTORY_FILE` at the test's temp dir. Individual tests
swap in a `FakeSandbox` where they do not want a real `bwrap` process.

---

## E2E journey map

The removed `tests/unit/` micro-tests are replaced by these end-to-end
journeys. Each journey drives the REAL code path named in the middle column;
only the LLM and the sandbox subprocess are faked.

| E2E module | Real code exercised | Replaces (former unit tests) |
|---|---|---|
| `test_e2e_cli_loop.py` | `main.main()` argparse → `Agent(...)` → `run()` | `test_main_cli.py`, `test_config.py` |
| `test_e2e_recovery.py` | `Agent._get_llm_response()` recovery branches | `test_agent_llm_response.py` |
| `test_e2e_special_commands.py` | `Agent._handle_special_command()` + termination | `test_agent_special_commands.py` |
| `test_e2e_context.py` | `ContextManager` trim/warn, scratchpad inject, `token_budget` | `test_context_pruning.py`, `test_context_warning.py`, `test_context_scratchpad.py`, `test_token_budget.py`, `test_model_context_limit.py` |
| `test_e2e_resume.py` | `--resume` load + UUID re-bind + persistence | `test_context_persistence.py`, `test_history.py` |
| `test_e2e_budget.py` | `_record_response_usage` / budget gate / `save_history` | `test_budget_persistence.py`, `test_agent_reasoning_effort.py` (usage paths) |
| `test_e2e_orchestrator.py` | orchestrator→subagent lifecycle, cost folding, `SubagentExit` | `test_subagent.py`, `test_orchestrator_prompts.py` |
| `test_e2e_supporting.py` | tokenizer routing, config.json precedence, ignore matcher, vision/transcribe/memo/history helpers, multimodal attach, block parsing | `test_tokenizer_routing.py`, `test_config_file.py`, `test_ignore_matcher.py`, `test_vision.py`, `test_transcribe.py`, `test_memo.py`, `test_agent_blocks.py`, `test_agent_output_format.py`, `test_fakes.py` |
| `integration/test_full_agent_loop.py` | whole stack, resume, orchestrator | `test_agent_pipeline.py` (loop-level) |

---

## Known-Bug Register

Bugs found by this suite are recorded here; fixes happen in separate,
deliberate commits — never as a side effect of adding tests.

| # | Status | Location | Description | Pinned by |
|---|--------|----------|-------------|-----------|
| 1 | **FIXED** | `agent._extract_blocks` | Cross-type fence pairs (e.g. `START_BASH`/`END_PYTHON`) could glue onto a later same-type END fence, producing one spanning garbage "script" containing fences and prose that would be executed as code. Both strict and relaxed patterns now use a tempered body forbidding command-fence markers inside a block body. | `e2e/test_e2e_supporting.py::TestAgentMultimodalAttach` (block-parsing path, indirect); full fence-mismatch case no longer has a dedicated test — see *Gaps* |
| 2 | **FIXED** | `sandbox.Sandbox.execute/_execute_python` | On `BASH_TIMEOUT`, `subprocess.run` SIGKILLs only the `bwrap` *client*; the sandboxed workload kept running on the host (verified empirically: a timed-out `sleep 999` survived). **Fixed** by `bwrap --die-with-parent` (plus `--unshare-pid`): the container dies with the client and the kernel reaps every process inside it. Verified: a real timed-out run returns 124 and leaves no stray process. | `integration/test_sandbox_bwrap.py::TestRealSandboxSmoke::test_timeout_maps_to_124` |
| 3 | **FIXED** | `search.get_all_files` | gitignore-style directory patterns (`build/`) were matched with bare `fnmatch`, which has no trailing-slash or ancestor semantics, so files under ignored trees were still indexed and embedded (wasted API cost). **Fixed** by normalizing trailing slashes in `get_ignore_patterns()` and pruning by basename, relpath, AND ancestor components during the walk. | `e2e/test_e2e_supporting.py::TestIgnoreMatcher` (matcher semantics; the `get_all_files` walk itself no longer has a dedicated test — see *Gaps*) |
| 4 | **FIXED** | `agent._get_lowest_reasoning_effort` | The "reasoning disabled" mode ("none") was unreachable for a model whose reasoning is NOT mandatory but whose catalog `supported_efforts` omits "none". That list enumerates levels available *while reasoning is ON*, so the lowest level came back as e.g. "low" and a requested `--reasoning-effort none` was silently bumped upward. **Fixed** by treating "none" as implicitly available whenever `reasoning_mandatory` is False. | `e2e/test_e2e_recovery.py::TestLengthRecovery` (exercises the lowest-effort recovery request); the explicit "none" bump case no longer has a dedicated test — see *Gaps* |

### Gaps (deliberately accepted by the refactor)

Moving from ~34 focused unit files to a handful of E2E journeys trades a
little pin-point precision for a lot of realistic coverage. The following
former unit assertions are no longer pinned by a dedicated test:

- `_extract_blocks` cross-type fence mismatch (bug #1).
- `search.get_all_files` directory-pruning walk (bug #3) — the matcher
  object itself is still covered.
- `_get_lowest_reasoning_effort` explicit `"none"` clamp (bug #4) — the
  lowest-effort recovery path is still covered.

If any of these regress, the fastest fix is to add a small E2E journey to
`tests/e2e/test_e2e_supporting.py` (or a new `test_e2e_*.py`) rather than
re-introducing a unit suite.

---

## Running the Tests

```bash
# Everything (E2E + integration; integration auto-skips without bwrap)
./venv/bin/python -m unittest discover -s tests -v

# E2E journeys only (offline, no bwrap, fastest)
./venv/bin/python -m unittest discover -s tests/e2e -v

# Integration only (requires bubblewrap; auto-skipped otherwise)
./venv/bin/python -m unittest discover -s tests/integration -v

# Single module
./venv/bin/python -m unittest tests.e2e.test_e2e_context -v
```

Environment requirements:
- Python ≥ 3.12 (declared in `pyproject.toml`; the source uses PEP 701 nested f-strings)
- `bwrap` (bubblewrap) available for integration tests (auto-skipped otherwise via
  `@unittest.skipUnless`)
- No API keys required; tests must pass with `OPENROUTER_API_KEY`
  **unset** and with no network access

---

## Design Notes

- **Seams already exist.** The codebase is unusually testable: `Sandbox` is
  injectable, `llm._CLIENT_CACHE` is a module-level dict that can be seeded
  with fakes, and `ContextManager` takes its UUID as a parameter. The suite
  exploits these seams rather than requiring refactors.
- **The fake LLM loop is the centerpiece.** `tests/e2e/harness.py` +
  `test_e2e_cli_loop.py` drive a complete `main() → Agent.run()` session —
  task → fenced bash block → output feedback → `exit` — with zero network.
  This validates the UUID protocol, output formatting, context persistence,
  and budget enforcement end-to-end.
- **Protocol fixtures are generated, not hardcoded.** Helpers build
  `---START_...-{uuid}---` fences from a live UUID so tests survive any future
  change to marker names in exactly one place.
- **Every loop is bounded.** `runaway_guard()` (SIGALRM) wraps each
  `Agent.run()` so a broken mock queue fails fast instead of hanging the
  harness.
