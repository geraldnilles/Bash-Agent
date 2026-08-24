# AGENTS.md — tests/ Directory

> Context for AI coding agents implementing or maintaining the Bash Agent test suite.

## What lives here

The test suite for the `bash_agent` package, implemented incrementally against
the authoritative inventory in `TEST_PLAN.md` (T-00 … T-41):

- `README.md` — running instructions, constraints, known-bug register
- `TEST_PLAN.md` — the authoritative test inventory, each entry with rationale
  and the mocking/seam strategy; its Progress Summary tracks implementation state
- `helpers/fakes.py` — shared offline fakes (T-00a–d): `chdir_tmp`,
  fenced-block builders, `FakeLLMClient`, `FakeSandbox`, `_make_agent()`
- `unit/` — fast pure-logic tests (e.g. `test_agent_blocks.py` for Group 1,
  `test_agent_special_commands.py` for Group 2, `test_agent_pipeline.py` for Group 3,
  `test_llm_adapter.py` for Group 5; `test_config_file.py` for Group 42 (persistent config.json); supporting-module groups live in
  `test_transcribe.py` (T-39), `test_memo.py` (T-40), and
  `test_search_helpers.py` (T-41))
- `integration/` — real processes, still offline (systemd-run, fake-LLM loop)

## Non-negotiable rules

1. **Offline only.** Never let a test hit openrouter.ai, Google APIs, or any
   URL. Mock at these exact seams:
   - `bash_agent.llm._CLIENT_CACHE` (seed with fake clients)
   - `urllib.request.urlopen` (capability probe in `agent.py`)
   - `requests.post` (rerank in `search.py`)
2. **Stdlib `unittest` only.** pytest is not installed. Do not add dependencies.
3. **Isolate the filesystem.** Every test that touches disk must run inside a
   temporary working directory (use the planned `helpers.fakes.chdir_tmp`).
   Modules compute `.bash_agent_tmp/` and `config.HISTORY_FILE` from the CWD
   *at import time* — changing directories mid-test affects new calls, not
   already-imported constants. Prefer patching `bash_agent.context.HISTORY_FILE` (context.py binds it via `from ... import`; patching bash_agent.config is NOT seen)
   directly for persistence tests.
4. **Never edit `bash_agent/` source to make a test pass.** Pin buggy behavior
   with `@unittest.expectedFailure` and record it in the README bug table.
5. **Update `TEST_PLAN.md` status column** whenever you implement a test group
   (`PROPOSED` → `IMPLEMENTED`), and update `README.md` if you discover a new
   latent bug.

## Useful facts for writing tests

- Session UUID appears in every protocol marker; build fences with a helper,
  never string-concatenate by hand.
- `Agent.__init__` performs a network capability probe — always patch
  `Agent._check_model_capabilities` (or `urllib.request.urlopen`) before
  constructing an `Agent`.
- `exit` and `copy-to-clipboard` special commands call `sys.exit()` in-process;
  assert with `assertRaises(SystemExit)`.
- Sandbox integration tests need a systemd **user** session; guard with
  `skipUnless(shutil.which("systemd-run"), ...)` and keep each invocation well
  under the 60s harness timeout.
- The venv interpreter is `./venv/bin/python` (3.14). Run the suite with it,
  not the system python.
- **systemd-run + PrivateTmp vs /tmp CWDs**: `--working-directory=` or
  `ReadWritePaths=` pointing under `/tmp` fails namespace setup (exit 226/200)
  because `PrivateTmp=yes` mounts a fresh tmpfs over `/tmp` inside the
  sandbox. Production always runs from a real project dir, so integration
  tests must too — use `helpers.fakes.chdir_repo_tmp()` (scratch dir under the
  repo's gitignored `.bash_agent_tmp/`), NOT `chdir_tmp()`.
- **Timeouts leak services**: killing the `systemd-run` client does NOT kill
  the transient unit (see bug #2 in README.md). Any test that triggers a
  sandbox timeout must clean up after itself; mark the workload with
  `exec -a <unique-marker> sleep 999` (a shell comment is stripped before
  exec and leaves the process unmatchable by `pkill -f`).
- **Protocol hygiene in test sources**: never write a contiguous fence marker
  literal (e.g. a full START/END token) inside code that will itself be
  transmitted through the UUID-fenced protocol — assemble expected fences
  piecewise (see `_cmd_start`/`_out_start` helpers in
  `integration/test_full_agent_loop.py`). A bare literal marker inside a
  transmitted block body can be mistaken for a real fence and split the block.
- **Bounding agent loops**: wrap every `Agent.run()` call in the
  `runaway_guard()` SIGALRM context manager from
  `integration/test_full_agent_loop.py`; a protocol regression otherwise
  hangs the harness until the 60s execution timeout.
