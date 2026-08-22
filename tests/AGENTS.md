# AGENTS.md — tests/ Directory

> Context for AI coding agents implementing or maintaining the Bash Agent test suite.

## What lives here

Design documents for a **not-yet-implemented** test suite:

- `README.md` — running instructions, constraints, known-bug register
- `TEST_PLAN.md` — the authoritative inventory of proposed tests (T-00 … T-41),
  each with rationale and the mocking/seam strategy

When tests are implemented, they follow the layout in `README.md`
(`helpers/fakes.py`, `unit/`, `integration/`).

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
   already-imported constants. Prefer patching `bash_agent.config.HISTORY_FILE`
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
