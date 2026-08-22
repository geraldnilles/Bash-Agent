# Bash Agent — Proposed Test Plan

> **STATUS: PROPOSED.** This document inventories the tests recommended for the
> `bash_agent` package. No test code exists yet. Each entry explains **what** is
> being tested, **why it matters**, and **how it shims into the existing code**
> (which seams are used, what gets mocked). All tests run locally with no
> network access.
>
> Conventions: IDs `T-nn`. Priority: P0 = protocol-critical / regression guard,
> P1 = important behavior, P2 = nice-to-have hardening. "Seam" describes the
> injection point that keeps the test offline and hermetic.

---

## Progress Summary

> Tick `- [x]` as each test is implemented and passing.
> Counts are derived from the plan; regenerate them if tests are added/removed.

| Group | Tests | Done |
|---|---|---|
| 0. Shared Test Infrastructure (`tests/helpers/fakes.py`) | 4 | 0 |
| 1. Protocol Parsing — `agent._extract_blocks` | 5 | 0 |
| 2. Special Commands — `agent._handle_special_command` | 6 | 0 |
| 3. Execution Pipeline — `parse_and_execute` / `_execute_script` | 6 | 0 |
| 4. Context Management — `context.ContextManager` | 5 | 0 |
| 5. LLM Adapter — `llm.py` | 4 | 0 |
| 6. Sandbox — `sandbox.Sandbox` | 3 | 0 |
| 7. Integration — real processes, still offline | 3 | 0 |
| 8. Supporting Modules | 9 | 0 |
| **Total** | **45** | **0** |

---

## 0. Shared Test Infrastructure (`tests/helpers/fakes.py`)

### T-00a — `chdir_tmp` context manager (P0)

- [ ] **Implemented**

Almost every module resolves paths against the process CWD at call time:
`Sandbox` writes temp scripts to `.bash_agent_tmp/`, `ContextManager` creates
the scratchpad there, `search.py` walks the CWD and reads `.gitignore`,
`utils.cleanup_tmp_folder()` deletes from `.bash_agent_tmp/`. This helper uses
`tempfile.TemporaryDirectory()` plus `os.chdir()` so each test runs in a
throwaway directory. It shims in as a plain context manager (or base-class
mixin) used by every filesystem-touching test; nothing in production code
changes. One subtlety documented for implementers: `config.HISTORY_FILE` is an
import-time constant, so persistence tests must patch
`bash_agent.config.HISTORY_FILE` rather than rely on chdir alone.

### T-00b — Fenced-block builders (P0)

- [ ] **Implemented**

The UUID-fenced protocol (`---START_BASH_COMMAND-{uuid}---` …) appears in
nearly every agent test. Helpers `bash_block(uuid, script)`,
`python_block(uuid, code)`, `output_block(uuid, exit_code, text)` generate
protocol-valid strings from a live UUID. They shim in as pure functions; if
marker names ever change in `agent.py`/`prompts.py`, only this file needs
editing. This protects ~20 downstream tests from protocol drift.

### T-00c — `FakeLLMClient` + cache seeding (P0)

- [ ] **Implemented**

`llm.get_llm_client()` caches clients in the module-level dict
`llm._CLIENT_CACHE`. A fake client object whose
`chat.completions.create(...)` returns scripted `ChatCompletion`-shaped
objects (with `choices[0].message.content`, `finish_reason`, `usage.cost`,
`model_dump()`) can be seeded via `llm._CLIENT_CACHE["openrouter"] = fake`.
This single seam makes `create_chat_completion`, the whole `Agent.run()` loop,
and budget accounting fully exercisable offline. The fake must also record
call kwargs so tests can assert on `extra_body` contents (reasoning effort,
provider whitelist).

### T-00d — `FakeSandbox` (P0)

- [ ] **Implemented**

`Agent.__init__` accepts no sandbox argument today, but the attribute is
assigned directly (`self.sandbox = Sandbox(...)`), so tests construct an
`Agent` then overwrite `agent.sandbox = FakeSandbox(...)`. The fake implements
`execute(script) -> (exit_code, output)` and `execute_python(code) ->
(exit_code, output)` returning canned results, recording scripts for
assertions. This decouples unit tests of the pipeline from systemd entirely;
real-sandbox behavior is covered separately by T-24/T-25.

---

## 1. Protocol Parsing — `agent._extract_blocks`

### T-01 — Valid bash and python block extraction (P0)

- [ ] **Implemented**

Feeds `_extract_blocks()` a response containing one well-formed bash block and
one python block built by T-00b, asserting it returns
`[("BASH", script), ("PYTHON", code)]` with whitespace stripped. This is the
front door of the entire agent: if parsing breaks, nothing else works. Shims
in directly — `_extract_blocks` is pure regex over a string and needs only an
`Agent` instance whose constructor side effects are neutralized via a shared `_make_agent()`
helper in `helpers/fakes.py` (patches `_check_model_capabilities`, swaps in
`FakeSandbox`).

### T-02 — Mixed prose and multiple blocks (P1)

- [ ] **Implemented**

Verifies blocks embedded inside explanatory markdown prose are still found,
that non-matching fences (e.g., `---START_ATTACHED_IMAGE-…---`) are ignored,
and that blocks appear in document order. Guards against regex regressions
like greedy matching swallowing two blocks into one. Pure-function test on the
same seam as T-01.

### T-03 — No-block response yields coaching warning (P1)

- [ ] **Implemented**

A response with no fences must return `([], warning)` where the warning
contains both example fence templates *with the live session UUID* — this
warning is how the LLM self-corrects, so its exactness matters. Asserts the
UUID interpolation actually happens (a stale or missing UUID here silently
breaks recovery).

### T-04 — Malformed UUID triggers relaxed-pattern rescue (P0)

- [ ] **Implemented**

Simulates the classic failure mode: the model emits correct fence structure
but a wrong/stale UUID (e.g., from a resumed session). Asserts the relaxed
fallback fires, the warning embeds a *corrected* block using the current
UUID, and the proposed script matches what the model wrote. This path is the
difference between a recoverable hiccup and a stuck session after `--resume`.

### T-05 — Cross-type fence mismatch rejected (P2)

- [ ] **Implemented**

A response where START says BASH but END says PYTHON must not match the strict
pattern (backreference `\1`) nor crash the relaxed fallback. Pins the
regex backreference behavior so nobody "simplifies" it into accepting
mismatched pairs.

---

## 2. Special Commands — `agent._handle_special_command`

### T-06 — `exit` terminates the process (P0)

- [ ] **Implemented**

Asserts `SystemExit(0)` is raised when a bash block contains exactly `exit`.
Uses `assertRaises(SystemExit)` around `parse_and_execute`; also asserts the
debug history was flushed first (order matters for resumability). Runs against
a FakeSandbox to prove interception happens *before* any real execution.

### T-07 — `reset` clears history but preserves system prompt (P0)

- [ ] **Implemented**

Seeds `context.history` with `[system, user, assistant]`, invokes the reset
command through `parse_and_execute`, and asserts exactly one message remains
and it is the system prompt. Also covers the empty-history edge case (fixed):
reset on an empty list must remain a no-op returning `EXIT_CODE_0`, never
raising `IndexError`.

### T-08 — `request-write` approval flows (P1)

- [ ] **Implemented**

Three sub-cases driven by patching `builtins.input`: approval appends the
absolute path to `sandbox.approved_write_paths` and returns formatted output
with `EXIT_CODE_0`; denial returns exit code 1; a free-text answer is echoed
back as a user message. Shims in via the same input-patching trick the
production code already relies on (`input()`), no other seams needed.

### T-09 — `ask-user` captures stdin answer (P1)

- [ ] **Implemented**

Patches `input` to return a canned answer and asserts the question was printed
and the answer comes back wrapped in a proper OUTPUT block. The EOF branch
(`input` raising `EOFError`) is tested separately to pin the
"[User provided no response / EOF]" fallback — important because the agent
runs non-interactively in CI-like environments.

### T-10 — `copy-to-clipboard` exits after copying (P1)

- [ ] **Implemented**

Patches `bash_agent.agent.copy_project_to_clipboard` (imported into the agent
namespace) with a recorder, asserts it received the comma-separated file list
verbatim, and asserts `SystemExit(0)`. Proves the clipboard tool is invoked
before termination and that arbitrary file lists pass through unmodified.

### T-11 — Non-special scripts fall through to execution (P0)

- [ ] **Implemented**

A bash block containing `echo hi` must return `handled=False` from
`_handle_special_command` and be routed to the sandbox. Guards against a
future refactor accidentally intercepting ordinary commands (e.g., a prefix
match like `request-write` catching `request-writeup.sh`). Uses FakeSandbox
recording to confirm dispatch.

---

## 3. Execution Pipeline — `parse_and_execute` / `_execute_script`

### T-12 — Full happy-path turn (P0)

- [ ] **Implemented**

One response containing a valid bash block → FakeSandbox returns
`(0, "hello")` → asserts: output block formatted with `EXIT_CODE_0`,
`(executed=True, feedback="")` returned, and a single user message appended to
context containing the fenced output. This is the smallest end-to-end proof
that parse→dispatch→format→commit works as a unit.

### T-13 — MAX_CODE_BLOCKS enforcement (P0)

- [ ] **Implemented**

With `MAX_CODE_BLOCKS=1` (current config), a response with two valid blocks
must execute only the first, append the cutoff warning mentioning counts
("first 1 of 2"), and still commit the executed output. Patch
`bash_agent.agent.MAX_CODE_BLOCKS` to 2 in a mirror test proving the limit is
read dynamically, not baked in. Prevents runaway multi-execution regressions.

### T-14 — Attached-image fence extraction (P0)

- [ ] **Implemented**

FakeSandbox returns output embedding
`---START_ATTACHED_IMAGE-{uuid}---data:image/png;base64,…---END_…---`. Asserts:
payload stripped from displayed output, `[Image attached…]` note added,
`agent._pending_multimodal_images` populated, and after
`_commit_execution_feedback` the context message content is a **list** with a
text part followed by an `image_url` part (the multimodal wire format), and
the pending list is cleared. This exercises the exact path `vision` uses.

### T-15 — `/tmp/` failure warning heuristic (P1)

- [ ] **Implemented**

Parametrized matrix over `_build_tmp_file_warning`: (exit≠0, "/tmp/" present,
error phrase present) → warning returned; each condition individually violated
→ `None`. Covers several phrasings from `_TMP_ERROR_PATTERN` ("No such file",
"does not exist", "failed to open"). Pure function — direct import, no mocks.
Protects a heuristic that keeps the LLM from repeatedly writing to host /tmp.

### T-16 — Output truncation formatting (P0)

- [ ] **Implemented**

`_format_output` with >10,000-char output must produce head+tail halves joined
by `...[Output Truncated]...`, a `VISIBLE_%` header below 100, and the trailing
advice line; small outputs pass through untouched with `VISIBLE_100%`. Verifies
the truncation point arithmetic (5,000/5,000 split) and that exit codes survive
in the header. Direct-call test on a pure method.

### T-17 — Scratchpad co-commit ordering (P1)

- [ ] **Implemented**

When the scratchpad changed during a turn (test writes to it mid-test),
`parse_and_execute` must prepend a fresh SCRATCHPAD block to the committed
user message and strip older scratchpad blocks from prior messages. Seeds a
prior message containing an old-format scratchpad fence, changes the file,
runs a turn, asserts old fence gone + new fence present exactly once. Exercises
`get_scratchpad_block` hash caching through the public pipeline.

---

## 4. Context Management — `context.ContextManager`

### T-18 — Multimodal content length accounting (P1)

- [ ] **Implemented**

`_content_length` on: plain string; list with text parts; list with N
`image_url` parts (each ≈6400 chars); mixed; non-str/non-list → 0. These
numbers feed pruning decisions, so drift here silently changes when trimming
kicks in. Static-method test, zero setup.

### T-19 — Hysteresis pruning ladder (P0)

- [ ] **Implemented**

Builds a history over `CONTEXT_LIMIT` (patch `bash_agent.config.CONTEXT_LIMIT`
to something tiny like 2,000 for speed) containing: system prompt, old
BASH_OUTPUT blocks, old command blocks, and plain messages. Asserts the
documented ladder: oldest outputs replaced with
`[BASH_OUTPUT DELETED TO SAVE CONTEXT]` first; commands truncated to 80 chars
with `...[TRUNCATED]`; index 0 never touched; loop terminates with total ≤
target (80%). This is the most intricate logic in the package and currently
has zero coverage.

### T-20 — Image-bearing messages dropped wholesale (P0)

- [ ] **Implemented**

Under pruning pressure, a message whose content is a list (multimodal) must be
removed entirely rather than regex-trimmed (which would raise TypeError).
Constructs a tiny-limit history mixing list-content and string messages and
asserts the list message is popped while string messages get the normal
ladder treatment. Regression guard for the fix in commit 78773ca.

### T-21 — Scratchpad hashing and VISIBLE math (P1)

- [ ] **Implemented**

Three cases: unchanged file between calls → second call returns `""` (hash
cache); changed file → new block emitted; oversized file (>SCRATCHPAD_LIMIT,
patch constant small) → truncated body of exactly SCRATCHPAD_LIMIT chars plus
`[ERROR]` suffix. Regression guard for the fixed VISIBLE math: the percentage
must be computed from the *pre-truncation* length (e.g., LIMIT=80k over 100k
original → `VISIBLE_80%`), not from the truncated body (which would always
yield 100%).

### T-22 — History persistence round-trip (P0)

- [ ] **Implemented**

Patch `bash_agent.config.HISTORY_FILE` into the tmpdir; save a history
containing string and list content; construct a fresh ContextManager with a
different UUID; `load_history()` must restore both fields and adopt the saved
UUID (resume re-binding depends on this). Corrupt-file case is a regression
guard for the fixed `import sys`: a malformed `history.json` must print
`[System Error] Failed to load history: …` to stderr and return `False`
(previously raised `NameError`). Also verifies `last_scratchpad_hash` resets
to None on load so the scratchpad re-injects after resume.

---

## 5. LLM Adapter — `llm.py`

### T-23 — Backend routing matrix (P0)

- [ ] **Implemented**

Table-driven: `google/gemini-x` + GEMINI_API_KEY set → `"gemini"`; same model,
key unset → `"openrouter"`; `openai/gpt-x` regardless of keys →
`"openrouter"`; None-safe. Implemented by setting/clearing env vars and
calling `get_backend` directly. Routing decides which API every request hits,
so mistakes here mean wrong endpoints.

### T-24 — OpenRouter payload normalization (P0)

- [ ] **Implemented**

Seed `llm._CLIENT_CACHE["openrouter"]` with FakeLLMClient; call
`create_chat_completion(model="deepseek/deepseek-v4-pro", reasoning_effort="low")`
and assert recorded kwargs contain `extra_body.reasoning.effort == "low"` and
`extra_body.provider.only == ["deepseek"]` (from MODEL_PROVIDERS); a model not
in the whitelist produces no `provider` key; `reasoning_effort=None` adds no
`reasoning` key. Offline via the cache seam; no HTTP anywhere.

### T-25 — Gemini payload stripping + cost monkey-patch (P0)

- [ ] **Implemented**

With GEMINI_API_KEY set and a gemini-routed fake client: assert the model name
arrived stripped of `google/`, that OpenRouter-only extras were dropped, and
that the returned response's patched `model_dump()` injects
`usage.cost` computed by `calculate_gemini_cost` from the usage token counts.
Also unit-tests `calculate_gemini_cost` tier selection by substring
("gemini-3-flash" tier vs default) including the unknown-model default path.
Guards the fragile monkey-patch called out in AGENTS.md pitfalls.

### T-26 — Client cache identity (P2)

- [ ] **Implemented**

Two `get_llm_client("openrouter")` calls return the same object; different
backends return different objects; seeding the cache bypasses construction.
Trivial but prevents accidental per-call client churn (connection storms).

---

## 6. Sandbox — `sandbox.Sandbox`

### T-27 — Construction defaults & write-path bookkeeping (P1)

- [ ] **Implemented**

New Sandbox: timeout falls back to `BASH_TIMEOUT`, approved_write_paths starts
as `[abspath(".")]`, uuid/multimodal stored. `request_write` driven by patched
`input` for y/n/message answers appending or not appending to the list.
Pure-python surface; no subprocess involved.

### T-28 — Command assembly inspection (P1)

- [ ] **Implemented**

Rather than executing, temporarily wrap `subprocess.run` with a recorder and
invoke `sandbox.execute("true")`; assert the argv contains the security
properties (`ProtectSystem=strict`, `ProtectHome=read-only`,
`PrivateTmp=yes`), working-directory flag, env forwarding
(BASH_AGENT_UUID, BASH_AGENT_MULTIMODAL, PATH, optional API key only when set),
one ReadWritePaths per approved path, and `/bin/bash <script>` tail. Same for
`execute_python` (venv python preferred when present, PYTHONPATH set, `.py`
suffix). This pins the security posture without needing systemd; if someone
drops a property, CI goes red before users do.

### T-29 — Temp-script hygiene (P1)

- [ ] **Implemented**

After a (recorded, mocked-run) execute call, assert the mkstemp'd script under
`.bash_agent_tmp/` was removed, had mode 0700 while it existed (check via
recorder hook), and that a `TimeoutExpired` maps to exit code 124 with the
"[SYSTEM ERROR] … timed out" banner plus partial stdout. Timeout mapping is
load-bearing: the LLM reads exit codes to decide retries.

---

## 7. Integration — real processes, still offline

### T-30 — Real systemd-run smoke tests (P1, skipUnless systemd-run)

- [ ] **Implemented**

Actual `Sandbox.execute` round-trips: `echo hello` → exit 0 with stdout;
`exit 3` → exit code 3; stderr merged into stdout (`ls /nonexistent` shows the
error inline); reading `/etc/hostname` succeeds (read-only allowed) while
`touch /etc/xyz` fails (ProtectSystem enforced); CWD visibility (project files
listable); BASH_AGENT_UUID visible inside the sandbox; timeout case via
`sleep 999` with `-t 2` → 124. These prove the isolation guarantees AGENTS.md
promises. Skipped automatically when no user session bus exists.

### T-31 — Full offline agent loop (P0, the centerpiece)

- [ ] **Implemented**

Constructs an Agent with capability probe patched out, seeds
`llm._CLIENT_CACHE` with a scripted FakeLLMClient whose responses are:
(1) a bash block running `echo step-one`, (2) a python block printing
context stats, (3) `exit`. Drives `agent.run("integration test task")` inside
`chdir_tmp`. Asserts across the whole stack: sandbox really ran the scripts
(real files created in tmpdir), OUTPUT blocks landed in history with correct
EXIT_CODE headers, `history.json` written and reloadable, scratchpad injected
on first message, budget/stats path exercised, clean SystemExit(0). This is
the closest thing to "run bagent" that requires no network, and it would have
caught bugs #1–#4 automatically.

### T-32 — Resume flow end-to-end (P1)

- [ ] **Implemented**

Continuation of T-31: after the first session persists history, build a second
Agent with `resume=True` (probe still patched), assert UUID re-bound from disk,
prior messages present, and a follow-up fake response referencing earlier
output executes correctly. Protects the `--resume` contract that humans rely
on daily.

---

## 8. Supporting Modules

### T-33 — `utils.is_binary_file` (P1)

- [ ] **Implemented**

Extension table (.png/.pdf/.zip…) → True; text file with null bytes in first
1024 bytes → True; plain text → False; empty file → False. Direct calls on
tmpdir fixtures. Clipboard quality depends on this filter.

### T-34 — `utils.cleanup_tmp_folder` whitelist (P0)

- [ ] **Implemented**

Populates `.bash_agent_tmp/` with whitelisted files (SCRATCHPAD.md,
history.json, ROLE.md, embeddings.json, search_disabled, vim_prompt.tmp,
clipboard_blacklist.txt) plus junk files/dirs; after cleanup only whitelisted
names remain. A bug here destroys user sessions — highest-value utils test.

### T-35 — `copy_project_to_clipboard` filtering (P1)

- [ ] **Implemented**

Patches the clipboard writers (`wl-copy`/`xclip` via subprocess.run recorder)
and builds a tmpdir tree with: normal files, a .gitignore'd file, a blacklisted
file (via clipboard_blacklist.txt), a binary file, a nested dir. Asserts
`<file path="…">` wrapping, correct exclusions, prefix/suffix presence, and
the `--files` subset mode. Fully offline because the clipboard boundary is
subprocess-based and mocked.

### T-36 — `get_system_prompt` composition (P1)

- [ ] **Implemented**

Calls `get_system_prompt(uuid, cwd, scratchpad_path, role_text=None,
multimodal_capabilities=["image"])` and asserts: UUID interpolated into
examples, CWD present, scratchpad path present, multimodal section included
(and absent for None), custom ROLE.md text embedded when role_text provided.
Snapshot-style assertions on key substrings rather than full equality so the
prompt can evolve.

### T-37 — `main.parse_args` flag mapping (P1)

- [ ] **Implemented**

Argparse-level tests: `--commit` implies resume+message (tested at `main()`
level with Agent/run mocked), `-x` writes clipboard content to SCRATCHPAD.md
(clipboard getter patched), `-s` clears scratchpad, `--copy-project` exits
before Agent construction (Agent class patched to raise if instantiated).
Shims in by importing `bash_agent.main` and patching its collaborators.

### T-38 — `vision.py` dual-mode (P1)

- [ ] **Implemented**

Generates a small PNG with Pillow in tmpdir. Sandbox mode: env
BASH_AGENT_UUID+BASH_AGENT_MULTIMODAL=image set → `main()` prints
ATTACHED_IMAGE fences containing a data URL and exits 0 without touching llm
(client cache poisoned with a failing fake to prove no call happens).
Oversize image (>MAX_PIXELS, tiny patched limit) → exit 1 with stderr message.
Fallback mode (no env): asserts the LLM message structure contains text +
image_url parts. Uses `runpy`/subprocess-free invocation via patched sys.argv.

### T-39 — `transcribe.py` helpers (P2)

- [ ] **Implemented**

`get_audio_format` extension mapping incl. extensionless → "unknown";
`encode_audio` base64 fidelity; `check_file_size` boundary with patched
threshold (oversize → SystemExit 1); context-file XML wrapping logic extracted
via a tiny driver that mimics main()'s prompt assembly. ffmpeg conversion
itself covered only if binary present (skipUnless), converting 0.5s of
generated silence.

### T-40 — `memo.py` pure helpers (P2)

- [ ] **Implemented**

`get_sources` parsing against canned `pactl list sources short` output
(subprocess patched); `find_source` substring and node-ID matching incl.
no-match error; `format_timestamp`/`format_duration` arithmetic (e.g., 65s →
"01:05"); `fmt_size` units. Recording itself is hardware-dependent and stays
untested by design.

### T-41 — `search.py` offline core (P1)

- [ ] **Implemented**

`get_ignore_patterns` merges .gitignore lines with hardcoded set;
`get_all_files` honors dir/file patterns; `get_file_hash` stability;
`load/save_embeddings_db` round-trip preserving `_model`;
`get_file_content` 100k truncation marker; `cosine_similarity` correctness on
known vectors and zero-vector → 0.0 guard. `rerank_documents` with
`requests.post` patched: success parses results sorted desc; HTTP failure and
missing-key paths return neutral 0.0 scores (graceful degradation contract).
`main()` early-exit when SEARCH_DISABLED_FLAG exists (exit 1, no embedding
calls — proven by poisoned fake client).

---

## Suggested Implementation Order

1. **T-00a–d** (helpers) — everything depends on these.
2. **P0 protocol/pipeline**: T-01, T-04, T-06, T-12, T-13, T-16, T-19, T-22, T-25, T-31.
3. **P0 guards**: T-20, T-34, T-28.
4. Remaining P1s, then P2s.
5. After each group lands, flip statuses here and move resolved rows in the
   README bug table when fixes are made.
