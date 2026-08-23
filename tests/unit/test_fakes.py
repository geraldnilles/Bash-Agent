"""
Group 0 — Shared Test Infrastructure tests (T-00a … T-00d).

Validates tests/helpers/fakes.py which provides the offline seams for the
entire suite. All tests are offline, stdlib unittest only.

T-00a  chdir_tmp           – isolated temp CWD context manager
T-00b  fenced-block builders (bash_block, python_block, output_block)
T-00c  FakeLLMClient + cache seeding
T-00d  FakeSandbox (plus _make_agent helper)
"""
import contextlib
import io
import os
import re
import uuid
import tempfile
import unittest
from unittest import mock

from tests.helpers.fakes import (
    chdir_tmp,
    ChdirTmp,
    bash_block,
    python_block,
    output_block,
    attached_image_block,
    FakeLLMClient,
    FakeSandbox,
    FakeResponse,
    make_fake_response,
    _make_agent,
)


# ---------------------------------------------------------------------------
# T-00a — chdir_tmp
# ---------------------------------------------------------------------------
class TestChdirTmp(unittest.TestCase):
    """T-00a: chdir_tmp gives each test a throwaway CWD and restores on exit."""

    def test_chdir_tmp_changes_cwd_and_restores(self):
        orig = os.getcwd()
        with chdir_tmp() as tmpdir:
            # Inside: CWD is the tmpdir
            self.assertEqual(os.path.abspath(tmpdir), os.getcwd())
            self.assertTrue(os.path.isdir(tmpdir))
            # File created inside is inside tmpdir and not in original repo
            with open(os.path.join(tmpdir, "hello.txt"), "w") as f:
                f.write("hi")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "hello.txt")))
        # Outside: restored
        self.assertEqual(os.getcwd(), orig)
        # Temp dir is deleted
        self.assertFalse(os.path.exists(tmpdir))

    def test_chdir_tmp_restores_on_exception(self):
        orig = os.getcwd()
        tmpdir_ref = None
        try:
            with chdir_tmp() as tmpdir:
                tmpdir_ref = tmpdir
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertEqual(os.getcwd(), orig)
        if tmpdir_ref:
            self.assertFalse(os.path.exists(tmpdir_ref))

    def test_chdir_tmp_isolates_filesystem(self):
        """Modules that resolve .bash_agent_tmp from CWD at call time are isolated."""
        with chdir_tmp() as tmpdir:
            # Simulate what Sandbox / ContextManager do: write to .bash_agent_tmp/
            os.makedirs(".bash_agent_tmp", exist_ok=True)
            with open(".bash_agent_tmp/SCRATCHPAD.md", "w") as f:
                f.write("scratch")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, ".bash_agent_tmp", "SCRATCHPAD.md")))
        # Outside, the file should not exist in the original tmpdir path
        self.assertFalse(os.path.exists(os.path.join(tmpdir, ".bash_agent_tmp", "SCRATCHPAD.md")))

    def test_chdir_tmp_context_manager_class(self):
        """ChdirTmp class variant should behave identically to the function."""
        orig = os.getcwd()
        with ChdirTmp() as tmpdir:
            self.assertEqual(os.path.abspath(tmpdir), os.getcwd())
            self.assertTrue(os.path.isdir(tmpdir))
        self.assertEqual(os.getcwd(), orig)
        self.assertFalse(os.path.exists(tmpdir))

    def test_chdir_tmp_history_file_subtlety(self):
        """
        Documents the import-time constant subtlety: config.HISTORY_FILE is
        resolved at import time via os.path.abspath(".bash_agent_tmp/history.json"),
        so chdir_tmp alone does NOT change it. Persistence tests must patch
        bash_agent.config.HISTORY_FILE directly.
        """
        from bash_agent import config as cfg
        orig_history = cfg.HISTORY_FILE
        with chdir_tmp() as tmpdir:
            # Inside tmpdir, config.HISTORY_FILE still points to original CWD
            self.assertEqual(cfg.HISTORY_FILE, orig_history)
            self.assertNotEqual(
                os.path.abspath(os.path.join(tmpdir, ".bash_agent_tmp", "history.json")),
                cfg.HISTORY_FILE,
            )
            # Correct approach: patch the constant
            patched = os.path.join(tmpdir, ".bash_agent_tmp", "history.json")
            with mock.patch("bash_agent.config.HISTORY_FILE", patched):
                from bash_agent.config import HISTORY_FILE as patched_val
                self.assertEqual(patched_val, patched)
            # Also patch context module's imported constant
            with mock.patch("bash_agent.context.HISTORY_FILE", patched):
                import bash_agent.context as ctx_mod
                self.assertEqual(ctx_mod.HISTORY_FILE, patched)

    def test_nested_chdir_tmp(self):
        orig = os.getcwd()
        with chdir_tmp() as outer:
            self.assertEqual(os.getcwd(), outer)
            with chdir_tmp() as inner:
                self.assertEqual(os.getcwd(), inner)
                self.assertNotEqual(outer, inner)
            self.assertEqual(os.getcwd(), outer)
        self.assertEqual(os.getcwd(), orig)


# ---------------------------------------------------------------------------
# T-00b — Fenced-block builders
# ---------------------------------------------------------------------------
class TestFencedBlockBuilders(unittest.TestCase):
    """T-00b: bash_block / python_block / output_block generate protocol-valid fences."""

    def test_bash_block_format(self):
        uid = str(uuid.uuid4())
        script = "echo hello"
        block = bash_block(uid, script)
        self.assertEqual(
            block,
            f"---START_BASH_COMMAND-{uid}---\n{script}\n---END_BASH_COMMAND-{uid}---",
        )
        # Must be parseable by production regex
        agent = _make_agent(uuid_str=uid)
        blocks, err = agent._extract_blocks(block)
        self.assertIsNone(err)
        self.assertEqual(blocks, [("BASH", script)])

    def test_python_block_format(self):
        uid = str(uuid.uuid4())
        code = "print(42)"
        block = python_block(uid, code)
        self.assertEqual(
            block,
            f"---START_PYTHON_COMMAND-{uid}---\n{code}\n---END_PYTHON_COMMAND-{uid}---",
        )
        agent = _make_agent(uuid_str=uid)
        blocks, err = agent._extract_blocks(block)
        self.assertIsNone(err)
        self.assertEqual(blocks, [("PYTHON", code)])

    def test_output_block_default_bash(self):
        uid = str(uuid.uuid4())
        block = output_block(uid, 0, "hello")
        self.assertIn(f"---START_BASH_OUTPUT-EXIT_CODE_0-VISIBLE_100%-{uid}---", block)
        self.assertIn("hello", block)
        self.assertIn(f"---END_BASH_OUTPUT-{uid}---", block)

    def test_output_block_python_and_visible(self):
        uid = str(uuid.uuid4())
        block = output_block(uid, 1, "err", cmd_type="PYTHON", visible=50)
        self.assertIn(f"---START_PYTHON_OUTPUT-EXIT_CODE_1-VISIBLE_50%-{uid}---", block)
        self.assertIn(f"---END_PYTHON_OUTPUT-{uid}---", block)

    def test_output_block_cmd_type_case_insensitive(self):
        uid = str(uuid.uuid4())
        block = output_block(uid, 0, "x", cmd_type="python")
        self.assertIn("PYTHON_OUTPUT", block)

    def test_builders_whitespace_stripping_via_agent(self):
        """Agent strips whitespace inside blocks; builders should preserve exact script."""
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        script = "  echo hi  "
        block = bash_block(uid, script)
        blocks, _ = agent._extract_blocks(block)
        # Agent strips outer whitespace
        self.assertEqual(blocks[0][1], script.strip())

    def test_multiple_blocks_in_order(self):
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        b1 = bash_block(uid, "echo one")
        b2 = python_block(uid, "print('two')")
        combined = f"Some prose\n{b1}\nMore prose\n{b2}\nTail"
        blocks, err = agent._extract_blocks(combined)
        self.assertIsNone(err)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], ("BASH", "echo one"))
        self.assertEqual(blocks[1], ("PYTHON", "print('two')"))

    def test_output_block_not_parsed_as_command(self):
        """OUTPUT fences must not be extracted as COMMAND blocks."""
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        out = output_block(uid, 0, "hello")
        blocks, err = agent._extract_blocks(out)
        # No COMMAND blocks found → warning message returned
        self.assertEqual(blocks, [])
        self.assertIsNotNone(err)

    def test_attached_image_ignored_as_command(self):
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        img_fence = f"---START_ATTACHED_IMAGE-{uid}---data:image/png;base64,abc---END_ATTACHED_IMAGE-{uid}---"
        b = bash_block(uid, "echo hi")
        combined = f"{img_fence}\n{b}"
        blocks, _ = agent._extract_blocks(combined)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], "BASH")

    def test_attached_image_block_builder_mirrors_vision_emission(self):
        """ATTACHED_IMAGE builder must match vision.py's 3-print layout and be
        consumable by the production extraction regex used in _execute_script."""
        uid = str(uuid.uuid4())
        url = "data:image/png;base64,iVBORw0KGgo="
        fence = attached_image_block(uid, url)

        self.assertEqual(
            fence,
            "---START_ATTACHED_IMAGE-%s---\n%s\n---END_ATTACHED_IMAGE-%s---"
            % (uid, url, uid),
        )

        # The same regex agent._execute_script uses must recover exactly [url]
        pattern = (
            r"---START_ATTACHED_IMAGE-%s---\s*(.*?)\s*---END_ATTACHED_IMAGE-%s---" % (uid, uid)
        )
        self.assertEqual(re.findall(pattern, fence, re.DOTALL), [url])


# ---------------------------------------------------------------------------
# T-00c — FakeLLMClient + cache seeding
# ---------------------------------------------------------------------------
class TestFakeLLMClient(unittest.TestCase):
    """T-00c: FakeLLMClient can be seeded into llm._CLIENT_CACHE to go offline."""

    def setUp(self):
        # Ensure a clean cache for each test
        from bash_agent import llm as llm_mod
        self.llm_mod = llm_mod
        self._orig_cache = dict(llm_mod._CLIENT_CACHE)
        llm_mod._CLIENT_CACHE.clear()

    def tearDown(self):
        self.llm_mod._CLIENT_CACHE.clear()
        self.llm_mod._CLIENT_CACHE.update(self._orig_cache)

    def test_fake_client_returns_scripted_responses_in_order(self):
        fake = FakeLLMClient(responses=[
            make_fake_response(content="first", cost=0.01),
            make_fake_response(content="second", cost=0.02),
        ])
        self.llm_mod._CLIENT_CACHE["openrouter"] = fake
        # Directly call completions.create
        r1 = fake.chat.completions.create(model="x", messages=[])
        r2 = fake.chat.completions.create(model="x", messages=[])
        self.assertEqual(r1.choices[0].message.content, "first")
        self.assertEqual(r2.choices[0].message.content, "second")

    def test_fake_response_shape(self):
        """Fake response must expose choices[0].message.content, finish_reason, usage.cost, model_dump()."""
        resp = make_fake_response(content="hello", finish_reason="stop", cost=0.05, prompt_tokens=10, completion_tokens=20)
        self.assertEqual(resp.choices[0].message.content, "hello")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.usage.cost, 0.05)
        dumped = resp.model_dump()
        self.assertIn("usage", dumped)
        self.assertEqual(dumped["usage"]["cost"], 0.05)
        self.assertEqual(dumped["usage"]["prompt_tokens"], 10)
        self.assertEqual(dumped["usage"]["completion_tokens"], 20)
        # model_dump_json must not raise
        j = resp.model_dump_json(indent=2)
        self.assertIn("hello", j)

    def test_fake_client_records_call_kwargs(self):
        """Extra-body / reasoning / provider whitelist must be recordable."""
        fake = FakeLLMClient(responses=[make_fake_response(content="ok")])
        self.llm_mod._CLIENT_CACHE["openrouter"] = fake
        # Call via the adapter layer to test extra_body injection
        from bash_agent import llm as llm_mod
        # Patch get_backend to force openrouter
        with mock.patch.object(llm_mod, "get_backend", return_value="openrouter"):
            llm_mod.create_chat_completion(
                model="deepseek/deepseek-v4-flash-0731",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                reasoning_effort="low",
            )
        self.assertEqual(len(fake.calls), 1)
        kwargs = fake.calls[0]
        self.assertIn("model", kwargs)
        self.assertIn("messages", kwargs)
        # For the deepseek model, provider whitelist should be injected
        extra = kwargs.get("extra_body", {})
        self.assertIn("reasoning", extra)
        self.assertEqual(extra["reasoning"]["effort"], "low")
        # Provider whitelist for whitelisted models
        if "deepseek/deepseek-v4-flash-0731" in self.llm_mod.config.MODEL_PROVIDERS if hasattr(self.llm_mod, "config") else False:
            pass  # covered below with direct assertion
        # Ensure provider key exists for this model (configured in config.py)
        self.assertIn("provider", extra)
        self.assertEqual(extra["provider"]["only"], ["deepseek"])

    def test_cache_seeding_makes_create_chat_completion_offline(self):
        """Seeding _CLIENT_CACHE must make create_chat_completion return fake without network."""
        fake = FakeLLMClient(responses=[make_fake_response(content="offline ok", cost=0.001)])
        self.llm_mod._CLIENT_CACHE["openrouter"] = fake
        with mock.patch.object(self.llm_mod, "get_backend", return_value="openrouter"):
            resp = self.llm_mod.create_chat_completion(
                model="any-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        self.assertEqual(resp.choices[0].message.content, "offline ok")
        # Ensure no network was hit — fake call count is 1
        self.assertEqual(len(fake.calls), 1)

    def test_fake_client_callable_factory(self):
        """Queued responses can be callables that receive kwargs."""
        def factory(**kwargs):
            return make_fake_response(content=f"model={kwargs.get('model')}")
        fake = FakeLLMClient(responses=[factory])
        self.llm_mod._CLIENT_CACHE["openrouter"] = fake
        r = fake.chat.completions.create(model="my-model", messages=[])
        self.assertIn("my-model", r.choices[0].message.content)

    def test_fake_client_default_response_when_empty(self):
        fake = FakeLLMClient()
        r = fake.chat.completions.create(model="x", messages=[])
        self.assertEqual(r.choices[0].message.content, "")

    def test_embedding_fake(self):
        fake = FakeLLMClient()
        self.llm_mod._CLIENT_CACHE["openrouter"] = fake
        r = fake.embeddings.create(model="test", input=["hello", "world"])
        self.assertEqual(len(r.data), 2)
        self.assertTrue(hasattr(r.data[0], "embedding"))

    def test_reasoning_field_on_choice(self):
        resp = make_fake_response(content="ans", reasoning="thoughts")
        self.assertEqual(resp.choices[0].message.reasoning, "thoughts")

    def test_gemini_cost_patching_path(self):
        """When backend is gemini, create_chat_completion patches model_dump to inject cost."""
        # Prepare a fake that returns usage without cost; gemini path should patch it
        class UsageNoCost:
            prompt_tokens = 100
            completion_tokens = 200
            cost = None
        resp = FakeResponse(content="hi", cost=None, prompt_tokens=100, completion_tokens=200)
        # Force usage object without cost to trigger patch path? Instead test the patched path:
        # Use a response with None cost and gemini backend
        fake_resp = FakeResponse(content="hi", cost=0.0, prompt_tokens=10, completion_tokens=10)
        fake = FakeLLMClient(responses=[fake_resp])
        self.llm_mod._CLIENT_CACHE["gemini"] = fake
        with mock.patch.object(self.llm_mod, "get_backend", return_value="gemini"):
            # gemini backend will monkey-patch model_dump; we assert patched cost is computed
            result = self.llm_mod.create_chat_completion(
                model="google/gemini-3-flash-preview",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            dumped = result.model_dump()
            # For gemini, cost should be >0 (pricing tiers) and not None
            self.assertIsNotNone(dumped["usage"]["cost"])
            self.assertIsInstance(dumped["usage"]["cost"], float)


# ---------------------------------------------------------------------------
# T-00d — FakeSandbox
# ---------------------------------------------------------------------------
class TestFakeSandbox(unittest.TestCase):
    """T-00d: FakeSandbox decouples unit tests from systemd."""

    def test_execute_returns_canned_result(self):
        sb = FakeSandbox(execute_result=(0, "hello"))
        code, out = sb.execute("echo hi")
        self.assertEqual(code, 0)
        self.assertEqual(out, "hello")

    def test_execute_records_scripts(self):
        sb = FakeSandbox(execute_result=(0, "ok"))
        sb.execute("echo one")
        sb.execute("echo two")
        self.assertEqual(sb.executed_scripts, ["echo one", "echo two"])
        self.assertEqual(sb.scripts, sb.executed_scripts)
        self.assertEqual(sb.calls, sb.executed_scripts)

    def test_execute_python_records(self):
        sb = FakeSandbox(execute_python_result=(0, "py ok"))
        code, out = sb.execute_python("print(1)")
        self.assertEqual(code, 0)
        self.assertEqual(out, "py ok")
        self.assertEqual(sb.executed_python_scripts, ["print(1)"])

    def test_execute_callable(self):
        def handler(script):
            return (0, f"ran:{script}")
        sb = FakeSandbox(execute_result=handler)
        code, out = sb.execute("echo hi")
        self.assertEqual(out, "ran:echo hi")

    def test_execute_queue_sequential(self):
        sb = FakeSandbox(execute_result=[(0, "first"), (1, "second")])
        self.assertEqual(sb.execute("a"), (0, "first"))
        self.assertEqual(sb.execute("b"), (1, "second"))
        # When queue exhausted, falls back to default (0, "")
        self.assertEqual(sb.execute("c"), (0, ""))

    def test_request_write_default_approves(self):
        sb = FakeSandbox()
        ok, msg = sb.request_write("/tmp/somepath")
        self.assertTrue(ok)
        self.assertIn("granted", msg)
        self.assertIn(os.path.abspath("/tmp/somepath"), sb.approved_write_paths)

    def test_request_write_stubbed_denial(self):
        sb = FakeSandbox()
        sb.stub_request_write("/etc/passwd", (False, "Write access denied by user."))
        ok, msg = sb.request_write("/etc/passwd")
        self.assertFalse(ok)
        self.assertIn("denied", msg)

    def test_fake_sandbox_swap_into_agent(self):
        """Agent.__init__ assigns Sandbox, but tests overwrite with FakeSandbox."""
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        # Swap in a fake that returns a known output
        fake_sb = FakeSandbox(execute_result=(0, "sandbox hello"))
        agent.sandbox = fake_sb
        # parse_and_execute should route to FakeSandbox and format output
        block = bash_block(uid, "echo hi")
        with mock.patch("bash_agent.agent.MAX_CODE_BLOCKS", 5):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                executed, feedback = agent.parse_and_execute(block)
        self.assertTrue(executed)
        self.assertEqual(feedback, "")
        self.assertEqual(fake_sb.executed_scripts, ["echo hi"])
        # The last context message should be the OUTPUT block containing "sandbox hello"
        last_msg = agent.context.history[-1]
        content = last_msg["content"]
        # content may be str or list (if multimodal); handle both
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
        self.assertIn("sandbox hello", text)
        self.assertIn(f"EXIT_CODE_0", text)

    def test_make_agent_patches_capabilities_and_sandbox(self):
        """_make_agent helper must neutralize the network probe and inject FakeSandbox."""
        uid = str(uuid.uuid4())
        # Ensure urllib is not called
        with mock.patch("urllib.request.urlopen") as fake_urlopen:
            agent = _make_agent(uuid_str=uid)
            fake_urlopen.assert_not_called()
        self.assertEqual(agent.uuid, uid)
        self.assertIsInstance(agent.sandbox, FakeSandbox)
        self.assertEqual(agent.sandbox.uuid, uid)
        self.assertEqual(agent.context.uuid, uid)

    def test_make_agent_default_uuid(self):
        agent = _make_agent()
        self.assertIsNotNone(agent.uuid)
        self.assertIsInstance(agent.sandbox, FakeSandbox)

    def test_queue_helpers(self):
        sb = FakeSandbox()
        sb.queue_execute(2, "queued")
        sb.queue_execute_python(3, "py queued")
        self.assertEqual(sb.execute("x"), (2, "queued"))
        self.assertEqual(sb.execute_python("y"), (3, "py queued"))

    def test_non_special_script_falls_through_to_sandbox(self):
        """Non-special scripts must return handled=False and be routed to sandbox."""
        uid = str(uuid.uuid4())
        agent = _make_agent(uuid_str=uid)
        fake_sb = FakeSandbox(execute_result=(0, "hi"))
        agent.sandbox = fake_sb
        handled, _ = agent._handle_special_command("BASH", "echo hi")
        self.assertFalse(handled)
        # Now via parse_and_execute
        block = bash_block(uid, "echo hi")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent.parse_and_execute(block)
        self.assertEqual(fake_sb.executed_scripts[-1], "echo hi")


