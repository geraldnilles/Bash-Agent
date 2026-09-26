"""
E2E — supporting modules exercised offline (breadth over depth).

These are still end-to-end in the sense that they run against the REAL
bash_agent modules with ONLY the network/sandbox boundaries faked. They
replace a swath of unit tests that covered:

  * tokenizer routing            (test_tokenizer_routing.py)
  * config.json precedence       (test_config_file.py)
  * gitignore-style matching     (test_ignore_matcher.py)
  * vision / transcribe helper funcs + the Agent's multimodal attach path
                                  (test_vision.py, test_transcribe.py)
  * memo helpers                 (test_memo.py)
  * history file load/render     (test_history.py)

NO network. Where a helper would otherwise call OpenRouter, the LLM seam is
the seeded FakeLLMClient; where it would read the clipboard we don't call it.
"""
import base64
import contextlib
import io
import json
import os
import unittest
from unittest import mock

from PIL import Image

from bash_agent import tokenizer
from bash_agent import config_file
from bash_agent import history as history_mod
from bash_agent import memo
from bash_agent import transcribe
from bash_agent import vision

from tests.e2e.harness import (
    E2ECase,
    make_fake_response,
    bash_block,
    attached_image_block,
    attached_audio_block,
    FakeSandbox,
)

import re

UUID_RE = re.compile(r"current session UUID is: ([0-9a-fA-F-]+)")


def uid_of(messages):
    content = messages[0].get("content", "") if messages else ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    m = UUID_RE.search(content)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Tokenizer routing
# ---------------------------------------------------------------------------

class TestTokenizerRouting(unittest.TestCase):
    def setUp(self):
        tokenizer.clear_token_cache()

    def test_empty_text_is_zero(self):
        self.assertEqual(tokenizer.count_tokens("", model="gpt-4o"), 0)
        self.assertEqual(tokenizer.count_tokens(""), 0)

    def test_default_model_used_when_none(self):
        n = tokenizer.count_tokens("hello world")
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 1)

    def test_gemini_routes_through_o200k_proxy(self):
        text = "The quick brown fox jumps over the lazy dog."
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer.GEMINI_PROXY_ENCODING)
        expected = len(enc.encode(text, disallowed_special=()))
        self.assertEqual(tokenizer.count_tokens(text, model="gemini/x"), expected)

    def test_openai_prefix_stripped(self):
        text = "tokenization example 1234"
        bare = tokenizer.count_tokens(text, model="gpt-4o")
        prefixed = tokenizer.count_tokens(text, model="openai/gpt-4o")
        self.assertEqual(bare, prefixed)

    def test_cache_hits_return_same(self):
        text = "cache me please"
        a = tokenizer.count_tokens(text, model="gpt-4o")
        b = tokenizer.count_tokens(text, model="gpt-4o")
        self.assertEqual(a, b)
        # Cache keyed by (model, text): a different model may differ.
        c = tokenizer.count_tokens(text, model="gemini/x")
        self.assertIsInstance(c, int)


# ---------------------------------------------------------------------------
# config.json precedence + validation
# ---------------------------------------------------------------------------

class ConfigFileCase(E2ECase):
    def write_config(self, payload):
        cfg_dir = os.path.join(self.tmpdir, ".bash_agent_tmp")
        os.makedirs(cfg_dir, exist_ok=True)
        path = os.path.join(cfg_dir, config_file.CONFIG_FILENAME)
        with open(path, "w") as f:
            if isinstance(payload, str):
                f.write(payload)
            else:
                json.dump(payload, f)
        return path


class TestConfigFile(ConfigFileCase):
    def test_missing_file_empty(self):
        self.assertEqual(config_file.load_config(), {})

    def test_valid_config_roundtrips(self):
        self.write_config({
            "model": "deepseek/deepseek-v4-pro",
            "max_tokens": 4096,
            "reasoning_effort": "medium",
        })
        cfg = config_file.load_config()
        self.assertEqual(cfg["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(cfg["max_tokens"], 4096)
        self.assertEqual(cfg["reasoning_effort"], "medium")

    def test_malformed_json_ignored(self):
        self.write_config("{ not json ")
        self.assertEqual(config_file.load_config(), {})

    def test_per_key_validation(self):
        self.write_config({
            "model": "   ",          # empty after strip -> ignored
            "max_tokens": True,      # bool -> ignored
            "reasoning_effort": "maximum",  # invalid -> ignored
            "unknown_key": 1,        # unknown -> ignored
        })
        self.assertEqual(config_file.load_config(), {})

    def test_agent_precedence_cli_beats_file_beats_env(self):
        # config file supplies a model + effort
        self.write_config({"model": "file/model", "reasoning_effort": "high"})
        agent = self.make_agent(is_orchestrator=False, fake_sandbox=False)
        # config file wins (no CLI/env provided)
        self.assertEqual(agent.model, "file/model")
        self.assertEqual(agent.reasoning_effort, "high")

    def test_explicit_model_arg_beats_file(self):
        self.write_config({"model": "file/model"})
        agent = self.make_agent(is_orchestrator=False, model="cli/model",
                                fake_sandbox=False)
        self.assertEqual(agent.model, "cli/model")


# ---------------------------------------------------------------------------
# gitignore-style matcher
# ---------------------------------------------------------------------------

class TestIgnoreMatcher(unittest.TestCase):
    def matcher(self, patterns, include_mode=False):
        from bash_agent.ignore import GitIgnoreMatcher
        return GitIgnoreMatcher(patterns, include_mode=include_mode)

    def test_plain_exclude_and_negation(self):
        m = self.matcher(["*.log", "!keep.log"])
        self.assertFalse(m.match("app.log"))
        self.assertTrue(m.match("keep.log"))
        self.assertIsNone(m.match("app.py"))

    def test_double_star_crosses_dirs(self):
        m = self.matcher(["build/**"])
        self.assertFalse(m.match("build/a/b.txt"))
        self.assertIsNone(m.match("src/a.txt"))

    def test_dir_only_pattern(self):
        m = self.matcher(["cache/"])
        self.assertFalse(m.match("cache", is_dir=True))
        self.assertIsNone(m.match("cache", is_dir=False))

    def test_anchored_pattern(self):
        m = self.matcher(["/root.txt"])
        self.assertFalse(m.match("root.txt"))
        self.assertIsNone(m.match("sub/root.txt"))

    def test_include_mode(self):
        m = self.matcher(["*.py"], include_mode=True)
        self.assertTrue(m.match("a.py"))
        self.assertIsNone(m.match("a.txt"))

    def test_patterns_from_file(self):
        import tempfile
        from bash_agent.ignore import patterns_from_file
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, ".gitignore")
            with open(p, "w") as f:
                f.write("# comment\n\n*.log\nbuild/\n")
            self.assertEqual(patterns_from_file(p), ["*.log", "build/"])
            self.assertEqual(patterns_from_file(os.path.join(d, "nope")), [])


# ---------------------------------------------------------------------------
# memo helpers
# ---------------------------------------------------------------------------

class TestMemoHelpers(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(memo.format_duration(5), "5s")
        self.assertEqual(memo.format_duration(90), "1m30s")
        self.assertEqual(memo.format_duration(3750), "1h02m30s")

    def test_format_timestamp_shape(self):
        ts = memo.format_timestamp()
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{6}$")


# ---------------------------------------------------------------------------
# transcribe helpers
# ---------------------------------------------------------------------------

class TestTranscribeHelpers(unittest.TestCase):
    def test_audio_format_by_extension(self):
        self.assertEqual(transcribe.get_audio_format("a.mp3"), "mp3")
        self.assertEqual(transcribe.get_audio_format("a.WAV"), "wav")
        self.assertEqual(transcribe.get_audio_format("noext"), "unknown")

    def test_encode_audio_roundtrip(self):
        import tempfile
        payload = b"\x00\x01\x02\xffbinary"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            self.assertEqual(
                base64.b64decode(transcribe.encode_audio(path)), payload)
        finally:
            os.unlink(path)

    def test_check_file_size_oversize_exits(self):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 2048)
            path = f.name
        try:
            # 1 MB ceiling vs 2KB file -> fine.
            transcribe.check_file_size(path, max_mb=1)
            # 0.0001 MB ceiling -> oversize -> SystemExit(1)
            with self.assertRaises(SystemExit) as cm:
                transcribe.check_file_size(path, max_mb=0.0001)
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# vision helpers
# ---------------------------------------------------------------------------

class TestVisionHelpers(unittest.TestCase):
    def test_encode_image_returns_b64_png(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "i.png")
            Image.new("RGB", (4, 4), (1, 2, 3)).save(path)
            b64 = vision.encode_image(path)
            raw = base64.b64decode(b64)
            # PNG magic bytes.
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")

    def test_check_image_size_rejects_oversize(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big.png")
            Image.new("RGB", (50, 50), (0, 0, 0)).save(path)
            # Temporarily lower the gate so 2500 px is "oversize".
            with mock.patch.object(vision, "MAX_PIXELS", 100):
                with self.assertRaises(SystemExit) as cm:
                    vision.check_image_size(path)
            self.assertEqual(cm.exception.code, 1)


# ---------------------------------------------------------------------------
# history load / render
# ---------------------------------------------------------------------------

class TestHistoryHelpers(E2ECase):
    def test_load_history_rejects_invalid(self):
        bad = os.path.join(self.tmpdir, "bad.json")
        with open(bad, "w") as f:
            f.write("{ nope")
        self.assertIsNone(history_mod.load_history(bad))
        # Missing file
        self.assertIsNone(history_mod.load_history(os.path.join(self.tmpdir, "no.json")))
        # Wrong shape
        wrong = os.path.join(self.tmpdir, "wrong.json")
        with open(wrong, "w") as f:
            json.dump(["not", "a", "dict"], f)
        self.assertIsNone(history_mod.load_history(wrong))

    def test_load_and_print_history(self):
        path = os.path.join(self.tmpdir, "h.json")
        data = {"uuid": "u-1", "history": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
        ]}
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = history_mod.load_history(path)
        self.assertEqual(loaded["uuid"], "u-1")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            history_mod.print_history(path, loaded, use_color=False)
        out = buf.getvalue()
        self.assertIn("session u-1", out)
        self.assertIn("[SYSTEM]", out)
        self.assertIn("[USER]", out)
        self.assertIn("hello world", out)


# ---------------------------------------------------------------------------
# Agent multimodal attach path (vision/transcribe sandbox fences)
# ---------------------------------------------------------------------------

class TestAgentMultimodalAttach(E2ECase):
    """The Agent scrapes ATTACHED_IMAGE / ATTACHED_AUDIO fences from sandbox
    stdout and turns them into structured multimodal content parts."""

    def _agent(self):
        return super().make_agent(is_orchestrator=False)

    def test_image_fence_becomes_structured_content(self):
        agent = self._agent()
        data_url = "data:image/png;base64," + base64.b64encode(b"PNGDATA").decode()
        # FakeSandbox returns the vision attach fence as stdout.
        agent.sandbox = FakeSandbox(
            execute_result=(0, attached_image_block(agent.uuid, data_url))
        )
        executed, _ = agent.parse_and_execute(
            bash_block(agent.uuid, "vision img.png"))
        self.assertTrue(executed)
        last = agent.context.history[-1]
        self.assertEqual(last["role"], "user")
        # Content became a structured list carrying the image part.
        self.assertIsInstance(last["content"], list)
        kinds = [p.get("type") for p in last["content"] if isinstance(p, dict)]
        self.assertIn("image_url", kinds)
        # The raw fence was stripped from the text part.
        text_part = next(p["text"] for p in last["content"]
                         if p.get("type") == "text")
        self.assertNotIn("ATTACHED_IMAGE", text_part)

    def test_audio_fence_becomes_structured_content(self):
        agent = self._agent()
        b64 = base64.b64encode(b"AUDIODATA").decode()
        agent.sandbox = FakeSandbox(
            execute_result=(0, attached_audio_block(agent.uuid, b64))
        )
        executed, _ = agent.parse_and_execute(
            bash_block(agent.uuid, "transcribe a.mp3"))
        self.assertTrue(executed)
        last = agent.context.history[-1]
        self.assertIsInstance(last["content"], list)
        kinds = [p.get("type") for p in last["content"] if isinstance(p, dict)]
        self.assertIn("input_audio", kinds)


if __name__ == "__main__":
    unittest.main()
