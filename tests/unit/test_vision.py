"""
Group 8 — dual-mode tests for bash_agent.vision.

T-38  vision.main() — multimodal attach mode vs LLM fallback mode (P1)

vision.py serves two masters depending on its environment:

  * SANDBOX ATTACH MODE — when invoked by the agent sandbox,
    BASH_AGENT_UUID is set and BASH_AGENT_MULTIMODAL includes "image".
    The tool prints the image as a fenced base64 payload (the same shape
    produced by helpers.attached_image_block) and exits 0 WITHOUT any LLM
    call; agent.py scrapes the fence out of sandbox stdout and attaches
    the image to its next request.
  * FALLBACK MODE — otherwise (standalone CLI, or a text-only model), the
    tool sends text + image_url content parts to the LLM and prints the
    textual answer.

A regression in either branch breaks vision silently: a stray LLM call in
attach mode wastes budget and duplicates context; a malformed fence is
dropped by agent.py's scanner; wrong message-part shape would 400 at the
provider. These tests pin both contracts offline.

Seam notes:
  * main() is invoked directly with patched sys.argv (subprocess-free;
    equivalent to the planned runpy approach).
  * The LLM boundary is the documented seam: bash_agent.llm._CLIENT_CACHE
    seeded with a fake client. Attach-mode tests POISON the cache with a
    loudly-failing client to prove no call happens; fallback tests seed
    scripted responses and assert on recorded kwargs.
  * vision.py does `from bash_agent.config import MAX_PIXELS`, binding the
    value at import time — the size gate MUST be patched at
    bash_agent.vision.MAX_PIXELS, never at bash_agent.config.
  * The agent harness itself exports BASH_AGENT_UUID/BASH_AGENT_MULTIMODAL
    while running these tests, so every invocation runs under a scrubbed
    environment (mock.patch.dict clear=True) with only the variables the
    scenario needs.
"""

import base64
import contextlib
import io
import re
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from bash_agent import llm
from bash_agent import vision
from bash_agent.vision import DEFAULT_PROMPT
from tests.helpers.fakes import (
    FakeLLMClient,
    attached_image_block,
    make_fake_response,
)

SESSION_UUID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
IMG_SIZE = (8, 6)  # 48 px total


def make_png(path, width=IMG_SIZE[0], height=IMG_SIZE[1], color=(180, 60, 20)):
    """Write a tiny real PNG so Pillow exercises actual decode/re-encode."""
    Image.new("RGB", (width, height), color).save(path, format="PNG")
    return path


class VisionCase(unittest.TestCase):
    """Shared harness: tmp image, scrubbed env, isolated LLM cache."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.img_path = make_png(os.path.join(self.tmpdir, "shot.png"))
        # Snapshot & empty the client cache so seeded/poisoned clients
        # cannot leak into (or come from) other tests.
        self._cache_backup = dict(llm._CLIENT_CACHE)
        llm._CLIENT_CACHE.clear()

    def tearDown(self):
        llm._CLIENT_CACHE.clear()
        llm._CLIENT_CACHE.update(self._cache_backup)
        self._tmp.cleanup()

    @property
    def tmpdir(self):
        return self._tmp.name

    def poison_cache(self):
        """Seed every backend with a client that fails LOUDLY if used."""
        poison = mock.MagicMock(name="poisoned_llm_client")
        poison.chat.completions.create.side_effect = AssertionError(
            "vision must NOT call the LLM in multimodal attach mode")
        # Seed the OpenRouter backend so a MODEL_ID flip cannot silently bypass.
        llm._CLIENT_CACHE["openrouter"] = poison
        return poison

    def seed_cache(self, response):
        fake = FakeLLMClient(responses=[response])
        llm._CLIENT_CACHE["openrouter"] = fake
        return fake

    def run_main(self, argv, extra_env=None):
        """
        Run vision.main() under patched sys.argv and a scrubbed environment
        (all BASH_AGENT_* vars removed unless re-added via extra_env).
        Returns dict with exit/SystemExit, stdout, stderr strings.
        """
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("BASH_AGENT_")}
        env.update(extra_env or {})
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(stdout_buf), \
                    contextlib.redirect_stderr(stderr_buf):
                with mock.patch("sys.argv", argv):
                    try:
                        vision.main()
                        exc = None
                    except SystemExit as e:
                        exc = e
        return {
            "exit": exc,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }

    def attach_argv(self, path=None):
        return ["vision", path or self.img_path]

    DATA_URL_RE = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")

    def extracted_data_url(self, stdout):
        match = self.DATA_URL_RE.search(stdout)
        self.assertIsNotNone(match, "no base64 data URL found on stdout")
        return match.group(0)


class TestSandboxAttachMode(VisionCase):
    """UUID + image modality -> fenced payload on stdout, zero LLM calls."""

    ATTACH_NOTE = "attached to conversation context"

    def test_emits_valid_fence_and_exits_0_without_llm_call(self):
        poison = self.poison_cache()
        res = self.run_main(
            self.attach_argv(),
            extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "image"})

        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 0)

        expected_payload = attached_image_block(
            SESSION_UUID, self.extracted_data_url(res["stdout"]))
        self.assertIn(expected_payload, res["stdout"])
        self.assertIn(f"Image '{self.img_path}' {self.ATTACH_NOTE}",
                      res["stdout"])

        # The heart of the contract: the LLM layer was never touched.
        poison.chat.completions.create.assert_not_called()

    def test_comma_separated_modality_list_still_attaches(self):
        # Sandbox exports e.g. "image,audio"; parsing must tolerate commas.
        self.poison_cache()
        res = self.run_main(
            self.attach_argv(),
            extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "audio,image"})
        self.assertEqual(res["exit"].code, 0)
        self.assertIn(f"---START_ATTACHED_IMAGE-{SESSION_UUID}---",
                      res["stdout"])
        self.assertIn("---END_ATTACHED_IMAGE-" + SESSION_UUID + "---",
                      res["stdout"])
        self.assertIn("data:image/png;base64,", res["stdout"])

    def test_data_url_decodes_back_to_the_source_image(self):
        self.poison_cache()
        res = self.run_main(
            self.attach_argv(),
            extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "image"})
        data_url = self.extracted_data_url(res["stdout"])
        raw = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, IMG_SIZE)

    def test_uuid_without_image_capability_falls_through_to_llm(self):
        # Gate is a CONJUNCTION: UUID present but modalities lack "image"
        # => standard fallback path (an LLM call DOES happen here).
        fake = self.seed_cache(make_fake_response(content="fallback answer"))
        res = self.run_main(
            self.attach_argv(),
            extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "audio"})
        self.assertIsNone(res["exit"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(res["stdout"].strip(), "fallback answer")


class TestOversizeGate(VisionCase):
    """check_image_size rejects images over the (patched) pixel budget."""

    def test_oversize_image_exits_1_with_diagnostic_stderr(self):
        self.poison_cache()
        with mock.patch.object(vision, "MAX_PIXELS", 10):
            res = self.run_main(
                self.attach_argv(),
                extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                           "BASH_AGENT_MULTIMODAL": "image"})

        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 1)
        self.assertIn("Error: Image is too large", res["stderr"])
        self.assertIn(f"{IMG_SIZE[0]}x{IMG_SIZE[1]} "
                      f"= {IMG_SIZE[0] * IMG_SIZE[1]} pixels", res["stderr"])
        self.assertNotIn("base64,", res["stdout"])  # nothing was emitted

    def test_exact_pixel_budget_passes_the_gate(self):
        # Strictly-greater comparison: an image exactly AT the limit is legal.
        self.poison_cache()
        with mock.patch.object(vision, "MAX_PIXELS",
                               IMG_SIZE[0] * IMG_SIZE[1]):
            res = self.run_main(
                self.attach_argv(),
                extra_env={"BASH_AGENT_UUID": SESSION_UUID,
                           "BASH_AGENT_MULTIMODAL": "image"})
        self.assertEqual(res["exit"].code, 0)


class TestMissingInput(VisionCase):
    """Bad path short-circuits before encoding or any network activity."""

    def test_missing_file_exits_1_without_touching_llm(self):
        poison = self.poison_cache()
        missing = os.path.join(self.tmpdir, "does-not-exist.png")
        res = self.run_main(self.attach_argv(missing))

        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 1)
        self.assertIn("Error: File", res["stderr"])
        self.assertIn("not found", res["stderr"])
        poison.chat.completions.create.assert_not_called()


class TestFallbackMode(VisionCase):
    """No sandbox env -> text+image_url parts go to the LLM; answer printed."""

    def test_message_structure_has_text_and_image_url_parts(self):
        fake = self.seed_cache(make_fake_response(content="## Chart notes"))
        res = self.run_main(self.attach_argv())

        self.assertIsNone(res["exit"])
        self.assertEqual(res["stdout"].strip(), "## Chart notes")

        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["model"], vision.MODEL_ID)

        message = call["messages"][0]
        self.assertEqual(message["role"], "user")
        text_part, image_part = message["content"]
        self.assertEqual(text_part,
                         {"type": "text", "text": DEFAULT_PROMPT})
        url = image_part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        # Payload decodes to a valid PNG of the source dimensions.
        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as decoded:
            self.assertEqual(decoded.size, IMG_SIZE)

    def test_custom_prompt_flows_into_text_part(self):
        fake = self.seed_cache(make_fake_response(content="ok"))
        res = self.run_main(["vision", "-p", "Describe the chart",
                             self.img_path])
        self.assertIsNone(res["exit"])
        text_part = fake.calls[0]["messages"][0]["content"][0]
        self.assertEqual(text_part["text"], "Describe the chart")

    def test_fallback_api_failure_exits_1_with_error_message(self):
        broken = mock.MagicMock(name="broken_llm_client")
        broken.chat.completions.create.side_effect = RuntimeError(
            "provider exploded")
        llm._CLIENT_CACHE["openrouter"] = broken

        res = self.run_main(self.attach_argv())

        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 1)
        self.assertIn("Error during API request", res["stderr"])
        self.assertIn("provider exploded", res["stderr"])
        self.assertEqual(res["stdout"], "")


if __name__ == "__main__":
    unittest.main()
