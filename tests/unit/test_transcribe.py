"""
Group 8 — helper tests for bash_agent.transcribe.

T-39  transcribe helpers (P2)

Covers the pieces of transcribe.py that can be exercised without hardware
or network:

  * get_audio_format   — extension mapping incl. extensionless -> "unknown"
  * encode_audio       — base64 fidelity on binary payloads
  * check_file_size    — boundary behavior with an explicit threshold
                         (oversize -> SystemExit(1) with a stderr message)
  * convert_to_mp3     — real ffmpeg conversion of generated silence
                         (skipUnless ffmpeg present), plus both failure
                         branches: bad input data and missing binary.
  * main() context XML — the <file path="...">/<context> prompt assembly is
                         driven END-TO-END through main() itself (stronger
                         than the planned "tiny driver mimic"): a real WAV
                         is converted by real ffmpeg, the LLM boundary is
                         patched at bash_agent.llm.create_chat_completion,
                         and the recorded kwargs are asserted.

Seam notes:
  * The LLM call site resolves llm.create_chat_completion at call time via
    the module object, so patching bash_agent.llm.create_chat_completion is
    seen by transcribe.main().
  * convert_to_mp3 writes its temp file via tempfile.NamedTemporaryFile;
    tests redirect that into the per-test tmpdir so cleanup can be asserted
    deterministically (main() unlinks the temp mp3 in its finally block).
  * Quirk pinned (not fixed): when -c/--context is supplied, main() REPLACES
    args.prompt entirely with its own reference-files template — a custom -p
    is silently discarded. The e2e test documents this current behavior.
"""

import base64
import contextlib
import glob
import io
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from unittest import mock

from bash_agent import llm
from bash_agent import transcribe
from tests.helpers.fakes import (
    attached_audio_block,
    make_fake_response,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def write_silence_wav(path, seconds=0.5, rate=8000):
    """Generate a tiny valid mono 16-bit PCM WAV of silence (stdlib only)."""
    frames = b"\x00\x00" * int(rate * seconds)
    with contextlib.closing(wave.open(path, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


class TestGetAudioFormat(unittest.TestCase):
    def test_extension_mapping(self):
        cases = {
            "clip.opus": "opus",
            "song.MP3": "mp3",          # case-insensitive
            "voice.WAV": "wav",
            "a.b.m4a": "m4a",           # multi-dot keeps last suffix
            "noext": "unknown",         # extensionless
            ".hidden": "unknown",       # dotfile has no "extension"
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(transcribe.get_audio_format(path), expected)


class TestEncodeAudio(unittest.TestCase):
    def test_base64_fidelity_on_binary_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audio.bin")
            raw = bytes(range(256)) + b"\x00\xff\xfe" + "éü".encode("utf-8")
            with open(path, "wb") as f:
                f.write(raw)

            encoded = transcribe.encode_audio(path)

            self.assertIsInstance(encoded, str)
            self.assertEqual(encoded, base64.b64encode(raw).decode("utf-8"))
            self.assertEqual(base64.b64decode(encoded), raw)


class TestCheckFileSize(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _make_file(self, size_bytes):
        path = os.path.join(self.tmpdir, "big.opus")
        with open(path, "wb") as f:
            f.truncate(size_bytes)  # sparse; getsize reports logical size
        return path

    @property
    def tmpdir(self):
        return self._tmp.name

    def test_under_limit_passes_silently(self):
        path = self._make_file(1024)
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), \
                contextlib.redirect_stderr(stderr_buf):
            # Returns None; must NOT raise SystemExit
            result = transcribe.check_file_size(path, max_mb=50)
        self.assertIsNone(result)
        self.assertEqual(stderr_buf.getvalue(), "")

    def test_exact_boundary_is_allowed(self):
        """Strictly-greater comparison: exactly max_mb MB is NOT oversize."""
        path = self._make_file(1024 * 1024)  # exactly 1.0 MB
        transcribe.check_file_size(path, max_mb=1)  # no SystemExit

    def test_oversize_exits_1_with_message(self):
        path = self._make_file(1024 * 1024 + 1)  # just over 1.0 MB
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as cm:
                transcribe.check_file_size(path, max_mb=1)
        self.assertEqual(cm.exception.code, 1)
        err = stderr_buf.getvalue()
        self.assertIn("too large", err)
        self.assertIn("Maximum allowed is 1 MB", err)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg binary not available")
class TestConvertToMp3Real(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    @property
    def tmpdir(self):
        return self._tmp.name

    def test_converts_generated_silence(self):
        wav_path = write_silence_wav(os.path.join(self.tmpdir, "in.wav"), seconds=0.5)
        out_path = transcribe.convert_to_mp3(wav_path)
        try:
            self.assertTrue(out_path.endswith(".mp3"))
            self.assertGreater(os.path.getsize(out_path), 0)
            # MP3 container/frame magic: ID3 tag or 0xFF frame sync
            with open(out_path, "rb") as f:
                head = f.read(3)
            self.assertTrue(head == b"ID3" or head[0] == 0xFF,
                            f"not an MP3 stream: {head!r}")
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_invalid_input_data_exits_1(self):
        """Non-audio input -> ffmpeg nonzero rc -> SystemExit(1)."""
        junk = os.path.join(self.tmpdir, "junk.txt")
        with open(junk, "w") as f:
            f.write("this is not audio\n")
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as cm:
                transcribe.convert_to_mp3(junk)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Error converting audio", stderr_buf.getvalue())


class TestConvertToMp3MissingBinary(unittest.TestCase):
    """The FileNotFoundError branch — forced offline, no ffmpeg needed."""

    def test_missing_ffmpeg_exits_1(self):
        stderr_buf = io.StringIO()
        with mock.patch("bash_agent.transcribe.subprocess.run",
                        side_effect=FileNotFoundError("nope")):
            with contextlib.redirect_stderr(stderr_buf):
                with self.assertRaises(SystemExit) as cm:
                    transcribe.convert_to_mp3("/unused/in.wav")
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("ffmpeg not found", stderr_buf.getvalue())


class TranscribeMainCase(unittest.TestCase):
    """Shared harness for driving transcribe.main() fully offline."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wav_path = write_silence_wav(
            os.path.join(self.tmpdir, "memo.wav"), seconds=0.3)
        self.captured = {}

    @property
    def tmpdir(self):
        return self._tmp.name

    def patch_llm(self, content="TRANSCRIBED TEXT"):
        def fake_create(**kwargs):
            self.captured.update(kwargs)
            return make_fake_response(content=content)
        return mock.patch("bash_agent.llm.create_chat_completion",
                          new=fake_create)

    def redirect_tempfiles(self):
        """Point NamedTemporaryFile at the per-test tmpdir for cleanup asserts."""
        real_ntf = transcribe.tempfile.NamedTemporaryFile

        def fake_ntf(*args, **kwargs):
            kwargs["dir"] = self.tmpdir
            return real_ntf(*args, **kwargs)

        return mock.patch.object(transcribe.tempfile, "NamedTemporaryFile",
                                 side_effect=fake_ntf)

    def run_main(self, argv):
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with self.patch_llm(), self.redirect_tempfiles():
            with contextlib.redirect_stdout(stdout_buf), \
                    contextlib.redirect_stderr(stderr_buf):
                with mock.patch("sys.argv", argv):
                    exc = None
                    try:
                        transcribe.main()
                    except SystemExit as e:
                        exc = e
        return {"exit": exc,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue()}


class TestMainContextAssembly(TranscribeMainCase):
    """T-39 core: the <context> XML wrapping logic, driven through main()."""

    def test_context_files_wrapped_as_xml_and_appended_to_prompt(self):
        c1 = os.path.join(self.tmpdir, "names.txt")
        c2 = os.path.join(self.tmpdir, "style.md")
        with open(c1, "w") as f:
            f.write("Gerald\nox-alpha")
        with open(c2, "w") as f:
            f.write("Use title case.")

        result = self.run_main(["transcribe", "-c", c1, c2, "--", self.wav_path])

        self.assertIsNone(result["exit"], msg=result["stderr"])
        self.assertEqual(result["stdout"], "TRANSCRIBED TEXT\n")

        messages = self.captured["messages"]
        self.assertEqual(len(messages), 1)
        parts = messages[0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        text = parts[0]["text"]

        # Both files wrapped in <file path="..."> elements inside <context>
        self.assertIn("<context>", text)
        self.assertIn(f'<file path="{c1}">\nGerald\nox-alpha\n</file>', text)
        self.assertIn(f'<file path="{c2}">\nUse title case.\n</file>', text)
        # Template instruction present (unique marker vs DEFAULT_PROMPT)
        self.assertIn("Use the following reference files", text)

        # Audio part: converted to mp3, payload is real base64 audio
        audio_part = parts[1]
        self.assertEqual(audio_part["type"], "input_audio")
        self.assertEqual(audio_part["input_audio"]["format"], "mp3")
        decoded = base64.b64decode(audio_part["input_audio"]["data"])
        self.assertGreater(len(decoded), 0)
        self.assertTrue(decoded[:3] == b"ID3" or decoded[0] == 0xFF,
                        f"payload not MP3: {decoded[:4]!r}")

        # Default model passed through unchanged
        self.assertEqual(self.captured["model"], transcribe.MODEL_ID)

        # Temp mp3 was cleaned up in main()'s finally block
        self.assertEqual(glob.glob(os.path.join(self.tmpdir, "*.mp3")), [])

    def test_custom_prompt_is_discarded_when_context_given(self):
        """Pins current behavior: -p is silently replaced by the -c template."""
        c1 = os.path.join(self.tmpdir, "ctx.txt")
        with open(c1, "w") as f:
            f.write("body\n")
        result = self.run_main(
            ["transcribe", "-p", "CUSTOM PROMPT X", "-c", c1, "--", self.wav_path])
        self.assertIsNone(result["exit"])
        text = self.captured["messages"][0]["content"][0]["text"]
        self.assertNotIn("CUSTOM PROMPT X", text)
        self.assertIn("Use the following reference files", text)


class TestMainBasicCall(TranscribeMainCase):
    def test_no_context_uses_default_prompt_and_cleans_temp(self):
        result = self.run_main(["transcribe", self.wav_path])
        self.assertIsNone(result["exit"], msg=result["stderr"])
        self.assertEqual(result["stdout"], "TRANSCRIBED TEXT\n")

        text = self.captured["messages"][0]["content"][0]["text"]
        self.assertEqual(text, transcribe.DEFAULT_PROMPT)
        self.assertEqual(self.captured["model"], transcribe.MODEL_ID)
        self.assertEqual(glob.glob(os.path.join(self.tmpdir, "*.mp3")), [])

    def test_missing_audio_file_exits_1(self):
        result = self.run_main(["transcribe", "/nonexistent/audio.opus"])
        self.assertIsInstance(result["exit"], SystemExit)
        self.assertEqual(result["exit"].code, 1)
        self.assertIn("not found", result["stderr"])


class TranscribeAttachCase(unittest.TestCase):
    """Harness for transcribe.main() multimodal sandbox attach mode.

    Differences from TranscribeMainCase:
      * environment is SCRUBBED of all BASH_AGENT_* vars, then selectively
        re-added via extra_env (the agent harness exports these while the
        unit suite runs),
      * LLM cache is POISONED in attach tests to prove no HTTP call happens,
      * real ffmpeg conversion still runs; NamedTemporaryFile is redirected
        into the per-test tmpdir so temp-MP3 cleanup can be asserted.
    """

    SESSION_UUID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wav_path = write_silence_wav(
            os.path.join(self.tmpdir, "memo.wav"), seconds=0.3)
        # Snapshot & empty the client cache so poisoned/seeded clients
        # cannot leak into (or come from) other tests.
        self._cache_backup = dict(llm._CLIENT_CACHE)
        llm._CLIENT_CACHE.clear()
        self.captured = {}

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
            "transcribe must NOT call the LLM in multimodal attach mode")
        llm._CLIENT_CACHE["openrouter"] = poison
        return poison

    def seed_llm(self, content="TRANSCRIBED TEXT"):
        """Patch llm.create_chat_completion for fallback-path assertions."""
        def fake_create(**kwargs):
            self.captured.update(kwargs)
            return make_fake_response(content=content)
        return mock.patch("bash_agent.llm.create_chat_completion",
                          new=fake_create)

    def redirect_tempfiles(self):
        """Point NamedTemporaryFile at the per-test tmpdir for cleanup asserts."""
        real_ntf = transcribe.tempfile.NamedTemporaryFile

        def fake_ntf(*args, **kwargs):
            kwargs["dir"] = self.tmpdir
            return real_ntf(*args, **kwargs)

        return mock.patch.object(transcribe.tempfile, "NamedTemporaryFile",
                                 side_effect=fake_ntf)

    def run_main(self, argv, extra_env=None):
        """
        Run transcribe.main() under patched sys.argv and a scrubbed
        environment (all BASH_AGENT_* vars removed unless re-added via
        extra_env). Returns dict with exit/SystemExit, stdout, stderr.
        """
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("BASH_AGENT_")}
        env.update(extra_env or {})
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True):
            with self.redirect_tempfiles():
                with contextlib.redirect_stdout(stdout_buf), \
                        contextlib.redirect_stderr(stderr_buf):
                    with mock.patch("sys.argv", argv):
                        exc = None
                        try:
                            transcribe.main()
                        except SystemExit as e:
                            exc = e
        return {"exit": exc,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue()}

    def extracted_payload(self, stdout):
        """Extract the base64 body of an ATTACHED_AUDIO fence from stdout."""
        m = re.search(
            rf"---START_ATTACHED_AUDIO-.*?---\n(.*?)\n---END_ATTACHED_AUDIO",
            stdout, re.DOTALL)
        self.assertIsNotNone(m, "no ATTACHED_AUDIO fence found on stdout")
        return m.group(1)


class TestSandboxAttachMode(TranscribeAttachCase):
    """UUID + audio modality -> fenced payload on stdout, zero LLM calls."""

    ATTACH_NOTE = "attached to conversation context"

    def test_emits_valid_fence_and_exits_0_without_llm_call(self):
        poison = self.poison_cache()
        res = self.run_main(
            ["transcribe", self.wav_path],
            extra_env={"BASH_AGENT_UUID": self.SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "audio"})

        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 0)

        # The exact helper-built fence must appear on stdout
        b64 = self.extracted_payload(res["stdout"])
        expected = attached_audio_block(self.SESSION_UUID, b64)
        self.assertIn(expected, res["stdout"])
        self.assertIn(f"Audio '{self.wav_path}' {self.ATTACH_NOTE}",
                      res["stdout"])

        # Payload decodes to real MP3 bytes (ID3 tag or 0xFF frame sync)
        decoded = base64.b64decode(b64)
        self.assertGreater(len(decoded), 0)
        self.assertTrue(decoded[:3] == b"ID3" or decoded[0] == 0xFF,
                        f"payload not MP3: {decoded[:4]!r}")

        # The heart of the contract: the LLM layer was never touched.
        poison.chat.completions.create.assert_not_called()

        # The attach path exits before try/finally; the explicit unlink in
        # the attach branch must have removed the converted temp MP3.
        self.assertEqual(glob.glob(os.path.join(self.tmpdir, "*.mp3")), [])

    def test_comma_separated_modality_list_still_attaches(self):
        # Sandbox exports e.g. "image,audio"; parsing must tolerate commas.
        self.poison_cache()
        res = self.run_main(
            ["transcribe", self.wav_path],
            extra_env={"BASH_AGENT_UUID": self.SESSION_UUID,
                       "BASH_AGENT_MULTIMODAL": "image,audio"})
        self.assertIsInstance(res["exit"], SystemExit)
        self.assertEqual(res["exit"].code, 0)
        self.assertIn(f"---START_ATTACHED_AUDIO-{self.SESSION_UUID}---",
                      res["stdout"])
        self.assertIn(f"---END_ATTACHED_AUDIO-{self.SESSION_UUID}---",
                      res["stdout"])

    def test_uuid_without_audio_capability_falls_through_to_llm(self):
        # Gate is a CONJUNCTION: UUID present but modalities lack "audio"
        # => standard fallback path (an LLM call DOES happen here).
        with self.seed_llm(content="fallback answer"):
            res = self.run_main(
                ["transcribe", self.wav_path],
                extra_env={"BASH_AGENT_UUID": self.SESSION_UUID,
                           "BASH_AGENT_MULTIMODAL": "image"})
        self.assertIsNone(res["exit"])
        self.assertEqual(res["stdout"], "fallback answer\n")

    def test_no_env_falls_back_to_llm(self):
        # No BASH_AGENT_* vars at all -> standard fallback path.
        with self.seed_llm(content="fallback answer"):
            res = self.run_main(["transcribe", self.wav_path])
        self.assertIsNone(res["exit"])
        self.assertEqual(res["stdout"], "fallback answer\n")


if __name__ == "__main__":
    unittest.main()
