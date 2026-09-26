"""
Group N — helper tests for bash_agent.history.

Covers the pure rendering helpers and the file-loading path without any
network/hardware:

  * _parse_data_url       — mime + decoded byte length from a data: URL
  * _format_image_block   — modern dict form, legacy bare-string form, URL form
  * _format_audio_block   — declared format + decoded size, legacy bare base64
  * _render_content       — plain string and mixed block list
  * load_history          — safe loading of missing / malformed / bogus files
  * resolve_paths         — defaults, explicit paths, and --all expansion

The base64-leak guarantee is pinned here: rendered placeholders never contain
the raw payload.
"""

import base64
import json
import os
import tempfile
import unittest

from bash_agent import config
from bash_agent import history as hist


IMG_B64 = base64.b64encode(os.urandom(1234)).decode()
AUD_B64 = base64.b64encode(os.urandom(4321)).decode()


class ParseDataUrlTests(unittest.TestCase):
    def test_direct_image_url_mime_and_size(self):
        mime, n = hist._parse_data_url(f"data:image/png;base64,{IMG_B64}")
        self.assertEqual(mime, "image/png")
        self.assertEqual(n, 1234)

    def test_non_data_url_returns_none(self):
        self.assertEqual(hist._parse_data_url("https://example.com/x.png"), (None, None))
        self.assertEqual(hist._parse_data_url(None), (None, None))


class ImageBlockTests(unittest.TestCase):
    def test_modern_dict_form(self):
        block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG_B64}"}}
        out = hist._format_image_block(block)
        self.assertEqual(out, "<Image payload (image/png, 1234 bytes)>")
        self.assertNotIn(IMG_B64, out)

    def test_legacy_bare_string_form(self):
        block = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{IMG_B64}"}
        out = hist._format_image_block(block)
        self.assertEqual(out, "<Image payload (image/jpeg, 1234 bytes)>")
        self.assertNotIn(IMG_B64, out)

    def test_remote_url_form(self):
        block = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        self.assertIn("https://x/y.png", hist._format_image_block(block))


class AudioBlockTests(unittest.TestCase):
    def test_dict_form_uses_declared_format(self):
        block = {"type": "input_audio", "input_audio": {"data": AUD_B64, "format": "mp3"}}
        out = hist._format_audio_block(block)
        self.assertEqual(out, "<Audio payload (mp3, 4321 bytes)>")
        self.assertNotIn(AUD_B64, out)

    def test_legacy_bare_base64_form(self):
        block = {"type": "input_audio", "input_audio": AUD_B64}
        out = hist._format_audio_block(block)
        self.assertTrue(out.startswith("<Audio payload ("))
        self.assertIn("4321 bytes", out)
        self.assertNotIn(AUD_B64, out)


class RenderContentTests(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(hist._render_content("hello", False), "hello")

    def test_none(self):
        self.assertEqual(hist._render_content(None, False), "")

    def test_block_list_and_unknown_type(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG_B64}"}},
            {"type": "mystery"},
        ]
        out = hist._render_content(content, False)
        self.assertIn("hi", out)
        self.assertIn("<Image payload (image/png, 1234 bytes)>", out)
        self.assertIn("<mystery payload>", out)

    def test_truncation(self):
        out = hist._render_content("x" * 100, False, truncate=10)
        self.assertTrue(out.startswith("x" * 10))
        self.assertIn("truncated", out)


class LoadHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def _write(self, name, obj):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f)
        return path

    def test_missing_file_returns_none(self):
        self.assertIsNone(hist.load_history(os.path.join(self.dir, "nope.json")))

    def test_malformed_json_returns_none(self):
        p = self._write("bad.json", "not json{")
        self.assertIsNone(hist.load_history(p))

    def test_wrong_shape_returns_none(self):
        p = self._write("shape.json", {"uuid": "x"})
        self.assertIsNone(hist.load_history(p))

    def test_valid_file_roundtrips(self):
        p = self._write("ok.json", {"uuid": "u1", "history": [{"role": "user", "content": "hi"}]})
        data = hist.load_history(p)
        self.assertEqual(data["uuid"], "u1")
        self.assertEqual(len(data["history"]), 1)


class ResolvePathsTests(unittest.TestCase):
    def test_default_includes_main_history(self):
        paths = hist.resolve_paths([], include_all=False)
        self.assertIn(os.path.abspath(config.HISTORY_FILE), paths)

    def test_explicit_glob(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("a.json", "b.json"):
                open(os.path.join(d, n), "w").close()
            paths = hist.resolve_paths([os.path.join(d, "*.json")], include_all=False)
            names = {os.path.basename(p) for p in paths}
            self.assertEqual(names, {"a.json", "b.json"})

    def test_missing_literal_kept(self):
        paths = hist.resolve_paths(["/no/such/file.json"], include_all=False)
        self.assertIn("/no/such/file.json", paths)

    def test_all_expands_subagent_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            main = os.path.join(d, "history.json")
            sub = os.path.join(d, config.SUBAGENT_HISTORY_PREFIX + "abcd1234.json")
            open(main, "w").close()
            open(sub, "w").close()
            orig = config.HISTORY_FILE
            try:
                hist.HISTORY_FILE = main
                paths = hist.resolve_paths([], include_all=True)
                self.assertIn(os.path.abspath(main), paths)
                self.assertIn(os.path.abspath(sub), paths)
            finally:
                hist.HISTORY_FILE = orig


if __name__ == "__main__":
    unittest.main()
