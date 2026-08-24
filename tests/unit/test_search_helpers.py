"""
Group 8 — offline-core tests for bash_agent.search.

T-41  search.py offline core (P1)

search.py is the semantic-search tool the agent harness exposes as `search`.
Its network boundary (OpenRouter embeddings via llm.create_embedding and
the /api/v1/rerank HTTP call) is fully mockable, so every behavior below is
pinned offline:

  * get_ignore_patterns   — .gitignore merge with hardcoded exclusions
  * get_all_files         — dir/file pattern filtering via os.walk
  * get_file_hash         — MD5 stability + content sensitivity
  * load/save_embeddings_db — JSON round-trip preserving the _model key
  * get_file_content      — 100k truncation marker; binary/missing -> None
  * cosine_similarity     — known-vector correctness; zero-vector guard
  * rerank_documents      — requests.post patched: success parsing (sorted
                            desc), graceful degradation on HTTP error,
                            missing key, and unexpected payload shape
  * main() early-exit     — SEARCH_DISABLED_FLAG present -> exit 1 with NO
                            embedding calls (proven by a poisoned fake)

Seam notes:
  * EMBEDDINGS_DB / SEARCH_DISABLED_FLAG are import-time abspaths computed
    from the repo CWD. Tests run under chdir_tmp() and patch BOTH at
    bash_agent.search.<name> for hermeticity.
  * The LLM seam is bash_agent.llm._CLIENT_CACHE seeded via FakeLLMClient;
    its default embeddings response yields deterministic per-index vectors
    [0.1*(i+1), 0.2*(i+1), 0.3, 0.4].
  * config.OPENROUTER_API_KEY is read inside rerank_documents at call time,
    so patching bash_agent.config.OPENROUTER_API_KEY is seen.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from bash_agent import config
from bash_agent import llm
from bash_agent import search
from tests.helpers.fakes import FakeLLMClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True) if \
        os.path.dirname(path) else None
    with open(path, "w") as f:
        f.write(text)
    return path


class SearchTmpCase(unittest.TestCase):
    """Base case: isolated CWD + patched DB/flag paths, restored on exit."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        # Import-time constants resolved against the repo CWD — rebind them.
        self.db_patcher = mock.patch.object(
            search, "EMBEDDINGS_DB",
            os.path.join(self.tmpdir, ".bash_agent_tmp", "embeddings.json"))
        self.flag_patcher = mock.patch.object(
            search, "SEARCH_DISABLED_FLAG",
            os.path.join(self.tmpdir, ".bash_agent_tmp", "search_disabled"))
        self.model_patcher = mock.patch.object(
            search, "EMBEDDING_MODEL", "test-embed-model")
        for p in (self.db_patcher, self.flag_patcher, self.model_patcher):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(os.chdir, self._orig_cwd)

    @property
    def tmpdir(self):
        return self._tmp.name


class TestGetIgnorePatterns(SearchTmpCase):
    def test_hardcoded_exclusions_always_present(self):
        patterns = search.get_ignore_patterns()
        self.assertTrue({".git", ".bash_agent_tmp", "__pycache__"} <= patterns)

    def test_merges_gitignore_lines(self):
        write(".gitignore", "build/\n*.log\n# comment\n\nignored_dir\n")
        patterns = search.get_ignore_patterns()
        self.assertIn("build", patterns)          # trailing slash normalized
        self.assertNotIn("build/", patterns)
        self.assertIn("*.log", patterns)
        self.assertIn("ignored_dir", patterns)
        # Comment and blank lines are skipped
        self.assertNotIn("# comment", patterns)

    def test_no_gitignore_returns_hardcoded_only(self):
        patterns = search.get_ignore_patterns()
        self.assertEqual(patterns - {".git", ".bash_agent_tmp", "__pycache__"},
                         set())


class TestGetAllFiles(SearchTmpCase):
    def test_walks_and_returns_relative_paths(self):
        write("src/a.py", "x=1")
        write("src/sub/b.py", "y=2")
        rel = sorted(search.get_all_files("."))
        self.assertIn("src/a.py", rel)
        self.assertIn("src/sub/b.py", rel)

    def test_respects_ignore_patterns_for_dirs_and_files(self):
        write("src/keep.py", "k")
        write("build/out.bin", "b")
        write("debug.log", "l")
        write(".gitignore", "build/\n*.log")
        names = sorted(search.get_all_files("."))
        self.assertIn("src/keep.py", names)
        self.assertNotIn("build/out.bin", names)   # dir pattern "build/"
        self.assertNotIn("debug.log", names)       # glob "*.log"

    def test_directory_pattern_prunes_nested_trees(self):
        """Regression (bug #3): files DEEP under an ignored dir were indexed."""
        write("keep.txt", "k")
        write("build/shallow.obj", "s")
        write("build/deep/nested.o", "n")
        write("vendor/pkg/lib.so", "v")
        write(".gitignore", "build/\nvendor/")
        names = sorted(search.get_all_files("."))
        self.assertEqual(names, [".gitignore", "keep.txt"])

    def test_pattern_without_trailing_slash_still_prunes_dir(self):
        write("keep.txt", "k")
        write("ignored_dir/f.txt", "f")
        write(".gitignore", "ignored_dir\n")
        names = sorted(search.get_all_files("."))
        self.assertEqual(names, [".gitignore", "keep.txt"])

    def test_gitignore_file_itself_is_indexed(self):
        """Pins CURRENT behavior: .gitignore is not self-excluded."""
        write(".gitignore", "*.log")
        write("app.py", "a")
        self.assertIn(".gitignore", search.get_all_files("."))


class TestGetFileHash(unittest.TestCase):
    def test_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "f.txt")
            with open(p, "w") as f:
                f.write("hello")
            h1 = search.get_file_hash(p)
            h2 = search.get_file_hash(p)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 32)  # md5 hexdigest length
            with open(p, "w") as f:
                f.write("world!")
            self.assertNotEqual(h1, search.get_file_hash(p))


class TestEmbeddingsDbRoundTrip(SearchTmpCase):
    def test_missing_db_loads_empty_dict(self):
        self.assertEqual(search.load_embeddings_db(), {})

    def test_save_then_load_round_trip_preserves_model_key(self):
        db = {"a.txt": {"hash": "h1", "embedding": [0.25, 0.5]}}
        search.save_embeddings_db(db)
        loaded = search.load_embeddings_db()
        self.assertEqual(loaded["_model"], "test-embed-model")
        self.assertEqual(loaded["a.txt"]["hash"], "h1")
        self.assertEqual(loaded["a.txt"]["embedding"], [0.25, 0.5])
        # Saving again must not duplicate or corrupt metadata
        search.save_embeddings_db(loaded)
        self.assertEqual(search.load_embeddings_db()["_model"],
                         "test-embed-model")

    def test_creates_bash_agent_tmp_directory_on_save(self):
        search.save_embeddings_db({})
        self.assertTrue(
            os.path.isdir(os.path.dirname(search.EMBEDDINGS_DB)))


class TestGetFileContent(SearchTmpCase):
    def test_reads_text_content(self):
        write("f.txt", "line one\nline two\n")
        self.assertEqual(search.get_file_content("f.txt", "."), "line one\nline two\n")

    def test_truncation_marker_over_100k(self):
        big = "x" * 100_001
        write("big.txt", big)
        content = search.get_file_content("big.txt", ".", max_chars=100000)
        self.assertTrue(content.startswith("x" * 100000))
        self.assertIn("... [truncated 1 characters]", content)

    def test_binary_or_missing_or_unreadable_returns_none(self):
        with open("blob.bin", "wb") as f:
            f.write(b"\xff\xfe\x00binary\xff")
        self.assertIsNone(search.get_file_content("blob.bin", "."))

    def test_missing_file_returns_none(self):
        self.assertIsNone(search.get_file_content("nope.txt", "."))

    def test_permission_error_returns_none(self):
        path = write("secret.txt", "s")
        os.chmod(path, 0o000)
        try:
            self.assertIsNone(search.get_file_content("secret.txt", "."))
        finally:
            os.chmod(path, 0o644)


class TestCosineSimilarity(unittest.TestCase):
    def test_known_vectors(self):
        # Orthogonal -> 0; parallel same direction -> 1; opposite -> -1
        self.assertAlmostEqual(search.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(search.cosine_similarity([2, 0], [3, 0]), 1.0)
        self.assertAlmostEqual(search.cosine_similarity([1, 0], [-5, 0]), -1.0)
        self.assertAlmostEqual(search.cosine_similarity([1, 1], [1, 0]), 2 ** 0.5 / 2)

    def test_zero_vector_guard_returns_0(self):
        self.assertEqual(search.cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)
        self.assertEqual(search.cosine_similarity([1, 2, 3], [0, 0, 0]), 0.0)


class TestRerankDocuments(unittest.TestCase):
    """rerank_documents(client, query, documents, top_n) — client unused."""

    DOCS = ["doc one", "doc two", "doc three"]

    def setUp(self):
        self.key_patcher = mock.patch.object(config, "OPENROUTER_API_KEY",
                                             "sk-test-key")
        self.key_patcher.start()
        self.addCleanup(self.key_patcher.stop)

    def _post_mock(self, json_data=None, raise_exc=None):
        m = mock.MagicMock(name="response")
        if raise_exc:
            m.raise_for_status.side_effect = raise_exc
            return m
        m.raise_for_status.return_value = None
        m.json.return_value = json_data
        return m

    def test_empty_documents_short_circuits_before_any_http(self):
        post = mock.patch("bash_agent.search.requests.post")
        with post as p:
            result = search.rerank_documents(mock.MagicMock(), "q", [], 5)
        p.assert_not_called()
        self.assertEqual(result, [])

    def test_success_parses_results_sorted_desc_by_score(self):
        payload = {"results": [
            {"index": 0, "relevance_score": 0.10},
            {"index": 2, "relevance_score": 0.90},
            {"index": 1, "relevance_score": 0.50},
        ]}
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock(payload)) as p:
            scored = search.rerank_documents(mock.MagicMock(), "query",
                                             self.DOCS, top_n=3)
        self.assertEqual(scored, [(2, 0.90), (1, 0.50), (0, 0.10)])
        args, kwargs = p.call_args
        url = args[0] if args else kwargs["url"]
        self.assertEqual(url, "https://openrouter.ai/api/v1/rerank")
        body = kwargs.get("json") or {}
        self.assertEqual(body["documents"], self.DOCS)
        self.assertEqual(body["model"], search.RERANK_MODEL)
        self.assertEqual(body["query"], "query")
        self.assertEqual(body["top_n"], 3)
        headers = kwargs.get("headers") or {}
        self.assertEqual(headers["Authorization"], "Bearer sk-test-key")

    def test_attribution_headers_present_on_rerank(self):
        payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock(payload)) as p:
            search.rerank_documents(mock.MagicMock(), "query",
                                    self.DOCS, top_n=1)
        headers = p.call_args.kwargs["headers"]
        self.assertEqual(headers["HTTP-Referer"],
                         "https://github.com/geraldnilles/Bash-Agent")
        self.assertEqual(headers["X-OpenRouter-Title"], "Bash Agent")
        self.assertEqual(headers["X-OpenRouter-Categories"], "cli-agent")
        # Original auth/content-type headers are preserved
        self.assertEqual(headers["Authorization"], "Bearer sk-test-key")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_top_n_clamped_to_document_count(self):
        payload = {"results": [{"index": i, "relevance_score": 0.5}
                               for i in range(3)]}
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock(payload)):
            search.rerank_documents(mock.MagicMock(), "q", self.DOCS, top_n=99)
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock(payload)) as p:
            search.rerank_documents(mock.MagicMock(), "q", self.DOCS, top_n=99)
        self.assertEqual(p.call_args.kwargs["json"]["top_n"], 3)

    def test_top_n_smaller_than_docs_requests_fewer(self):
        payload = {"results": [{"index": 1, "relevance_score": 0.9}]}
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock(payload)) as p:
            search.rerank_documents(mock.MagicMock(), "q", self.DOCS, top_n=1)
        self.assertEqual(p.call_args.kwargs["json"]["top_n"], 1)

    def test_http_error_degrades_to_neutral_scores(self):
        resp = self._post_mock(raise_exc=__import__("requests").HTTPError())
        stderr_buf = io.StringIO()
        with mock.patch("bash_agent.search.requests.post", return_value=resp):
            with contextlib.redirect_stderr(stderr_buf):
                scored = search.rerank_documents(mock.MagicMock(), "q",
                                                 self.DOCS, top_n=2)
        self.assertEqual(scored, [(0, 0.0), (1, 0.0), (2, 0.0)])
        self.assertIn("Reranking failed", stderr_buf.getvalue())

    def test_missing_api_key_degrades_to_neutral_scores(self):
        with mock.patch.object(config, "OPENROUTER_API_KEY", None):
            stderr_buf = io.StringIO()
            with contextlib.redirect_stderr(stderr_buf):
                scored = search.rerank_documents(mock.MagicMock(), "q",
                                                 self.DOCS, top_n=2)
        self.assertEqual(scored, [(i, 0.0) for i in range(3)])
        self.assertIn("OPENROUTER_API_KEY not set", stderr_buf.getvalue())

    def test_unexpected_payload_shape_degrades_to_neutral_scores(self):
        stderr_buf = io.StringIO()
        with mock.patch("bash_agent.search.requests.post",
                        return_value=self._post_mock({"surprise": True})):
            with contextlib.redirect_stderr(stderr_buf):
                scored = search.rerank_documents(mock.MagicMock(), "q",
                                                 self.DOCS, top_n=2)
        self.assertEqual(scored, [(i, 0.0) for i in range(3)])
        self.assertIn("Unexpected rerank response format", stderr_buf.getvalue())

    def test_timeout_exception_degrades_gracefully(self):
        import requests as requests_mod
        stderr_buf = io.StringIO()
        with mock.patch("bash_agent.search.requests.post",
                        side_effect=requests_mod.Timeout()):
            with contextlib.redirect_stderr(stderr_buf):
                scored = search.rerank_documents(mock.MagicMock(), "q",
                                                 self.DOCS, top_n=2)
        self.assertEqual(scored, [(i, 0.0) for i in range(3)])
        self.assertIn("Reranking failed", stderr_buf.getvalue())


class TestMainDisabledEarlyExit(SearchTmpCase):
    """The SEARCH_DISABLED_FLAG contract: exit 1, zero embedding calls."""

    def test_flag_present_exits_1_without_llm_calls(self):
        flag_path = search.SEARCH_DISABLED_FLAG
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as f:
            f.write("")
        poison = FakeLLMClient()
        poison.chat.completions.create = mock.Mock(
            side_effect=AssertionError("disabled mode must not call chat"))
        poison.embeddings.create = mock.Mock(
            side_effect=AssertionError("disabled mode must not embed"))
        llm._CLIENT_CACHE["openrouter"] = poison

        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        argv = ["search", "-n", "3", "anything"]
        with mock.patch("sys.argv", argv):
            with contextlib.redirect_stdout(stdout_buf), \
                    contextlib.redirect_stderr(stderr_buf):
                with self.assertRaises(SystemExit) as cm:
                    search.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("user has disabled search", stderr_buf.getvalue())
        self.assertEqual(stdout_buf.getvalue(), "")
        poison.chat.completions.create.assert_not_called()
        poison.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
