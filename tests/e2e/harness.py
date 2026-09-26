"""
Shared harness for the offline E2E suite (tests/e2e).

Every E2E test drives the REAL `bash_agent` harness code end-to-end. The only
things faked are the *network boundaries* and the sandbox subprocess:

  MOCK BOUNDARIES (complete list — nothing else touches the network):
   1. `bash_agent.llm._CLIENT_CACHE["openrouter"]`
        -> seeded with `tests.helpers.fakes.FakeLLMClient` (chat + embeddings).
   2. `urllib.request.urlopen`
        -> patched with a fake that serves a canned OpenRouter /models
           catalog (or raises). Exercises the REAL `Agent._get_models_catalog`
           + capability/reasoning/context probe code with no HTTP.
   3. `requests.post`
        -> patched (search rerank). Provided here for search journeys.
   4. `subprocess.run` (bwrap) — replaced by `FakeSandbox`, OR left real for
      the guarded bwrap journeys (local process, still no network).

`bash_agent.context.HISTORY_FILE` is patched to the throwaway CWD so history
persistence lands inside the test's temp dir.

PROTOCOL HYGIENE: expected fence strings are assembled PIECEWISE
(`cmd_start`/`cmd_end`/`out_start`/`out_end`/...) so no contiguous command
fence literal ever appears in a transmitted block body.

Every agent loop is wrapped in `runaway_guard()` (SIGALRM) so a broken mock
queue fails fast instead of hanging the harness.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sys
import unittest
from unittest import mock

from bash_agent import llm
from bash_agent import agent as agent_module
from bash_agent.agent import Agent

from tests.helpers.fakes import (
    FakeLLMClient,
    FakeSandbox,
    make_fake_response,
    bash_block,
    python_block,
    output_block,
    attached_image_block,
    attached_audio_block,
    chdir_repo_tmp,
    deterministic_count_tokens,
    DeterministicTokenCounts,
    _stub_model_reasoning_info,
    _stub_model_context_info,
    bwrap_available,
)


# ---------------------------------------------------------------------------
# Fence-string builders (kept PIECEWISE — see module docstring)
# ---------------------------------------------------------------------------

def cmd_start(kind: str, uid: str) -> str:
    return "---START_" + kind + "_COMMAND-" + uid + "---"


def cmd_end(kind: str, uid: str) -> str:
    return "---END_" + kind + "_COMMAND-" + uid + "---"


def out_start(kind: str, exit_code: int, uid: str, visible: int = 100) -> str:
    return (
        "---START_" + kind + "_OUTPUT-EXIT_CODE_"
        + str(exit_code) + "-VISIBLE_" + str(visible) + "%-" + uid + "---"
    )


def out_end(kind: str, uid: str) -> str:
    return "---END_" + kind + "_OUTPUT-" + uid + "---"


def scratch_start(uid: str) -> str:
    return "---START_" + "SCRATCHPAD" + ".md-VISIBLE_100%-" + uid + "---"


def scratch_end(uid: str) -> str:
    return "---END_" + "SCRATCHPAD" + ".md-" + uid


@contextlib.contextmanager
def runaway_guard(seconds: float = 30):
    """Fail fast if an agent loop runs away instead of hanging the harness."""
    def _handler(signum, frame):
        raise AssertionError(
            "agent loop exceeded %ss — runaway loop? Check that the fake LLM "
            "queue ends with an 'exit' turn." % seconds
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Fake urllib (serves the OpenRouter /models catalog)
# ---------------------------------------------------------------------------

DEFAULT_CATALOG = []  # default: model not catalogued -> probe fallbacks


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_urlopen(catalog):
    """Return a fake ``urlopen`` that serves ``catalog`` as the /models JSON."""
    payload = json.dumps({"data": catalog}).encode("utf-8")

    def _urlopen(req, timeout=None, *a, **kw):
        return _FakeHTTPResponse(payload)

    return _urlopen


# ---------------------------------------------------------------------------
# Shared base TestCase
# ---------------------------------------------------------------------------

class E2ECase(unittest.TestCase):
    """Common offline harness: repo-local CWD, patched net seams, clean cache."""

    def setUp(self):
        self._chdir_cm = chdir_repo_tmp(prefix="e2e")
        self.tmpdir = self._chdir_cm.__enter__()

        self.stdout_buf = io.StringIO()
        self._stdout_cm = contextlib.redirect_stdout(self.stdout_buf)
        self._stdout_cm.__enter__()

        # History lands in this test's throwaway CWD.
        self.hist_file = os.path.join(self.tmpdir, ".bash_agent_tmp", "history.json")
        self._hist_patch = mock.patch("bash_agent.context.HISTORY_FILE", self.hist_file)
        self._hist_patch.start()

        # --- Network seam #2: models catalog (offline) -------------------
        # Default catalog is empty (model unknown) -> probes use their safe
        # fallbacks, matching the reference integration test.
        self._urlopen_patch = mock.patch(
            "urllib.request.urlopen", make_urlopen(list(DEFAULT_CATALOG))
        )
        self._urlopen_patch.start()
        self._reset_models_cache()

        # --- Network seam #3: requests.post (search rerank) --------------
        self._post_calls = []
        self._post_patch = mock.patch("bash_agent.search.requests.post")
        self._post_mock = self._post_patch.start()
        self._post_mock.side_effect = self._fake_post

        # Remember / restore the LLM client cache (network seam #1).
        self._old_client = llm._CLIENT_CACHE.get("openrouter")
        llm._CLIENT_CACHE.pop("openrouter", None)

    def tearDown(self):
        if self._old_client is None:
            llm._CLIENT_CACHE.pop("openrouter", None)
        else:
            llm._CLIENT_CACHE["openrouter"] = self._old_client
        self._reset_models_cache()
        self._post_patch.stop()
        self._urlopen_patch.stop()
        self._hist_patch.stop()
        self._stdout_cm.__exit__(None, None, None)
        self._chdir_cm.__exit__(None, None, None)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _reset_models_cache():
        agent_module._MODELS_CACHE["data"] = None
        agent_module._MODELS_CACHE["fetched_at"] = 0.0

    def _fake_post(self, *a, **kw):
        self._post_calls.append((a, kw))
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"results": []}
        return resp

    def set_catalog(self, models):
        """Install a canned /models catalog for the next construction."""
        self._urlopen_patch.stop()
        self._urlopen_patch = mock.patch(
            "urllib.request.urlopen", make_urlopen(models)
        )
        self._urlopen_patch.start()
        self._reset_models_cache()

    def seed_llm(self, responses, embeddings=None):
        fake = FakeLLMClient(responses=responses, embedding_responses=embeddings)
        llm._CLIENT_CACHE["openrouter"] = fake
        return fake

    def make_agent(self, uuid_str=None, fake_sandbox=True, **kwargs):
        """Construct an Agent with the network probes served offline."""
        agent_module._MODELS_CACHE["data"] = None
        agent_module._MODELS_CACHE["fetched_at"] = 0.0
        with mock.patch("bash_agent.agent.cleanup_tmp_folder", return_value=None):
            agent = Agent(**kwargs)
        if fake_sandbox:
            agent.sandbox = FakeSandbox()
        if uuid_str is not None:
            agent.uuid = uuid_str
            agent.context.uuid = uuid_str
            try:
                agent.sandbox.uuid = uuid_str
            except Exception:
                pass
        return agent

    def history(self, agent=None):
        return (agent or self.agent).context.history

    def messages_text(self, agent=None):
        """Concatenate all message text for coarse assertions."""
        parts = []
        for m in self.history(agent):
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(
                    b.get("text", "") for b in c if isinstance(b, dict)
                )
            parts.append(c)
        return "\n".join(parts)

    def on_disk(self):
        with open(self.hist_file) as f:
            return json.load(f)


__all__ = [
    "E2ECase",
    "runaway_guard",
    "cmd_start",
    "cmd_end",
    "out_start",
    "out_end",
    "scratch_start",
    "scratch_end",
    "make_urlopen",
    "DEFAULT_CATALOG",
    "FakeLLMClient",
    "FakeSandbox",
    "make_fake_response",
    "bash_block",
    "python_block",
    "output_block",
    "attached_image_block",
    "attached_audio_block",
    "chdir_repo_tmp",
    "deterministic_count_tokens",
    "DeterministicTokenCounts",
    "_stub_model_reasoning_info",
    "_stub_model_context_info",
    "bwrap_available",
]
