"""
Shared test infrastructure for Bash Agent — Group 0.

Implements:
  T-00a  chdir_tmp           – isolated temp CWD context manager
  T-00b  fenced-block builders (bash_block, python_block, output_block)
  T-00c  FakeLLMClient + cache seeding helpers
  T-00d  FakeSandbox

Also provides _make_agent() helper used by downstream tests (T-01 etc.)
to construct an Agent without network or systemd side effects.

All helpers are pure offline fakes — no network, no systemd.
"""
from __future__ import annotations

import os
import re
import uuid
import tempfile
import contextlib
from collections import deque
from typing import Any, Dict, List, Tuple, Optional
from unittest import mock

# ---------------------------------------------------------------------------
# T-00a — chdir_tmp
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def chdir_tmp():
    """
    Context manager that creates a temporary directory, chdirs into it,
    yields the path, then restores the original CWD and cleans up.

    Usage:
        with chdir_tmp() as tmpdir:
            # CWD is now tmpdir
            open("file.txt", "w").write("hi")
        # CWD restored, tmpdir deleted

    Implementation notes (per TEST_PLAN):
      * Uses tempfile.TemporaryDirectory() + os.chdir()
      * Modules that use import-time constants (config.HISTORY_FILE) must be
        patched separately; chdir alone is not sufficient for those.
    """
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(original)


# Backwards-compat alias: some tests may expect a class or different name.
# Provide ChdirTmp as an alternative context manager class.
class ChdirTmp:
    """Class-based alternative to chdir_tmp() — also usable as `with ChdirTmp():`."""

    def __init__(self):
        self._tmp = None
        self._orig = None
        self.tmpdir = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self._orig = os.getcwd()
        os.chdir(self.tmpdir)
        return self.tmpdir

    def __exit__(self, *exc):
        os.chdir(self._orig)
        self._tmp.cleanup()
        return False


# ---------------------------------------------------------------------------
# T-00b — Fenced-block builders
# ---------------------------------------------------------------------------

def bash_block(uuid_str: str, script: str) -> str:
    """Return a protocol-valid BASH command block for the given uuid/script."""
    return f"---START_BASH_COMMAND-{uuid_str}---\n{script}\n---END_BASH_COMMAND-{uuid_str}---"


def python_block(uuid_str: str, code: str) -> str:
    """Return a protocol-valid PYTHON command block for the given uuid/code."""
    return f"---START_PYTHON_COMMAND-{uuid_str}---\n{code}\n---END_PYTHON_COMMAND-{uuid_str}---"


def output_block(
    uuid_str: str,
    exit_code: int,
    text: str,
    cmd_type: str = "BASH",
    visible: int = 100,
) -> str:
    """
    Return a protocol-valid OUTPUT block.

    Mirrors Agent._format_output() fence format:
      ---START_{TYPE}_OUTPUT-EXIT_CODE_{code}-VISIBLE_{visible}%-{uuid}---
      {text}
      ---END_{TYPE}_OUTPUT-{uuid}---

    Defaults to BASH / 100% to match the most common test usage.
    Tests that need PYTHON or truncation can pass explicit arguments.
    """
    cmd_type = cmd_type.upper()
    return (
        f"---START_{cmd_type}_OUTPUT-EXIT_CODE_{exit_code}-VISIBLE_{visible}%-{uuid_str}---\n"
        f"{text}\n"
        f"---END_{cmd_type}_OUTPUT-{uuid_str}---"
    )


def attached_image_block(uuid_str: str, data_url: str) -> str:
    """
    Return a protocol-valid ATTACHED_IMAGE payload fence, byte-for-byte the
    emission shape of bash_agent/vision.py in multimodal sandbox mode:

        ---START_ATTACHED_IMAGE-{uuid}---
        {data_url}
        ---END_ATTACHED_IMAGE-{uuid}---

    Agent._execute_script scans sandbox stdout for these fences, strips the
    payloads, and queues {"url": ...} dicts onto _pending_multimodal_images;
    _commit_execution_feedback then converts them into image_url content
    parts. Used by T-14 (and later T-38).
    """
    return (
        f"---START_ATTACHED_IMAGE-{uuid_str}---\n"
        f"{data_url}\n"
        f"---END_ATTACHED_IMAGE-{uuid_str}---"
    )


# ---------------------------------------------------------------------------
# T-00c — FakeLLMClient + cache seeding
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost: float = 0.0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost


class _FakeMessage:
    def __init__(self, content: Optional[str], reasoning: Optional[str] = None):
        self.content = content
        self.reasoning = reasoning
        # model_extra is used by some OpenRouter reasoning flows
        self.model_extra = {"reasoning": reasoning} if reasoning else None


class _FakeChoice:
    def __init__(
        self,
        content: Optional[str],
        finish_reason: str = "stop",
        reasoning: Optional[str] = None,
    ):
        self.message = _FakeMessage(content, reasoning=reasoning)
        self.finish_reason = finish_reason


class _FakeResponse:
    """
    Minimal ChatCompletion-shaped object.

    Supports:
      - response.choices[0].message.content
      - response.choices[0].finish_reason
      - response.choices[0].message.reasoning
      - response.usage.cost / prompt_tokens / completion_tokens
      - response.model_dump()  -> dict with usage.cost etc.
      - response.model_dump_json(indent=2) (used in agent error logging)
    """

    def __init__(
        self,
        content: Optional[str] = "",
        finish_reason: str = "stop",
        cost: float = 0.0,
        prompt_tokens: int = 5,
        completion_tokens: int = 5,
        provider: Optional[str] = None,
        reasoning: Optional[str] = None,
        usage: Optional[Any] = None,
    ):
        self.choices = [_FakeChoice(content, finish_reason=finish_reason, reasoning=reasoning)]
        if usage is not None:
            self.usage = usage
        else:
            self.usage = _FakeUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
        self._provider = provider
        # Keep raw values for model_dump
        self._cost = cost
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._content = content
        self._finish_reason = finish_reason

    def model_dump(self) -> Dict[str, Any]:
        # Be tolerant: if usage is an object with attributes, extract them
        if hasattr(self.usage, "cost"):
            cost = getattr(self.usage, "cost", self._cost)
            prompt = getattr(self.usage, "prompt_tokens", self._prompt_tokens)
            completion = getattr(self.usage, "completion_tokens", self._completion_tokens)
        elif isinstance(self.usage, dict):
            cost = self.usage.get("cost", self._cost)
            prompt = self.usage.get("prompt_tokens", self._prompt_tokens)
            completion = self.usage.get("completion_tokens", self._completion_tokens)
        else:
            cost = self._cost
            prompt = self._prompt_tokens
            completion = self._completion_tokens
        return {
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost": cost,
                "total_tokens": (prompt or 0) + (completion or 0),
            },
            "provider": self._provider,
        }

    def model_dump_json(self, indent: int = None) -> str:
        import json as _json

        return _json.dumps(self.model_dump(), indent=indent)


class _FakeCompletions:
    def __init__(self, parent: "FakeLLMClient"):
        self._parent = parent

    def create(self, **kwargs):
        # Record call for assertion on extra_body / reasoning / provider
        # calls and create_calls are aliases to the same list, so append once
        self._parent.calls.append(kwargs)
        if self._parent._responses:
            resp = self._parent._responses.popleft()
            # Allow callable factories
            if callable(resp):
                return resp(**kwargs)
            return resp
        # Default: echo-like response if no scripted response
        # Try to infer from last queued or return empty
        return _FakeResponse(content="", finish_reason="stop")


class _FakeChat:
    def __init__(self, parent: "FakeLLMClient"):
        self.completions = _FakeCompletions(parent)


class _FakeEmbeddings:
    def __init__(self, parent: "FakeLLMClient"):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.embedding_calls.append(kwargs)
        # Return fixed vectors if no scripted embeddings
        if self._parent._embedding_responses:
            resp = self._parent._embedding_responses.popleft()
            if callable(resp):
                return resp(**kwargs)
            return resp
        # Default fake embedding response
        texts = kwargs.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        # Return deterministic fake vectors (length 4)
        class _Vec:
            def __init__(self, idx):
                self.embedding = [0.1 * (idx + 1), 0.2 * (idx + 1), 0.3, 0.4]

        class _Resp:
            def __init__(self, n):
                self.data = [_Vec(i) for i in range(n)]

        return _Resp(len(texts))


class FakeLLMClient:
    """
    Fake OpenAI client for offline tests.

    Mirrors the subset of the OpenAI client used by bash_agent:
      client.chat.completions.create(**kwargs) -> ChatCompletion
      client.embeddings.create(**kwargs) -> Embeddings

    Seeding:
      from bash_agent import llm
      fake = FakeLLMClient(responses=[FakeResponse(...), ...])
      llm._CLIENT_CACHE["openrouter"] = fake
      llm._CLIENT_CACHE["gemini"] = fake  # if needed

    The fake records every call in `calls` / `create_calls` so tests can
    assert on extra_body (reasoning effort, provider whitelist, etc.)
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        embedding_responses: Optional[List[Any]] = None,
    ):
        self._responses: deque = deque(responses or [])
        self._embedding_responses: deque = deque(embedding_responses or [])
        self.calls: List[Dict[str, Any]] = []
        # Alias for backwards compatibility with different naming conventions
        self.create_calls: List[Dict[str, Any]] = self.calls
        self.embedding_calls: List[Dict[str, Any]] = []
        self.chat = _FakeChat(self)
        self.embeddings = _FakeEmbeddings(self)

    def queue_response(self, response: Any):
        """Append a scripted response to be returned on next create() call."""
        self._responses.append(response)

    def queue_responses(self, responses: List[Any]):
        for r in responses:
            self.queue_response(r)

    # Compatibility: some code may call client.chat.completions directly
    # Provide a helper to seed the llm cache
    def seed_cache(self, backend: str = "openrouter"):
        from bash_agent import llm as _llm

        _llm._CLIENT_CACHE[backend] = self
        return self


def make_fake_response(
    content: str = "",
    finish_reason: str = "stop",
    cost: float = 0.0,
    prompt_tokens: int = 5,
    completion_tokens: int = 5,
    reasoning: Optional[str] = None,
) -> _FakeResponse:
    """Convenience factory for a ChatCompletion-shaped fake response."""
    return _FakeResponse(
        content=content,
        finish_reason=finish_reason,
        cost=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning=reasoning,
    )


# Expose the internal classes for tests that need to customize
FakeResponse = _FakeResponse
FakeChoice = _FakeChoice
FakeMessage = _FakeMessage
FakeUsage = _FakeUsage


# ---------------------------------------------------------------------------
# T-00d — FakeSandbox
# ---------------------------------------------------------------------------

class FakeSandbox:
    """
    Offline stand-in for bash_agent.sandbox.Sandbox.

    Usage:
        agent = _make_agent()
        agent.sandbox = FakeSandbox(execute_result=(0, "hello"))
        # or with per-script logic:
        def my_exec(script):
            return (0, f"ran: {script}")
        agent.sandbox = FakeSandbox(execute_result=my_exec)

    Records every script passed to execute / execute_python for assertions.
    """

    def __init__(
        self,
        execute_result: Any = None,
        execute_python_result: Any = None,
        approved_write_paths: Optional[List[str]] = None,
    ):
        # execute_result can be:
        #   - None              -> default (0, "")
        #   - tuple[int,str]    -> always return that
        #   - callable(script) -> (exit_code, output)
        #   - list of tuples    -> popped sequentially
        self._execute_result = execute_result
        self._execute_python_result = execute_python_result
        self._execute_queue: deque = deque()
        self._execute_python_queue: deque = deque()
        if isinstance(execute_result, list):
            self._execute_queue = deque(execute_result)
            self._execute_result = None
        if isinstance(execute_python_result, list):
            self._execute_python_queue = deque(execute_python_result)
            self._execute_python_result = None

        self.executed_scripts: List[str] = []
        self.executed_python_scripts: List[str] = []
        # Unified alias expected by some tests: `scripts` / `calls`
        self.scripts: List[str] = self.executed_scripts
        self.calls: List[str] = self.executed_scripts
        self.approved_write_paths: List[str] = approved_write_paths if approved_write_paths is not None else [os.path.abspath(".")]

        # For request_write simulation
        self._request_write_results: Dict[str, Tuple[bool, str]] = {}

    def _resolve_result(self, result, queue, script: str):
        if queue:
            # Queued list-of-results mode
            item = queue.popleft()
            if callable(item):
                return item(script)
            return item
        if result is None:
            return (0, "")
        if callable(result):
            return result(script)
        return result

    def execute(self, script: str) -> Tuple[int, str]:
        self.executed_scripts.append(script)
        return self._resolve_result(self._execute_result, self._execute_queue, script)

    def execute_python(self, code: str) -> Tuple[int, str]:
        self.executed_python_scripts.append(code)
        return self._resolve_result(self._execute_python_result, self._execute_python_queue, code)

    def request_write(self, path: str) -> Tuple[bool, str]:
        abs_path = os.path.abspath(path)
        # Check if a stubbed result exists for this path
        if abs_path in self._request_write_results:
            return self._request_write_results[abs_path]
        if path in self._request_write_results:
            return self._request_write_results[path]
        # Default: approve and record
        self.approved_write_paths.append(abs_path)
        return True, "Write access granted."

    def stub_request_write(self, path: str, result: Tuple[bool, str]):
        """Pre-program request_write for a given path."""
        self._request_write_results[os.path.abspath(path)] = result
        self._request_write_results[path] = result

    def queue_execute(self, exit_code: int, output: str):
        """Queue a specific (exit_code, output) for next execute() call."""
        self._execute_queue.append((exit_code, output))

    def queue_execute_python(self, exit_code: int, output: str):
        self._execute_python_queue.append((exit_code, output))


# ---------------------------------------------------------------------------
# Shared _make_agent helper (used by T-01 and other downstream tests)
# ---------------------------------------------------------------------------

def _make_agent(uuid_str: Optional[str] = None, **agent_kwargs) -> Any:
    """
    Construct a bash_agent.agent.Agent without network or systemd effects.

    - Patches Agent._check_model_capabilities to avoid urllib probe
    - Swaps in a FakeSandbox after construction
    - Optionally sets a deterministic UUID
    - Ensures context/history isolation (uses chdir_tmp-friendly paths)

    Returns the Agent instance with .sandbox replaced by FakeSandbox.
    """
    from bash_agent.agent import Agent

    # Default: no network probe
    patch_target = "bash_agent.agent.Agent._check_model_capabilities"
    with mock.patch(patch_target, return_value=None):
        # Also avoid filesystem pollution from cleanup_tmp_folder during construction
        # by patching it; caller can opt out by passing keep_tmp=True behavior,
        # but we patch by default to keep tests hermetic unless inside chdir_tmp.
        with mock.patch("bash_agent.agent.cleanup_tmp_folder", return_value=None):
            agent = Agent(**agent_kwargs)
    # Swap sandbox
    fake_sandbox = FakeSandbox()
    agent.sandbox = fake_sandbox
    # Optionally force a known UUID (must also update context uuid for consistency)
    if uuid_str is not None:
        agent.uuid = uuid_str
        # ContextManager uuid is separate; keep them in sync for fence matching
        try:
            agent.context.uuid = uuid_str
        except Exception:
            pass
        # Sandbox uuid should also match
        try:
            agent.sandbox.uuid = uuid_str
        except Exception:
            pass
    # Also ensure cleanup_tmp_folder etc. do not delete the temp dir when using chdir_tmp
    return agent


# Public alias — some downstream tests import `make_agent`
make_agent = _make_agent


__all__ = [
    "chdir_tmp",
    "ChdirTmp",
    "bash_block",
    "python_block",
    "output_block",
    "attached_image_block",
    "FakeLLMClient",
    "FakeResponse",
    "FakeUsage",
    "FakeChoice",
    "FakeMessage",
    "make_fake_response",
    "FakeSandbox",
    "_make_agent",
    "make_agent",
]


# ---------------------------------------------------------------------------
# Integration-test helper — repo-local scratch CWD
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def chdir_repo_tmp(prefix: str = "t30"):
    """
    Context manager that chdirs into a fresh scratch directory under the
    REPO's .bash_agent_tmp/ folder (gitignored, auto-cleanable) instead of
    /tmp.

    Why this exists (T-30): systemd-run with PrivateTmp=yes mounts a fresh
    tmpfs over /tmp inside the sandbox, so --working-directory= or
    ReadWritePaths= pointing under /tmp cannot resolve -> namespace setup
    fails (exit 226/200). Production always runs from a real project dir, so
    integration tests must too. The directory is removed on exit.
    """
    import shutil as _shutil

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    base = os.path.join(repo_root, ".bash_agent_tmp", f"test-{prefix}")
    _shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)

    original = os.getcwd()
    try:
        os.chdir(base)
        yield base
    finally:
        os.chdir(original)
        _shutil.rmtree(base, ignore_errors=True)


__all__.append("chdir_repo_tmp")


def systemd_user_bus_available() -> bool:
    """
    True when systemd-run exists AND a user-session round-trip succeeds.
    Used as skipUnless guard by all Group 7 integration tests.
    """
    import shutil as _shutil
    import subprocess as _sp

    if not _shutil.which("systemd-run"):
        return False
    try:
        r = _sp.run(
            ["systemd-run", "--user", "--quiet", "--wait", "--collect",
             "--pipe", "/bin/true"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


__all__.append("systemd_user_bus_available")
