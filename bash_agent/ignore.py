"""Gitignore-style pattern matching shared by the --copy-project feature.

All of the selection knobs (--include, --ignore, .gitignore contents, and the
clipboard blacklist) are matched with the SAME rules git uses in .gitignore:

  *  matches any characters except '/'
  ** matches across directory boundaries
  ?  matches any single character except '/'
  [...] character class ('[!...]' acts as negation)
  pattern/      trailing slash means "directory only"
  /pattern or a  pattern containing a '/' is anchored to the repo root
  pattern without '/' matches at any depth (basename or any segment)
  !pattern      negates: "do NOT ignore (re-include)"

Matching follows git's rules: the LAST matching pattern wins, and negation
(!) re-includes what an earlier pattern excluded.
"""

import os
import re

__all__ = ["GitIgnoreMatcher"]


def _translate(pattern):
    """Convert one gitignore pattern into (regex, negated, dir_only).

    Returns None for blank lines and '#'-comments.
    """
    line = pattern.rstrip()
    if not line or line.lstrip().startswith("#"):
        return None

    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:].lstrip()

    dir_only = False
    if line.endswith("/"):
        dir_only = True
        line = line.rstrip("/")

    anchored = False
    if line.startswith("/"):
        anchored = True
        line = line[1:]  # leading slash just anchors; content kept
    elif "/" in line:
        anchored = True

    # Convert glob tokens to a regex (escaped literal for everything else).
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "*":
            if i + 1 < n and line[i + 1] == "*":
                # "**" or "**/" -> cross segment boundaries
                if i + 2 < n and line[i + 2] == "/":
                    out.append("(?:.*/)?")
                    i += 3
                else:
                    out.append(".*")
                    i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            # Grab the full class "[...]" (with optional leading [!]).
            j = i + 1
            if j < n and line[j] in ("!", "^"):
                j += 1
            if j < n and line[j] == "]":
                j += 1
            while j < n and line[j] != "]":
                j += 1
            if j < n:
                cls = line[i + 1:j]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                elif cls.startswith("^"):
                    cls = "\\" + cls
                out.append("[" + cls + "]")
                i = j + 1
            else:
                out.append(re.escape(c))
                i += 1
        else:
            out.append(re.escape(c))
            i += 1

    body = "".join(out)
    if anchored:
        regex = "^" + body + "$"
    else:
        # No '/': matches at the basename or any segment boundary.
        regex = "(?:^|/)" + body + "$"
    return regex, negated, dir_only


class GitIgnoreMatcher:
    """Compiled set of gitignore-style patterns with git's last-match-wins."""

    def __init__(self, patterns, include_mode=False):
        self._rules = []  # list of (compiled_regex, negated, dir_only)
        self.include_mode = include_mode
        for p in patterns:
            if p is None:
                continue
            p = p.strip()
            if not p:
                continue
            t = _translate(p)
            if t is None:
                continue
            regex, negated, dir_only = t
            self._rules.append((re.compile(regex), negated, dir_only))

    def __bool__(self):
        return bool(self._rules)

    def match(self, relpath, is_dir=False):
        """Return True (include), False (exclude), or None (no pattern matched).

        Git rule: the LAST matching pattern decides. A dir-only pattern only
        applies when is_dir is True.
        """
        relpath = relpath.replace(os.sep, "/")
        result = None
        for regex, negated, dir_only in self._rules:
            if dir_only and not is_dir:
                continue
            if regex.search(relpath):
                # ignore mode: plain pattern excludes (False), ! re-includes (True)
                # include mode: plain pattern includes (True), ! excludes (False)
                result = not negated if self.include_mode else negated
        return result


def patterns_from_file(path):
    """Read patterns from a gitignore-style file (skips blanks/comments)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
