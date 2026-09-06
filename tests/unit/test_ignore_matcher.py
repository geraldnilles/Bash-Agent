"""
Group 9 — gitignore-style matcher (bash_agent.ignore).

T-40  GitIgnoreMatcher  – the one matcher behind --include/--ignore/.gitignore
                         and the clipboard blacklist (P1)

The copy-project feature unified every selection knob on gitignore-syntax
globs.  This pins Git's matching semantics so the whole stack depends on a
single, well-tested rule engine rather than ad-hoc fnmatch.

Semantics covered (mirroring git check-ignore):
  * '*'  must not cross '/'
  * '**' crosses directory boundaries (incl. zero segments)
  * '?'  matches a single non-'/' char
  * '[..]' character class; '[!..]' negation; rooted on non-'/' too
  * trailing '/'  -> directory-only match
  * leading '/'   -> anchored to repo root
  * pattern containing '/'   -> anchored (no implicit basename match)
  * pattern without '/' -> matches basename OR any segment (any depth)
  * LAST matching rule wins; '!' re-includes
"""

import os
import unittest

from bash_agent.ignore import GitIgnoreMatcher, _translate, patterns_from_file


class TranslateTest(unittest.TestCase):
    def test_comments_and_blanks_are_skipped(self):
        self.assertIsNone(_translate(""))
        self.assertIsNone(_translate("   "))
        self.assertIsNone(_translate("# comment"))

    def test_basic_negation_and_dir_only_flags(self):
        self.assertEqual(_translate("*.log"),
                         ("(?:^|/)[^/]*\\.log$", False, False))
        self.assertEqual(_translate("!keep.txt")[1], True)     # negated
        self.assertEqual(_translate("build/")[2], True)        # dir_only
        self.assertEqual(_translate("/root.txt")[2], False)

    def test_star_does_not_cross_slash(self):
        m = GitIgnoreMatcher(["*.py"])
        self.assertIs(m.match("a.py", False), False)
        self.assertIs(m.match("sub/a.py", False), False)   # any depth
        self.assertIs(m.match("sub/b.py.txt", False), None)

    def test_double_star_crosses_directory_boundaries(self):
        m = GitIgnoreMatcher(["src/**/gen.c"])
        self.assertIs(m.match("src/gen.c", False), False)          # zero dirs
        self.assertIs(m.match("src/a/gen.c", False), False)
        self.assertIs(m.match("src/a/b/gen.c", False), False)
        self.assertIs(m.match("other/gen.c", False), None)         # not under src/

    def test_question_mark_single_non_slash(self):
        m = GitIgnoreMatcher(["a?c"])
        self.assertIs(m.match("abc", False), False)
        self.assertIs(m.match("a/c", False), None)

    def test_bracket_class(self):
        m = GitIgnoreMatcher(["[ab].txt", "[!c].txt"])
        self.assertIs(m.match("a.txt", False), False)
        self.assertIs(m.match("b.txt", False), False)
        # [!c] matches any single non-'c' char -> d.txt excluded, c.txt NOT.
        self.assertIs(m.match("c.txt", False), None)
        self.assertIs(m.match("d.txt", False), False)

    def test_dir_only_ignores_files_with_same_name(self):
        m = GitIgnoreMatcher(["build/"])
        self.assertIs(m.match("build", True), False)      # the dir itself
        self.assertIs(m.match("build/app.o", False), None)  # file itself: no rule
        # (the walker prunes the dir, so children never surface)

    def test_anchored_only_matches_root(self):
        m = GitIgnoreMatcher(["/rooted.txt"])
        self.assertIs(m.match("rooted.txt", False), False)
        self.assertIs(m.match("sub/rooted.txt", False), None)

    def test_slash_anchors_basename_pattern(self):
        m = GitIgnoreMatcher(["sub/deep.txt"])
        self.assertIs(m.match("sub/deep.txt", False), False)
        self.assertIs(m.match("other/deep.txt", False), None)  # only under sub/

    def test_unanchored_matches_any_segment(self):
        m = GitIgnoreMatcher(["deep.txt"])
        self.assertIs(m.match("deep.txt", False), False)
        self.assertIs(m.match("a/deep.txt", False), False)
        self.assertIs(m.match("a/b/deep.txt", False), False)
        self.assertIs(m.match("a/b/deep.txt.bak", False), None)

    def test_last_match_wins_and_negation_reincludes(self):
        m = GitIgnoreMatcher(["*.log", "!important.log", "deep/*.txt"])
        self.assertIs(m.match("x.log", False), False)
        self.assertIs(m.match("important.log", False), True)   # ! re-includes
        self.assertIs(m.match("deep/a.txt", False), False)

    def test_negation_after_exclusion(self):
        # git rule: last rule wins, so a later ! on a wider pattern re-includes
        m = GitIgnoreMatcher(["*.log", "!important.log"])
        self.assertIs(m.match("important.log", False), True)
        # re-include then re-exclude => excluded again
        m2 = GitIgnoreMatcher(["*.log", "!important.log", "important.log"])
        self.assertIs(m2.match("important.log", False), False)


class IncludeModeTest(unittest.TestCase):
    """Same engine, flipped polarity: plain pattern INCLUDES (True)."""

    def test_plain_pattern_includes(self):
        m = GitIgnoreMatcher(["src/**/*.py"], include_mode=True)
        self.assertIs(m.match("src/a.py", False), True)
        self.assertIs(m.match("src/a/b.py", False), True)
        self.assertIs(m.match("src/a.txt", False), None)
        self.assertIs(m.match("other/a.py", False), None)

    def test_negation_in_include_mode_excludes(self):
        m = GitIgnoreMatcher(["*.py", "!setup.py"], include_mode=True)
        self.assertIs(m.match("main.py", False), True)
        self.assertIs(m.match("setup.py", False), False)   # ! excludes in include mode


class PatternsFromFileTest(unittest.TestCase):
    def test_reads_non_comment_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".gitignore",
                                         delete=False) as f:
            f.write("# c\n*.log\n\nbuild/\n")
            name = f.name
        try:
            self.assertEqual(patterns_from_file(name), ["*.log", "build/"])
        finally:
            os.unlink(name)

    def test_missing_file_returns_empty(self):
        self.assertEqual(patterns_from_file("/no/such/file"), [])


if __name__ == "__main__":
    unittest.main()
