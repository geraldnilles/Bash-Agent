import os
from bash_agent.prompts import COPY_PROJECT_PREFIX, COPY_PROJECT_SUFFIX
import subprocess

def cleanup_tmp_folder():
    """Delete all contents of .bash_agent_tmp/ folder."""
    tmp_path = os.path.abspath(".bash_agent_tmp")
    if os.path.exists(tmp_path):
        for item in os.listdir(tmp_path):
            item_path = os.path.join(tmp_path, item)
            if item in [ "SCRATCHPAD.md", "vim_prompt.tmp", "ROLE.md", "embeddings.json", "search_disabled", "history.json", "clipboard_blacklist.txt", "config.json" ]:
                continue  # Keep the scratchpad
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)



def get_clipboard_content():
    """Read content from clipboard using wl-paste (Wayland) or xclip (X11)."""
    try:
        # Try wl-paste first (Wayland)
        result = subprocess.run(["wl-paste"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    
    try:
        # Try xclip (X11)
        result = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    
    raise RuntimeError("Could not read from clipboard. Please ensure wl-paste (Wayland) or xclip (X11) is installed.")




def is_binary_file(file_path):
    """
    Determines if a file is binary based on its extension or content inspection.
    """
    # 1. Quick extension check for common binary files/images
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico',
        '.pdf', '.zip', '.tar', '.gz', '.mp3', '.opus', '.mp4', '.exe', '.dll', '.so', '.bin'
    }
    _, ext = os.path.splitext(file_path.lower())
    if ext in binary_extensions:
        return True

    # 2. Fallback: Inspect the first 1024 bytes for a null byte
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        # If we can't read it, treat it cautiously
        return True


def _parse_glob_list(s):
    """Split a comma/space separated glob list, trimming empties."""
    parts = s.replace(",", " ").split()
    return [p for p in parts if p]


def copy_project_to_clipboard(file_paths=None, ignore=None, include=None):
    """
    Copies project files to the clipboard as XML-like tagged blocks.

    Selection is unified around gitignore-style glob patterns (the same
    syntax .gitignore uses). include ("--files"/"--include"), ignore,
    `.gitignore` contents, and the clipboard blacklist all use git's exact
    matching rules:

      *   matches any characters except '/'
      **  crosses directory boundaries
      ?   matches a single non-'/' character
      [..] character class; [!/^..] negates
      trailing '/'  -> directory-only
      leading '/' or pattern containing '/' -> anchored to repo root
      pattern without '/' -> matches at any depth
      !pattern      -> negation

    The ignore layers (.gitignore, clipboard blacklist, user --ignore) are
    MERGED into ONE ordered rule list with the user's patterns LAST and
    git's last-match-wins applied across all of them. The layers are treated
    as a single source, NOT as independent precedence tiers: a user
    `!pattern` (a later rule) can re-include a file that `.gitignore`
    excluded, just as a later .gitignore rule can re-include an earlier one.
    This is intentionally simpler than git's per-file-then-per-source
    hierarchy: with one list, the rule you wrote LAST always decides.

    Args:
        file_paths: Glob patterns of files to COPY (gitignore syntax), e.g.
                    "src/**/*.py,README.md". A file is copied if it matches
                    ANY pattern. If None, copies the whole project.
        ignore:     Glob patterns to EXCLUDE, e.g. "*.log,build/". Applied on
                    top of `.gitignore` and the clipboard blacklist. '!'
                    re-includes.
        include:    Alias for file_paths (the --include CLI flag).
    """
    from bash_agent.ignore import GitIgnoreMatcher, patterns_from_file

    output = []

    # --- Internals (never overridable, even by '!') -------------------------
    always = GitIgnoreMatcher([".git/", ".bash_agent_tmp/"])

    # --- Per-layer matchers (kept for diagnostics) --------------------------
    gitignore_pats = patterns_from_file(".gitignore")
    blacklist_pats = patterns_from_file(os.path.join(
        os.getcwd(), ".bash_agent_tmp", "clipboard_blacklist.txt"))
    user_pats = _parse_glob_list(ignore or "")

    gitignore = GitIgnoreMatcher(gitignore_pats)
    blacklist = GitIgnoreMatcher(blacklist_pats)
    usrign = GitIgnoreMatcher(user_pats)

    # --- Merged matcher: last-match-wins across the whole flattening -------
    # Order: .gitignore, blacklist, then user --ignore (user wins ties, and a
    # user '!pattern' can re-include a file .gitignore excluded).
    ignore_rules = GitIgnoreMatcher(gitignore_pats + blacklist_pats + user_pats)

    # --- Optional include masks --------------------------------------------
    if file_paths is None:
        file_paths = include
    inc = None
    inc_pats = None
    if file_paths:
        inc_pats = _parse_glob_list(file_paths)
        inc = GitIgnoreMatcher(inc_pats, include_mode=True)
        print(f"Copying files matching: {inc_pats}")

    root_dir = os.getcwd()

    def ignored_reason(rel, is_dir):
        """'--ignore' | 'blacklist' | '.gitignore' | 'always' | None.

        Decision comes from the MERGED matcher (last-match-wins); the layer
        named is the highest-precedence one that alone also excludes it,
        which is consistent because user patterns are appended LAST.
        """
        if always.match(rel, is_dir) is False:
            return "always"
        if ignore_rules.match(rel, is_dir) is not False:
            return None  # not excluded (e.g. user '!' re-included it)
        for label, m in (("--ignore", usrign),
                         ("blacklist", blacklist),
                         (".gitignore", gitignore)):
            if m.match(rel, is_dir) is False:
                return label
        return None

    def emit_file(rel_path, full_path):
        if is_binary_file(full_path):
            print(f"Warning: Binary file skipped: {rel_path}")
            return False
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Binary file skipped: {rel_path}")
            return False
        except PermissionError:
            print(f"Warning: Permission denied: {rel_path}")
            return False
        output.append(f'<file path="{rel_path}">\n{content}\n</file>')
        return True

    # --- 0. Directory tree (full-project copies only) -----------------------
    if inc is None:
        tree_filter_args = ["tree", "--gitignore"]
        for p in sorted(user_pats):
            tree_filter_args += ["-I", p]
        if ignore:
            tree_filter_args += ["--prune"]
        try:
            tree_result = subprocess.run(tree_filter_args,
                                         capture_output=True, text=True)
            output.append("=== DIRECTORY TREE ===")
            output.append(tree_result.stdout if tree_result.returncode == 0
                          else "(tree command failed)")
        except FileNotFoundError:
            output.append("=== DIRECTORY TREE ===")
            output.append("(tree command not installed)")

    # --- 1. Walk & emit ------------------------------------------------------
    files_seen = []
    for root, dirs, files in os.walk(root_dir):
        kept = []
        for d in dirs:
            rel_d = os.path.relpath(os.path.join(root, d), root_dir)
            if ignored_reason(rel_d, is_dir=True) is not None:
                continue
            kept.append(d)
        dirs[:] = kept

        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), root_dir)
            files_seen.append(rel_path)

            if inc is not None and inc.match(rel_path, is_dir=False) is not True:
                continue

            reason = ignored_reason(rel_path, is_dir=False)
            if reason is None:
                emit_file(rel_path, os.path.join(root, file))
                continue

            # Diagnostics: blacklist surfaces in BOTH modes; .gitignore and
            # --ignore only in subset mode (so full-project stays quiet).
            if reason == "blacklist":
                print(f"Info: Ignored by clipboard blacklist: {rel_path}")
            elif inc is not None:
                if reason == ".gitignore":
                    print(f"Warning: Ignored by .gitignore pattern: {rel_path}")
                elif reason == "--ignore":
                    print(f"Warning: Ignored by --ignore pattern: {rel_path}")

    # Unmatched include patterns -> "File not found" warning (subset mode).
    if inc is not None:
        for pat in inc_pats:
            m = GitIgnoreMatcher([pat], include_mode=True)
            if not any(m.match(f, is_dir=False) is True for f in files_seen):
                print(f"Warning: File not found: {pat}")

    full_text = (COPY_PROJECT_PREFIX + "\n\n" + "\n\n".join(output)
                 + "\n\n" + COPY_PROJECT_SUFFIX)

    # --- 2. Copy to clipboard ------------------------------------------------
    try:
        subprocess.run(["wl-copy"], input=full_text, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=full_text, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"Error copying to clipboard: {e}")


def get_vim_prompt(prompt_text: str = "OBJECTIVE:") -> str:
    """Launch vim to get user input using a persistent file in .bash_agent_tmp."""
    tmp_dir = os.path.abspath(".bash_agent_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "vim_prompt.tmp")
    
    # Write initial content only if the file doesn't exist
    if not os.path.exists(tmp_file):
        with open(tmp_file, 'w') as f:
            f.write(f"{prompt_text}")
    
    # Launch vim with clean settings
    subprocess.run(
        ["vim", "-c", "set noswapfile", "-c", "set spell", tmp_file],
    )
    
    # Read the content
    with open(tmp_file, 'r') as f:
        result = f.read()
    
    return result



