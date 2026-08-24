import os
from bash_agent.prompts import COPY_PROJECT_PREFIX, COPY_PROJECT_SUFFIX
import subprocess
import fnmatch

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


def _matches_any_ignore(path, patterns):
    """Return True if path (or its basename) matches any glob pattern."""
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(os.path.basename(path), p) for p in patterns)

def copy_project_to_clipboard(file_paths=None, ignore=None):
    """
    Copies files in the working directory to the clipboard, 
    respecting .gitignore and excluding .git, .bash_agent_tmp, and
    any additional user-supplied ignore patterns.

    Args:
        file_paths: Optional comma-separated string of specific file paths to copy.
                   If None, copies the entire directory.
        ignore: Optional comma-separated string of glob patterns (files or
                directories) to exclude from the copy, e.g. "*.log,node_modules".
                Patterns are matched against the path relative to the working
                directory and against the basename.
    """
    import subprocess
    import fnmatch

    output = []

    # Load optional clipboard blacklist patterns
    blacklist_patterns = set()
    blacklist_path = os.path.abspath(os.path.join(os.getcwd(), ".bash_agent_tmp", "clipboard_blacklist.txt"))
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    blacklist_patterns.add(line)

    # Parse extra ignore patterns (e.g. "--ignore '*.log,node_modules'")
    extra_ignore_patterns = set()
    if ignore:
        for p in ignore.split(','):
            p = p.strip()
            if p:
                extra_ignore_patterns.add(p)
    
    # Parse file_paths if provided
    specific_files = None
    if file_paths:
        specific_files = [p.strip() for p in file_paths.split(',') if p.strip()]
        print(f"Copying specific files: {specific_files}")

    ignore_patterns = {".git", ".bash_agent_tmp"} | extra_ignore_patterns

    # 0. Get directory tree structure using tree --gitignore (only if copying entire project)
    if specific_files is None:
        # Build tree args, adding explicit --prune for each extra ignore pattern
        tree_filter_args = ["tree", "--gitignore"]
        for p in sorted(extra_ignore_patterns):
            tree_filter_args += ["-I", p]
        if extra_ignore_patterns:
            tree_filter_args += ["--prune"]
        try:
            tree_result = subprocess.run(tree_filter_args, capture_output=True, text=True)
            if tree_result.returncode == 0:
                output.append("=== DIRECTORY TREE ===")
                output.append(tree_result.stdout)
            else:
                output.append("=== DIRECTORY TREE ===")
                output.append("(tree command not available or failed)")
        except FileNotFoundError:
            output.append("=== DIRECTORY TREE ===")
            output.append("(tree command not installed)")

    # 1. Determine ignore patterns from .gitignore
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_patterns.add(line)

    root_dir = os.getcwd()

    # 2. Copy files
    if specific_files:
        # Copy only the specified files
        for path in specific_files:
            full_path = os.path.abspath(os.path.join(root_dir, path))
            rel_path = os.path.relpath(full_path, root_dir)
            
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {rel_path}")
                continue
            
            if not os.path.isfile(full_path):
                print(f"Warning: Not a file: {rel_path}")
                continue
            
            # Check if file matches any ignore pattern (user --ignore first)
            if _matches_any_ignore(rel_path, extra_ignore_patterns):
                print(f"Warning: Ignored by --ignore pattern: {rel_path}")
                continue

            # Check if file matches any .gitignore/blacklist pattern
            if _matches_any_ignore(rel_path, ignore_patterns):
                print(f"Warning: Ignored by .gitignore pattern: {rel_path}")
                continue

            # Check if file matches any clipboard blacklist pattern
            if _matches_any_ignore(rel_path, blacklist_patterns):
                print(f"Info: Ignored by clipboard blacklist: {rel_path}")
                continue
            
            # New binary guard check
            if is_binary_file(full_path):
                print(f"Warning: Binary file skipped: {rel_path}")
                continue
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                output.append(f'<file path="{rel_path}">\n{content}\n</file>')
            except UnicodeDecodeError:
                print(f"Warning: Binary file skipped: {rel_path}")
                continue
            except PermissionError:
                print(f"Warning: Permission denied: {rel_path}")
                continue
    else:
        # Walk the directory (original behavior)
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to prevent os.walk from descending into
            # ignored directories (.gitignore + user --ignore patterns)
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in ignore_patterns) and not _matches_any_ignore(os.path.relpath(os.path.join(root, d), root_dir), extra_ignore_patterns)]
            
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                
                # Check if file matches any user --ignore pattern
                if _matches_any_ignore(rel_path, extra_ignore_patterns):
                    continue

                # Check if file matches any .gitignore pattern
                if _matches_any_ignore(rel_path, ignore_patterns):
                    continue

                # Check if file matches any clipboard blacklist pattern
                if _matches_any_ignore(rel_path, blacklist_patterns):
                    print(f"Info: Ignored by clipboard blacklist: {rel_path}")
                    continue
                
                # New binary guard check
                if is_binary_file(full_path):
                    print(f"Warning: Binary file skipped: {rel_path}")
                    continue
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    output.append(f'<file path="{rel_path}">\n{content}\n</file>')
                except (UnicodeDecodeError, PermissionError):
                    # Skip binary files or inaccessible files
                    continue

    full_text = COPY_PROJECT_PREFIX + "\n\n" + "\n\n".join(output) + "\n\n" + COPY_PROJECT_SUFFIX

    # 3. Copy to clipboard
    try:
        # Try wl-copy first
        subprocess.run(["wl-copy"], input=full_text, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            # Fallback to xclip
            subprocess.run(["xclip", "-selection", "clipboard"], input=full_text, text=True, check=True)
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



