#!/usr/bin/env python3

import sys
import os
import argparse
from bash_agent.agent import Agent
from bash_agent.config import DEFAULT_MODEL, DEFAULT_AUDIO_MODEL
from bash_agent.config_file import load_config
from bash_agent.tokenizer import count_tokens
from bash_agent.utils import copy_project_to_clipboard, get_clipboard_content

def _parse_token_budget_arg(value: str) -> str:
    """argparse receiver: validate a --token-budget spec, return raw string.

    Accepts absolute token counts ("327680", "256K", "1.5M") or a percentage
    of the model context window ("25%"). Invalid values abort with a clean
    usage error (SystemExit 2) instead of a traceback later.
    """
    from bash_agent.token_budget import parse_token_budget
    try:
        parse_token_budget(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Bash Agent")
    parser.add_argument("-m", dest="message", type=str, help="User message to send (instead of reading from stdin)")
    parser.add_argument("-p", "--paste", action="store_true", help="Read the first user message from clipboard instead of stdin")
    parser.add_argument("-x", "--execute", action="store_true", help="Paste clipboard into SCRATCHPAD.md and instruct the agent to execute the plan")
    parser.add_argument("-s", "--clear-scratchpad", action="store_true", help="Clear the SCRATCHPAD.md file before running the agent")
    parser.add_argument("-k", "--keep-tmp", action="store_true", help="Keep the contents of .bash_agent_tmp/ folder")
    parser.add_argument("-d", "--debug", action="store_true", help="to /tmp/bash_agent_log.txt on every LLM ping")
    parser.add_argument("--model", type=str, default=None, help="OpenRouter model name (overrides OPENROUTER_MODEL env var)")
    parser.add_argument("--audio", action="store_true", help="Use the audio-enabled default model (xiaomi/mimo-v2.5) instead of the standard default. Ignored when --model/config.json/OPENROUTER_MODEL select a model.")
    parser.add_argument("--reasoning-effort", type=str, choices=['none', 'minimal', 'low', 'medium', 'high', 'default'], default=None, help="Reasoning effort (none, minimal, low, medium, high, or default to use model's built-in default)")
    parser.add_argument("--max-tokens", type=int, default=None, help='Override max output tokens for LLM')
    parser.add_argument(
        "--token-budget", type=_parse_token_budget_arg, default=None, metavar="SPEC",
        help="Context-window token budget: absolute tokens (e.g. 327680, 256K, 1.5M) or a percentage of the model's context window ending in %% (e.g. 25%%). Defaults to 25%% of the model window.",
    )
    parser.add_argument("-t", "--timeout", type=int, default=None, help="Session default command timeout in seconds (default: 60). A single BASH/PYTHON block may override this up to 600s via a first-line `# timeout: N` directive.")
    parser.add_argument("--no-sfx", action="store_true", help="Disable subtle sound effects (also BAGENT_SFX=0 / BAGENT_NO_SFX=1)")
    parser.add_argument("-b", "--budget", type=float, default=0.10, help="Total session cost budget in USD (default: 0.10)")
    parser.add_argument("--commit", action="store_true", help="Resume the last session and create a git commit message for the changes.")
    parser.add_argument("-r","--resume", action="store_true", help="Restore the history from the last agent conversation and append the new prompt.")
    parser.add_argument("-c", "--copy-project", action="store_true",
                        help="Copy project to clipboard and exit")
    parser.add_argument("--include", "--files", dest="include", type=str,
                        default=None,
                        help="Glob patterns of files to copy (gitignore syntax; "
                             "e.g. 'src/**/*.py,README.md'). Omit to copy the "
                             "whole project. --files is an alias.")
    parser.add_argument("-i", "--ignore", type=str, default=None,
                        help="Glob patterns to exclude from the copy "
                             "(gitignore syntax; e.g. '*.log,build/'). Applied "
                             "on top of .gitignore and the clipboard blacklist. "
                             "'!' negates.")
    return parser.parse_args()

def _resolve_model(cli_model, audio=False):
    """Resolve the model to count tokens against for --copy-project.

    Mirrors Agent.__init__'s precedence exactly:
        CLI --model > .bash_agent_tmp/config.json > OPENROUTER_MODEL >
        (DEFAULT_AUDIO_MODEL when audio=True) > DEFAULT_MODEL
    Returns the model slug string (never None).
    """
    if cli_model:
        return cli_model
    try:
        cfg = load_config()
        if "model" in cfg:
            return cfg["model"]
    except Exception:
        pass
    env_model = os.environ.get("OPENROUTER_MODEL")
    if env_model:
        return env_model
    if audio:
        return DEFAULT_AUDIO_MODEL
    return DEFAULT_MODEL


def main():
    args = parse_args()

    # Handle --commit flag: resume last session with commit message
    if args.commit:
        args.resume = True
        args.message = "Commit the change."
    
    # Check for --copy-project flag first (exits before LLM requests)
    if args.copy_project:
        text = copy_project_to_clipboard(args.include, ignore=args.ignore)
        # Print the token count of the entire copied string as the selected
        # model's tokenizer would see it (CLI > config.json > env > default).
        if isinstance(text, str):
            model = _resolve_model(args.model, audio=args.audio)
            try:
                n = count_tokens(text, model=model)
            except Exception:
                n = None
            if n is not None:
                print(f"Copied project token count ({model}): {n:,} tokens")
            else:
                print("Copied project token count: unavailable")
        print("Project copied to clipboard. Exiting.")
        sys.exit(0)
    
    # Get the initial task based on arguments
    if args.execute:
        try:
            clipboard_data = get_clipboard_content()
            tmp_dir = os.path.abspath(".bash_agent_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            scratchpad_path = os.path.join(tmp_dir, "SCRATCHPAD.md")
            with open(scratchpad_path, "w", encoding="utf-8") as f:
                f.write(clipboard_data)
                
            print("[System] Clipboard content written to SCRATCHPAD.md")
            initial_task = "OBJECTIVE: Please read the plan detailed in the SCRATCHPAD.md file and execute it step-by-step."
        except RuntimeError as e:
            print(f"[Clipboard Error] {e}")
            sys.exit(1)
    elif args.message:
        initial_task = args.message
    elif args.paste:
        try:
            initial_task = get_clipboard_content()
            print("[Reading from clipboard]")
        except RuntimeError as e:
            print(f"[Clipboard Error] {e}")
            initial_task = None
    else:
        initial_task = None
    
    # Clear SCRATCHPAD.md if requested
    if args.clear_scratchpad:
        tmp_dir = os.path.abspath(".bash_agent_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        scratchpad_path = os.path.join(tmp_dir, "SCRATCHPAD.md")
        with open(scratchpad_path, "w", encoding="utf-8") as f:
            f.write("")
        print("[System] SCRATCHPAD.md cleared.")
    
    if args.no_sfx:
        os.environ["BAGENT_NO_SFX"] = "1"
    agent = Agent(keep_tmp=args.keep_tmp, debug=args.debug, model=args.model, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens, token_budget=args.token_budget, timeout=args.timeout, resume=args.resume, budget=args.budget, audio=args.audio)
    agent.run(initial_task)

if __name__ == "__main__":
    main()
