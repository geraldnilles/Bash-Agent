#!/usr/bin/env python3

import sys
import os
import argparse
from bash_agent.agent import Agent
from bash_agent.utils import copy_project_to_clipboard, get_clipboard_content

def parse_args():
    parser = argparse.ArgumentParser(description="Bash Agent")
    parser.add_argument("-m", dest="message", type=str, help="User message to send (instead of reading from stdin)")
    parser.add_argument("-p", "--paste", action="store_true", help="Read the first user message from clipboard instead of stdin")
    parser.add_argument("-x", "--execute", action="store_true", help="Paste clipboard into SCRATCHPAD.md and instruct the agent to execute the plan")
    parser.add_argument("-s", "--clear-scratchpad", action="store_true", help="Clear the SCRATCHPAD.md file before running the agent")
    parser.add_argument("-k", "--keep-tmp", action="store_true", help="Keep the contents of .bash_agent_tmp/ folder")
    parser.add_argument("-d", "--debug", action="store_true", help="to /tmp/bash_agent_log.txt on every LLM ping")
    parser.add_argument("--model", type=str, default=None, help="OpenRouter model name (overrides OPENROUTER_MODEL env var)")
    parser.add_argument("--reasoning-effort", type=str, choices=['none', 'minimal', 'low', 'medium', 'high', 'default'], default=None, help="Reasoning effort (none, minimal, low, medium, high, or default to use model's built-in default)")
    parser.add_argument("--max-tokens", type=int, default=None, help='Override max output tokens for LLM')
    parser.add_argument("-t", "--timeout", type=int, default=None, help="Command timeout in seconds (default: 60)")
    parser.add_argument("-b", "--budget", type=float, default=0.10, help="Total session cost budget in USD (default: 0.10)")
    parser.add_argument("--commit", action="store_true", help="Resume the last session and create a git commit message for the changes.")
    parser.add_argument("-r","--resume", action="store_true", help="Restore the history from the last agent conversation and append the new prompt.")
    parser.add_argument("-c", "--copy-project", action="store_true", help="Copy project to clipboard and exit")
    parser.add_argument("--files", type=str, default=None, help="Comma-separated list of file paths to copy (use with --copy-project)")
    parser.add_argument("-i", "--ignore", type=str, default=None, help="Comma-separated glob patterns of files/directories to exclude (use with --copy-project)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Handle --commit flag: resume last session with commit message
    if args.commit:
        args.resume = True
        args.message = "Commit the change."
    
    # Check for --copy-project flag first (exits before LLM requests)
    if args.copy_project:
        copy_project_to_clipboard(args.files, ignore=args.ignore)
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
    
    agent = Agent(keep_tmp=args.keep_tmp, debug=args.debug, model=args.model, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens, timeout=args.timeout, resume=args.resume, budget=args.budget)
    agent.run(initial_task)

if __name__ == "__main__":
    main()
