#!/usr/bin/env python3
"""
Transcribe audio using OpenRouter's Gemini 3 Flash Preview model.
Converts audio to text via an LLM with audio input capabilities.
"""
import os
import sys
import argparse
import base64
import subprocess
import tempfile

from bash_agent import config
from bash_agent import llm

# Constants
MODEL_ID = "google/gemini-3-flash-preview"
#MODEL_ID = "xiaomi/mimo-v2.5"
#MODEL_ID = "mistralai/voxtral-small-24b-2507"
#MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_PROMPT = "Transcribe the audio. Skip filler words (um, uh, er, like, you know, etc.), false starts, and other non-informative content. Focus on capturing the meaningful content and key points. Return only the cleaned transcription, with no additional text, commentary, or explanation."


def get_audio_format(file_path):
    """Determine audio format from file extension. Accepts any file type."""
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    return ext if ext else "unknown"


def encode_audio(audio_path):
    """Read audio file and encode to base64 string."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def check_file_size(audio_path, max_mb=50):
    """Check if file size is within reasonable limits."""
    size_bytes = os.path.getsize(audio_path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_mb:
        print(f"Error: Audio file is too large ({size_mb:.1f} MB). Maximum allowed is {max_mb} MB.", file=sys.stderr)
        sys.exit(1)


def convert_to_mp3(audio_path):
    """Convert audio file to mono 128kbps MP3 using ffmpeg."""
    # Create a temporary file with .mp3 extension
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        cmd = [
            "ffmpeg", "-y",  # -y to overwrite output file if exists
            "-i", audio_path,
            "-ac", "1",      # mono
            "-b:a", "128k",  # 128 kbps bitrate
            "-map", "0:a:0", # take first audio stream only (ignore video)
            "-vn",           # no video
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting audio: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        return tmp_path
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during audio conversion: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using an LLM on OpenRouter",
        usage="%(prog)s [-h] [-p PROMPT] [-m MODEL] [--max-size MAX_SIZE] [-c CONTEXT [CONTEXT ...] [--]] audio",
        epilog="Note: If you provide context files before the audio file, use '--' to separate them so the parser knows where the file list ends.\nExample: transcribe -c file1.md file2.md -- audio.opus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Custom prompt instructing the model how to handle the audio")
    parser.add_argument("-m", "--model", type=str, default=MODEL_ID,
                        help=f"Model to use (default: {MODEL_ID})")
    parser.add_argument("--max-size", type=int, default=50,
                        help="Maximum file size in MB (default: 50)")
    parser.add_argument("-c", "--context", type=str, nargs="+", default=None,
                        help="One or more text files to use as context (e.g., reference transcripts, style guides). "
                             "Contents are formatted as XML and appended to the prompt.")
    parser.add_argument("audio", type=str, help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: File {args.audio} not found.", file=sys.stderr)
        sys.exit(1)

    # Check file size
    check_file_size(args.audio, args.max_size)

    # Convert input file to MP3 for broader backend compatibility
    args.audio = convert_to_mp3(args.audio)
    audio_format = "mp3"
    cleanup_temp = True

    # Encode audio
    base64_audio = encode_audio(args.audio)

    # Load context files if provided
    prompt_text = args.prompt
    if args.context:
        context_parts = []
        for ctx_path in args.context:
            if not os.path.exists(ctx_path):
                print(f"Error: Context file {ctx_path} not found.", file=sys.stderr)
                sys.exit(1)
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctx_content = f.read()
            context_parts.append(f'<file path="{ctx_path}">\n{ctx_content}\n</file>')
        context_xml = "\n".join(context_parts)
        prompt_text = (
            f"Transcribe the audio. Skip filler words (um, uh, er, like, you know, etc.), false starts, and other non-informative content. "
            f"Use the following reference files for correct spelling, terminology, and formatting. "
            f"Focus on capturing the meaningful content and key points. Return only the cleaned transcription, with no additional text, commentary, or explanation.\n\n"
            f"<context>\n{context_xml}\n</context>"
        )

    try:
        response = llm.create_chat_completion(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error during API request: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up temp file if conversion happened
        if cleanup_temp:
            try:
                if os.path.exists(args.audio):
                    os.unlink(args.audio)
            except Exception:
                pass

if __name__ == "__main__":
    main()
