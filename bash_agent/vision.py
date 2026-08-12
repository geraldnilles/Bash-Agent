#!/usr/bin/env python3
"""
Vision tool that processes an image using OpenRouter's Gemma 4 31b model.
"""
import os
import sys
import argparse
import base64
import io
from PIL import Image

from bash_agent import config
from bash_agent import llm
from bash_agent.config import MAX_PIXELS

# Constants
MODEL_ID = "google/gemma-4-31b-it"
MODEL_ID = "xiaomi/mimo-v2.5"
DEFAULT_PROMPT = "Convert the image into Markdown text."


def encode_image(image_path):
    """Convert image to PNG and encode to base64 string."""
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def check_image_size(image_path):
    """Check if image total pixels are within the limit."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            total_pixels = width * height
            if total_pixels > MAX_PIXELS:
                print(f"Error: Image is too large ({width}x{height} = {total_pixels} pixels). Maximum allowed is {MAX_PIXELS} pixels.", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"Error opening image file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process an image using Gemma 4 31b")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_PROMPT, help="Custom prompt for the model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File {args.image} not found.", file=sys.stderr)
        sys.exit(1)

    # Validate image size
    check_image_size(args.image)

    # Encode image
    base64_image = encode_image(args.image)

    # --- MULTIMODAL SANDBOX MODE ---
    # When executed inside the agent sandbox with BASH_AGENT_MULTIMODAL=1 and a
    # session UUID, emit the image as a fenced base64 payload on stdout. The
    # agent harness parses these fences and attaches the image to its next LLM
    # request. No HTTP call to OpenRouter is needed.
    is_sandbox_multimodal = os.environ.get("BASH_AGENT_MULTIMODAL") == "1"
    session_uuid = os.environ.get("BASH_AGENT_UUID")
    if is_sandbox_multimodal and session_uuid:
        data_url = f"data:image/png;base64,{base64_image}"
        print(f"---START_ATTACHED_IMAGE-{session_uuid}---")
        print(data_url)
        print(f"---END_ATTACHED_IMAGE-{session_uuid}---")
        print(f"Image '{args.image}' attached to conversation context.")
        sys.exit(0)

    # --- TEXT-ONLY FALLBACK / STANDALONE CLI ---
    try:
        response = llm.create_chat_completion(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
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

if __name__ == "__main__":
    main()
