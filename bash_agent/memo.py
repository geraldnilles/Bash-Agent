#!/usr/bin/env python3
"""
Record a voice memo from your default PipeWire microphone and save as Opus.

This script wraps pw-record + ffmpeg to:
1. List available audio sources (so you can pick one)
2. Record a WAV via PipeWire (pw-record)
3. Convert to Opus
4. Save to the current working directory with ISO timestamp + duration in filename

Usage:
  ./memo.py                        # interactive: choose source, record until Ctrl+C
  ./memo.py --duration 30          # record for 30 seconds
  ./memo.py --list                 # list audio source names
  ./memo.py --list-all             # list ALL sources (including monitors)
  ./memo.py --source "Mic1"        # record from a specific source (substring match)
  ./memo.py --source 54            # record from source node ID 54
  ./memo.py --output mynote.opus   # specify output path
  ./memo.py --format mp3           # output as MP3 (128kbps mono) instead of Opus
"""
import argparse
import datetime
import os
import subprocess
import sys
import signal
import shutil
import time
import tempfile

def get_sources(include_monitors=False):
    """Return list of (node_id, name) from pactl source list."""
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True, timeout=10
    )
    sources = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(None, 3)
        if len(parts) < 2:
            continue
        node_id = parts[0]
        name = parts[1]
        # Skip monitor sinks unless explicitly asked — they're output loopbacks, not mics
        if not include_monitors and ".monitor" in name:
            continue
        sources.append((node_id, name))
    return sources

def find_source(source_spec, sources):
    """
    Match source_spec against available sources.
    - If it's a number, try matching node_id.
    - Otherwise, case-insensitive substring match on the name.
    Returns (node_id, name) or raises ValueError.
    """
    # Try numeric match first
    if source_spec.isdigit():
        for node_id, name in sources:
            if node_id == source_spec:
                return (node_id, name)

    # Substring match on name
    matches = []
    spec_lower = source_spec.lower()
    for node_id, name in sources:
        if spec_lower in name.lower():
            matches.append((node_id, name))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print("Multiple sources match your query. Please be more specific:")
        for node_id, name in matches:
            print(f"  {node_id}: {name}")
        print("\nUsing the first match:", matches[0][1])
        return matches[0]
    else:
        raise ValueError(f"No source found matching '{source_spec}'")

def format_timestamp():
    """Return ISO timestamp string like 2026-04-28T153045."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")


def format_duration(seconds):
    """Format seconds as a string like 2m30s or 1h05m12s."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"

def pick_source_interactive(sources):
    """Let the user pick a source from a numbered list. Returns (node_id, name)."""
    print("Available audio input sources:")
    for i, (node_id, name) in enumerate(sources, 1):
        # Pretty-print: extract the last meaningful part of the name
        if "HiFi__" in name:
            short = name.split("HiFi__")[1].replace("__", " > ").replace("_source", "")
        else:
            short = name.split(".")[-1] if "." in name else name
        print(f"  {i:2d}. [{node_id}] {short}")
    print()

    while True:
        try:
            choice = input(f"Choose source [1-{len(sources)}, default=1]: ").strip()
            if choice == "":
                return sources[0]
            idx = int(choice) - 1
            if 0 <= idx < len(sources):
                return sources[idx]
            print(f"Please enter a number between 1 and {len(sources)}.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print()
            return sources[0]

def record_wav(output_wav, source_name=None, duration=None):
    """
    Record audio using pw-record.
    Returns actual recording duration in seconds on success, None on failure.
    """
    cmd = ["pw-record"]

    # Use --target with the pactl source name to select the mic
    if source_name:
        cmd.extend(["--target", source_name])

    cmd.extend([
        "--rate", "48000",
        "--channels", "2",          # mono is fine for voice memos
        "--format", "s16",
        "--volume", "1.0",
        output_wav
    ])

    print(f"🎤 Recording... ", end="", flush=True)
    if duration:
        print(f"({duration}s)", flush=True)
    else:
        print("(press Ctrl+C to stop)", flush=True)

    start_time = time.time()
    try:
        if duration:
            subprocess.run(cmd, timeout=duration, check=True,
                          capture_output=True, text=True)
        else:
            subprocess.run(cmd, check=True,
                          capture_output=True, text=True)
        elapsed = time.time() - start_time
        print("✅ Recording complete.")
        return elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print("✅ Recording complete (time limit reached).")
        return elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        if e.returncode == -signal.SIGINT or e.returncode == 1:
            # pw-record exits with 1/128+SIGINT when Ctrl+C'd
            print("✅ Recording stopped.")
            return elapsed
        print(f"❌ Recording failed: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return None
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("⏹️  Stopped.")
        return elapsed

def convert_to_opus(wav_path, opus_path):
    """
    Convert WAV to Opus using ffmpeg with voice-optimized settings.
    16 kbps Opus at 16kHz mono is the sweet spot: tiny files with zero
    transcription quality loss.
    Returns True on success, False on failure.
    """
    print(f"🔄 Converting to Opus (16k, 16kHz mono)... ", end="", flush=True)

    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", wav_path,
        "-c:a", "libopus",
        "-b:a", "16k",
        "-ar", "16000",
        "-ac", "1",
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-application", "voip",
        opus_path
    ], capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr.strip()}", file=sys.stderr)
        return False

    print("✅ Done.")
    return True

def convert_to_mp3(wav_path, mp3_path):
    """
    Convert WAV to MP3 using ffmpeg with voice-optimized settings.
    128 kbps MP3 at 16kHz mono.
    Returns True on success, False on failure.
    """
    print(f"🔄 Converting to MP3 (128k, 16kHz mono)... ", end="", flush=True)

    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", wav_path,
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        #"-ar", "16000",
        "-ac", "1",
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
        mp3_path
    ], capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr.strip()}", file=sys.stderr)
        return False

    print("✅ Done.")
    return True


def fmt_size(path):
    """Return human-readable file size."""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def main():
    parser = argparse.ArgumentParser(
        description="Record a voice memo and save as Opus. Requires PipeWire and ffmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available audio sources (microphones) and exit")
    parser.add_argument("--list-all", "-L", action="store_true",
                        help="List ALL sources (including output monitors) and exit")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="Source name or ID to record from (default: pick interactively)")
    parser.add_argument("--duration", "-d", type=int, default=None,
                        help="Recording duration in seconds (default: record until Ctrl+C)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output Opus path (default: <cwd>/ISOdate_duration.opus)")
    parser.add_argument("--format", "-f", type=str, default="opus", choices=["opus", "mp3"],
                        help="Output format: opus (default) or mp3 (128kbps mono)")
    args = parser.parse_args()

    # Check dependencies
    if not shutil.which("pw-record"):
        print("❌ Error: 'pw-record' not found. Is PipeWire installed?", file=sys.stderr)
        print("   Install with: sudo apt install pipewire-utils", file=sys.stderr)
        sys.exit(1)
    if not shutil.which("ffmpeg"):
        print("❌ Error: 'ffmpeg' not found.", file=sys.stderr)
        print("   Install with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Get available sources
    sources = get_sources(include_monitors=args.list_all)
    if not sources:
        print("❌ No audio input sources found. Check your microphone.", file=sys.stderr)
        sys.exit(1)

    # --list / --list-all modes
    if args.list or args.list_all:
        header = "ALL audio sources (microphones + monitors):" if args.list_all else "Available audio input sources (microphones):"
        print(header)
        for node_id, name in sources:
            # Pretty-print name
            if "HiFi__" in name:
                short = name.split("HiFi__")[1].replace("__", " > ").replace("_source", "").replace("_sink.monitor", " (monitor)")
            else:
                short = name
            print(f"  {node_id}: {short}")
        sys.exit(0)

    # Resolve source
    if args.source:
        try:
            node_id, source_name = find_source(args.source, sources)
        except ValueError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if len(sources) == 1:
            # Only one source — auto-select it
            node_id, source_name = sources[0]
        else:
            # Interactive pick
            node_id, source_name = pick_source_interactive(sources)

    # Pretty-print the selected source
    if "HiFi__" in source_name:
        friendly_name = source_name.split("HiFi__")[1].replace("__", " > ").replace("_source", "")
    else:
        friendly_name = source_name
    print(f"🎙️  Source: {friendly_name} (ID {node_id})")

    # Determine output path
    if args.output:
        opus_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(opus_path) or ".", exist_ok=True)
        has_custom_output = True
    else:
        has_custom_output = False

    # Record to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # Record
        actual_duration = record_wav(wav_path, source_name=source_name, duration=args.duration)
        if actual_duration is None:
            sys.exit(1)

        # Build filename if not custom output
        if not has_custom_output:
            ts = format_timestamp()
            dur = format_duration(actual_duration)
            ext = "mp3" if args.format == "mp3" else "opus"
            opus_path = os.path.join(os.getcwd(), f"{ts}_{dur}.{ext}")

        # Convert
        if args.format == "mp3":
            if not convert_to_mp3(wav_path, opus_path):
                sys.exit(1)
        else:
            if not convert_to_opus(wav_path, opus_path):
                sys.exit(1)

        # Print result
        opus_size = fmt_size(opus_path)
        print(f"\n📁 Saved: {opus_path}")
        print(f"📏 Size:  {opus_size}")
        print(f"🔊 Try:   transcribe {opus_path}")


    finally:
        # Clean up temp WAV
        if os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == "__main__":
    main()
