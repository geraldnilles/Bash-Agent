"""Subtle sound effects — generated programmatically, no wav assets.

PipeWire-native: tries pw-play > paplay > aplay.
All playback is fire-and-forget on a daemon thread so the agent loop never blocks.
Set BAGENT_SFX=0 or BAGENT_NO_SFX=1 (or --no-sfx) to mute.
"""

import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
import wave

SAMPLE_RATE = 48000  # PipeWire default

def _enabled() -> bool:
    if os.environ.get("BAGENT_NO_SFX", "").strip().lower() in ("1","true","yes","on"):
        return False
    if os.environ.get("BAGENT_SFX", "").strip().lower() in ("0","false","off","no"):
        return False
    return True

def _player_cmd():
    for c in ("pw-play", "paplay", "aplay"):
        if shutil.which(c):
            return c
    return None

def _wav_bytes(samples, sample_rate=SAMPLE_RATE) -> bytes:
    """samples: iterable of float in [-1,1] -> 16-bit mono wav bytes"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        # convert
        import struct
        frames = struct.pack("<" + "h"*len(samples), *(max(-32767, min(32767, int(s*32767))) for s in samples))
        w.writeframes(frames)
    return buf.getvalue()

def _sine_samples(freq, duration, volume=0.2, sample_rate=SAMPLE_RATE, attack=0.005, release=0.03, decay_exp=None):
    n = int(sample_rate * duration)
    out = [0.0]*n
    for i in range(n):
        t = i / sample_rate
        # envelope: linear attack, sustain, exponential-ish release
        if t < attack:
            env = t / attack
        elif t > duration - release:
            env = max(0.0, (duration - t) / release)
            if decay_exp is not None:
                # extra exponential damp for click
                env = env * math.exp(-decay_exp * (t - (duration - release)))
        else:
            env = 1.0
            if decay_exp is not None:
                # full-duration exponential decay (click)
                env = math.exp(-decay_exp * t)
                # still apply attack
                if t < attack:
                    env *= (t/attack)
        out[i] = math.sin(2*math.pi*freq*t) * volume * env
    return out

def _mix(*tracks):
    if not tracks:
        return []
    L = max(len(t) for t in tracks)
    out = [0.0]*L
    for tr in tracks:
        for i, v in enumerate(tr):
            out[i] += v
    # normalize to avoid clipping
    peak = max(abs(v) for v in out) or 1.0
    if peak > 1.0:
        out = [v/peak for v in out]
    return out

def _click_samples():
    # soft subtle click: short high tick with fast exponential decay
    # 1200 Hz sine, 45ms, very quiet, sharp decay
    s = _sine_samples(freq=600, duration=0.045, volume=0.18, attack=0.001, release=0.035, decay_exp=90)
    # add faint second harmonic for woodiness
    s2 = _sine_samples(freq=1200, duration=0.045, volume=0.05, attack=0.001, release=0.035, decay_exp=110)
    return _mix(s, s2)

def _chime_samples():
    # pleasant chime: gentle arpeggio C6 -> E6 -> G6 with lingering chord
    # frequencies: C6 1046.5, E6 1318.5, G6 1568 (C major)
    # play as overlapping notes with soft attack
    sr = SAMPLE_RATE
    total = 0.85
    n = int(sr*total)
    out = [0.0]*n
    notes = [
        (1046.50, 0.00, 0.45, 0.22),  # C6
        #(1318.51, 0.12, 0.50, 0.20),  # E6 delayed
        (1567.98, 0.12, 0.55, 0.18),  # G6
    ]
    for freq, offset, dur, vol in notes:
        seg = _sine_samples(freq, dur, volume=vol, attack=0.015, release=0.25)
        # also add octave shimmer very quiet
        shimmer = _sine_samples(freq*2, dur, volume=vol*0.12, attack=0.02, release=0.25)
        seg = _mix(seg, shimmer)
        off = int(offset*sr)
        for i, v in enumerate(seg):
            if off+i < n:
                # gentle bell-like decay already via envelope; add slight overall fade
                out[off+i] += v
    # master fade + soft limiter
    for i in range(n):
        t = i/sr
        # final 150ms fade out
        if t > total - 0.15:
            out[i] *= max(0, (total - t)/0.15)
    peak = max(abs(v) for v in out) or 1.0
    # keep chime soft: scale to 0.35 peak
    out = [v/peak*0.35 for v in out]
    return out

_click_wav = None
_chime_wav = None

def _get_click_wav():
    global _click_wav
    if _click_wav is None:
        _click_wav = _wav_bytes(_click_samples())
    return _click_wav

def _get_chime_wav():
    global _chime_wav
    if _chime_wav is None:
        _chime_wav = _wav_bytes(_chime_samples())
    return _chime_wav

def _play_bytes(wav_bytes: bytes):
    player = _player_cmd()
    if not player:
        return
    # write to temp file; pw-play/paplay/aplay need a file
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes)
            path = tf.name
    except Exception:
        return
    def _run():
        try:
            # suppress stdout/stderr
            subprocess.run([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            pass
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    th = threading.Thread(target=_run, daemon=True)
    th.start()

def click():
    if not _enabled():
        return
    try:
        _play_bytes(_get_click_wav())
    except Exception:
        pass

def chime():
    if not _enabled():
        return
    try:
        _play_bytes(_get_chime_wav())
    except Exception:
        pass

# fire-and-forget with small delay variant for sync chime on exit
def chime_sync(timeout: float = 1.2):
    """Blocking chime for exit — waits briefly so sound finishes before process dies."""
    if not _enabled():
        return
    player = _player_cmd()
    if not player:
        return
    try:
        wav = _get_chime_wav()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav)
            path = tf.name
        try:
            subprocess.run([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        except Exception:
            pass
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception:
        pass

def main():
    """CLI harness for isolated testing: python -m bash_agent.sfx [--click] [--chime]"""
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Test bash_agent sfx (programmatic PipeWire tones)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--click", action="store_true", help="play only the subtle click")
    g.add_argument("--chime", action="store_true", help="play only the pleasant chime")
    parser.add_argument("--loop", type=int, default=0, help="repeat N times (0=once)")
    parser.add_argument("--delay", type=float, default=0.4, help="delay between click+chime in seconds (default 0.4)")
    parser.add_argument("--force", action="store_true", help="ignore BAGENT_SFX/BAGENT_NO_SFX mute env vars")
    args = parser.parse_args()

    # optionally force-enable even if muted
    if args.force:
        os.environ.pop("BAGENT_SFX", None)
        os.environ.pop("BAGENT_NO_SFX", None)

    if not _enabled():
        print(f"[sfx] muted (BAGENT_SFX={os.environ.get('BAGENT_SFX')!r} BAGENT_NO_SFX={os.environ.get('BAGENT_NO_SFX')!r}); use --force to override or unset those env vars.")
        # still show info
    player = _player_cmd()
    print(f"[sfx] player: {player or 'NONE (install pw-play/paplay/aplay)'}  sample_rate={SAMPLE_RATE}")
    print(f"[sfx] click: {len(_get_click_wav())} bytes  chime: {len(_get_chime_wav())} bytes")

    if player is None:
        print("[sfx] no audio player found — wav generation still works, just can't play.")
        return 0

    def play_once():
        if args.click:
            print("[sfx] playing click...")
            chime_sync.__doc__  # keep linter happy
            # use sync for testing so we actually hear it before exit
            # click is short; reuse chime_sync path but for click wav
            wav = _get_click_wav()
            # blocking play for click too
            import tempfile, subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav)
                path = tf.name
            try:
                subprocess.run([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            finally:
                try: os.unlink(path)
                except: pass
        elif args.chime:
            print("[sfx] playing chime...")
            chime_sync(timeout=1.5)
        else:
            print("[sfx] playing click...")
            # blocking click
            wav = _get_click_wav()
            import tempfile, subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav)
                path = tf.name
            try:
                subprocess.run([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            finally:
                try: os.unlink(path)
                except: pass
            time.sleep(args.delay)
            print("[sfx] playing chime...")
            chime_sync(timeout=1.5)
        print("[sfx] done.")

    if args.loop > 0:
        for i in range(args.loop):
            print(f"[sfx] loop {i+1}/{args.loop}")
            play_once()
            if i < args.loop - 1:
                time.sleep(0.6)
    else:
        play_once()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
