"""listen/transcribe.py — speech in, local Whisper, no network.

Frozen CLI (see PLAN_v2):
    python listen/transcribe.py [--device N]
        push-to-talk capture (SPACE starts, SPACE stops), 16 kHz mono,
        transcribes with mlx-whisper, prints ONE line to stdout, exit 0.
    python listen/transcribe.py --text "some query"
        bypasses the microphone entirely and echoes the string through the
        identical downstream path. This is the stage fallback for a loud
        room, not a debug flag.

Nothing but the transcript goes to stdout — F pipes stdout straight into
decide.py. All status/diagnostic output goes to stderr.
"""
import argparse
import sys

import numpy as np

SAMPLE_RATE = 16000
WHISPER_MODEL = "mlx-community/whisper-small.en-mlx"
DEFAULT_DEVICE = 0  # MIC_DEFAULT_DEVICE_INDEX per coordination/FACTS.md


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def record_push_to_talk(device):
    """Hold/press SPACE to start, press SPACE again to stop. 'q' aborts."""
    import termios
    import tty

    import sounddevice as sd

    info = sd.query_devices(device)
    log(f"[listen] device {device}: {info['name']}")
    log("[listen] press SPACE to start recording, SPACE again to stop (q aborts)")

    frames = []
    state = {"recording": False, "done": False, "aborted": False}

    def callback(indata, frame_count, time_info, status):
        if status:
            log(f"[listen] stream status: {status}")
        if state["recording"]:
            frames.append(indata.copy())

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    )
    try:
        tty.setcbreak(fd)
        stream.start()
        while not state["done"]:
            ch = sys.stdin.read(1)
            if ch == " ":
                if not state["recording"]:
                    state["recording"] = True
                    log("[listen] recording...")
                else:
                    state["recording"] = False
                    state["done"] = True
                    log("[listen] stopped")
            elif ch.lower() == "q":
                state["done"] = True
                state["aborted"] = True
                log("[listen] aborted")
    finally:
        stream.stop()
        stream.close()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    if state["aborted"] or not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten().astype(np.float32)


def record_fixed_seconds(device, seconds):
    """Non-interactive capture used by the smoke test: record for a fixed duration."""
    import sounddevice as sd

    info = sd.query_devices(device)
    log(f"[listen] device {device}: {info['name']}")
    log(f"[listen] recording {seconds}s (fixed-duration smoke-test capture)...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", device=device)
    sd.wait()
    log("[listen] stopped")
    return audio.flatten().astype(np.float32)


def transcribe_audio(audio):
    if audio.size == 0:
        return ""
    import mlx_whisper

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, verbose=False)
    return " ".join(result["text"].strip().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="bypass the mic; echo this string through the same path (stage fallback)")
    ap.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="sounddevice input index (FACTS: 0 = MacBook Pro Microphone)")
    ap.add_argument("--seconds", type=float, default=None,
                     help="record a fixed duration instead of push-to-talk (used by smoke.py)")
    args = ap.parse_args()

    if args.text is not None:
        transcript = " ".join(args.text.strip().split())
    else:
        if args.seconds is not None:
            audio = record_fixed_seconds(args.device, args.seconds)
        else:
            audio = record_push_to_talk(args.device)
        transcript = transcribe_audio(audio)

    print(transcript)
    return 0


if __name__ == "__main__":
    sys.exit(main())
