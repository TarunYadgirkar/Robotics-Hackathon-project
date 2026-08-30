"""Frame sources for live feedback ingest: real webcam (spacebar start/stop) and a
synthetic fallback used only when the camera cannot be opened (WEBCAM_STATUS in
FACTS.md is currently `blocked` - macOS TCC camera-permission denial, not a
hardware-absence signal).
"""
import os
import sys
import time

import cv2
import numpy as np

W, H = 640, 360
FPS = 2.0
SPACE = 32
QUIT = ord("q")


def open_camera(device=None):
    if device is None:
        device = int(os.environ.get("CAMERA_INDEX", "0"))
    """Returns an opened cv2.VideoCapture, or None if it cannot be opened.

    Mirrors the exact check used in the P0 probe (`cv2.VideoCapture(0).isOpened()`)
    so callers can distinguish "no camera / no permission" from "camera works".
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def record_take_interactive(cap, max_seconds=10.0, w=W, h=H, target_fps=FPS, take_label=""):
    """Spacebar start/stop capture from an already-opened camera.

    Blocks on a preview window. First SPACE starts recording; a second SPACE (or
    the max_seconds cap) stops it. 'q' aborts the whole take (raises KeyboardInterrupt).
    Returns a list of RGB uint8 frames at (h, w, 3), sampled at target_fps.
    """
    win = f"feedback ingest {take_label} - SPACE start/stop, q quit"
    print(f"[capture] take {take_label}: press SPACE to start recording (q aborts)", file=sys.stderr)
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed while waiting to start")
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == SPACE:
            break
        if key == QUIT:
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("aborted before recording started")

    print(f"[capture] take {take_label}: recording (SPACE to stop early, cap {max_seconds:.0f}s)", file=sys.stderr)
    frames = []
    t0 = time.time()
    interval = 1.0 / target_fps
    next_due = t0
    while time.time() - t0 < max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if now >= next_due:
            resized = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            next_due += interval
        if key == SPACE:
            break
        if key == QUIT:
            cv2.destroyWindow(win)
            raise KeyboardInterrupt(f"aborted during take {take_label}")
    cv2.destroyWindow(win)
    print(f"[capture] take {take_label}: {len(frames)} frames captured", file=sys.stderr)
    return frames


def record_take_auto(cap, duration_s=5.0, w=W, h=H, target_fps=FPS):
    """Non-interactive capture for --selftest: grabs frames for duration_s, no GUI, no keypress."""
    frames = []
    t0 = time.time()
    interval = 1.0 / target_fps
    next_due = t0
    while time.time() - t0 < duration_s:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now >= next_due:
            resized = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            next_due += interval
    return frames


def synthetic_frame_sequence(duration_s=5.0, w=W, h=H, target_fps=FPS, seed=0):
    """Deterministic synthetic frames for the no-webcam selftest branch.

    Draws a moving skin-tone blob against a dark background. This exercises the
    identical MediaPipe extraction call path end to end (frame in, landmarks or
    no-detection out, table written) without a camera. It is not expected to
    produce real hand detections - MediaPipe's hand landmark model looks for
    actual hand geometry, not a colored disc - and callers must not treat a
    zero detection_rate on this path as a failure.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(round(duration_s * target_fps)))
    frames = []
    for i in range(n):
        frame = np.full((h, w, 3), 25, np.uint8)
        noise = rng.integers(0, 8, size=(h, w, 3), dtype=np.uint8)
        frame = frame + noise
        cx = int(w * (0.2 + 0.6 * i / max(1, n - 1)))
        cy = h // 2
        cv2.circle(frame, (cx, cy), 45, (196, 156, 128), -1)
        for k in range(5):
            fx = cx - 30 + k * 15
            fy = cy - 60
            cv2.circle(frame, (fx, fy), 12, (196, 156, 128), -1)
        frames.append(frame)
    return frames
