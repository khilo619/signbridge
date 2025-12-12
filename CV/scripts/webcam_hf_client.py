"""Webcam client that streams clips to a remote SignBridge I3D API (e.g. Hugging Face Space).

This script captures frames from a local webcam, maintains a sliding window,
periodically encodes that window as a short video clip, and sends it to a
remote FastAPI `/sign/predict` endpoint. Predictions are overlaid on the
webcam feed with basic debouncing and confidence thresholding.

Before running, set `HF_BASE_URL` to your deployed Space / API base URL.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import requests


# Deployed API / Hugging Face Space base URL (no trailing /docs)
# Docs live at HF_BASE_URL + "/docs"; the API itself is rooted at HF_BASE_URL.
# HF_BASE_URL = "https://khalood619-signbridge-demo-55.hf.space"  # 55-class demo (alt)
HF_BASE_URL = "https://khalood619-signbridge-api.hf.space"  # 100-class (87% model)

SIGN_PREDICT_URL = HF_BASE_URL.rstrip("/") + "/sign/predict"

# Real-time configuration
CLIP_FRAMES = 32          # number of frames per clip sent to the API
CLIP_FPS = 25.0           # nominal FPS when encoding the clip
TOP_K = 5                 # how many alternatives to request from the API
MIN_CONFIDENCE = 0.6      # minimum probability to treat a prediction as valid
# Cooldown between completed clips; controls how fast READY turns green again
REQUEST_INTERVAL = 1.5    # seconds between remote requests
MIN_MOTION_SCORE = 2.3    # motion threshold to START capturing a sign


@dataclass
class PredictionState:
    """Holds the current prediction state for overlay and debouncing."""

    current_gloss: Optional[str] = None
    current_prob: float = 0.0
    last_update: float = 0.0
    last_error: Optional[str] = None
    inflight: bool = False
    transcript: List[str] = field(default_factory=list)
    last_capture_len: int = 0  # number of frames in last captured clip

    def update_from_response(self, payload: dict) -> None:
        """Update internal state from the API JSON payload."""
        self.last_error = None

        gloss = payload.get("gloss")
        prob = float(payload.get("probability", 0.0))

        # Apply confidence threshold
        if gloss is None or prob < MIN_CONFIDENCE:
            self.current_gloss = None
            self.current_prob = prob
            self.last_update = time.time()
            return

        now = time.time()
        # Append to transcript only when the gloss changes and is confident
        if gloss != self.current_gloss:
            self.transcript.append(gloss)

        self.current_gloss = gloss
        self.current_prob = prob
        self.last_update = now


def encode_clip_to_temp_video(frames: List["cv2.Mat"], fps: float) -> str:
    """Encode a list of BGR frames into a temporary video file and return its path."""
    if not frames:
        raise ValueError("No frames provided to encode_clip_to_temp_video")

    height, width = frames[0].shape[:2]

    # Use a widely supported codec; small clips, so size is not critical.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        os.remove(path)
        raise RuntimeError("Failed to open VideoWriter for temporary clip")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return path


def compute_motion_score(frames: List["cv2.Mat"]) -> float:
    """Compute a simple motion score over a sequence of frames.

    We downscale frames to 64x64, convert to grayscale, and average the
    absolute differences between consecutive frames. This is a cheap proxy
    for "how much the scene is changing".
    """

    if len(frames) < 2:
        return 0.0

    # Prepare first frame
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (64, 64))

    motion_sum = 0.0
    count = 0

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        diff = cv2.absdiff(gray, prev)
        motion_sum += float(diff.mean())
        prev = gray
        count += 1

    if count == 0:
        return 0.0

    return motion_sum / count


def send_clip_for_prediction(
    frames: List["cv2.Mat"], state: PredictionState, top_k: int = TOP_K
) -> None:
    """Background worker: encode frames, send to remote API, update state."""
    try:
        clip_path = encode_clip_to_temp_video(frames, fps=CLIP_FPS)
    except Exception as exc:  # pylint: disable=broad-except
        state.last_error = f"Failed to encode clip: {exc}"
        state.inflight = False
        return

    try:
        with open(clip_path, "rb") as f:
            files = {"video": (os.path.basename(clip_path), f, "video/mp4")}
            data = {"top_k": str(top_k)}

            resp = requests.post(SIGN_PREDICT_URL, files=files, data=data, timeout=60)

        os.remove(clip_path)

        if resp.status_code != 200:
            state.last_error = f"API error {resp.status_code}: {resp.text[:200]}"
            return

        payload = resp.json()
        state.update_from_response(payload)
    except Exception as exc:  # pylint: disable=broad-except
        state.last_error = f"Request failed: {exc}"
    finally:
        state.inflight = False


def draw_overlay(
    frame: "cv2.Mat",
    state: PredictionState,
    ready_to_sign: bool,
    time_to_next: float,
    motion_score: float,
    capturing: bool,
    capture_len: int,
) -> "cv2.Mat":
    """Draw current prediction, transcript, and timing cues on top of the frame."""

    overlay = frame.copy()

    # Basic styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color_ok = (0, 255, 0)
    color_dim = (200, 200, 200)
    color_err = (0, 0, 255)
    color_ready = (0, 255, 255)
    thickness = 2

    y = 30

    # Status line
    status_text = "Status: "
    status_color = color_dim

    if state.inflight:
        status_text += "SENDING clip to API..."
        status_color = color_err
    elif capturing:
        status_text += "CAPTURING sign – keep signing"
        status_color = color_ready
    elif ready_to_sign:
        status_text += "READY – perform next sign"
        status_color = color_ok
    else:
        status_text += f"waiting ({time_to_next:.1f}s) / motion={motion_score:.1f}"
        status_color = color_ready

    cv2.putText(overlay, status_text, (10, y), font, scale, status_color, thickness, cv2.LINE_AA)
    y += 30

    # Current prediction
    if state.current_gloss is not None:
        pred_text = f"Current: {state.current_gloss} ({state.current_prob:.2f})"
        cv2.putText(overlay, pred_text, (10, y), font, scale, color_ok, thickness, cv2.LINE_AA)
        y += 30

    # Live capture length while capturing, otherwise show last capture length
    if capturing and capture_len > 0:
        capture_text = f"Capturing: {capture_len}/{CLIP_FRAMES} frames"
        cv2.putText(overlay, capture_text, (10, y), font, 0.6, color_ready, 1, cv2.LINE_AA)
        y += 25
    elif (not capturing) and state.last_capture_len > 0:
        capture_text = f"Last capture: {state.last_capture_len}/{CLIP_FRAMES} frames"
        cv2.putText(overlay, capture_text, (10, y), font, 0.6, color_dim, 1, cv2.LINE_AA)
        y += 25

    # Transcript (last few words)
    if state.transcript:
        tail = state.transcript[-6:]
        transcript_text = "Transcript: " + " ".join(tail)
        cv2.putText(overlay, transcript_text, (10, y), font, scale, color_dim, 1, cv2.LINE_AA)
        y += 30

    # Last error (if any)
    if state.last_error:
        err_text = f"Error: {state.last_error[:80]}"
        cv2.putText(overlay, err_text, (10, y), font, 0.5, color_err, 1, cv2.LINE_AA)

    return overlay


def main() -> None:
    if "your-space-url" in HF_BASE_URL:
        print("[ERROR] Please set HF_BASE_URL in webcam_hf_client.py to your Space / API URL before running.")
        return

    print("==============================")
    print(" SignBridge I3D Webcam → Remote API Demo")
    print("==============================")
    print(f"Using remote endpoint: {SIGN_PREDICT_URL}")
    print("Press 'q' in the video window to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        return

    # State machine for capture based on instantaneous motion
    state = PredictionState()
    last_request_time = 0.0
    window_name = "SignBridge - Remote I3D Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prev_gray = None
    capturing = False
    # Fixed-length buffer of frames that will actually be fed to the model
    capture_buffer: deque[np.ndarray] = deque(maxlen=CLIP_FRAMES)
    low_motion_frames = 0
    capture_start_time = 0.0

    # Require a short stable period after the sign ends (~0.2–0.3s at typical webcam FPS)
    LOW_MOTION_FRAMES = 3
    # Safety cap on capture duration in seconds (prevents getting stuck if motion never truly goes low)
    MAX_CAPTURE_SECONDS = 2.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            # Compute instantaneous motion between consecutive frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (64, 64))

            if prev_gray is None:
                motion_this = 0.0
            else:
                diff = cv2.absdiff(small_gray, prev_gray)
                motion_this = float(diff.mean())

            prev_gray = small_gray

            # Timing diagnostics
            now = time.time()
            time_since_last = now - last_request_time
            time_to_next = max(0.0, REQUEST_INTERVAL - time_since_last)

            # Can we start capturing a new sign?
            can_start_capture = (
                not capturing
                and not state.inflight
                and time_to_next <= 0.0
                and motion_this >= MIN_MOTION_SCORE
            )

            if can_start_capture:
                capturing = True
                capture_buffer.clear()
                low_motion_frames = 0
                capture_start_time = now

            # Continue capturing while motion is present
            if capturing:
                # Always keep only the most recent CLIP_FRAMES frames
                capture_buffer.append(frame.copy())

                # Track short neutral period after the sign to decide when to cut
                if motion_this < MIN_MOTION_SCORE:
                    low_motion_frames += 1
                else:
                    low_motion_frames = 0

                # Finish once we have a full buffer and either:
                #  - a short stable (low-motion) period, OR
                #  - we have been capturing for longer than MAX_CAPTURE_SECONDS
                enough_frames = len(capture_buffer) >= CLIP_FRAMES
                stable_enough = low_motion_frames >= LOW_MOTION_FRAMES
                time_long = (now - capture_start_time) >= MAX_CAPTURE_SECONDS
                should_finish = (enough_frames and stable_enough) or time_long

                if should_finish and len(capture_buffer) > 0:
                    frames_for_clip = list(capture_buffer)
                    capturing = False
                    low_motion_frames = 0

                    # Record how many frames were captured for this sign (up to CLIP_FRAMES)
                    state.last_capture_len = len(frames_for_clip)

                    if not state.inflight:
                        state.inflight = True
                        last_request_time = now

                        worker = threading.Thread(
                            target=send_clip_for_prediction,
                            args=(frames_for_clip, state),
                            daemon=True,
                        )
                        worker.start()

            # Ready = user can start the next sign now (cooldown finished, no inflight request)
            ready_to_sign = (not capturing) and (not state.inflight) and (time_to_next <= 0.0)

            # Draw overlay and display
            capture_len = len(capture_buffer) if capturing else 0
            annotated = draw_overlay(
                frame,
                state,
                ready_to_sign,
                time_to_next,
                motion_this,
                capturing,
                capture_len,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
