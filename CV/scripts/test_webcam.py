"""
<<<<<<< HEAD
Local webcam demo using the 55-class demo I3D model (no remote API).

This script mirrors the behavior of ``webcam_hf_client.py`` but runs
inference locally via ``SignRecognizer`` instead of sending clips to
the Hugging Face Space. It uses motion-based capture, debouncing, and
an on-screen transcript to provide a smooth real-time experience.
=======
Local webcam demo for the I3D sign recognition model.

Run this script to test the trained 100-class model on your webcam.
>>>>>>> khaled
"""

from __future__ import annotations

<<<<<<< HEAD
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configure the demo seed (55-class) BEFORE importing CV modules
# ---------------------------------------------------------------------------

CV_DIR = Path(__file__).resolve().parents[1]  # .../CV
ASSETS_DIR = CV_DIR / "assets"
CHECKPOINTS_DIR = CV_DIR / "checkpoints"

os.environ.setdefault("SIGNBRIDGE_LABEL_MAP", str(ASSETS_DIR / "label_mapping_demo.json"))
os.environ.setdefault("SIGNBRIDGE_CHECKPOINT", str(CHECKPOINTS_DIR / "demo_model_55class.pth"))

from CV.inference.sign_recognizer import SignRecognizer


# Real-time configuration (mirroring webcam_hf_client.py)
CLIP_FRAMES = 32          # number of frames per clip sent to the model
TOP_K = 5                 # how many alternatives to compute
MIN_CONFIDENCE = 0.6      # minimum probability to treat a prediction as valid
REQUEST_INTERVAL = 1.5    # seconds between predictions
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

    def update_from_result(self, gloss: Optional[str], prob: float) -> None:
        """Update state from a local model prediction.

        Applies the same confidence threshold and transcript update
        behavior as ``webcam_hf_client.PredictionState.update_from_response``.
        """

        self.last_error = None

        if gloss is None or prob < MIN_CONFIDENCE:
            self.current_gloss = None
            self.current_prob = prob
            self.last_update = time.time()
            return

        now = time.time()

        if gloss != self.current_gloss:
            self.transcript.append(gloss)

        self.current_gloss = gloss
        self.current_prob = prob
        self.last_update = now


def compute_motion_score(frames: List["cv2.Mat"]) -> float:
    """Compute a simple motion score over a sequence of frames.

    Downscale frames to 64x64, convert to grayscale, and average the
    absolute differences between consecutive frames.
    """

    if len(frames) < 2:
        return 0.0

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


def run_local_prediction(
    frames: List["cv2.Mat"],
    state: PredictionState,
    recognizer: SignRecognizer,
    top_k: int = TOP_K,
) -> None:
    """Background worker: run local model inference and update state."""

    try:
        result = recognizer.predict_clip(frames, topk=top_k)
        gloss = result.gloss
        prob = float(result.probability)
        state.update_from_result(gloss, prob)
    except Exception as exc:  # pylint: disable=broad-except
        state.last_error = f"Model inference failed: {exc}"
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
        status_text += "RUNNING local model..."
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
=======
import cv2
import threading
from queue import Queue

from CV import config
from CV.data.video_reader import open_webcam, release_webcam
from CV.inference.sign_recognizer import SignRecognizer


def prediction_worker(recognizer, frame_queue: Queue) -> None:
    """Worker thread that runs predictions on collected frames."""
    while True:
        frames = frame_queue.get()
        if frames is None:  # Signal to stop
            break
        try:
            result = recognizer.predict_clip(frames, topk=5)
            print(
                f"Prediction: {result.gloss} (label={result.label}, prob={result.probability:.2f})"
            )
            print(
                "  Top-5: "
                + ", ".join(
                    f"{g} ({p:.2f})" for g, p in zip(result.topk_glosses, result.topk_probabilities)
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"Prediction error: {e}")
>>>>>>> khaled


def main() -> None:
    recognizer = SignRecognizer()
<<<<<<< HEAD

    print("==============================")
    print(" SignBridge I3D Webcam Demo (55-class local model)")
    print("==============================")
    print("Press 'q' in the video window to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        return

    state = PredictionState()
    last_request_time = 0.0
    window_name = "SignBridge - Local I3D Webcam (Demo Seed)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prev_gray = None
    capturing = False
    capture_buffer: deque[np.ndarray] = deque(maxlen=CLIP_FRAMES)
    low_motion_frames = 0
    capture_start_time = 0.0

    LOW_MOTION_FRAMES = 3
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

            now = time.time()
            time_since_last = now - last_request_time
            time_to_next = max(0.0, REQUEST_INTERVAL - time_since_last)

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

            if capturing:
                capture_buffer.append(frame.copy())

                if motion_this < MIN_MOTION_SCORE:
                    low_motion_frames += 1
                else:
                    low_motion_frames = 0

                enough_frames = len(capture_buffer) >= CLIP_FRAMES
                stable_enough = low_motion_frames >= LOW_MOTION_FRAMES
                time_long = (now - capture_start_time) >= MAX_CAPTURE_SECONDS
                should_finish = (enough_frames and stable_enough) or time_long

                if should_finish and len(capture_buffer) > 0:
                    frames_for_clip = list(capture_buffer)
                    capturing = False
                    low_motion_frames = 0

                    state.last_capture_len = len(frames_for_clip)

                    if not state.inflight:
                        state.inflight = True
                        last_request_time = now

                        worker = threading.Thread(
                            target=run_local_prediction,
                            args=(frames_for_clip, state, recognizer),
                            daemon=True,
                        )
                        worker.start()

            ready_to_sign = (not capturing) and (not state.inflight) and (time_to_next <= 0.0)

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
=======
    num_frames_per_clip = getattr(config, "NUM_FRAMES", 32)

    print("==============================")
    print(" SignBridge I3D Webcam Demo")
    print("==============================")
    print("Press 'q' in the video window to quit.")
    print(f"Collecting {num_frames_per_clip} frames per clip before each prediction.\n")

    cap = None
    try:
        cap = open_webcam(0)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error opening webcam: {e}")
        return

    # Start prediction worker thread
    frame_queue = Queue(maxsize=1)
    worker_thread = threading.Thread(target=prediction_worker, args=(recognizer, frame_queue), daemon=True)
    worker_thread.start()

    frames = []
    window_name = "SignBridge - Webcam"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        # Show live preview
        cv2.imshow(window_name, frame)

        # Accumulate frames for the current clip
        frames.append(frame)

        if len(frames) >= num_frames_per_clip:
            # Send frames to worker thread (non-blocking)
            try:
                frame_queue.put_nowait(frames.copy())
            except:  # pylint: disable=bare-except
                pass  # Skip if queue is full
            frames = []

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    frame_queue.put(None)  # Signal worker to stop
    worker_thread.join(timeout=5)
    release_webcam(cap)
    cv2.destroyAllWindows()
>>>>>>> khaled


if __name__ == "__main__":
    main()