"""obs_virtual_cam.py
--------------------------------
Run emotion detection using the existing pipeline and stream the
post‑processed frames directly to the **OBS Virtual Camera** via
*pyvirtualcam*.  No GUI windows are shown – everything runs quietly in
the background so that Zoom/Teams can pick up the augmented feed
immediately.

## Quick Setup
```bash
# 1. Install/upgrade dependencies
pip install --upgrade pyvirtualcam opencv-python

# 2. Start OBS ➜ "Start Virtual Camera" (first time会弹系统扩展授权)

# 3. Launch the pipeline
python obs_virtual_cam.py --cam-id 0 --width 1280 --height 720 --fps 30
```

Press **Ctrl‑C** to stop.
"""

import argparse
import logging
import os
import sys
import time
import json
from collections import deque

import cv2
import numpy as np
import pyvirtualcam
from fer import FER

from camera_utils import get_webcam  # noqa: F401 (kept for future use)
from visual_utils import draw_emotion_data


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def init_detector() -> FER:
    """Initialise the FER detector with graceful fallback."""
    try:
        logging.info("Loading FER detector (mtcnn=True)…")
        return FER(mtcnn=True)
    except Exception as e:
        logging.warning("MTCNN failed: %s – falling back to OpenCV cascade", e)
        return FER(mtcnn=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FER pipeline and stream to OBS Virtual Camera")
    parser.add_argument("--cam-id", type=int, default=0,
                        help="Physical webcam index")
    parser.add_argument("--width", type=int, default=1280,
                        help="Frame width")
    parser.add_argument("--height", type=int, default=720,
                        help="Frame height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second")
    parser.add_argument("--skip", type=int, default=2,
                        help="Process every N‑th frame to save CPU")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # ── Physical webcam ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        logging.error("Could not open webcam – check --cam-id")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info("Webcam initialised @ %dx%d %d FPS", width, height, args.fps)

    # ── OBS Virtual Camera ────────────────────────────────────────────────
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=args.fps,
                                  print_fps=False, backend="obs")
    except Exception as e:
        logging.error(
            "pyvirtualcam failed to open OBS Virtual Camera.\n"
            , e)
        sys.exit(2)

    logging.info("Streaming to virtual camera device: %s", cam.device)

    detector = init_detector()
    config = load_config()
    emotion_history: deque = deque(maxlen=30)
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                logging.warning("Webcam frame grab failed – retrying…")
                time.sleep(0.05)
                continue

            # Resize if needed
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))

            # Heavy FER only every --skip frames
            if frame_idx % args.skip == 0:
                emotions = detector.detect_emotions(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            else:
                emotions = []

            # Draw overlay
            frame_bgr = draw_emotion_data(
                frame_bgr, emotions, emotion_history, config.get("emoji_paths", {}))

            cam.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()
            frame_idx += 1

    except KeyboardInterrupt:
        logging.info("Interrupted – shutting down…")

    finally:
        cap.release()
        cam.close()
        logging.info("Camera resources released.")


if __name__ == "__main__":
    main()
