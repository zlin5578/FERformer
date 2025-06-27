import cv2
import torch
import numpy as np
import argparse
import time
import os
from datetime import datetime

# Dummy emotion inference (replace with real detector like FER/DeepFace)
def detect_emotions(frame):
    # Simulate one face with emotion probabilities
    return [{
        "box": [50, 50, 150, 150],  # x, y, w, h
        "emotions": {
            "angry": np.random.rand(),
            "disgust": np.random.rand(),
            "fear": np.random.rand(),
            "happy": np.random.rand(),
            "sad": np.random.rand(),
            "surprise": np.random.rand(),
            "neutral": np.random.rand()
        }
    }]

# Emotion color map
emotion_colors = {
    "angry": (0, 0, 255),
    "disgust": (0, 255, 0),
    "fear": (255, 0, 255),
    "happy": (0, 255, 255),
    "sad": (255, 255, 0),
    "surprise": (255, 128, 0),
    "neutral": (200, 200, 200),
}

# Draw emotions with bars
def draw_emotion_data(frame, emotions_data):
    for i, face_data in enumerate(emotions_data):
        x, y, w, h = face_data["box"]
        emotions = face_data["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        # Face bounding box
        color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label
        label = f"{top_emotion.upper()}: {emotions[top_emotion]*100:.1f}%"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Emotion bars
        bar_x = x + w + 10
        bar_y = y
        bar_width = 100
        bar_height = 15
        spacing = 5

        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        for j, (emotion, score) in enumerate(sorted_emotions):
            bar_length = int(score * bar_width)
            bar_color = emotion_colors.get(emotion, (255, 255, 255))

            # Background
            cv2.rectangle(frame, (bar_x, bar_y + j * (bar_height + spacing)),
                          (bar_x + bar_width, bar_y + j * (bar_height + spacing) + bar_height),
                          (50, 50, 50), -1)

            # Filled bar
            cv2.rectangle(frame, (bar_x, bar_y + j * (bar_height + spacing)),
                          (bar_x + bar_length, bar_y + j * (bar_height + spacing) + bar_height),
                          bar_color, -1)

            # Label text
            cv2.putText(frame, f"{emotion}: {int(score * 100)}%",
                        (bar_x + bar_width + 5, bar_y + j * (bar_height + spacing) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

# ---------------- Main YOLO + Emotion --------------------

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/live_detector/weights/best.pt')
model.conf = 0.3
model.iou = 0.7

# Argument parser
parser = argparse.ArgumentParser(description="YOLOv5 + Emotion Display")
parser.add_argument("-f", "--file", type=str, help="Path to input video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name (optional)")
args = parser.parse_args()

# Choose source
if args.file:
    vs = cv2.VideoCapture(args.file)
    if not vs.isOpened():
        print(f"[ERROR] Cannot open video file: {args.file}")
        exit(1)
else:
    vs = cv2.VideoCapture(0)
    if not vs.isOpened():
        print("[ERROR] Cannot access webcam.")
        exit(1)

# Warm up
time.sleep(2.0)
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_filename = args.out if args.out else f"output_{timestamp}.avi"

# Set up writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height))

print("[INFO] Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)
    detections = results.xyxy[0]
    names = model.names

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls)].upper()
        conf_text = f"{label}: {int(conf * 100)}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, conf_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Emotion detection + visualization
    emotion_data = detect_emotions(frame)
    frame = draw_emotion_data(frame, emotion_data)

    # FPS overlay
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    # out.write(frame)
    cv2.imshow("YOLO + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.release()
out.release()
cv2.destroyAllWindows()