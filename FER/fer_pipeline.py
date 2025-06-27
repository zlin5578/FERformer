import cv2
import numpy as np
from fer import FER
import time
import os
from emoji_utils import overlay_emoji
from camera_utils import get_webcam
from visual_utils import draw_emotion_data, draw_status_text
from csv_logger import EmotionCSVLogger

def run_fer_loop():
    print("Loading FER emotion detector...")
    try:
        detector = FER(mtcnn=True)
        print("✓ FER detector loaded successfully!")
    except Exception as e:
        print(f"Error loading FER with MTCNN: {e}")
        try:
            detector = FER(mtcnn=False)
            print("FER detector loaded with OpenCV cascade!")
        except Exception as e2:
            print(f"Error loading FER: {e2}")
            print("Please install: pip install fer tensorflow")
            return

    emoji_paths = {
        "angry": "emojis/angry.png",
        "disgust": "emojis/disgust.png",
        "fear": "emojis/fear.png",
        "happy": "emojis/happy.png",
        "sad": "emojis/sad.png",
        "surprise": "emojis/surprised.png",
        "neutral": "emojis/neutral.png",
    }

    try:
        cap = get_webcam()
    except IOError as e:
        print(e)
        return

    prev_time = time.time()
    last_emotion_time = 0
    emotion_interval = 0.2  # seconds
    emotion_history = {}
    frame_count = 0
    emotions_data = []  # persist between frames
    logging_active = False
    csv_logger = EmotionCSVLogger()

    print("Starting FER emotion detection...")
    print("Press 'r' to toggle logging, 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame_count += 1
        curr_time = time.time()

        # Detect emotion every N seconds
        if curr_time - last_emotion_time >= emotion_interval:
            try:
                emotions_data = detector.detect_emotions(frame)
                last_emotion_time = curr_time

                if emotions_data:
                    dominant_emotion = emotions_data[0]['emotions']
                    top_emotion = max(dominant_emotion, key=dominant_emotion.get)
                    print(f"[{time.strftime('%H:%M:%S')}] Dominant Emotion: {top_emotion}")

                    if logging_active:
                        csv_logger.log(person_id="Unknown", emotion=top_emotion)

            except Exception as e:
                print(f"Error in emotion detection: {e}")

        # Always draw latest emotions (even if not updated this frame)
        frame = draw_emotion_data(frame, emotions_data, emotion_history, emoji_paths)

        # Show face count
        cv2.putText(frame, f"Faces detected: {len(emotions_data)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show logging status
        if logging_active:
            cv2.putText(frame, "LOGGING ACTIVE", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS calculation
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        draw_status_text(frame, fps, frame_count)

        cv2.imshow("FER Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key == ord('r'):
            logging_active = not logging_active
            if logging_active:
                csv_logger.start_new_log()
                print("Emotion logging started.")
            else:
                csv_logger.stop()
                print("Emotion logging stopped.")

    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print("FER emotion detection ended.")