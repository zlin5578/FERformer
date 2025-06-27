import cv2
import numpy as np
import time
import os
from deepface import DeepFace

# Optional: Try to use virtual webcam
USE_VIRTUAL_CAM = True
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    USE_VIRTUAL_CAM = False


def overlay_emoji(frame, emoji_png_path, pos=(0, 0), scale=1.0):
    emoji = cv2.imread(emoji_png_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        return frame

    emoji_h, emoji_w = emoji.shape[:2]
    emoji = cv2.resize(emoji, (int(emoji_w * scale), int(emoji_h * scale)), interpolation=cv2.INTER_AREA)

    if emoji.shape[2] == 4:
        alpha_emoji = emoji[:, :, 3] / 255.0
        rgb_emoji = emoji[:, :, :3]
    else:
        alpha_emoji = np.ones(emoji.shape[:2])
        rgb_emoji = emoji

    x, y = pos
    h, w = rgb_emoji.shape[:2]

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    for c in range(3):
        frame[y:y + h, x:x + w, c] = (
            alpha_emoji * rgb_emoji[:, :, c] +
            (1 - alpha_emoji) * frame[y:y + h, x:x + w, c]
        )
    return frame


def draw_emotion_bars(frame, emotions, top_left=(10, 10), bar_width=150, bar_height=20, spacing=10):
    x, y = top_left
    for i, (emotion, score) in enumerate(emotions.items()):
        bar_len = int(bar_width * score / 100)
        color = (0, int(255 * (score / 100)), int(255 * (1 - score / 100)))

        cv2.rectangle(frame, (x, y + i * (bar_height + spacing)),
                      (x + bar_len, y + i * (bar_height + spacing) + bar_height),
                      color, -1)

        cv2.putText(frame, f"{emotion}: {score:.1f}%", (x + bar_width + 10, y + i * (bar_height + spacing) + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

    emotion_to_emoji_path = {
        "angry": "emojis/angry.png",
        "disgust": "emojis/disgust.png",
        "fear": "emojis/fear.png",
        "happy": "emojis/happy.png",
        "neutral": "emojis/neutral.png",
        "sad": "emojis/sad.png",
        "surprise": "emojis/surprised.png",
    }

    print("Starting webcam stream...")

    def run_loop(cam=None):
        last_inference_time = 0
        result_text = "Analyzing..."
        emoji_path = None
        emotion_scores = {}
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Calculate FPS
            current_time = time.time()
            fps_display = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Emotion detection every .1s
            if current_time - last_inference_time > 0.2:
                try:
                    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

                    if isinstance(result, list) and len(result) > 0:
                        dominant_emotion = result[0]["dominant_emotion"]
                        emotion_scores = result[0]["emotion"]
                        result_text = f"{dominant_emotion.capitalize()} ({emotion_scores[dominant_emotion]:.1f}%)"
                        emoji_path = emotion_to_emoji_path.get(dominant_emotion)

                        # Draw bounding box
                        region = result[0].get("region", {})
                        if region:
                            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    else:
                        result_text = "No face detected"
                        emoji_path = None
                        emotion_scores = {}
                except Exception as e:
                    print(f"Error during inference: {e}")
                    result_text = "Error"
                    emoji_path = None
                    emotion_scores = {}
                last_inference_time = current_time

            annotated = frame.copy()

            # Draw emoji and emotion label
            (text_width, text_height), baseline = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = (width - text_width) // 2
            text_y = int(height * 2 / 3) + text_height

            if emoji_path and os.path.isfile(emoji_path):
                annotated = overlay_emoji(annotated, emoji_path, pos=(text_x, text_y - 80), scale=0.3)

            cv2.putText(annotated, result_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw emotion bars
            if emotion_scores:
                draw_emotion_bars(annotated, emotion_scores, top_left=(10, 10))

            # Draw FPS
            cv2.putText(annotated, f"FPS: {fps_display:.2f}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Output to screen or virtual cam
            if cam:
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()
            else:
                cv2.imshow("Emotion Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if USE_VIRTUAL_CAM and VIRTUAL_CAM_AVAILABLE:
        try:
            with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.RGB) as cam:
                print(f"Virtual camera started: {cam.device}")
                run_loop(cam)
        except Exception as e:
            print(f"Virtual camera failed: {e}")
            print("Switching to OpenCV window mode")
            run_loop()
    else:
        run_loop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()