import cv2

emotion_colors = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 128),
    "fear": (128, 0, 128),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral": (128, 128, 128)
}

def draw_emotion_data(frame, emotions_data, emotion_history, emoji_paths):
    from emoji_utils import overlay_emoji
    for i, face_data in enumerate(emotions_data):
        (x, y, w, h) = face_data["box"]
        emotions = face_data["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        face_id = f"face_{i}"
        if face_id in emotion_history:
            alpha = 0.6
            for emotion in emotions:
                if emotion in emotion_history[face_id]:
                    emotions[emotion] = (alpha * emotions[emotion] +
                                         (1 - alpha) * emotion_history[face_id][emotion])
            top_emotion = max(emotions, key=emotions.get)
        emotion_history[face_id] = emotions.copy()

        color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label = f"{top_emotion.upper()}: {emotions[top_emotion]:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        emoji_path = emoji_paths.get(top_emotion.lower())
        if emoji_path:
            frame = overlay_emoji(frame, emoji_path, x, y + h + 10, scale=0.4)

        bar_x = x + w + 10
        bar_y = y
        bar_width = 120
        bar_height = 15
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)

        for j, (emotion, score) in enumerate(sorted_emotions[:7]):
            bar_length = int(score * bar_width)
            bar_color = emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (bar_x, bar_y + j * 20),
                          (bar_x + bar_width, bar_y + j * 20 + bar_height),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y + j * 20),
                          (bar_x + bar_length, bar_y + j * 20 + bar_height),
                          bar_color, -1)
            cv2.putText(frame, f"{emotion}: {int(score * 100)}%",
                        (bar_x + bar_width + 5, bar_y + j * 20 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def draw_status_text(frame, fps, frame_count):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit, 'r' to log", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "FER Detection: Active", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)