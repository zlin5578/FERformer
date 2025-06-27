import cv2
import os
import json

emotion_colors = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 128),
    "fear": (128, 0, 128),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral": (80, 80, 80)
}

EMOJI_RESOLUTION = 160

emotion_tally = {emotion: 0 for emotion in emotion_colors.keys()}

def draw_emotion_data(frame, emotions_data, emotion_history, emoji_paths):
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    from emoji_utils import overlay_emoji

    global emotion_tally

    if not emotions_data:
        return frame

    frame_height, frame_width = frame.shape[:2]

    left_x = int(frame_width * 1/20)
    right_x = int(frame_width * 7/8)
    top_y = int(frame_height * 1/20)
    bottom_y = int(frame_height * 7/8)

    emoji_scale = 0.4
    emoji_size = int(EMOJI_RESOLUTION * emoji_scale)

    face_data = emotions_data[0]
    emotions = face_data["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    emotion_tally[top_emotion] += 1

    for i, face_data in enumerate(emotions_data):
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

        label = f"{top_emotion.capitalize()}: {int(emotions[top_emotion] * 100)}%"

        emoji_path = emoji_paths.get(top_emotion.lower())
        if config.get("emoji_toggle", True) and emoji_path:
            location = config.get("overlay_location", 1)

            if location == 0:  # Top right
                emoji_x = right_x
                emoji_y = top_y
                bars_below = True
            elif location == 1:  # Top left
                emoji_x = left_x
                emoji_y = top_y
                bars_below = True
            elif location == 2:  # Bottom right
                emoji_x = left_x
                emoji_y = bottom_y
                bars_below = False
            elif location == 3:  # Bottom left
                emoji_x = right_x
                emoji_y = bottom_y
                bars_below = False
            else:
                emoji_x = right_x
                emoji_y = top_y
                bars_below = True

            frame = overlay_emoji(frame, emoji_path, emoji_x, emoji_y, scale=emoji_scale)

        bar_x = emoji_x
        bar_width = 125
        bar_height = 25
        font_scale = 0.5
        font_thickness = 1
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)

        if bars_below:
            bar_y_start = emoji_y + emoji_size + (frame_height // 100)
            bar_direction = 1
        else:
            bar_y_start = emoji_y - (frame_height // 100)
            bar_direction = -1

        if config.get("emotion_likelihood_toggle", True):
            for j, (emotion, score) in enumerate(sorted_emotions[:7]):
                bar_length = int(score * bar_width)
                y_offset = bar_y_start + (j * (bar_height + 5)) * bar_direction
                bar_color = emotion_colors.get(emotion, (255, 255, 255))

                cv2.rectangle(frame, (bar_x, y_offset),
                            (bar_x + bar_width, y_offset + bar_height * bar_direction),
                            (50, 50, 50), -1)

                cv2.rectangle(frame, (bar_x, y_offset),
                            (bar_x + bar_length, y_offset + bar_height * bar_direction),
                            bar_color, -1)

                text = f"{emotion.capitalize()} (%)"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = bar_x + 5
                text_y = y_offset + int(bar_height * 0.75 * bar_direction)
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        label_y = bar_y_start + (len(sorted_emotions[:7]) * (bar_height + 5)) * bar_direction + (15 * bar_direction)

        if config.get("emotion_label_toggle", True):
            label = f"{top_emotion.capitalize()}: {emotions[top_emotion]:.2f}"
            cv2.putText(frame, label, (bar_x, label_y + (frame_height // 100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_colors.get(top_emotion, (255, 255, 255)), 2)

        tally_start_y = label_y + (25 * bar_direction)
        max_count = max(emotion_tally.values()) if emotion_tally.values() else 1

        if config.get("local_history_toggle", True):
            for k, (emotion, count) in enumerate(sorted(emotion_tally.items(), key=lambda x: x[1], reverse=True)):
                bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
                y_offset = tally_start_y + (k * (bar_height + 5)) * bar_direction
                bar_color = emotion_colors.get(emotion, (255, 255, 255))

                cv2.rectangle(frame, (bar_x, y_offset),
                            (bar_x + bar_width, y_offset + bar_height * bar_direction),
                            (50, 50, 50), -1)

                cv2.rectangle(frame, (bar_x, y_offset),
                            (bar_x + bar_length, y_offset + bar_height * bar_direction),
                            bar_color, -1)

                text = f"{emotion.capitalize()} (#)"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = bar_x + 5
                text_y = y_offset + int(bar_height * 0.75 * bar_direction)
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

def draw_status_text(frame, fps, frame_count):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)