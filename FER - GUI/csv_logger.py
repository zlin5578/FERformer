import csv
import os
from datetime import datetime
from process_emotion import process_emotion_csv

class EmotionCSVLogger:
    def __init__(self):
        self.session_dir = None
        self.timestamp = None
        self.raw_csv_path = None
        self.processed_csv_path = None
        self.active = False

    def start_new_log(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join("logs", f"session_{self.timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.raw_csv_path = os.path.join(self.session_dir, f"raw_emotion_log_{self.timestamp}.csv")
        self.processed_csv_path = os.path.join(self.session_dir, f"processed_emotion_log_{self.timestamp}.csv")

        with open(self.raw_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "person_id", "dominant_emotion"])

        self.active = True

    def stop(self):
        self.active = False
        process_emotion_csv(self.raw_csv_path, self.processed_csv_path)

    def log(self, person_id: str, emotion: str):
        if not self.active or not self.raw_csv_path:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.raw_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, person_id, emotion])