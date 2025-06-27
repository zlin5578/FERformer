# camera_utils.py

import cv2

def get_webcam(width=640, height=360, fps=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Error: Could not open webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    return cap