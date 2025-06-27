# FERformer: Real-Time Facial Emotion Recognition Pipeline

## Overview

FERformer is a modular real-time facial emotion recognition (FER) pipeline designed for seamless integration with video conferencing platforms (such as Zoom, Teams) and local GUI applications. By leveraging **OBS Virtual Camera** or other virtual camera solutions as a bridge, FERformer enables the output of post-processed video streams—including emotion overlays and visual augmentations—directly into any application that accepts webcam input.

The system supports multiple state-of-the-art facial emotion recognition backbones, including **YOLO**, **DeepFace**, and **FER**, allowing users to flexibly choose or benchmark different emotion detection algorithms according to their requirements.

## Key Features

- **Virtual Camera Integration:** Stream augmented video feeds via OBS Virtual Camera or pyvirtualcam, making emotion recognition results instantly available in third-party applications (e.g., Zoom, Teams).
- **Multi-Backend Support:** Easily switch between YOLO, DeepFace, and FER as the underlying emotion recognition engine.
- **Real-Time Processing:** Efficient frame-by-frame inference and overlay, supporting live deployments.
- **Customizable Visualization:** Overlay emotion bars, bounding boxes, and emojis on detected faces for intuitive feedback.
- **Flexible Deployment:** Run as a background service (headless) or with a local GUI for direct interaction and visualization.

## Requirements

- Python <= 3.11
- OBS Studio
- OBS Virtual Camera
- DeepFace
- OpenCV
- (Optional) PyTorch (for YOLOv5 backend)

## Usage

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Start OBS and enable Virtual Camera.**

3. **Run the pipeline:**
    - For headless streaming to OBS Virtual Cam:
        ```bash
        python obs_virtual_cam.py --cam-id 0 --width 1280 --height 720 --fps 30
        ```
    - For local GUI preview:
        ```bash
        python main.py
        ```

4. **Select "OBS Virtual Camera" as your webcam in Zoom, Teams, or any other application.**

## Backbones

- **YOLOv5:** Utilized for robust face detection and can be extended for emotion classification.
- **DeepFace:** Provides high-accuracy emotion recognition using deep learning models.
- **FER:** Lightweight and fast, suitable for real-time applications.

## Applications

- Online meetings and presentations with live emotion feedback.
- Human-computer interaction research.
- Real-time emotion analytics for education, gaming, and entertainment.

---

For detailed configuration and advanced usage, refer to the documentation in each submodule.