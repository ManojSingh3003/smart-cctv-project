# Smart CCTV with Threat Detection using YOLOv8 & MediaPipe

![Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)
*(Recommendation: Record a short GIF of your project in action and replace the link above)*

This project is a smart surveillance system that uses real-time video to detect suspicious objects and actions. It leverages the **YOLOv8** model for state-of-the-art object detection and **MediaPipe** for human pose estimation, triggering audio-visual alerts when a threat is identified.

---

## Features
- **Real-time Object Detection**: Identifies objects from a live camera feed using YOLOv8x.
- **Suspicious Object Alerts**: Specifically detects and flags `knives` and `guns`.
- **Action Recognition**: Analyzes human poses with MediaPipe to detect threatening gestures like "stabbing" or "sudden arm extension".
- **Audio-Visual Alerts**: Provides on-screen alerts and speaks warnings using a text-to-speech engine.
- **Configurable**: Easily change suspicious object classes, alert cooldowns, and detection confidence thresholds.

---

## Technology Stack
- **Python 3**
- **OpenCV** for video capture and processing.
- **Ultralytics YOLOv8** for object detection.
- **Google MediaPipe** for pose estimation.
- **pyttsx3** for text-to-speech alerts.

---

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/smart-cctv-project.git](https://github.com/your-username/smart-cctv-project.git)
cd smart-cctv-project
```

### 2. Create a Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
All required libraries are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Download the YOLOv8 Model
This project uses the `yolov8x.pt` model. It is not included in the repository due to its size. You can download it automatically, as the Ultralytics library will fetch it on the first run.

If you wish to download it manually, you can find it in the [YOLOv8 official repository](https://github.com/ultralytics/ultralytics).

---

## How to Run
Once the setup is complete, run the main script from your terminal:
```bash
python smart_cctv.py
```
Press the `q` key to exit the video stream.

---

## How It Works
The application pipeline is as follows:
1.  **Frame Capture**: Captures video frames from the webcam using OpenCV.
2.  **Object Detection**: Each frame is passed to the YOLOv8 model, which identifies objects and their bounding boxes.
3.  **Pose Estimation**: The frame is also processed by MediaPipe to extract 33 human pose landmarks if a person is present.
4.  **Threat Analysis**:
    * The system checks if any detected objects are in the `SUSPICIOUS_CLASSES` list.
    * A buffer of recent pose landmarks is analyzed to detect suspicious movements based on simple heuristics.
5.  **Alert Generation**: If a threat is detected and the alert cooldown has passed, an on-screen message is displayed in red, and a spoken warning is issued.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.