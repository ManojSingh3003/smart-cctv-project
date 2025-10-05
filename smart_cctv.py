import cv2
import time
import pyttsx3
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from collections import deque

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO("yolov8x.pt")  # Ensure this model file is in your working directory

# Define suspicious objects
SUSPICIOUS_CLASSES = ['knife', 'gun']
ALERT_COOLDOWN = 10  # seconds
last_alert_time = 0

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Buffer to store recent pose landmarks for action detection
POSE_BUFFER_SIZE = 30
pose_landmarks_buffer = deque(maxlen=POSE_BUFFER_SIZE)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def extract_keypoints(landmarks):
    """
    Extract normalized keypoints (x, y) from pose landmarks list.
    Returns a flattened np.array of shape (33*2,) for 33 landmarks.
    """
    keypoints = []
    for lm in landmarks.landmark:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return np.array(keypoints)

def detect_suspicious_action(pose_buffer):
    """
    Simple heuristic for suspicious action detection over a buffer of poses:
    - Detect rapid movement of right wrist towards neck area (simulate stabbing/kidnapping)
    - Detect sudden arm extension
    
    Input: pose_buffer is deque of keypoint arrays.
    Output: (bool, label)
    """

    if len(pose_buffer) < POSE_BUFFER_SIZE:
        return False, ""

    # Convert buffer to numpy array (frames, keypoints)
    data = np.array(pose_buffer)  # shape: (30, 66)

    # Extract right wrist and right shoulder landmarks over time (indexes 16, 12)
    # Landmark index from MediaPipe:
    # 16 - RIGHT_WRIST, 12 - RIGHT_SHOULDER, 2 - NOSE (used as neck approx)

    right_wrist_y = data[:, 16*2+1]  # y coordinate
    right_wrist_x = data[:, 16*2]    # x coordinate
    right_shoulder_y = data[:, 12*2+1]
    right_shoulder_x = data[:, 12*2]
    nose_y = data[:, 0*2+1]
    nose_x = data[:, 0*2]

    # Check if wrist moves quickly from below shoulder to near neck (nose)
    # Calculate vertical distance changes
    wrist_to_shoulder_dist = right_wrist_y - right_shoulder_y  # positive means wrist below shoulder
    wrist_to_nose_dist = np.sqrt((right_wrist_x - nose_x)**2 + (right_wrist_y - nose_y)**2)

    # Heuristic 1: Rapid upward movement of right wrist in last 10 frames
    recent_wrist_y = right_wrist_y[-10:]
    if recent_wrist_y[-1] < recent_wrist_y[0] - 0.1:  # moved up by >0.1 normalized units
        # If wrist gets close to nose (neck/head area)
        if wrist_to_nose_dist[-1] < 0.15:
            return True, "stabbing/kidnapping gesture"

    # Heuristic 2: Right arm extension (wrist far from shoulder horizontally)
    wrist_shoulder_x_diff = np.abs(right_wrist_x - right_shoulder_x)
    if wrist_shoulder_x_diff[-1] > 0.3:
        return True, "arm extension"

    return False, ""

def detect_and_alert(frame):
    global last_alert_time

    # YOLOv8 object detection
    results = model(frame)[0]
    detected_classes = set()
    threat_detected = False
    threat_label = ""
    people_count = 0

    for r in results.boxes.data.tolist():
        class_id = int(r[5])
        confidence = r[4]
        class_name = model.names[class_id]

        if confidence < 0.6:
            continue

        detected_classes.add(class_name)

        if class_name == 'person':
            people_count += 1

        if class_name in SUSPICIOUS_CLASSES:
            threat_detected = True
            threat_label = class_name

        x1, y1, x2, y2 = map(int, r[:4])
        color = (0, 255, 0) if class_name == 'person' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # MediaPipe pose estimation
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        keypoints = extract_keypoints(results_pose.pose_landmarks)
        pose_landmarks_buffer.append(keypoints)

        action_detected, action_label = detect_suspicious_action(pose_landmarks_buffer)
        if action_detected:
            threat_detected = True
            threat_label = action_label

    # Overlay information
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    info_text = "Seeing: " + ", ".join(sorted(detected_classes)) if detected_classes else "Seeing: Nothing"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Bottom-left box with people count and threat alert
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (10, frame.shape[0] - 70), (600, frame.shape[0] - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)

    bottom_text = f"People detected: {people_count}"
    if threat_detected:
        bottom_text += f" | ALERT: {threat_label.upper()} DETECTED"

    color_text = (0, 255, 0) if not threat_detected else (0, 0, 255)
    cv2.putText(frame, bottom_text, (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)

    # Speak if threat detected and cooldown passed
    current_time = time.time()
    if threat_detected and (current_time - last_alert_time > ALERT_COOLDOWN):
        alert_msg = f"Warning! Suspicious activity detected: {threat_label}."
        print(alert_msg)
        speak(alert_msg)
        last_alert_time = current_time

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        frame = detect_and_alert(frame)
        cv2.imshow("Smart CCTV - YOLOv8x + Action Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
