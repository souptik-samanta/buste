import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import numpy as np
import dlib
import random

# Load dlib face detector and shape predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_symmetry_score(landmarks):
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    left = points[:34]
    right = points[34:][::-1]  # Mirror the second half
    diffs = np.linalg.norm(left - right, axis=1)
    score = 10 - np.clip(np.mean(diffs) / 5.0, 0, 5)
    return round(score + random.uniform(-0.5, 0.5), 2)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            shape = predictor(gray, face)
            face_score = get_symmetry_score(shape)
            cv2.putText(img, f"💄 Model Score: {face_score}/10", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Pose Detection
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                lm = results.pose_landmarks.landmark
                left_shoulder = lm[11]
                right_shoulder = lm[12]
                left_wrist = lm[15]
                right_wrist = lm[16]

                # Convert to pixel coords
                x1, y1 = int(left_shoulder.x * w), int(left_shoulder.y * h)
                x2, y2 = int(right_shoulder.x * w), int(right_shoulder.y * h)
                xw1, yw1 = int(left_wrist.x * w), int(left_wrist.y * h)
                xw2, yw2 = int(right_wrist.x * w), int(right_wrist.y * h)

                # Improved bust estimate: shoulder width + wrist-chest depth
                shoulder_width = np.linalg.norm([x1 - x2, y1 - y2])
                chest_ratio = np.linalg.norm([x1 - xw1, y1 - yw1]) + np.linalg.norm([x2 - xw2, y2 - yw2])
                bust_score = round((shoulder_width * 0.7 + chest_ratio * 0.3) / w * 100 + random.uniform(0, 5), 2)

                cv2.putText(img, f"🍉 Bust Score: {bust_score}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (147, 20, 255), 2)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            except Exception as e:
                print("Pose error:", e)

        return img

# Streamlit UI
st.set_page_config(page_title="Bustness & Model Score App")
st.title("🔮 Bustness & Model Look Scanner")
st.write("🔍 Works live in browser using webcam. Parody/educational use only.")

webrtc_streamer(key="bust-model-app", video_processor_factory=VideoTransformer)
