import cv2
import math
import io
import numpy as np
import streamlit as st
import requests
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fpdf import FPDF

# --- EXPLICIT MEDIAPIPE IMPORTS (Fixes AttributeError on Streamlit Cloud) ---
import mediapipe as mp
import mediapipe.solutions.pose as mp_pose
import mediapipe.solutions.drawing_utils as mp_drawing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="REBA Ergonomic Risk Evaluator",
    page_icon="🦾",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
if "log_trunk" not in st.session_state:
    st.session_state.log_trunk = []
if "log_neck" not in st.session_state:
    st.session_state.log_neck = []
if "log_legs" not in st.session_state:
    st.session_state.log_legs = []
if "log_upper_arm" not in st.session_state:
    st.session_state.log_upper_arm = []
if "log_lower_arm" not in st.session_state:
    st.session_state.log_lower_arm = []
if "log_wrist" not in st.session_state:
    st.session_state.log_wrist = []
if "log_reba" not in st.session_state:
    st.session_state.log_reba = []

# --- HELPER FUNCTIONS FOR ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- REBA SCORING TABLES & LOGIC ---
TABLE_A = [
    # Trunk 1
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 4, 5, 6]],
    # Trunk 2
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    # Trunk 3
    [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    # Trunk 4
    [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
    # Trunk 5
    [[6, 7, 8, 9], [7, 8, 9, 9], [8, 9, 9, 9], [9, 9, 9, 9]]
]

TABLE_B = [
    # Lower Arm 1
    [[1, 2, 2], [1, 2, 3]],
    # Lower Arm 2
    [[1, 2, 3], [2, 3, 4]],
    # Lower Arm 3
    [[3, 4, 5], [4, 5, 6]],
    # Lower Arm 4
    [[4, 5, 6], [5, 6, 7]],
    # Lower Arm 5
    [[6, 7, 8], [7, 8, 9]],
    # Lower Arm 6
    [[7, 8, 9], [8, 9, 9]]
]

TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
]

def get_reba_score(trunk, neck, legs, upper_arm, lower_arm, wrist, load=0, coupling=0, activity=0):
    """Computes final REBA score based on posture ratings."""
    try:
        score_a = TABLE_A[trunk - 1][neck - 1][legs - 1] + load
        score_b = TABLE_B[upper_arm - 1][lower_arm - 1][wrist - 1] + coupling
        score_c = TABLE_C[score_a - 1][score_b - 1]
        final_reba = score_c + activity
        return min(final_reba, 12)
    except IndexError:
        return 1

# --- VIDEO PROCESSOR CLASS ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = img.shape

            # Key points
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]

            # Angles
            trunk_angle = calculate_angle([hip[0], hip[1] - 100], hip, shoulder)
            neck_angle = calculate_angle(shoulder, ear, [ear[0], ear[1] - 100])
            upper_arm_angle = calculate_angle(hip, shoulder, elbow)
            lower_arm_angle = calculate_angle(shoulder, elbow, wrist)
            leg_angle = calculate_angle(hip, knee, [knee[0], knee[1] + 100])

            # Heuristic REBA Score mapping
            trunk_score = 1 if trunk_angle < 10 else (2 if trunk_angle < 20 else 3)
            neck_score = 1 if neck_angle < 20 else 2
            legs_score = 1 if leg_angle < 30 else 2
            upper_arm_score = 1 if upper_arm_angle < 20 else (2 if upper_arm_angle < 45 else 3)
            lower_arm_score = 1 if 60 <= lower_arm_angle <= 100 else 2
            wrist_score = 1

            reba_score = get_reba_score(trunk_score, neck_score, legs_score, upper_arm_score, lower_arm_score, wrist_score)

            # Draw Landmarks
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Draw Overlay Info
            cv2.putText(img, f"REBA Score: {reba_score}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# --- STREAMLIT USER INTERFACE ---
st.title("🦾 REBA Ergonomic Risk Evaluator")
st.markdown("Automated real-time Rapid Entire Body Assessment (REBA) computer vision pipeline.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Stream")
    webrtc_streamer(
        key="reba-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=REBAProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("Manual Score Overrides")
    trunk = st.slider("Trunk Score", 1, 5, 1)
    neck = st.slider("Neck Score", 1, 3, 1)
    legs = st.slider("Legs Score", 1, 4, 1)
    upper_arm = st.slider("Upper Arm Score", 1, 6, 1)
    lower_arm = st.slider("Lower Arm Score", 1, 3, 1)
    wrist = st.slider("Wrist Score", 1, 3, 1)
    
    calculated_score = get_reba_score(trunk, neck, legs, upper_arm, lower_arm, wrist)
    
    st.metric("Manual REBA Assessment", calculated_score)
