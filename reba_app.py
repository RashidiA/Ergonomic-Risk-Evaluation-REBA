import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import tempfile
import os
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- HELPER: ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    """Calculates the angle at point 'b' given points 'a' and 'c'."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- REBA SCORING ENGINE ---
def score_trunk(angle):
    # Deviation from vertical (assuming 180 is upright)
    dev = abs(180 - angle)
    if dev <= 5: return 1
    if dev <= 20: return 2
    if dev <= 60: return 3
    return 4

def score_neck(angle):
    # Angle between shoulder, ear, and a vertical reference
    # 0-20 deg = 1, >20 deg = 2
    if angle <= 20: return 1
    return 2

def score_upper_arm(angle):
    # 0-20 deg = 1, 21-45 deg = 2, 45-90 deg = 3, >90 deg = 4
    if angle <= 20: return 1
    if angle <= 45: return 2
    if angle <= 90: return 3
    return 4

# --- FIREWALL BYPASS (METERED.CA) ---
@st.cache_data(ttl=3600)
def get_ice_servers():
    try:
        api_key = st.secrets["METERED_API_KEY"]
        url = f"https://openrelay.metered.ca/api/v1/turn/credentials?apiKey={api_key}"
        return requests.get(url, timeout=5).json()
    except:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- VIDEO PROCESSOR ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_frame = None
        self.results_data = {"trunk": 0, "neck": 0, "arm": 0, "total": 0}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # 1. TRUNK ANGLE (Shoulder-Hip-Knee)
            shld = [lm[11].x * w, lm[11].y * h]
            hip = [lm[23].x * w, lm[23].y * h]
            knee = [lm[25].x * w, lm[25].y * h]
            t_angle = calculate_angle(shld, hip, knee)
            t_score = score_trunk(t_angle)
            
            # 2. UPPER ARM ANGLE (Hip-Shoulder-Elbow)
            elbw = [lm[13].x * w, lm[13].y * h]
            a_angle = calculate_angle(hip, shld, elbw)
            a_score = score_upper_arm(a_angle)
            
            # 3. NECK ANGLE (Nose-Shoulder-Hip)
            nose = [lm[0].x * w, lm[0].y * h]
            n_angle = calculate_angle(nose, shld, hip)
            n_score = score_neck(n_angle)
            
            # Update UI Data
            self.results_data = {
                "trunk": t_score, "neck": n_score, "arm": a_score, 
                "total": t_score + n_score + a_score
            }
            
            # Visual Overlays
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA: {self.results_data['total']}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.set_page_config(page_title="REBA AI Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

with st.sidebar:
    st.header("Settings")
    op_id = st.text_input("Operator ID", "OP-001")
    st.info("AI measuring Trunk, Neck, and Arms.")

ctx = webrtc_streamer(
    key="reba-ai",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_processor:
    data = ctx.video_processor.results_data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trunk Score", data['trunk'])
    col2.metric("Neck Score", data['neck'])
    col3.metric("Arm Score", data['arm'])
    col4.metric("Total Risk", data['total'])

if st.button("📸 Generate Audit Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, f"REBA AUDIT: {op_id}", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, ctx.video_processor.latest_frame)
            pdf.image(tmp.name, x=40, y=30, w=130)
            pdf.ln(135)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Final Risk Score: {data['total']}", ln=True)
            st.download_button("📥 Download PDF", pdf.output(dest='S').encode('latin-1'), f"Audit_{op_id}.pdf")
        os.unlink(tmp.name)
