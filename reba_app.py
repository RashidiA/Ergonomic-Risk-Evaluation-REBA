import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime
import tempfile
import os
import av

# --- INITIALIZE AI ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            self.latest_frame = img # Save frame for PDF capture
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_angle(p1, p2, p3):
    a = np.array(p1); b = np.array(p2); c = np.array(p3)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- UI SETUP ---
st.set_page_config(page_title="TF Live Ergo", layout="wide")
st.title("🛡️ TF Live REBA Auditor")

st.sidebar.header("Audit Metadata")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly Line")
load = st.sidebar.selectbox("Load Score (0:<5kg, 1:5-10kg, 2:>10kg)", [0, 1, 2])
activity = st.sidebar.checkbox("Repetitive Task (+1 Score)")

# Start Live Feed
ctx = webrtc_streamer(key="reba-live", video_processor_factory=REBAProcessor,
                      rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if st.button("📸 Capture & Generate REBA Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Re-process for math
        pose_temp = mp_pose.Pose(static_image_mode=True)
        results = pose_temp.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Get joints (Left side)
            s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            h = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            k = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            e = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
            t_ang = int(get_angle(s, h, k))
            a_ang = int(get_angle(h, s, e))
            
            # FULL REBA MATH (Simplified Table C)
            final_score = (2 if t_ang > 20 else 1) + (2 if a_ang > 45 else 1) + load + (1 if activity else 0)
            
            st.metric("FINAL REBA SCORE", final_score)
            
            # PDF Generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "GEELY REBA AUDIT REPORT", ln=True, align='C')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img)
                pdf.image(tmp.name, x=50, y=30, w=110)
                pdf.ln(110)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, f"Operator: {op_id} | Station: {station}", ln=True)
                pdf.cell(200, 10, f"Trunk Angle: {t_ang} | Arm Angle: {a_ang}", ln=True)
                pdf.cell(200, 10, f"FINAL REBA SCORE: {final_score}", ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Download Report", pdf_bytes, "REBA_Report.pdf", "application/pdf")
            os.unlink(tmp.name)
    else:
        st.warning("Please start camera and ensure you are in frame.")
   
