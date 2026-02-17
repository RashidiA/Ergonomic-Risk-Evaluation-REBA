import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import av

# --- FAIL-SAFE INITIALIZATION ---
# Using the legacy API requires version <= 0.10.30
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        st.error("❌ MediaPipe 'solutions' missing. Version conflict detected.")
        st.info("Fix: Update requirements.txt to 'mediapipe==0.10.14' and reboot.")
        st.stop()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    READY = True
except Exception as e:
    st.error(f"❌ MediaPipe Failed: {e}")
    READY = False

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- VIDEO ENGINE ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) if READY else None
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip for selfie-view
        
        if self.pose:
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI INTERFACE ---
st.set_page_config(page_title="REBA Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

if not READY:
    st.warning("⚠️ AI engine is offline. Please check your deployment logs.")
else:
    # Sidebar for data
    op_id = st.sidebar.text_input("Operator ID", "OP-001")
    station = st.sidebar.text_input("Station", "Assembly")

    # Streamer
    ctx = webrtc_streamer(
        key="reba-main", 
        video_processor_factory=REBAProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if st.button("📸 Capture & Create Report"):
        if ctx.video_processor and ctx.video_processor.latest_frame is not None:
            img = ctx.video_processor.latest_frame
            st.success("Snapshot captured!")
            
            # PDF Creation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "GEELY REBA AUDIT REPORT", ln=True, align='C')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img)
                pdf.image(tmp.name, x=45, y=35, w=120)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Download PDF", pdf_bytes, f"REBA_{op_id}.pdf")
            os.unlink(tmp.name)
        else:
            st.warning("Start the camera first.")
