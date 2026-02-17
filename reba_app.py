import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import av

# --- DELAYED SYSTEM CHECK ---
# We use st.cache_resource to ensure this only runs once and doesn't crash the UI loop
@st.cache_resource
def load_mediapipe():
    try:
        import mediapipe as mp
        return mp.solutions.pose, mp.solutions.drawing_utils, True
    except Exception as e:
        return None, None, str(e)

mp_pose, mp_drawing, mp_status = load_mediapipe()

# --- UI ERROR HANDLING ---
st.set_page_config(page_title="Live Ergo", layout="wide")
st.title("🛡️ Live REBA Auditor")

if mp_status is not True:
    st.error(f"❌ AI Engine Error: {mp_status}")
    st.warning("This error usually means 'packages.txt' was not found or ignored during deployment.")
    st.info("Try deleting the app from your Streamlit dashboard and re-deploying it from scratch.")
    st.stop()

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- VIDEO LOGIC ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for mobile use
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- APP INTERFACE ---
st.sidebar.header("Audit Details")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly")

ctx = webrtc_streamer(
    key="reba-stream", 
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if st.button("📸 Capture & Generate Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        st.success("Snapshot captured!")
        
        # Simple PDF Generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "GEELY REBA AUDIT REPORT", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=40, y=40, w=130)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Download PDF", pdf_bytes, f"REBA_{op_id}.pdf")
        os.unlink(tmp.name)
