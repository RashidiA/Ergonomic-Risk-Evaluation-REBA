import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF
import av
import tempfile
import os

# --- FORCE INITIALIZATION ---
# This helps prevent the AttributeError by checking the module directly
if not hasattr(mp.solutions, 'pose'):
    st.error("System Error: Mediapipe Pose module not found. Please ensure packages.txt contains libgl1.")
else:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # AI Processing
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.latest_frame = img # Save for PDF
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Trim and Final Ergo Auditor")
st.title("🛡️ Live REBA Auditor")

# Metadata sidebar
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly")

# Live Feed
ctx = webrtc_streamer(key="reba", video_processor_factory=REBAProcessor)

if st.button("📸 Capture & Generate PDF Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        st.success("Analysis Captured!")
        
        # Simple PDF logic
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, f"REBA Audit: {station}", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=40, y=40, w=130)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Download Report", pdf_bytes, f"REBA_{op_id}.pdf")
        os.unlink(tmp.name)
    else:
        st.warning("Please start the camera and ensure someone is in view.")
