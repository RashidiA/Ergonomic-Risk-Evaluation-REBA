import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import cv2
import numpy as np
from fpdf import FPDF
import datetime
import tempfile
import os
import av

# Safety check for Mediapipe components
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    st.error("Mediapipe failed to load. Please check your packages.txt and requirements.txt.")

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
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            self.latest_frame = img 
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- STREAMLIT UI ---
st.set_page_config(page_title="Trim & Final Ergo Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

st.sidebar.header("Audit Details")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly")

# Live Stream
ctx = webrtc_streamer(key="reba", video_processor_factory=REBAProcessor)

if st.button("📸 Capture & Generate PDF Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        st.success("Snapshot captured!")
        
        # Simple PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, f"REBA Audit Report: {station}", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=40, y=40, w=130)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Download Report", pdf_bytes, f"REBA_{op_id}.pdf")
        os.unlink(tmp.name)
    else:
        st.warning("Start camera and ensure a person is in the frame first.")
