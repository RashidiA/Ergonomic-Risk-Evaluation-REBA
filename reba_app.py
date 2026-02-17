import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF
import av
import tempfile
import os

# --- HEAVY DUTY INITIALIZATION ---
try:
    # We define these globally so the Processor can find them
    MP_POSE = mp.solutions.pose
    MP_DRAWING = mp.solutions.drawing_utils
    POSE_MODEL = MP_POSE.Pose(static_image_mode=False, min_detection_confidence=0.5)
    READY = True
except Exception as e:
    st.error(f"MediaPipe Initialization Failed: {e}")
    READY = False

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if READY:
            # AI Processing
            results = POSE_MODEL.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                MP_DRAWING.draw_landmarks(img, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS)
                self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="TF Ergo Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

if not READY:
    st.warning("⚠️ The AI engine (MediaPipe) is not responding. Please check 'Manage App' -> 'Logs' to see if 'libGL.so.1' is missing.")
else:
    # Live Feed
    ctx = webrtc_streamer(key="reba", video_processor_factory=REBAProcessor)

    if st.button("📸 Capture & Generate Report"):
        if ctx.video_processor and ctx.video_processor.latest_frame is not None:
            # (Keep your existing PDF generation logic here)
            st.success("Analysis Captured!")
        else:
            st.info("Start the camera and stand in the side-view profile.")
