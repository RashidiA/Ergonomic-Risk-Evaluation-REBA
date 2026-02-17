import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- SYSTEM HEALTH CHECK ---
def check_ai_engine():
    try:
        # Check if the legacy 'solutions' module exists
        if hasattr(mp.solutions, 'pose'):
            return mp.solutions.pose, mp.solutions.drawing_utils, True
        return None, None, "MediaPipe 'solutions' missing. You are likely on Python 3.12+. Switch to 3.11 in Advanced Settings."
    except Exception as e:
        return None, None, str(e)

mp_pose, mp_drawing, mp_status = check_ai_engine()

# --- APP START ---
st.set_page_config(page_title="Live REBA", layout="wide")
st.title("🛡️ Live REBA Auditor")

if mp_status is not True:
    st.error(f"❌ System Mismatch: {mp_status}")
    st.stop()

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect for mobile
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.PoseConnections)
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Sidebar
st.sidebar.header("Audit Setup")
op_id = st.sidebar.text_input("Operator ID", "OP-001")

# Live Stream
ctx = webrtc_streamer(key="reba-live", video_processor_factory=REBAProcessor)

if st.button("📸 Capture & PDF"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        # PDF logic...
        st.success("Report generated!")
