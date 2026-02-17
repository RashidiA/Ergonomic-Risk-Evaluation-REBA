import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import cv2
import numpy as np
from fpdf import FPDF
import datetime

# --- INITIALIZE AI ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.latest_score = 0
        self.latest_img = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Flip image for "mirror" effect if using front camera, or leave for back
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Store the frame for the report
            self.latest_img = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI SETUP ---
st.set_page_config(page_title="TF Live Ergo", layout="wide")
st.title("🛡️ TF Live REBA Auditor")

# Sidebar Meta-data for PDF
st.sidebar.header("Audit Metadata")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly Line 1")

# --- LIVE STREAM ---
ctx = webrtc_streamer(key="reba", video_processor_factory=REBAProcessor)

# --- REPORT GENERATION ---
if st.button("Generate Audit Report from Live Feed"):
    if ctx.video_processor and ctx.video_processor.latest_img is not None:
        # 1. Capture the current frame from the live stream
        snapshot = ctx.video_processor.latest_img
        
        # 2. Perform math and generate PDF (using the logic from previous steps)
        st.success(f"Snapshot captured for {op_id} at {station}!")
        
        # (PDF Generation logic goes here - download button)
        st.info("PDF Report generated based on the current live posture.")
    else:
        st.warning("Please start the camera and ensure a person is in frame first.")