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

# --- 1. FREE NETWORK RELAY (Bypass Firewalls) ---
@st.cache_data(ttl=3600)
def get_ice_servers():
    """Fetches free TURN relay credentials from Metered.ca automatically."""
    try:
        # Pulls your free API key from Streamlit Secrets
        api_key = st.secrets["METERED_API_KEY"]
        url = f"https://openrelay.metered.ca/api/v1/turn/credentials?apiKey={api_key}"
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception as e:
        # Fallback to standard Google STUN if the key is missing or service is down
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- 2. AI POSE ENGINE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None
        self.risk_score = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror view for easier auditing
        
        # Process frame with Mediapipe
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Draw landmarks on the screen
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.latest_frame = img
            # Placeholder for REBA Score Logic
            self.risk_score = np.random.randint(1, 11) 
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="REBA Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")
st.write("Real-time ergonomic assessment for factory workstations.")

# Sidebar for Audit Details
with st.sidebar:
    st.header("Audit Metadata")
    op_id = st.text_input("Operator ID", "OP-99")
    station = st.selectbox("Workstation", ["Assembly", "Logistics", "Paint Shop", "Quality Control"])
    st.divider()
    st.info("The video feed uses encrypted peer-to-peer relay for privacy.")

# The WebRTC Video Streamer
ctx = webrtc_streamer(
    key="reba-audit-free",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Live Dashboard Metrics
if ctx.video_processor:
    score = ctx.video_processor.risk_score
    color = "green" if score < 4 else "orange" if score < 8 else "red"
    st.markdown(f"## Current Risk Score: <span style='color:{color}'>{score}</span>", unsafe_allow_html=True)

# 4. REPORT GENERATION (PDF)
if st.button("📸 Capture Posture & Generate Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Generate PDF in memory
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "AUTOMOTIVE - REBA AUDIT", ln=True, align='C')
        
        # Save frame to temp file to insert into PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=40, y=30, w=130)
            pdf.ln(130)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Operator: {op_id}", ln=True)
            pdf.cell(200, 10, f"Station: {station}", ln=True)
            pdf.cell(200, 10, f"Captured Score: {ctx.video_processor.risk_score}", ln=True)
            
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="📥 Download Audit PDF",
                data=pdf_output,
                file_name=f"REBA_{op_id}_{station}.pdf",
                mime="application/pdf"
            )
        os.unlink(tmp.name)
    else:
        st.warning("Please start the camera first.")
