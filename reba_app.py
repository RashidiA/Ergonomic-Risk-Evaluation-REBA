import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import tempfile
import os
import datetime
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- AUTOMATIC FIREWALL BYPASS (ICE SERVERS) ---
def get_ice_servers():
    """Fetch Twilio tokens automatically so users don't have to do anything."""
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Using standard connection (Twilio not configured): {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- AI INITIALIZATION ---
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    READY = True
except:
    READY = False

# --- VIDEO PROCESSING ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for users
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI APP ---
st.set_page_config(page_title="REBA Assessement", layout="wide")
st.title("🛡️ Live REBA Auditor")

if not READY:
    st.error("AI Engine Error. Check Python version in Advanced Settings (needs 3.11).")
    st.stop()

# Sidebar Setup
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Workstation", "Trim & Final")

# The Live Feed (Now with Auto-Config)
ctx = webrtc_streamer(
    key="reba-public",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

# --- PDF REPORTING ---
if st.button("📸 Capture & Create Audit"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Calculate Mock REBA (Simplified for demo)
        score = np.random.randint(1, 10) 
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "REBA AUDIT REPORT", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=45, y=35, w=120)
            pdf.ln(120)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Operator: {op_id} | Station: {station}", ln=True)
            pdf.cell(200, 10, f"REBA Risk Score: {score}", ln=True)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Download Report", pdf_bytes, f"REBA_{op_id}.pdf")
        os.unlink(tmp.name)
