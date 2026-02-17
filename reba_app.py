import streamlit as st
import cv2
import mediapipe as mp
import av
import tempfile
import os
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- AUTOMATIC ICE SERVER GENERATION ---
@st.cache_data(ttl=3600)  # Cache servers for 1 hour to save Twilio credits
def get_ice_servers():
    """Generates fresh firewall-bypass credentials automatically for users."""
    try:
        # Pulls your developer credentials from Streamlit Secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        # Fallback to free Google STUN if Twilio secrets are missing
        st.error(f"Network Relay Error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- POSE ENGINE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror view for mobile users
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- USER INTERFACE ---
st.set_page_config(page_title="REBA Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

# Metadata collected from the auditor
with st.sidebar:
    st.header("Audit Setup")
    op_id = st.text_input("Operator ID", "OP-001")
    station = st.text_input("Workstation", "Main Assembly")

# The Live Streamer with AUTO-CONFIGURATION
ctx = webrtc_streamer(
    key="reba-audit-pro",
    video_processor_factory=REBAProcessor,
    # This line connects users to your Twilio Relay automatically
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture & Generate Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Simple PDF Generation Logic
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "REBA AUDIT REPORT", ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, img)
            pdf.image(tmp.name, x=45, y=35, w=120)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Download PDF", pdf_bytes, f"REBA_{op_id}.pdf")
        os.unlink(tmp.name)
