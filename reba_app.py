import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import av

# --- EMERGENCY SYSTEM CHECK FOR MEDIAPIPE ---
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
        st.error("❌ CRITICAL ERROR: MediaPipe is installed but the Pose module is inaccessible.")
        st.info("Please ensure 'packages.txt' exists in GitHub with 'libgl1' and 'libglib2.0-0'.")
        st.stop()
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"❌ MediaPipe Failed to Load: {e}")
    st.stop()

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- MATHEMATIC FUNCTIONS ---
def get_angle(p1, p2, p3):
    """Calculates angle at p2"""
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- VIDEO PROCESSING CLASS ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Flip for user convenience
        img = cv2.flip(img, 1)
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- USER INTERFACE ---
st.set_page_config(page_title="Ergonomic Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

st.sidebar.header("Audit Configuration")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Workstation", "Trim & Final")
load_val = st.sidebar.selectbox("Load/Force Score", [0, 1, 2], help="0:<5kg, 1:5-10kg, 2:>10kg")
activity = st.sidebar.checkbox("Repetitive Task (+1)")

# Live Video Feed
ctx = webrtc_streamer(
    key="reba-stream", 
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- REPORT GENERATION LOGIC ---
if st.button("📸 Capture & Generate Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Re-run pose for the static image to get precise coordinates for math
        with mp_pose.Pose(static_image_mode=True) as pose_static:
            res = pose_static.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Left side joints
                s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                h = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                k = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                e = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                t_ang = int(get_angle(s, h, k))
                a_ang = int(get_angle(h, s, e))
                
                # Final REBA Score Calculation
                reba_score = (2 if t_ang > 20 else 1) + (2 if a_ang > 45 else 1) + load_val + (1 if activity else 0)
                
                # UI Display
                st.metric("Final REBA Score", reba_score)
                
                # PDF Generation
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, "GEELY REBA AUDIT REPORT", ln=True, align='C')
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, img)
                    pdf.image(tmp.name, x=40, y=40, w=130)
                    pdf.ln(135)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, f"Date: {datetime.date.today()}", ln=True)
                    pdf.cell(200, 10, f"Operator: {op_id} | Station: {station}", ln=True)
                    pdf.cell(200, 10, f"Trunk Angle: {t_ang} | Arm Angle: {a_ang}", ln=True)
                    pdf.cell(200, 10, f"REBA Score: {reba_score}", ln=True)
                    
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    st.download_button("📥 Download PDF Report", pdf_bytes, f"REBA_{op_id}.pdf")
                os.unlink(tmp.name)
            else:
                st.error("Skeleton not detected in the capture. Please stand in side-view.")
    else:
        st.warning("Please start the camera first.")
