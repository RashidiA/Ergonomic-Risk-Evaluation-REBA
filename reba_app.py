import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import av

# --- SYSTEM INITIALIZATION ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    READY = True
except Exception:
    READY = False

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- MATH LOGIC ---
def get_angle(p1, p2, p3):
    """Calculates the angle at joint p2"""
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- VIDEO ENGINE ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) if READY else None
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for mobile
        
        if self.pose:
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.latest_frame = img
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- USER INTERFACE ---
st.set_page_config(page_title="Live REBA", layout="wide")
st.title("🛡️ Live REBA Auditor")

if not READY:
    st.error("❌ System Error: MediaPipe failed to initialize.")
    st.info("Please DELETE this app from your Streamlit Dashboard and re-deploy it to refresh the system drivers.")
    st.stop()

st.sidebar.header("Audit Metadata")
op_id = st.sidebar.text_input("Operator ID", "OP-001")
station = st.sidebar.text_input("Station", "Assembly Line")
load_score = st.sidebar.selectbox("Load/Force", [0, 1, 2], help="0:<5kg, 1:5-10kg, 2:>10kg")

ctx = webrtc_streamer(
    key="reba-live", 
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- CAPTURE & MATH ---
if st.button("📸 Capture & Generate REBA Report"):
    if ctx.video_processor and ctx.video_processor.latest_frame is not None:
        img = ctx.video_processor.latest_frame
        
        # Static Pose detection for precise math
        with mp_pose.Pose(static_image_mode=True) as pose_static:
            res = pose_static.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Use left side landmarks
                s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                h = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                k = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                trunk_ang = int(get_angle(s, h, k))
                final_score = (2 if trunk_ang > 20 else 1) + load_score
                
                st.metric("Detected Trunk Angle", f"{trunk_ang}°")
                st.metric("Preliminary REBA Score", final_score)

                # Generate PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, "GEELY REBA AUDIT REPORT", ln=True, align='C')
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    cv2.imwrite(tmp.name, img)
                    pdf.image(tmp.name, x=45, y=30, w=120)
                    pdf.ln(115)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, f"Date: {datetime.date.today()} | Operator: {op_id}", ln=True)
                    pdf.cell(200, 10, f"Station: {station} | Trunk Angle: {trunk_ang} deg", ln=True)
                    pdf.cell(200, 10, f"FINAL SCORE: {final_score}", ln=True)
                    
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    st.download_button("📥 Download PDF Report", pdf_bytes, f"REBA_{op_id}.pdf")
                os.unlink(tmp.name)
            else:
                st.error("Could not find a person in the frame. Please adjust your side-view position.")
    else:
        st.warning("Please start the camera and stand in frame.")
