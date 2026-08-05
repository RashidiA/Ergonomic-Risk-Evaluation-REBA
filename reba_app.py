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

# --- HELPER: ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- REBA SCORING ENGINE ---
def score_trunk(angle):
    dev = abs(180 - angle)
    if dev <= 5: return 1
    if dev <= 20: return 2
    if dev <= 60: return 3
    return 4

def score_neck(angle):
    if angle <= 20: return 1
    return 2

def score_upper_arm(angle):
    if angle <= 20: return 1
    if angle <= 45: return 2
    if angle <= 90: return 3
    return 4

# --- MANUAL WEIGHT LIFTING MATRIX ---
LIFTING_MATRIX = {
    "Male": {
        "Above Shoulder": {"Close": 10.0, "Far": 5.0},
        "Shoulder to Elbow": {"Close": 20.0, "Far": 10.0},
        "Elbow to Knuckle": {"Close": 25.0, "Far": 15.0},
        "Knuckle to Mid-Leg": {"Close": 20.0, "Far": 10.0},
        "Below Mid-Leg": {"Close": 10.0, "Far": 5.0}
    },
    "Female": {
        "Above Shoulder": {"Close": 7.0, "Far": 3.0},
        "Shoulder to Elbow": {"Close": 13.0, "Far": 7.0},
        "Elbow to Knuckle": {"Close": 16.0, "Far": 10.0},
        "Knuckle to Mid-Leg": {"Close": 13.0, "Far": 7.0},
        "Below Mid-Leg": {"Close": 7.0, "Far": 3.0}
    }
}

# --- FIREWALL BYPASS (METERED.CA) ---
@st.cache_data(ttl=3600)
def get_ice_servers():
    try:
        api_key = st.secrets["METERED_API_KEY"]
        app_name = "rashidi"
        url = f"https://{app_name}.metered.live/api/v1/turn/credentials?apiKey={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]

# --- SAFE 2-PAGE PDF GENERATOR ---
def generate_2page_pdf(operator_id, profile, actual_weight, data, img_frame):
    pdf = FPDF()
    
    # PAGE 1: REBA AUDIT OVERLAY
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"REBA POSTURE AUDIT: {operator_id}", ln=True, align='C')
    pdf.ln(5)

    if img_frame is None:
        img_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img_frame, "No Frame Captured", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img_frame)
        pdf.image(tmp.name, x=35, y=30, w=140)
        tmp_path = tmp.name

    pdf.ln(100)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Posture Sub-Scores & Risk Evaluation", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Trunk Score: {data.get('trunk', 1)}", ln=True)
    pdf.cell(0, 6, f"Neck Score: {data.get('neck', 1)}", ln=True)
    pdf.cell(0, 6, f"Arm Score: {data.get('arm', 1)}", ln=True)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, f"Total REBA Score: {data.get('total', 3)}", ln=True)

    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 1 of 2 - REBA Posture Risk Evaluation", align='L')

    # PAGE 2: MANUAL WEIGHT LIFTING AUDIT
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "MANUAL WEIGHT LIFTING AUDIT", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Operator ID: {operator_id} | Profile: {profile}", ln=True)
    pdf.ln(4)

    auto_zone = data.get("auto_zone", "Elbow to Knuckle")
    auto_reach = data.get("auto_reach", "Close")
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Auto-Detected Zone: {auto_zone} ({auto_reach})", ln=True)
    pdf.cell(0, 6, f"Actual Weight Lifted: {actual_weight:.1f} kg", ln=True)
    pdf.cell(0, 6, f"Recommended Max Limit: {max_limit:.1f} kg", ln=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f"STATUS: {status}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, f"Recommended Weight Matrix Standard ({profile})", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(60, 7, "Height Zone", border=1)
    pdf.cell(60, 7, "Close Reach Limit (kg)", border=1)
    pdf.cell(60, 7, "Far Reach Limit (kg)", border=1, ln=True)

    pdf.set_font("Arial", size=9)
    for z_name, vals in LIFTING_MATRIX[profile].items():
        prefix = "-> " if z_name == auto_zone else ""
        pdf.cell(60, 7, f"{prefix}{z_name}", border=1)
        pdf.cell(60, 7, f"{vals['Close']} kg", border=1)
        pdf.cell(60, 7, f"{vals['Far']} kg", border=1, ln=True)

    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 2 of 2 - Recommended Weight Limits Matrix", align='L')

    pdf_out = pdf.output(dest='S').encode('latin-1')
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    return pdf_out

# --- VIDEO PROCESSOR ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_frame = None
        self.results_data = {
            "trunk": 1, "neck": 1, "arm": 1, "total": 3,
            "auto_zone": "Elbow to Knuckle", "auto_reach": "Close"
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            shld = [lm[11].x * w, lm[11].y * h]
            hip = [lm[23].x * w, lm[23].y * h]
            knee = [lm[25].x * w, lm[25].y * h]
            t_angle = calculate_angle(shld, hip, knee)
            t_score = score_trunk(t_angle)
            
            elbw = [lm[13].x * w, lm[13].y * h]
            a_angle = calculate_angle(hip, shld, elbw)
            a_score = score_upper_arm(a_angle)
            
            nose = [lm[0].x * w, lm[0].y * h]
            n_angle = calculate_angle(nose, shld, hip)
            n_score = score_neck(n_angle)

            wrst = [lm[15].x * w, lm[15].y * h]
            if wrst[1] < shld[1]:
                detected_zone = "Above Shoulder"
            elif shld[1] <= wrst[1] < elbw[1]:
                detected_zone = "Shoulder to Elbow"
            elif elbw[1] <= wrst[1] < hip[1]:
                detected_zone = "Elbow to Knuckle"
            elif hip[1] <= wrst[1] < knee[1]:
                detected_zone = "Knuckle to Mid-Leg"
            else:
                detected_zone = "Below Mid-Leg"

            arm_reach_dist = abs(wrst[0] - shld[0])
            detected_reach = "Far" if arm_reach_dist > (w * 0.25) else "Close"
            
            self.results_data = {
                "trunk": t_score, "neck": n_score, "arm": a_score, 
                "total": t_score + n_score + a_score,
                "auto_zone": detected_zone, "auto_reach": detected_reach
            }
            
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA: {self.results_data['total']}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        self.latest_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- INITIALIZE SESSION STATE ---
if "saved_results_data" not in st.session_state:
    st.session_state.saved_results_data = {
        "trunk": 1, "neck": 1, "arm": 1, "total": 3,
        "auto_zone": "Elbow to Knuckle", "auto_reach": "Close"
    }
if "saved_latest_frame" not in st.session_state:
    st.session_state.saved_latest_frame = None

# --- STREAMLIT UI ---
st.set_page_config(page_title="REBA AI Auditor", layout="wide")
st.title("🛡️ Live REBA Auditor")

with st.sidebar:
    st.header("Settings & MMH Inputs")
    op_id = st.text_input("Operator ID", "OP-001")
    profile = st.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
    actual_wt = st.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

ctx = webrtc_streamer(
    key="reba-ai",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

# Continuously save live updates into session state while stream is running
if ctx.video_processor:
    st.session_state.saved_results_data = ctx.video_processor.results_data
    if ctx.video_processor.latest_frame is not None:
        st.session_state.saved_latest_frame = ctx.video_processor.latest_frame

# Display Metrics from Session State (Works live AND when stopped)
data = st.session_state.saved_results_data
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trunk Score", data['trunk'])
col2.metric("Neck Score", data['neck'])
col3.metric("Arm Score", data['arm'])
col4.metric("Total Risk", data['total'])

st.caption(f"📍 Auto-Detected Lifting Zone: **{data.get('auto_zone', 'Elbow to Knuckle')} ({data.get('auto_reach', 'Close')})**")

# Report generation now pulls from session state
if st.button("📸 Generate 2-Page Audit Report"):
    if st.session_state.saved_latest_frame is not None or ctx.video_processor is not None:
        with st.spinner("Generating PDF Report..."):
            pdf_bytes = generate_2page_pdf(
                op_id, profile, actual_wt, 
                st.session_state.saved_results_data, 
                st.session_state.saved_latest_frame
            )
            st.download_button(
                label="📥 Download 2-Page PDF Report", 
                data=pdf_bytes, 
                file_name=f"Audit_{op_id}.pdf", 
                mime="application/pdf"
            )
    else:
        st.error("No captured frame found. Please start the camera briefly to record a frame before stopping.")
