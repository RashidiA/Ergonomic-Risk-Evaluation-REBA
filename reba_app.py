import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import time
import tempfile
import os
import requests
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fpdf import FPDF

# --- INITIALIZE SESSION STATE ---
if "log_trunk" not in st.session_state:
    st.session_state.log_trunk = []
if "log_neck" not in st.session_state:
    st.session_state.log_neck = []
if "log_arm" not in st.session_state:
    st.session_state.log_arm = []
if "detected_zones" not in st.session_state:
    st.session_state.detected_zones = []
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "recording_duration" not in st.session_state:
    st.session_state.recording_duration = 1.0

# --- HELPER: ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

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

# --- ERGONOMIC WEIGHT LIFTING LIMITS ---
WEIGHT_LIMITS = {
    "Male": {
        "Above Shoulder (Far)": 5.0, "Above Shoulder (Close)": 10.0,
        "Shoulder to Elbow (Far)": 10.0, "Shoulder to Elbow (Close)": 20.0,
        "Elbow to Knuckle (Far)": 15.0, "Elbow to Knuckle (Close)": 25.0,
        "Knuckle to Mid-Leg (Far)": 10.0, "Knuckle to Mid-Leg (Close)": 20.0,
        "Below Mid-Leg (Far)": 5.0, "Below Mid-Leg (Close)": 10.0
    },
    "Female": {
        "Above Shoulder (Far)": 3.0, "Above Shoulder (Close)": 7.0,
        "Shoulder to Elbow (Far)": 7.0, "Shoulder to Elbow (Close)": 13.0,
        "Elbow to Knuckle (Far)": 10.0, "Elbow to Knuckle (Close)": 16.0,
        "Knuckle to Mid-Leg (Far)": 7.0, "Knuckle to Mid-Leg (Close)": 13.0,
        "Below Mid-Leg (Far)": 3.0, "Below Mid-Leg (Close)": 7.0
    }
}

# --- FIREWALL BYPASS ---
@st.cache_data(ttl=3600)
def get_ice_servers():
    api_key = st.secrets.get("METERED_API_KEY", "")
    app_name = "rashidi"
    try:
        url = f"https://{app_name}.metered.live/api/v1/turn/credentials?apiKey={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]

# --- VIDEO PROCESSOR WITH CONTINUOUS LOGGING ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_frame = None
        self.results_data = {"trunk": 1, "neck": 1, "arm": 1, "total": 3}
        
        # Continuous rolling logs (keeps last 300 frames)
        self.log_trunk = []
        self.log_neck = []
        self.log_arm = []
        self.detected_zones = []
        self.start_time = time.time()

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
            elbw = [lm[13].x * w, lm[13].y * h]
            wrist = [lm[15].x * w, lm[15].y * h]
            nose = [lm[0].x * w, lm[0].y * h]

            t_score = score_trunk(calculate_angle(shld, hip, knee))
            a_score = score_upper_arm(calculate_angle(hip, shld, elbw))
            n_score = score_neck(calculate_angle(nose, shld, hip))
            total_score = t_score + n_score + a_score

            self.results_data = {
                "trunk": t_score, "neck": n_score, "arm": a_score, "total": total_score
            }

            # Pose-based lifting zone detection
            wrist_y = wrist[1]
            if wrist_y < shld[1]:
                zone = "Above Shoulder (Close)"
            elif wrist_y < (shld[1] + hip[1]) / 2:
                zone = "Shoulder to Elbow (Close)"
            elif wrist_y < hip[1]:
                zone = "Elbow to Knuckle (Close)"
            elif wrist_y < knee[1]:
                zone = "Knuckle to Mid-Leg (Close)"
            else:
                zone = "Below Mid-Leg (Close)"
            
            # Continuously append logs (limit to last 300 frames)
            self.log_trunk.append(t_score)
            self.log_neck.append(n_score)
            self.log_arm.append(a_score)
            self.detected_zones.append(zone)

            if len(self.log_trunk) > 300:
                self.log_trunk.pop(0)
                self.log_neck.pop(0)
                self.log_arm.pop(0)
                self.detected_zones.pop(0)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"STATUS: AR LIVE | REBA SCORE: {total_score}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            self.latest_frame = img

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- PDF REPORT GENERATOR ---
def generate_pdf_report(op_id, duration, gender, actual_weight, 
                        trunk_stats, neck_stats, arm_stats, detected_zone, max_weight, frame_img):
    pdf = FPDF()
    
    # PAGE 1
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 10, "REBA POSTURE AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 6, f"Operator: {op_id} | Total Duration: {duration:.1f} sec", ln=True, align='C')
    pdf.ln(5)
    
    if frame_img is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame_img)
            pdf.image(tmp.name, x=45, y=32, w=120)
            os.unlink(tmp.name)
        pdf.ln(95)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 8, "Posture Duration Analysis Breakdown", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    
    headers = ["Body Part", "Score 1-2 (%)", "Score 3-4 (%)", "Score 5+ (%)"]
    w_col = [45, 45, 45, 45]
    for i, h in enumerate(headers):
        pdf.cell(w_col[i], 8, h, border=1, align='C')
    pdf.ln()

    body_data = [("Trunk", trunk_stats), ("Neck", neck_stats), ("Upper Arm", arm_stats)]
    for label, stats in body_data:
        pdf.cell(w_col[0], 8, label, border=1)
        pdf.cell(w_col[1], 8, f"{stats.get('low', 0):.1f}%", border=1, align='C')
        pdf.cell(w_col[2], 8, f"{stats.get('mid', 0):.1f}%", border=1, align='C')
        pdf.cell(w_col[3], 8, f"{stats.get('high', 0):.1f}%", border=1, align='C')
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(190, 6, "Page 1 of 2 — Posture Risk Evaluation", align='C')

    # PAGE 2
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 10, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 6, f"Operator: {op_id} | Evaluation Profile: {gender}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 8, "Manual Material Handling Evaluation", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(190, 8, f"Automatically Evaluated Zone: {detected_zone}", border='B', ln=True)
    pdf.cell(190, 8, f"Actual Weight Lifted: {actual_weight:.1f} kg", border='B', ln=True)
    pdf.cell(190, 8, f"Max Recommended Limit: {max_weight:.1f} kg", border='B', ln=True)
    pdf.ln(8)
    
    is_exceeded = actual_weight > max_weight
    pdf.set_font("Arial", 'B', 14)
    if is_exceeded:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(190, 12, f"SAFETY STATUS: EXCEEDED RECOMMENDED LIMIT (+{(actual_weight - max_weight):.1f} kg)", border=1, align='C', ln=True)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(190, 12, "SAFETY STATUS: WITHIN SAFE ERGONOMIC LIMIT", border=1, align='C', ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, "Ergonomic Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    if is_exceeded:
        pdf.multi_cell(190, 6, "1. Reduce load weight or utilize mechanical lifting assistance (e.g., hoist, vacuum lifter).\n2. Reposition storage height closer to elbow/knuckle level to increase threshold.\n3. Implement job rotation or dual-operator lifting protocols.")
    else:
        pdf.multi_cell(190, 6, "1. Load weight remains safe for standard execution in this zone.\n2. Maintain current reach distance and vertical placement guidelines.")

    pdf.ln(40)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(190, 6, "Page 2 of 2 — Weight Limits Based on Recommended Ergonomic Standards", align='C')

    return pdf.output(dest='S').encode('latin-1')

def compute_percentage_breakdown(log_list):
    if not log_list:
        return {"low": 100.0, "mid": 0.0, "high": 0.0}
    total = len(log_list)
    counts = Counter(log_list)
    return {
        "low": sum(counts[s] for s in [1, 2]) / total * 100.0,
        "mid": sum(counts[s] for s in [3, 4]) / total * 100.0,
        "high": sum(counts[s] for s in counts if s >= 5) / total * 100.0
    }

# --- UI SETUP ---
st.set_page_config(page_title="REBA & Lifting Ergonomic Auditor", layout="wide")
st.title("🛡️ REBA & Weight Lifting Ergonomic Auditor")

with st.sidebar:
    st.header("📋 Session Parameters")
    op_id = st.text_input("Operator ID", "OP-001")
    st.markdown("---")
    st.header("🏋️ Manual Lifting Settings")
    gender = st.selectbox("Operator Gender", ["Male", "Female"])
    actual_weight = st.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

ctx = webrtc_streamer(
    key="reba-ai",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_processor:
    data = ctx.video_processor.results_data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trunk Score", data['trunk'])
    c2.metric("Neck Score", data['neck'])
    c3.metric("Arm Score", data['arm'])
    c4.metric("Total REBA Risk", data['total'])

st.markdown("---")
st.header("📄 Audit PDF Generation")

if st.button("📸 Generate 2-Page PDF Audit Report"):
    # Multi-tier Data Extraction (Guarantees PDF Generation)
    log_trunk = []
    log_neck = []
    log_arm = []
    detected_zones = []
    frame_img = None
    duration = 5.0

    if ctx.video_processor:
        log_trunk = list(ctx.video_processor.log_trunk)
        log_neck = list(ctx.video_processor.log_neck)
        log_arm = list(ctx.video_processor.log_arm)
        detected_zones = list(ctx.video_processor.detected_zones)
        frame_img = ctx.video_processor.latest_frame
        duration = max(1.0, time.time() - ctx.video_processor.start_time)
    
    # Fallback to current processor state if log is empty
    if not log_trunk and ctx.video_processor:
        d = ctx.video_processor.results_data
        log_trunk = [d['trunk']]
        log_neck = [d['neck']]
        log_arm = [d['arm']]

    t_stats = compute_percentage_breakdown(log_trunk)
    n_stats = compute_percentage_breakdown(log_neck)
    a_stats = compute_percentage_breakdown(log_arm)
    
    if detected_zones:
        detected_zone = Counter(detected_zones).most_common(1)[0][0]
    else:
        detected_zone = "Elbow to Knuckle (Close)"
        
    max_rec_weight = WEIGHT_LIMITS[gender].get(detected_zone, 10.0)

    pdf_bytes = generate_pdf_report(
        op_id=op_id,
        duration=duration,
        gender=gender,
        actual_weight=actual_weight,
        trunk_stats=t_stats,
        neck_stats=n_stats,
        arm_stats=a_stats,
        detected_zone=detected_zone,
        max_weight=max_rec_weight,
        frame_img=frame_img
    )
    
    st.download_button(
        label="📥 Download 2-Page Audit PDF",
        data=pdf_bytes,
        file_name=f"REBA_Lifting_Audit_{op_id}.pdf",
        mime="application/pdf"
    )
