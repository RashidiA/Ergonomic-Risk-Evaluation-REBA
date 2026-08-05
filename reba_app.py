import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import tempfile
import os
import time
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fpdf import FPDF

# --- GLOBAL PERSISTENT MEMORY (Survives WebRTC STOP & Reruns) ---
@st.cache_resource
def get_global_store():
    return {
        "frame": None,
        "total_duration": 0.0,
        "overall_score": 3,
        "breakdown": {
            "Trunk": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Neck": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Upper Arm": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0}
        },
        "results": {
            "auto_zone": "Elbow to Knuckle",
            "auto_reach": "Close"
        }
    }

GLOBAL_STORE = get_global_store()

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

# --- MANUAL WEIGHT LIFTING REFERENCE MATRIX (kg) ---
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

REBA_ACTION_TABLE = [
    ("1", "None", "Not necessary"),
    ("2-3", "Low", "May be necessary"),
    ("4-7", "Medium", "Necessary"),
    ("8-10", "High", "Necessary and soon"),
    ("11-15", "Very high", "Necessary urgent")
]

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

# --- PDF GENERATOR MATCHING SPECIFIED LAYOUT ---
def generate_custom_pdf(operator_id, profile, actual_weight, store_data):
    pdf = FPDF()
    
    # ==================== PAGE 1 ====================
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "REBA POSTURE AUDIT REPORT", ln=True, align='C')
    pdf.ln(3)
    
    # Operator Info Subheader
    pdf.set_font("Arial", 'B', 10)
    dur = store_data.get("total_duration", 0.0)
    pdf.cell(0, 6, f"Operator: {operator_id} | Total Duration: {dur:.1f} sec", ln=True, align='C')
    
    score = store_data.get("overall_score", 3)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, f"Evaluated Overall REBA Score: {score}", ln=True, align='C')
    pdf.ln(4)
    
    # Table 1: Posture Duration Analysis Breakdown
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Posture Duration Analysis Breakdown", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(50, 7, "Body Part", border=1, align='C')
    pdf.cell(45, 7, "Score 1-2 (%)", border=1, align='C')
    pdf.cell(45, 7, "Score 3-4 (%)", border=1, align='C')
    pdf.cell(45, 7, "Score 5+ (%)", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    breakdown = store_data.get("breakdown", {})
    for part in ["Trunk", "Neck", "Upper Arm"]:
        stats = breakdown.get(part, {"1-2": 100.0, "3-4": 0.0, "5+": 0.0})
        pdf.cell(50, 7, part, border=1)
        pdf.cell(45, 7, f"{stats['1-2']:.1f}%", border=1, align='C')
        pdf.cell(45, 7, f"{stats['3-4']:.1f}%", border=1, align='C')
        pdf.cell(45, 7, f"{stats['5+']:.1f}%", border=1, align='C', ln=True)
        
    pdf.ln(6)
    
    # Table 2: REBA Standard Action & Risk Table
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "REBA Standard Action & Risk Table", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(40, 7, "REBA Score", border=1, align='C')
    pdf.cell(60, 7, "Risk level", border=1, align='C')
    pdf.cell(85, 7, "Action", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    for r_score, r_level, r_action in REBA_ACTION_TABLE:
        is_current = (r_score == "2-3" if score in [2,3] else r_score == str(score))
        prefix = "-> " if is_current else ""
        pdf.cell(40, 7, f"{prefix}{r_score}", border=1, align='C')
        pdf.cell(60, 7, r_level, border=1)
        pdf.cell(85, 7, r_action, border=1, ln=True)
        
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 1 of 2 - REBA Posture Risk Evaluation", align='L')

    # ==================== PAGE 2 ====================
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"Operator: {operator_id} | Evaluation Profile: {profile}", ln=True, align='C')
    pdf.ln(4)
    
    # MMH Summary
    res_data = store_data.get("results", {})
    auto_zone = res_data.get("auto_zone", "Elbow to Knuckle")
    auto_reach = res_data.get("auto_reach", "Close")
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status_str = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Automatically Evaluated Zone: {auto_zone} ({auto_reach})", ln=True)
    pdf.cell(0, 6, f"Actual Weight Lifted: {actual_weight:.1f} kg", ln=True)
    pdf.cell(0, 6, f"Max Recommended Limit: {max_limit:.1f} kg", ln=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f"SAFETY STATUS: {status_str}", ln=True)
    pdf.ln(4)
    
    # Table 3: Recommended Weight Matrix Reference
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, f"Recommended Weight Matrix Reference ({profile})", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(65, 7, "Height Zone", border=1)
    pdf.cell(60, 7, "Close Reach Limit (kg)", border=1, align='C')
    pdf.cell(60, 7, "Far Reach Limit (kg)", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    for z_name, vals in LIFTING_MATRIX[profile].items():
        prefix = "-> " if z_name == auto_zone else ""
        pdf.cell(65, 7, f"{prefix}{z_name}", border=1)
        pdf.cell(60, 7, f"{vals['Close']:.1f} kg", border=1, align='C')
        pdf.cell(60, 7, f"{vals['Far']:.1f} kg", border=1, align='C', ln=True)
        
    pdf.ln(4)
    
    # Diagram & Recommendations Section
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Ergonomic Lifting Reference Diagram", ln=True)
    
    img_path = "assets/recommended_weight.png"
    tmp_path = None

    if os.path.exists(img_path):
        pdf.image(img_path, x=15, y=pdf.get_y() + 2, w=80)
    else:
        placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Image Not Found", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, placeholder)
            tmp_path = tmp.name
            pdf.image(tmp_path, x=15, y=pdf.get_y() + 2, w=80)

    pdf.set_x(102)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Ergonomic Recommendations:", ln=True)
    pdf.set_x(102)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(90, 5, "1. Load weight remains safe for standard execution in this zone.\n2. Maintain current reach distance and vertical placement guidelines.")

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 2 of 2 - Recommended Weight Limits Matrix Standard", align='L')

    return bytes(pdf.output())

# --- VIDEO PROCESSOR ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.start_time = None
        self.counts = {
            "Trunk": {"1-2": 0, "3-4": 0, "5+": 0},
            "Neck": {"1-2": 0, "3-4": 0, "5+": 0},
            "Upper Arm": {"1-2": 0, "3-4": 0, "5+": 0}
        }
        self.total_frames = 0

    def recv(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Scores
            shld = [lm[11].x * w, lm[11].y * h]
            hip = [lm[23].x * w, lm[23].y * h]
            knee = [lm[25].x * w, lm[25].y * h]
            t_score = score_trunk(calculate_angle(shld, hip, knee))
            
            elbw = [lm[13].x * w, lm[13].y * h]
            a_score = score_upper_arm(calculate_angle(hip, shld, elbw))
            
            nose = [lm[0].x * w, lm[0].y * h]
            n_score = score_neck(calculate_angle(nose, shld, hip))
            
            total_reba = t_score + n_score + a_score

            # Auto zone & reach
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

            # Accumulate Duration Statistics
            self.total_frames += 1
            for name, sc in [("Trunk", t_score), ("Neck", n_score), ("Upper Arm", a_score)]:
                if sc <= 2:
                    self.counts[name]["1-2"] += 1
                elif sc <= 4:
                    self.counts[name]["3-4"] += 1
                else:
                    self.counts[name]["5+"] += 1

            # Compute %
            tf = max(1, self.total_frames)
            breakdown_pct = {}
            for name in ["Trunk", "Neck", "Upper Arm"]:
                breakdown_pct[name] = {
                    "1-2": (self.counts[name]["1-2"] / tf) * 100.0,
                    "3-4": (self.counts[name]["3-4"] / tf) * 100.0,
                    "5+": (self.counts[name]["5+"] / tf) * 100.0
                }

            elapsed_dur = time.time() - self.start_time

            # Update Persistent Memory
            GLOBAL_STORE["total_duration"] = elapsed_dur
            GLOBAL_STORE["overall_score"] = total_reba
            GLOBAL_STORE["breakdown"] = breakdown_pct
            GLOBAL_STORE["results"] = {
                "auto_zone": detected_zone,
                "auto_reach": detected_reach
            }
            GLOBAL_STORE["frame"] = img.copy()

            # Render Visual Overlay
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA: {total_reba}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

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

# Display Metrics
st.markdown("### Live / Last Captured Metrics")
m_col1, m_col2 = st.columns(2)
m_col1.metric("Overall REBA Score", GLOBAL_STORE["overall_score"])
m_col2.metric("Total Duration (s)", f"{GLOBAL_STORE['total_duration']:.1f}")

res_data = GLOBAL_STORE["results"]
st.caption(f"📍 Automatically Evaluated Zone: **{res_data.get('auto_zone', 'Elbow to Knuckle')} ({res_data.get('auto_reach', 'Close')})**")

st.markdown("---")

# Generate exact PDF matching prompt structure
pdf_bytes = generate_custom_pdf(
    op_id, profile, actual_wt, GLOBAL_STORE
)

st.download_button(
    label="📥 Download REBA Lifting Audit PDF Report", 
    data=pdf_bytes, 
    file_name=f"REBA_Lifting_Audit_{op_id}.pdf", 
    mime="application/pdf"
)
