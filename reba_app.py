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
from ultralytics import YOLO

# --- LOAD LIGHTWEIGHT YOLOv8 NANO MODEL ---
@st.cache_resource
def load_yolo():
    # Downloads ~6MB yolov8n.pt on first run in Streamlit Cloud
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# --- GLOBAL PERSISTENT MEMORY ---
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
            "auto_reach": "Close",
            "detected_object": "None"
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

# COCO Classes relevant for manual handling (backpack, handbag, suitcase, bottle, box, etc.)
TARGET_OBJECT_CLASSES = [24, 26, 28, 39, 41, 63, 67] 

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

# --- PDF GENERATOR (STRICT 2-PAGE LAYOUT + YELLOW HIGHLIGHT) ---
def generate_custom_pdf(operator_id, profile, actual_weight, store_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)  # Strict page bounding to prevent page 3 overflow
    
    # ==================== PAGE 1 ====================
    pdf.add_page()
    
    # Title & Subheader
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 8, "REBA POSTURE AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    dur = store_data.get("total_duration", 0.0)
    pdf.cell(0, 5, f"Operator: {operator_id} | Total Duration: {dur:.1f} sec", ln=True, align='C')
    
    score = store_data.get("overall_score", 3)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, f"Evaluated Overall REBA Score: {score}", ln=True, align='C')
    pdf.ln(3)
    
    # Table 1: Posture Duration Analysis Breakdown
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Posture Duration Analysis Breakdown", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(50, 6, "Body Part", border=1, align='C')
    pdf.cell(45, 6, "Score 1-2 (%)", border=1, align='C')
    pdf.cell(45, 6, "Score 3-4 (%)", border=1, align='C')
    pdf.cell(45, 6, "Score 5+ (%)", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    breakdown = store_data.get("breakdown", {})
    for part in ["Trunk", "Neck", "Upper Arm"]:
        stats = breakdown.get(part, {"1-2": 100.0, "3-4": 0.0, "5+": 0.0})
        pdf.cell(50, 6, part, border=1)
        pdf.cell(45, 6, f"{stats['1-2']:.1f}%", border=1, align='C')
        pdf.cell(45, 6, f"{stats['3-4']:.1f}%", border=1, align='C')
        pdf.cell(45, 6, f"{stats['5+']:.1f}%", border=1, align='C', ln=True)
        
    pdf.ln(4)
    
    # Table 2: REBA Standard Action & Risk Table (With Yellow Highlight)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "REBA Standard Action & Risk Table", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(40, 6, "REBA Score", border=1, align='C')
    pdf.cell(60, 6, "Risk level", border=1, align='C')
    pdf.cell(85, 6, "Action", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    for r_score, r_level, r_action in REBA_ACTION_TABLE:
        if r_score == "1":
            is_match = (score == 1)
        elif r_score == "2-3":
            is_match = (score in [2, 3])
        elif r_score == "4-7":
            is_match = (4 <= score <= 7)
        elif r_score == "8-10":
            is_match = (8 <= score <= 10)
        else:
            is_match = (score >= 11)

        prefix = "-> " if is_match else ""
        
        if is_match:
            pdf.set_fill_color(255, 255, 0)
            fill_flag = True
        else:
            fill_flag = False

        pdf.cell(40, 6, f"{prefix}{r_score}", border=1, align='C', fill=fill_flag)
        pdf.cell(60, 6, r_level, border=1, fill=fill_flag)
        pdf.cell(85, 6, r_action, border=1, ln=True, fill=fill_flag)
        
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 1 of 2 - REBA Posture Risk Evaluation", align='L')

    # ==================== PAGE 2 ====================
    pdf.add_page()
    
    # Title & Subheader
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Evaluation Profile: {profile}", ln=True, align='C')
    pdf.ln(3)
    
    # MMH Summary
    res_data = store_data.get("results", {})
    auto_zone = res_data.get("auto_zone", "Elbow to Knuckle")
    auto_reach = res_data.get("auto_reach", "Close")
    detected_obj = res_data.get("detected_object", "None")
    
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status_str = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Automatically Evaluated Zone: {auto_zone} ({auto_reach})", ln=True)
    pdf.cell(0, 5, f"YOLO Detected Object: {detected_obj.title()}", ln=True)
    pdf.cell(0, 5, f"Actual Weight Lifted: {actual_weight:.1f} kg", ln=True)
    pdf.cell(0, 5, f"Max Recommended Limit: {max_limit:.1f} kg", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, f"SAFETY STATUS: {status_str}", ln=True)
    pdf.ln(3)
    
    # Table 3: Recommended Weight Matrix Reference (With Yellow Highlight)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"Recommended Weight Matrix Reference ({profile})", ln=True)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(65, 6, "Height Zone", border=1)
    pdf.cell(60, 6, "Close Reach Limit (kg)", border=1, align='C')
    pdf.cell(60, 6, "Far Reach Limit (kg)", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=9)
    for z_name, vals in LIFTING_MATRIX[profile].items():
        is_active_zone = (z_name == auto_zone)
        prefix = "-> " if is_active_zone else ""
        
        pdf.cell(65, 6, f"{prefix}{z_name}", border=1)
        
        if is_active_zone and auto_reach == "Close":
            pdf.set_fill_color(255, 255, 0)
            pdf.cell(60, 6, f"{vals['Close']:.1f} kg", border=1, align='C', fill=True)
        else:
            pdf.cell(60, 6, f"{vals['Close']:.1f} kg", border=1, align='C', fill=False)

        if is_active_zone and auto_reach == "Far":
            pdf.set_fill_color(255, 255, 0)
            pdf.cell(60, 6, f"{vals['Far']:.1f} kg", border=1, align='C', fill=True, ln=True)
        else:
            pdf.cell(60, 6, f"{vals['Far']:.1f} kg", border=1, align='C', fill=False, ln=True)
        
    pdf.ln(3)
    
    # Diagram & Recommendations Section
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Ergonomic Lifting Reference Diagram", ln=True)
    
    img_path = "assets/recommended_weight.png"
    tmp_path = None
    y_pos = pdf.get_y() + 2

    if os.path.exists(img_path):
        pdf.image(img_path, x=15, y=y_pos, w=75)
    else:
        placeholder = np.zeros((250, 350, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Image Not Found", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, placeholder)
            tmp_path = tmp.name
            pdf.image(tmp_path, x=15, y=y_pos, w=75)

    pdf.set_xy(98, y_pos)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Ergonomic Recommendations:", ln=True)
    pdf.set_x(98)
    pdf.set_font("Arial", size=8.5)
    pdf.multi_cell(95, 4.5, "1. Load weight remains safe for standard execution in this zone.\n2. Maintain current reach distance and vertical placement guidelines.")

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 2 of 2 - Recommended Weight Limits Matrix Standard", align='L')

    return bytes(pdf.output())

# --- HYBRID VIDEO PROCESSOR (MEDIAPIPE + YOLO FRAME SKIPPING) ---
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
        self.last_detected_obj = "None"

    def recv(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.total_frames += 1
        
        # --- 1. YOLO OBJECT DETECTION (SKIPPED TO RUN ONCE EVERY 10 FRAMES FOR SPEED) ---
        if self.total_frames % 10 == 0:
            try:
                yolo_res = yolo_model(img, verbose=False, conf=0.35)[0]
                detected_boxes = yolo_res.boxes
                
                found_obj = "None"
                for box in detected_boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in TARGET_OBJECT_CLASSES or cls_id != 0: # Exclude person (0)
                        found_obj = yolo_model.names[cls_id]
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                        cv2.putText(img, f"Object: {found_obj}", (b[0], max(20, b[1]-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        break
                self.last_detected_obj = found_obj
            except Exception:
                pass

        # --- 2. MEDIAPIPE POSE EVALUATION (EVERY FRAME FOR SMOOTH TRACKING) ---
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Keypoints
            shld = [lm[11].x * w, lm[11].y * h]
            hip = [lm[23].x * w, lm[23].y * h]
            knee = [lm[25].x * w, lm[25].y * h]
            elbw = [lm[13].x * w, lm[13].y * h]
            nose = [lm[0].x * w, lm[0].y * h]
            wrst = [lm[15].x * w, lm[15].y * h]

            # REBA Scores
            t_score = score_trunk(calculate_angle(shld, hip, knee))
            a_score = score_upper_arm(calculate_angle(hip, shld, elbw))
            n_score = score_neck(calculate_angle(nose, shld, hip))
            total_reba = t_score + n_score + a_score

            # Auto Zone Calculation
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
            for name, sc in [("Trunk", t_score), ("Neck", n_score), ("Upper Arm", a_score)]:
                if sc <= 2:
                    self.counts[name]["1-2"] += 1
                elif sc <= 4:
                    self.counts[name]["3-4"] += 1
                else:
                    self.counts[name]["5+"] += 1

            # Compute Duration %
            tf = max(1, self.total_frames)
            breakdown_pct = {}
            for name in ["Trunk", "Neck", "Upper Arm"]:
                breakdown_pct[name] = {
                    "1-2": (self.counts[name]["1-2"] / tf) * 100.0,
                    "3-4": (self.counts[name]["3-4"] / tf) * 100.0,
                    "5+": (self.counts[name]["5+"] / tf) * 100.0
                }

            elapsed_dur = time.time() - self.start_time

            # Sync Global Store
            GLOBAL_STORE["total_duration"] = elapsed_dur
            GLOBAL_STORE["overall_score"] = total_reba
            GLOBAL_STORE["breakdown"] = breakdown_pct
            GLOBAL_STORE["results"] = {
                "auto_zone": detected_zone,
                "auto_reach": detected_reach,
                "detected_object": self.last_detected_obj
            }
            GLOBAL_STORE["frame"] = img.copy()

            # Visual Overlays
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA Score: {total_reba}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.set_page_config(page_title="REBA + YOLO AI Auditor", layout="wide")
st.title("🛡️ Live REBA & Object-Aware Ergonomic Auditor")

with st.sidebar:
    st.header("Settings & MMH Inputs")
    op_id = st.text_input("Operator ID", "OP-001")
    profile = st.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
    actual_wt = st.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

ctx = webrtc_streamer(
    key="reba-yolo-ai",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

# Live Metrics Display
st.markdown("### Live / Last Captured Metrics")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Overall REBA Score", GLOBAL_STORE["overall_score"])
m_col2.metric("Total Duration (s)", f"{GLOBAL_STORE['total_duration']:.1f}")
res_data = GLOBAL_STORE["results"]
m_col3.metric("YOLO Object Detected", res_data.get("detected_object", "None").title())

st.caption(f"📍 Automatically Evaluated Zone: **{res_data.get('auto_zone', 'Elbow to Knuckle')} ({res_data.get('auto_reach', 'Close')})**")

st.markdown("---")

# Download PDF Report
pdf_bytes = generate_custom_pdf(
    op_id, profile, actual_wt, GLOBAL_STORE
)

st.download_button(
    label="📥 Download REBA Lifting Audit PDF Report", 
    data=pdf_bytes, 
    file_name=f"REBA_Lifting_Audit_{op_id}.pdf", 
    mime="application/pdf"
)
