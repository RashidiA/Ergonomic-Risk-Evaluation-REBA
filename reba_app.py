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

# --- LOAD & PRE-WARM YOLO MODEL ---
@st.cache_resource
def load_and_warmup_yolo():
    model = YOLO("yolov8n.pt")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = model(dummy_img, verbose=False)
    return model

yolo_model = load_and_warmup_yolo()

# --- GLOBAL PERSISTENT MEMORY ---
@st.cache_resource
def get_global_store():
    return {
        "frame": None,
        "total_duration": 0.0,
        "overall_score": 3,
        "last_detected_object": "None (Hands Free)",  # Persistent Latch
        "breakdown": {
            "Trunk": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Neck": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Upper Arm": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Legs": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0},
            "Wrists": {"1-2": 100.0, "3-4": 0.0, "5+": 0.0}
        },
        "results": {
            "auto_zone": "Elbow to Knuckle",
            "auto_reach": "Close",
            "detected_object": "None (Hands Free)"
        },
        "niosh": {
            "h_cm": 30.0, "v_cm": 75.0, "d_cm": 25.0, "angle_deg": 0.0,
            "hm": 0.83, "vm": 1.00, "dm": 1.00, "am": 1.00, "fm": 0.95, "cm": 1.00,
            "rwl": 18.2, "li": 0.44, "status": "SAFE (LI <= 1.0)"
        }
    }

GLOBAL_STORE = get_global_store()

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

def score_legs(knee_angle):
    flexion = abs(180.0 - knee_angle)
    if flexion <= 30: return 1
    if flexion <= 60: return 2
    return 3

def score_wrists(wrist_angle):
    dev = abs(180.0 - wrist_angle)
    if dev <= 15: return 1
    return 2

# --- NIOSH CALCULATOR ENGINE ---
def compute_niosh(h_cm, v_cm, d_cm, angle_deg, actual_weight, fm=0.95, cm=1.00):
    lc = 23.0  # Load Constant in kg
    
    h_cm = max(25.0, min(63.0, h_cm))
    hm = 25.0 / h_cm
    
    v_cm = max(0.0, min(175.0, v_cm))
    vm = max(0.0, 1.0 - (0.003 * abs(v_cm - 75.0)))
    
    d_cm = max(25.0, min(175.0, d_cm))
    dm = 0.82 + (4.5 / d_cm)
    
    angle_deg = max(0.0, min(135.0, angle_deg))
    am = max(0.0, 1.0 - (0.0032 * angle_deg))
    
    rwl = lc * hm * vm * dm * am * fm * cm
    li = actual_weight / max(0.1, rwl)
    status = "SAFE (LI <= 1.0)" if li <= 1.0 else "UNSAFE / HIGH RISK (LI > 1.0)"
    
    return {
        "h_cm": h_cm, "v_cm": v_cm, "d_cm": d_cm, "angle_deg": angle_deg,
        "hm": hm, "vm": vm, "dm": dm, "am": am, "fm": fm, "cm": cm,
        "rwl": rwl, "li": li, "status": status
    }

# --- HELPER: BOUNDING BOX INTERSECTION ---
def check_hand_object_intersection(hand_box, obj_box):
    hx1, hy1, hx2, hy2 = hand_box
    ox1, oy1, ox2, oy2 = obj_box
    ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
    ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
    return (ix1 < ix2) and (iy1 < iy2)

# --- ADVANCED FINGER-BASED NON-COCO UNKNOWN OBJECT DETECTOR ---
def detect_object_near_fingers(img, finger_pts, img_w, img_h):
    if not finger_pts:
        return False, None

    xs = [pt[0] for pt in finger_pts]
    ys = [pt[1] for pt in finger_pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    margin_x = int(img_w * 0.22)
    margin_y = int(img_h * 0.25)
    
    fx1 = max(0, int(min_x - margin_x))
    fy1 = max(0, int(min_y - margin_y))
    fx2 = min(img_w, int(max_x + margin_x))
    fy2 = min(img_h, int(max_y + margin_y))
    
    roi = img[fy1:fy2, fx1:fx2]
    if roi.size == 0 or (fx2 - fx1) < 20 or (fy2 - fy1) < 20:
        return False, None

    # 1. Skin tone filtering (HSV space)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    non_skin_mask = cv2.bitwise_not(skin_mask)

    # 2. Thresholding non-skin visual structures (Otsu Adaptive)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    combined = cv2.bitwise_and(otsu_thresh, non_skin_mask)
    
    # 3. Morphological closing to fill object solid shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi_area = (fx2 - fx1) * (fy2 - fy1)
    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area > (0.08 * roi_area):
            x, y, w, h = cv2.boundingRect(c)
            abs_box = (fx1 + x, fy1 + y, fx1 + x + w, fy1 + y + h)
            return True, abs_box

    return False, None

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

REBA_ACTION_TABLE = [
    ("1", "None", "Not necessary"),
    ("2-3", "Low", "May be necessary"),
    ("4-7", "Medium", "Necessary"),
    ("8-10", "High", "Necessary and soon"),
    ("11-15", "Very high", "Necessary urgent")
]

# --- FIREWALL BYPASS ---
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

# --- 3-PAGE PDF GENERATOR ---
def generate_custom_pdf(operator_id, profile, actual_weight, store_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    
    # PAGE 1: REBA POSTURE AUDIT
    pdf.add_page()
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 8, "REBA POSTURE AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    dur = store_data.get("total_duration", 0.0)
    pdf.cell(0, 5, f"Operator: {operator_id} | Total Duration: {dur:.1f} sec", ln=True, align='C')
    
    score = store_data.get("overall_score", 3)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, f"Evaluated Overall REBA Score: {score}", ln=True, align='C')
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "Full-Body Posture Duration Breakdown", ln=True)
    pdf.set_font("Arial", 'B', 8.5)
    pdf.cell(50, 5, "Body Part", border=1, align='C')
    pdf.cell(45, 5, "Score 1-2 (%)", border=1, align='C')
    pdf.cell(45, 5, "Score 3-4 (%)", border=1, align='C')
    pdf.cell(45, 5, "Score 5+ (%)", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=8.5)
    breakdown = store_data.get("breakdown", {})
    for part in ["Trunk", "Neck", "Upper Arm", "Legs", "Wrists"]:
        stats = breakdown.get(part, {"1-2": 100.0, "3-4": 0.0, "5+": 0.0})
        pdf.cell(50, 5, part, border=1)
        pdf.cell(45, 5, f"{stats['1-2']:.1f}%", border=1, align='C')
        pdf.cell(45, 5, f"{stats['3-4']:.1f}%", border=1, align='C')
        pdf.cell(45, 5, f"{stats['5+']:.1f}%", border=1, align='C', ln=True)
        
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "REBA Standard Action & Risk Table", ln=True)
    pdf.set_font("Arial", 'B', 8.5)
    pdf.cell(40, 5, "REBA Score", border=1, align='C')
    pdf.cell(60, 5, "Risk level", border=1, align='C')
    pdf.cell(85, 5, "Action", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=8.5)
    for r_score, r_level, r_action in REBA_ACTION_TABLE:
        if r_score == "1": is_match = (score == 1)
        elif r_score == "2-3": is_match = (score in [2, 3])
        elif r_score == "4-7": is_match = (4 <= score <= 7)
        elif r_score == "8-10": is_match = (8 <= score <= 10)
        else: is_match = (score >= 11)

        prefix = "-> " if is_match else ""
        fill_flag = is_match
        if fill_flag: pdf.set_fill_color(255, 255, 0)

        pdf.cell(40, 5, f"{prefix}{r_score}", border=1, align='C', fill=fill_flag)
        pdf.cell(60, 5, r_level, border=1, fill=fill_flag)
        pdf.cell(85, 5, r_action, border=1, ln=True, fill=fill_flag)
        
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 1 of 3 - REBA Posture Risk Evaluation", align='L')

    # PAGE 2: MANUAL MATERIAL HANDLING AUDIT
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Evaluation Profile: {profile}", ln=True, align='C')
    pdf.ln(3)
    
    res_data = store_data.get("results", {})
    auto_zone = res_data.get("auto_zone", "Elbow to Knuckle")
    auto_reach = res_data.get("auto_reach", "Close")
    
    # Use Persistent Latched Object for Report
    detected_obj = store_data.get("last_detected_object", res_data.get("detected_object", "None (Hands Free)"))
    
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status_str = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Automatically Evaluated Zone: {auto_zone} ({auto_reach})", ln=True)
    pdf.cell(0, 5, f"YOLO / Sensor Detected Object: {detected_obj.title()}", ln=True)
    pdf.cell(0, 5, f"Actual Weight Lifted: {actual_weight:.1f} kg", ln=True)
    pdf.cell(0, 5, f"Max Recommended Limit: {max_limit:.1f} kg", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, f"SAFETY STATUS: {status_str}", ln=True)
    pdf.ln(3)
    
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
    pdf.cell(0, 10, "Page 2 of 3 - Recommended Weight Limits Matrix Standard", align='L')

    # PAGE 3: NIOSH LIFTING EQUATION AUDIT
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "NIOSH LIFTING EQUATION ASSESSMENT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Trigger Source: Object Detection", ln=True, align='C')
    pdf.ln(3)
    
    nd = store_data.get("niosh", {})
    obj_name_str = store_data.get("last_detected_object", res_data.get("detected_object", "None (Hands Free)"))
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "1. Object & Load Condition", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Object Detected: {obj_name_str.title()}", ln=True)
    pdf.cell(0, 5, f"Actual Object Weight: {actual_weight:.1f} kg", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "2. NIOSH Multipliers & Spatial Geometry", ln=True)
    
    pdf.set_font("Arial", 'B', 8.5)
    pdf.cell(60, 5, "Parameter / Multiplier", border=1)
    pdf.cell(40, 5, "Measured Value", border=1, align='C')
    pdf.cell(40, 5, "Multiplier Factor", border=1, align='C')
    pdf.cell(45, 5, "Formula / Standard", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=8.5)
    pdf.cell(60, 5, "Load Constant (LC)", border=1); pdf.cell(40, 5, "23.0 kg", border=1, align='C'); pdf.cell(40, 5, "1.00", border=1, align='C'); pdf.cell(45, 5, "Baseline Load", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Horizontal Multiplier (HM)", border=1); pdf.cell(40, 5, f"{nd['h_cm']:.1f} cm", border=1, align='C'); pdf.cell(40, 5, f"{nd['hm']:.2f}", border=1, align='C'); pdf.cell(45, 5, "25 / H", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Vertical Multiplier (VM)", border=1); pdf.cell(40, 5, f"{nd['v_cm']:.1f} cm", border=1, align='C'); pdf.cell(40, 5, f"{nd['vm']:.2f}", border=1, align='C'); pdf.cell(45, 5, "1 - 0.003|V - 75|", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Distance Multiplier (DM)", border=1); pdf.cell(40, 5, f"{nd['d_cm']:.1f} cm", border=1, align='C'); pdf.cell(40, 5, f"{nd['dm']:.2f}", border=1, align='C'); pdf.cell(45, 5, "0.82 + (4.5 / D)", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Asymmetric Multiplier (AM)", border=1); pdf.cell(40, 5, f"{nd['angle_deg']:.1f} deg", border=1, align='C'); pdf.cell(40, 5, f"{nd['am']:.2f}", border=1, align='C'); pdf.cell(45, 5, "1 - 0.0032(A)", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Frequency Multiplier (FM)", border=1); pdf.cell(40, 5, "Moderate", border=1, align='C'); pdf.cell(40, 5, f"{nd['fm']:.2f}", border=1, align='C'); pdf.cell(45, 5, "Lifting Table", border=1, align='C', ln=True)
    pdf.cell(60, 5, "Coupling Multiplier (CM)", border=1); pdf.cell(40, 5, "Good", border=1, align='C'); pdf.cell(40, 5, f"{nd['cm']:.2f}", border=1, align='C'); pdf.cell(45, 5, "Container Grip", border=1, align='C', ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "3. NIOSH Final Safety Assessment", ln=True)
    
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Recommended Weight Limit (RWL): {nd['rwl']:.2f} kg", ln=True)
    pdf.cell(0, 5, f"Lifting Index (LI = Actual Weight / RWL): {nd['li']:.2f}", ln=True)
    
    fill_color = (144, 238, 144) if nd['li'] <= 1.0 else (255, 182, 193)
    pdf.set_fill_color(*fill_color)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, f"NIOSH EVALUATION: {nd['status']}", border=1, align='C', fill=True, ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=8.5)
    pdf.multi_cell(0, 4.5, "Engineering Notes:\n- LI <= 1.0 indicates task is safe for most healthy industrial workers.\n- LI > 1.0 indicates increased risk of lower back strain; ergonomic redesign or mechanical lift assist is recommended.")

    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 3 of 3 - NIOSH Lifting Equation Assessment Report", align='L')

    return bytes(pdf.output())

# --- HYBRID VIDEO PROCESSOR WITH DETECT-LATCH MEMORY ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.start_time = None
        self.counts = {
            "Trunk": {"1-2": 0, "3-4": 0, "5+": 0},
            "Neck": {"1-2": 0, "3-4": 0, "5+": 0},
            "Upper Arm": {"1-2": 0, "3-4": 0, "5+": 0},
            "Legs": {"1-2": 0, "3-4": 0, "5+": 0},
            "Wrists": {"1-2": 0, "3-4": 0, "5+": 0}
        }
        self.total_frames = 0
        self.hand_box = None

    def recv(self, frame):
        if self.start_time is None: self.start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.total_frames += 1

        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            shld = [lm[11].x * w, lm[11].y * h]
            hip = [lm[23].x * w, lm[23].y * h]
            elbw = [lm[13].x * w, lm[13].y * h]
            nose = [lm[0].x * w, lm[0].y * h]
            
            l_wrist = [lm[15].x * w, lm[15].y * h]
            r_wrist = [lm[16].x * w, lm[16].y * h]
            l_index = [lm[19].x * w, lm[19].y * h]
            r_index = [lm[20].x * w, lm[20].y * h]
            l_pinky = [lm[17].x * w, lm[17].y * h]
            r_pinky = [lm[18].x * w, lm[18].y * h]

            knee = [lm[25].x * w, lm[25].y * h]
            ankle = [lm[27].x * w, lm[27].y * h]

            finger_landmarks = [l_wrist, r_wrist, l_index, r_index, l_pinky, r_pinky]

            margin = int(w * 0.18)
            hx1 = int(max(0, min(l_wrist[0], r_wrist[0]) - margin))
            hy1 = int(max(0, min(l_wrist[1], r_wrist[1]) - margin))
            hx2 = int(min(w, max(l_wrist[0], r_wrist[0]) + margin))
            hy2 = int(min(h, max(l_wrist[1], r_wrist[1]) + margin))
            self.hand_box = (hx1, hy1, hx2, hy2)

            t_score = score_trunk(calculate_angle(shld, hip, knee))
            a_score = score_upper_arm(calculate_angle(hip, shld, elbw))
            n_score = score_neck(calculate_angle(nose, shld, hip))
            l_score = score_legs(calculate_angle(hip, knee, ankle))
            w_score = score_wrists(calculate_angle(elbw, l_wrist, l_index))

            total_reba = t_score + n_score + a_score + l_score + w_score

            avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2.0
            if avg_wrist_y < shld[1]: detected_zone = "Above Shoulder"
            elif shld[1] <= avg_wrist_y < elbw[1]: detected_zone = "Shoulder to Elbow"
            elif elbw[1] <= avg_wrist_y < hip[1]: detected_zone = "Elbow to Knuckle"
            elif hip[1] <= avg_wrist_y < knee[1]: detected_zone = "Knuckle to Mid-Leg"
            else: detected_zone = "Below Mid-Leg"

            arm_reach_dist = abs(l_wrist[0] - shld[0])
            detected_reach = "Far" if arm_reach_dist > (w * 0.25) else "Close"

            scale_px_to_cm = 50.0 / max(1.0, abs(hip[1] - shld[1]))
            h_cm = abs(l_wrist[0] - ankle[0]) * scale_px_to_cm
            v_cm = abs(ankle[1] - l_wrist[1]) * scale_px_to_cm
            d_cm = abs(shld[1] - l_wrist[1]) * scale_px_to_cm
            angle_deg = abs(180.0 - calculate_angle(shld, hip, knee))

            # --- DUAL-ENGINE OBJECT DETECTION (EVERY 5 FRAMES) ---
            if self.total_frames % 5 == 0:
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    yolo_res = yolo_model(rgb_img, verbose=False, conf=0.15)[0]
                    found_obj = None
                    obj_box_draw = None

                    # 1. Primary Pass: COCO YOLO
                    for box in yolo_res.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id != 0:
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            obj_b = (b[0], b[1], b[2], b[3])
                            if self.hand_box and check_hand_object_intersection(self.hand_box, obj_b):
                                found_obj = yolo_model.names[cls_id]
                                obj_box_draw = obj_b
                                break

                    # 2. Secondary Pass: Finger-based foreign object sensor
                    if not found_obj:
                        has_unknown, unknown_b = detect_object_near_fingers(img, finger_landmarks, w, h)
                        if has_unknown:
                            found_obj = "Unidentified Object"
                            obj_box_draw = unknown_b

                    # --- LATCHING LOGIC FOR REPORT PERSISTENCE ---
                    if found_obj:
                        actual_wt = GLOBAL_STORE.get("actual_weight", 8.0)
                        niosh_res = compute_niosh(h_cm, v_cm, d_cm, angle_deg, actual_wt)
                        
                        GLOBAL_STORE["niosh"] = niosh_res
                        GLOBAL_STORE["results"]["detected_object"] = found_obj
                        # Persist latched object state to guarantee memory in PDF
                        GLOBAL_STORE["last_detected_object"] = found_obj

                        if obj_box_draw:
                            cv2.rectangle(img, (obj_box_draw[0], obj_box_draw[1]), 
                                          (obj_box_draw[2], obj_box_draw[3]), (0, 0, 255), 3)
                            cv2.putText(img, f"LOAD DETECTED: {found_obj.upper()} | RWL: {niosh_res['rwl']:.1f}kg", 
                                        (obj_box_draw[0], max(20, obj_box_draw[1]-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    else:
                        # Fallback to latched detected object if previously captured in current run
                        latched = GLOBAL_STORE.get("last_detected_object", "None (Hands Free)")
                        GLOBAL_STORE["results"]["detected_object"] = latched

                except Exception:
                    pass

            for name, sc in [("Trunk", t_score), ("Neck", n_score), ("Upper Arm", a_score), 
                             ("Legs", l_score), ("Wrists", w_score)]:
                if sc <= 2: self.counts[name]["1-2"] += 1
                elif sc <= 4: self.counts[name]["3-4"] += 1
                else: self.counts[name]["5+"] += 1

            tf = max(1, self.total_frames)
            breakdown_pct = {
                name: {
                    "1-2": (self.counts[name]["1-2"] / tf) * 100.0,
                    "3-4": (self.counts[name]["3-4"] / tf) * 100.0,
                    "5+": (self.counts[name]["5+"] / tf) * 100.0
                } for name in ["Trunk", "Neck", "Upper Arm", "Legs", "Wrists"]
            }

            GLOBAL_STORE["total_duration"] = time.time() - self.start_time
            GLOBAL_STORE["overall_score"] = total_reba
            GLOBAL_STORE["breakdown"] = breakdown_pct
            GLOBAL_STORE["results"]["auto_zone"] = detected_zone
            GLOBAL_STORE["results"]["auto_reach"] = detected_reach
            GLOBAL_STORE["frame"] = img.copy()

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA Score: {total_reba}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.set_page_config(page_title="REBA + NIOSH AI Auditor", layout="wide")
st.title("🛡️ Full REBA, MMH & NIOSH Lifting Equation Auditor")

with st.sidebar:
    st.header("Settings & MMH Inputs")
    op_id = st.text_input("Operator ID", "OP-001")
    profile = st.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
    actual_wt = st.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
    GLOBAL_STORE["actual_weight"] = actual_wt

    if st.button("🔄 Reset Object Latch Memory"):
        GLOBAL_STORE["last_detected_object"] = "None (Hands Free)"
        GLOBAL_STORE["results"]["detected_object"] = "None (Hands Free)"
        st.success("Object detection state reset to Hands Free!")

ctx = webrtc_streamer(
    key="reba-niosh-ai",
    video_processor_factory=REBAProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False}
)

st.markdown("### Live / Last Captured Metrics")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Overall REBA Score", GLOBAL_STORE["overall_score"])
m_col2.metric("Total Duration (s)", f"{GLOBAL_STORE['total_duration']:.1f}")

res_data = GLOBAL_STORE["results"]
latched_obj = GLOBAL_STORE.get("last_detected_object", res_data.get("detected_object", "None (Hands Free)"))
m_col3.metric("Object on Hand", latched_obj.title())

niosh_data = GLOBAL_STORE.get("niosh", {})
m_col4.metric("NIOSH RWL / LI", f"{niosh_data.get('rwl', 0.0):.1f} kg (LI: {niosh_data.get('li', 0.0):.2f})")

st.caption(f"📍 Automatically Evaluated Zone: **{res_data.get('auto_zone', 'Elbow to Knuckle')} ({res_data.get('auto_reach', 'Close')})** | NIOSH Status: **{niosh_data.get('status', 'Pending Object Detection')}**")

st.markdown("---")

pdf_bytes = generate_custom_pdf(op_id, profile, actual_wt, GLOBAL_STORE)

st.download_button(
    label="📥 Download 3-Page REBA + NIOSH Audit PDF Report", 
    data=pdf_bytes, 
    file_name=f"REBA_NIOSH_Audit_{op_id}.pdf", 
    mime="application/pdf"
)
