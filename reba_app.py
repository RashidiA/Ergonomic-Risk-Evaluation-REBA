import cv2
import math
import io
import numpy as np
import streamlit as st
import queue
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration
)
from fpdf import FPDF
import mediapipe as mp

# --- SAFE MEDIAPIPE INITIALIZATION ---
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="REBA Ergonomic Risk Evaluator",
    page_icon="🦾",
    layout="wide"
)

# Shared State & Queues
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if "log_reba" not in st.session_state:
    st.session_state.log_reba = []
if "log_trunk" not in st.session_state:
    st.session_state.log_trunk = []
if "log_neck" not in st.session_state:
    st.session_state.log_neck = []
if "log_upper_arm" not in st.session_state:
    st.session_state.log_upper_arm = []

# --- STANDARD STUN CONFIGURATION ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# REBA LOOKUP TABLES
TABLE_A = [
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 4, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
    [[6, 7, 8, 9], [7, 8, 9, 9], [8, 9, 9, 9], [9, 9, 9, 9]]
]

TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 6]],
    [[4, 5, 6], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 9]],
    [[7, 8, 9], [8, 9, 9]]
]

TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
]

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

def get_reba_score(trunk, neck, legs, upper_arm, lower_arm, wrist, load=0, coupling=0, activity=0):
    try:
        score_a = TABLE_A[trunk - 1][neck - 1][legs - 1] + load
        score_b = TABLE_B[upper_arm - 1][lower_arm - 1][wrist - 1] + coupling
        score_c = TABLE_C[score_a - 1][score_b - 1]
        return min(score_c + activity, 15)
    except IndexError:
        return 1

def get_risk_level(score):
    if score == 1:
        return "None", "Not necessary"
    elif 2 <= score <= 3:
        return "Low", "May be necessary"
    elif 4 <= score <= 7:
        return "Medium", "Necessary"
    elif 8 <= score <= 10:
        return "High", "Necessary and soon"
    else:
        return "Very high", "Necessary urgent"

def calc_pct(log):
    if not log:
        return "0.0%", "0.0%", "0.0%"
    total = len(log)
    s12 = sum(1 for x in log if x in [1, 2]) / total * 100
    s34 = sum(1 for x in log if x in [3, 4]) / total * 100
    s5p = sum(1 for x in log if x >= 5) / total * 100
    return f"{s12:.1f}%", f"{s34:.1f}%", f"{s5p:.1f}%"

def generate_2page_pdf(operator_id, profile, actual_weight, height_zone, reach, reba_logs, trunk_logs, neck_logs, arm_logs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "REBA POSTURE AUDIT REPORT", ln=True)
    pdf.set_font("Arial", size=10)
    duration = len(reba_logs) * 0.1
    eval_reba = max(reba_logs) if reba_logs else 1
    pdf.cell(0, 8, f"Operator: {operator_id} | Duration: {duration:.1f}s | Evaluated REBA: {eval_reba}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Posture Duration Breakdown", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(45, 7, "Body Part", border=1)
    pdf.cell(45, 7, "Score 1-2 (%)", border=1)
    pdf.cell(45, 7, "Score 3-4 (%)", border=1)
    pdf.cell(45, 7, "Score 5+ (%)", border=1, ln=True)

    pdf.set_font("Arial", size=9)
    for b_name, b_log in [("Trunk", trunk_logs), ("Neck", neck_logs), ("Upper Arm", arm_logs)]:
        s12, s34, s5p = calc_pct(b_log)
        pdf.cell(45, 7, b_name, border=1)
        pdf.cell(45, 7, s12, border=1)
        pdf.cell(45, 7, s34, border=1)
        pdf.cell(45, 7, s5p, border=1, ln=True)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "MANUAL WEIGHT LIFTING AUDIT", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Operator: {operator_id} | Profile: {profile}", ln=True)
    
    max_limit = LIFTING_MATRIX[profile][height_zone][reach]
    safety_status = "WITHIN SAFE LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE LIMIT"
    pdf.cell(0, 6, f"Evaluated Zone: {height_zone} ({reach})", ln=True)
    pdf.cell(0, 6, f"Actual Weight: {actual_weight:.1f} kg | Limit: {max_limit:.1f} kg", ln=True)
    pdf.cell(0, 8, f"STATUS: {safety_status}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# --- LAZY / SAFE WEBRTC PROCESSOR ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Lazy initialization inside recv to prevent model download blocking on stream launch
        if self.pose is None:
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=0
                )
            except Exception:
                return frame.from_ndarray(img, format="bgr24")

        try:
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results and results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                hp = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                kn = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
                ea = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]

                trunk_a = calculate_angle([hp[0], hp[1] - 100], hp, sh)
                neck_a = calculate_angle(sh, ea, [ea[0], ea[1] - 100])
                upper_a = calculate_angle(hp, sh, el)
                lower_a = calculate_angle(sh, el, wr)
                leg_a = calculate_angle(hp, kn, [kn[0], kn[1] + 100])

                t_s = 1 if trunk_a < 10 else (2 if trunk_a < 20 else 3)
                n_s = 1 if neck_a < 20 else 2
                l_s = 1 if leg_a < 30 else 2
                u_s = 1 if upper_a < 20 else (2 if upper_a < 45 else 3)
                lo_s = 1 if 60 <= lower_a <= 100 else 2
                w_s = 1

                reba = get_reba_score(t_s, n_s, l_s, u_s, lo_s, w_s)

                st.session_state.result_queue.put({
                    "reba": reba, "trunk": t_s, "neck": n_s, 
                    "upper_arm": u_s, "lower_arm": lo_s, "legs": l_s
                })

                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(img, f"REBA: {reba}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        except Exception:
            pass

        return frame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.title("🦾 REBA Ergonomic Risk & Manual Lifting Audit")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Live Assessment Stream")
    webrtc_streamer(
        key="reba-live-stream-v5",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=REBAProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    with st.expander("🏋️ Manual Weight Lifting Parameters", expanded=True):
        op_id = st.text_input("Operator ID", value="OP-001")
        profile = st.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
        actual_wt = st.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

    st.markdown("---")
    st.subheader("📊 Dynamic Metrics")
    
    latest_data = {"reba": 1, "trunk": 1, "neck": 1, "upper_arm": 1, "lower_arm": 1, "legs": 1}
    while not st.session_state.result_queue.empty():
        latest_data = st.session_state.result_queue.get()
        st.session_state.log_reba.append(latest_data["reba"])
        st.session_state.log_trunk.append(latest_data["trunk"])
        st.session_state.log_neck.append(latest_data["neck"])
        st.session_state.log_upper_arm.append(latest_data["upper_arm"])

    reba_score = latest_data["reba"]
    risk, action = get_risk_level(reba_score)

    m1, m2 = st.columns(2)
    m1.metric("REBA Score", f"{reba_score} / 12")
    m2.metric("Risk Level", risk)

    st.info(f"Necessary action: {action}")

    st.markdown("---")
    st.subheader("Sub-Score Breakdown")
    s1, s2, s3 = st.columns(3)
    s1.metric("Trunk", latest_data["trunk"])
    s2.metric("Neck", latest_data["neck"])
    s3.metric("Legs", latest_data["legs"])

    s4, s5 = st.columns(2)
    s4.metric("Upper Arm", latest_data["upper_arm"])
    s5.metric("Lower Arm", latest_data["lower_arm"])

    st.markdown("---")
    if st.button("Generate Full 2-Page Audit PDF"):
        pdf_data = generate_2page_pdf(
            op_id, profile, actual_wt, "Elbow to Knuckle", "Close",
            st.session_state.log_reba, st.session_state.log_trunk,
            st.session_state.log_neck, st.session_state.log_upper_arm
        )
        st.download_button(
            label="💾 Download 2-Page REBA & MMH Report (.pdf)",
            data=pdf_data,
            file_name=f"REBA_Lifting_Audit_{op_id}.pdf",
            mime="application/pdf"
        )
