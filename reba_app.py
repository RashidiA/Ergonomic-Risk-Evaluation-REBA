import cv2
import math
import io
import numpy as np
import streamlit as st
import queue
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fpdf import FPDF

# --- EXPLICIT MEDIAPIPE IMPORTS ---
import mediapipe as mp
import mediapipe.solutions.pose as mp_pose
import mediapipe.solutions.drawing_utils as mp_drawing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="REBA Ergonomic Risk Evaluator",
    page_icon="🦾",
    layout="wide"
)

# Shared Queue for Thread-Safe Communication between WebRTC thread & Streamlit UI
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if "log_reba" not in st.session_state:
    st.session_state.log_reba = []

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# --- REBA TABLES ---
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

def get_reba_score(trunk, neck, legs, upper_arm, lower_arm, wrist, load=0, coupling=0, activity=0):
    try:
        score_a = TABLE_A[trunk - 1][neck - 1][legs - 1] + load
        score_b = TABLE_B[upper_arm - 1][lower_arm - 1][wrist - 1] + coupling
        score_c = TABLE_C[score_a - 1][score_b - 1]
        return min(score_c + activity, 12)
    except IndexError:
        return 1

def get_risk_level(score):
    if score == 1:
        return "Negligible", "🟢 Necessary action: None"
    elif 2 <= score <= 3:
        return "Low", "🟡 Necessary action: May be necessary"
    elif 4 <= score <= 7:
        return "Medium", "🟠 Necessary action: Necessary"
    elif 8 <= score <= 10:
        return "High", "🔴 Necessary action: Soon"
    else:
        return "Very High", "🚨 Necessary action: Immediate"

# --- PDF GENERATOR ---
def generate_pdf_report(logs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "REBA Ergonomic Risk Evaluation Summary", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Total Frames Evaluated: {len(logs)}", ln=True)
    
    if logs:
        avg_score = round(sum(logs) / len(logs), 2)
        max_score = max(logs)
        pdf.cell(0, 10, f"Average REBA Score: {avg_score}", ln=True)
        pdf.cell(0, 10, f"Peak REBA Score: {max_score}", ln=True)
        
        risk_level, action = get_risk_level(int(avg_score))
        pdf.cell(0, 10, f"Overall Risk Category: {risk_level}", ln=True)
        pdf.cell(0, 10, f"Recommendation: {action}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# --- WEBRTC PROCESSOR ---
class REBAProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Left side keypoints
            sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
            hp = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            kn = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            ea = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]

            # Angles
            trunk_a = calculate_angle([hp[0], hp[1] - 100], hp, sh)
            neck_a = calculate_angle(sh, ea, [ea[0], ea[1] - 100])
            upper_a = calculate_angle(hp, sh, el)
            lower_a = calculate_angle(sh, el, wr)
            leg_a = calculate_angle(hp, kn, [kn[0], kn[1] + 100])

            # Scores
            t_s = 1 if trunk_a < 10 else (2 if trunk_a < 20 else 3)
            n_s = 1 if neck_a < 20 else 2
            l_s = 1 if leg_a < 30 else 2
            u_s = 1 if upper_a < 20 else (2 if upper_a < 45 else 3)
            lo_s = 1 if 60 <= lower_a <= 100 else 2
            w_s = 1

            reba = get_reba_score(t_s, n_s, l_s, u_s, lo_s, w_s)

            # Send metrics safely to Streamlit thread
            st.session_state.result_queue.put({
                "reba": reba, "trunk": t_s, "neck": n_s, 
                "upper_arm": u_s, "lower_arm": lo_s, "legs": l_s
            })

            # Video Overlay
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"REBA: {reba}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.title("🦾 REBA Ergonomic Risk Evaluator")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Live Stream Assessment")
    webrtc_streamer(
        key="reba-processor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=REBAProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("📊 Live Posture Metrics")
    
    # Retrieve latest calculation from queue
    latest_data = {"reba": 1, "trunk": 1, "neck": 1, "upper_arm": 1, "lower_arm": 1, "legs": 1}
    while not st.session_state.result_queue.empty():
        latest_data = st.session_state.result_queue.get()
        st.session_state.log_reba.append(latest_data["reba"])

    reba_score = latest_data["reba"]
    risk, action = get_risk_level(reba_score)

    # Metrics Grid
    m1, m2 = st.columns(2)
    m1.metric("REBA Score", f"{reba_score} / 12")
    m2.metric("Risk Level", risk)

    st.info(action)

    st.markdown("---")
    st.subheader("Sub-Scores")
    s1, s2, s3 = st.columns(3)
    s1.metric("Trunk", latest_data["trunk"])
    s2.metric("Neck", latest_data["neck"])
    s3.metric("Legs", latest_data["legs"])

    s4, s5 = st.columns(2)
    s4.metric("Upper Arm", latest_data["upper_arm"])
    s5.metric("Lower Arm", latest_data["lower_arm"])

    st.markdown("---")
    st.subheader("📄 Report Export")
    if st.button("Generate Summary PDF"):
        pdf_bytes = generate_pdf_report(st.session_state.log_reba)
        st.download_button(
            label="💾 Download REBA Report (.pdf)",
            data=pdf_bytes,
            file_name="REBA_Ergonomic_Report.pdf",
            mime="application/pdf"
        )
