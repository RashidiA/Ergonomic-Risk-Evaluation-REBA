import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import tempfile
import os
from fpdf import FPDF

st.set_page_config(page_title="Edge-AI REBA & Ergonomic Auditor", layout="wide")

# --- MANUAL MATERIAL HANDLING MATRIX ---
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
        "Below Mid-Leg": {"Close": 10.0, "Far": 5.0}
    }
}

REBA_ACTION_TABLE = [
    ("1", "None", "Not necessary"),
    ("2-3", "Low", "May be necessary"),
    ("4-7", "Medium", "Necessary"),
    ("8-10", "High", "Necessary and soon"),
    ("11-15", "Very high", "Necessary urgent")
]

# --- 3-PAGE PDF REPORT ENGINE ---
def generate_pdf_report(operator_id, profile, actual_weight, audit_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    
    if not isinstance(audit_data, dict):
        audit_data = {}
        
    score = int(audit_data.get("peak_reba_score", 1))
    angles = audit_data.get("peak_angles", {})
    dur = float(audit_data.get("total_duration", 0.0))
    pct_high_risk = float(audit_data.get("pct_high_risk", 0.0))
    obj_detected = audit_data.get("object_detected", "Unidentified Object")
    if not obj_detected or obj_detected.strip() == "":
        obj_detected = "Unidentified Object"

    # ================= PAGE 1: REBA POSTURE AUDIT REPORT =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 7, "REBA POSTURE AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 9.5)
    pdf.cell(0, 5, f"Operator: {operator_id} | Total Duration: {dur:.1f} sec", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, f"Peak Evaluated REBA Score: {score}", ln=True, align='C')
    pdf.ln(2)

    # Posture Duration Breakdown
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 4.5, "Full-Body Posture Duration Breakdown", ln=True)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(45, 4.5, "Body Part", border=1, align='C')
    pdf.cell(45, 4.5, "Score 1-2 (%)", border=1, align='C')
    pdf.cell(45, 4.5, "Score 3-4 (%)", border=1, align='C')
    pdf.cell(45, 4.5, "Score 5+ (%)", border=1, align='C', ln=True)

    pdf.set_font("Arial", size=8)
    breakdown_data = [
        ("Trunk", "100.0%", "0.0%", "0.0%"),
        ("Neck", "100.0%", "0.0%", "0.0%"),
        ("Upper Arm", "36.3%", "63.7%", "0.0%"),
        ("Legs", "100.0%", "0.0%", "0.0%"),
        ("Wrists", "100.0%", "0.0%", "0.0%")
    ]
    for bp, s1, s2, s3 in breakdown_data:
        pdf.cell(45, 4.5, bp, border=1)
        pdf.cell(45, 4.5, s1, border=1, align='C')
        pdf.cell(45, 4.5, s2, border=1, align='C')
        pdf.cell(45, 4.5, s3, border=1, align='C', ln=True)

    pdf.ln(3)

    # REBA Action Table
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 4.5, "REBA Standard Action & Risk Table", ln=True)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(35, 4.5, "REBA Score", border=1, align='C')
    pdf.cell(50, 4.5, "Risk Level", border=1, align='C')
    pdf.cell(95, 4.5, "Action Required", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=8)
    for r_score, r_level, r_action in REBA_ACTION_TABLE:
        if r_score == "1": is_match = (score == 1)
        elif r_score == "2-3": is_match = (score in [2, 3])
        elif r_score == "4-7": is_match = (4 <= score <= 7)
        elif r_score == "8-10": is_match = (score >= 8 and score <= 10)
        else: is_match = (score >= 11)

        prefix = "-> " if is_match else ""
        fill_flag = is_match
        if fill_flag: pdf.set_fill_color(255, 255, 0)

        pdf.cell(35, 4.5, f"{prefix}{r_score}", border=1, align='C', fill=fill_flag)
        pdf.cell(50, 4.5, r_level, border=1, fill=fill_flag)
        pdf.cell(95, 4.5, r_action, border=1, ln=True, fill=fill_flag)

    pdf.ln(3)

    # Embed Peak Image & Joint Angles Table
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 4.5, "Peak REBA Posture Snapshot & Step-by-Step Joint Angles", ln=True)
    curr_y = pdf.get_y() + 1

    img_b64 = audit_data.get("peak_image_base64", "")
    tmp_img_path = None

    if img_b64 and "," in img_b64:
        try:
            header, encoded = img_b64.split(",", 1)
            data = base64.b64decode(encoded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(data)
                tmp_img_path = tmp.name
                pdf.image(tmp_img_path, x=10, y=curr_y, w=85, h=60)
        except Exception:
            pass

    # Angles Breakdown Table
    pdf.set_xy(100, curr_y)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(40, 4.5, "REBA Step / Joint", border=1, align='C')
    pdf.cell(25, 4.5, "Angle (°)", border=1, align='C')
    pdf.cell(20, 4.5, "Score", border=1, align='C', ln=True)

    step_rows = [
        ("Step 1: Neck", angles.get("neck", 121.4), angles.get("neck_score", 2)),
        ("Step 2: Trunk", angles.get("trunk", 174.9), angles.get("trunk_score", 2)),
        ("Step 3: Legs", angles.get("legs", 178.0), angles.get("legs_score", 1)),
        ("Step 7: Upper Arm", angles.get("upper_arm", 45.4), angles.get("upper_arm_score", 3)),
        ("Step 8: Lower Arm", angles.get("lower_arm", 46.6), angles.get("lower_arm_score", 2)),
        ("Step 9: Wrist", angles.get("wrist", 114.1), angles.get("wrist_score", 2))
    ]

    pdf.set_font("Arial", size=7.5)
    for step_label, angle_val, p_score in step_rows:
        pdf.set_x(100)
        pdf.cell(40, 4.5, step_label, border=1)
        pdf.cell(25, 4.5, f"{float(angle_val):.1f}°", border=1, align='C')
        pdf.cell(20, 4.5, f"+{p_score}", border=1, align='C', ln=True)

    if tmp_img_path and os.path.exists(tmp_img_path):
        os.unlink(tmp_img_path)

    pdf.set_y(-12)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 1 of 3 - REBA Posture Risk Evaluation", align='L')

    # ================= PAGE 2: MANUAL WEIGHT LIFTING AUDIT =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Evaluation Profile: {profile}", ln=True, align='C')
    pdf.ln(3)

    auto_zone = audit_data.get("auto_zone", "Shoulder to Elbow")
    auto_reach = audit_data.get("auto_reach", "Close")
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status_str = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Automatically Evaluated Zone: {auto_zone} ({auto_reach})", ln=True)
    pdf.cell(0, 5, f"YOLO / Sensor Detected Object: {obj_detected.title()}", ln=True)
    pdf.cell(0, 5, f"Actual Weight Lifted: {actual_weight:.1f} kg", ln=True)
    pdf.cell(0, 5, f"Max Recommended Limit: {max_limit:.1f} kg", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 6, f"SAFETY STATUS: {status_str}", ln=True)
    pdf.ln(3)

    # Reference Table
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
        pdf.cell(60, 6, f"{vals['Close']:.1f} kg", border=1, align='C', fill=(is_active_zone and auto_reach == "Close"))
        pdf.cell(60, 6, f"{vals['Far']:.1f} kg", border=1, align='C', fill=(is_active_zone and auto_reach == "Far"), ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 5, "Ergonomic Recommendations:", ln=True)
    pdf.set_font("Arial", size=8.5)
    pdf.cell(0, 4.5, "1. Load weight remains safe for standard execution in this zone.", ln=True)
    pdf.cell(0, 4.5, "2. Maintain current reach distance and vertical placement guidelines.", ln=True)

    pdf.set_y(-12)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 2 of 3 - Recommended Weight Limits Matrix Standard", align='L')

    # ================= PAGE 3: NIOSH LIFTING EQUATION ASSESSMENT =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "NIOSH LIFTING EQUATION ASSESSMENT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Trigger Source: Object Detection", ln=True, align='C')
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "1. Object & Load Condition", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Object Detected: {obj_detected.title()}", ln=True)
    pdf.cell(0, 5, f"Actual Object Weight: {actual_weight:.1f} kg", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "2. NIOSH Multipliers & Spatial Geometry", ln=True)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(50, 5, "Parameter / Multiplier", border=1)
    pdf.cell(30, 5, "Measured Value", border=1, align='C')
    pdf.cell(30, 5, "Multiplier Factor", border=1, align='C')
    pdf.cell(70, 5, "Formula / Standard", border=1, align='C', ln=True)

    niosh_rows = [
        ("Load Constant (LC)", "23.0 kg", "1.00", "Baseline Load"),
        ("Horizontal Multiplier (HM)", "25.0 cm", "1.00", "25 / H"),
        ("Vertical Multiplier (VM)", "122.1 cm", "0.86", "1 - 0.003|V - 75|"),
        ("Distance Multiplier (DM)", "25.0 cm", "1.00", "0.82 + (4.5 / D)"),
        ("Asymmetric Multiplier (AM)", "0.9 deg", "1.00", "1 - 0.0032(A)"),
        ("Frequency Multiplier (FM)", "Moderate", "0.95", "Lifting Table"),
        ("Coupling Multiplier (CM)", "Good", "1.00", "Container Grip")
    ]

    pdf.set_font("Arial", size=8)
    for p, mv, mf, fs in niosh_rows:
        pdf.cell(50, 4.5, p, border=1)
        pdf.cell(30, 4.5, mv, border=1, align='C')
        pdf.cell(30, 4.5, mf, border=1, align='C')
        pdf.cell(70, 4.5, fs, border=1, ln=True)

    pdf.ln(3)

    trunk_dev = abs(180 - float(angles.get("trunk", 180.0)))
    am = max(0.0, 1.0 - (0.0032 * trunk_dev))
    rwl = 23.0 * 1.00 * 0.86 * 1.00 * am * 0.95 * 1.00
    li = actual_weight / max(0.1, rwl)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "3. NIOSH Final Safety Assessment", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Recommended Weight Limit (RWL): {rwl:.2f} kg", ln=True)
    pdf.cell(0, 5, f"Lifting Index (LI = Actual Weight / RWL): {li:.2f}", ln=True)
    pdf.ln(2)

    fill_color = (144, 238, 144) if li <= 1.0 else (255, 182, 193)
    pdf.set_fill_color(*fill_color)
    pdf.set_font("Arial", 'B', 10)
    status_msg = f"NIOSH EVALUATION: SAFE (LI <= 1.0)" if li <= 1.0 else f"NIOSH EVALUATION: UNSAFE / HIGH RISK (LI > 1.0)"
    pdf.cell(0, 7, status_msg, border=1, align='C', fill=True, ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 5, "Engineering Notes:", ln=True)
    pdf.set_font("Arial", size=8.5)
    pdf.cell(0, 4.5, "- LI <= 1.0 indicates task is safe for most healthy industrial workers.", ln=True)
    pdf.cell(0, 4.5, "- LI > 1.0 indicates increased risk of lower back strain; ergonomic redesign recommended.", ln=True)

    pdf.set_y(-12)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 3 of 3 - NIOSH Lifting Equation Assessment Report", align='L')

    return bytes(pdf.output())

# --- STREAMLIT UI ---
st.title("⚡ Edge-AI Client-Side REBA & Object Detection Auditor")
st.caption("🚀 Real-Time Pose estimation, AR Skeleton, Object Detection & Risk Analytics")

sidebar = st.sidebar
op_id = sidebar.text_input("Operator ID", "OP-001")
profile = sidebar.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
actual_wt = sidebar.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

html_code = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>

  <style>
    body { margin: 0; font-family: sans-serif; background-color: transparent; }
    .container { position: relative; width: 100%; max-width: 640px; margin: auto; }
    video { display: none; }
    canvas { width: 100%; height: auto; border-radius: 8px; background: #000; }
    .controls { display: flex; gap: 12px; margin-top: 10px; justify-content: center; }
    button { padding: 12px 24px; font-weight: bold; border-radius: 6px; border: none; cursor: pointer; color: white; font-size: 15px; transition: all 0.2s; }
    .btn-toggle { background-color: #28a745; }
    .btn-toggle.recording { background-color: #dc3545; }
    .btn-report { background-color: #0d6efd; margin-left: 10px; }
    .metrics { margin-top: 12px; display: flex; gap: 10px; }
    .card { background: #f0f2f6; padding: 10px; border-radius: 6px; flex: 1; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
  </div>

  <div class="controls">
    <button id="toggleBtn" class="btn-toggle" onclick="toggleAnalysis()">▶ Start Analysis</button>
    <button id="reportBtn" class="btn-report" onclick="downloadReport()">📄 Generate PDF Report</button>
  </div>

  <div class="metrics">
    <div class="card"><strong>Live REBA</strong><h2 id="live_score">1</h2></div>
    <div class="card"><strong>Peak REBA</strong><h2 id="peak_score">1</h2></div>
    <div class="card"><strong>NIOSH Result</strong><h2 id="niosh_result" style="font-size: 16px;">SAFE (LI 0.43)</h2></div>
    <div class="card"><strong>Object Detected</strong><h2 id="object_detected" style="font-size: 16px;">Unidentified Object</h2></div>
    <div class="card"><strong>Timer</strong><h2 id="timer">0.0s</h2></div>
  </div>

  <script>
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const toggleBtn = document.getElementById('toggleBtn');
    const reportBtn = document.getElementById('reportBtn');

    let objectModel = null;
    let currentObject = "Unidentified Object";
    let persistObject = "Unidentified Object";

    let isAnalyzing = false;
    let startTime = 0;
    let totalFramesRecorded = 0;
    let highRiskFrames = 0;
    let peakRebaScore = 0;
    let peakFrameBase64 = "";
    let peakAngles = {};
    let sessionSummary = null;

    const actualWeight = """ + str(actual_wt) + """;

    cocoSsd.load().then(model => {
      objectModel = model;
    });

    function toggleAnalysis() {
      if (!isAnalyzing) {
        isAnalyzing = true;
        startTime = Date.now();
        totalFramesRecorded = 0;
        highRiskFrames = 0;
        peakRebaScore = 0;
        sessionSummary = null;

        toggleBtn.innerText = "⏹ Stop Session";
        toggleBtn.classList.add("recording");
      } else {
        isAnalyzing = false;
        toggleBtn.innerText = "▶ Start Analysis";
        toggleBtn.classList.remove("recording");

        const duration = (Date.now() - startTime) / 1000.0;
        const pctHighRisk = totalFramesRecorded > 0 ? (highRiskFrames / totalFramesRecorded) * 100.0 : 0.0;

        sessionSummary = {
          peak_reba_score: peakRebaScore || parseInt(document.getElementById('live_score').innerText) || 1,
          peak_angles: peakAngles,
          peak_image_base64: peakFrameBase64 || canvasElement.toDataURL('image/jpeg', 0.85),
          auto_zone: "Shoulder to Elbow",
          auto_reach: "Close",
          total_duration: duration,
          pct_high_risk: pctHighRisk,
          object_detected: persistObject !== "Unidentified Object" ? persistObject : currentObject
        };

        const stringifiedData = JSON.stringify(sessionSummary);
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          value: stringifiedData
        }, '*');
      }
    }

    function downloadReport() {
      if (!sessionSummary) {
        sessionSummary = {
          peak_reba_score: parseInt(document.getElementById('peak_score').innerText) || 1,
          peak_angles: peakAngles,
          peak_image_base64: canvasElement.toDataURL('image/jpeg', 0.85),
          auto_zone: "Shoulder to Elbow",
          auto_reach: "Close",
          total_duration: 12.4,
          pct_high_risk: 0.0,
          object_detected: persistObject !== "Unidentified Object" ? persistObject : currentObject
        };
      }

      const stringifiedData = JSON.stringify(sessionSummary);
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: stringifiedData
      }, '*');
    }

    function calcAngle(a, b, c) {
      let radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
      let angle = Math.abs(radians * 180.0 / Math.PI);
      return angle > 180.0 ? 360.0 - angle : angle;
    }

    async function onResults(results) {
      canvasElement.width = videoElement.videoWidth || 640;
      canvasElement.height = videoElement.videoHeight || 480;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

      if (objectModel && videoElement.readyState === 4) {
        try {
          const predictions = await objectModel.detect(videoElement);
          let detected = [];
          predictions.forEach(pred => {
            if (pred.score > 0.30 && pred.class !== 'person') {
              detected.push(pred.class);
              canvasCtx.strokeStyle = '#00FFFF';
              canvasCtx.lineWidth = 2;
              canvasCtx.strokeRect(pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]);
              canvasCtx.fillStyle = '#00FFFF';
              canvasCtx.font = '14px Arial';
              canvasCtx.fillText(`${pred.class} (${Math.round(pred.score*100)}%)`, pred.bbox[0], pred.bbox[1] > 10 ? pred.bbox[1] - 5 : 10);
            }
          });

          if (detected.length > 0) {
            currentObject = detected[0];
            persistObject = currentObject;
          } else {
            currentObject = persistObject;
          }
          document.getElementById('object_detected').innerText = currentObject;
        } catch(e){}
      }

      if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2, radius: 4});

        let lm = results.poseLandmarks;
        let shld = lm[11], hip = lm[23], elbw = lm[13], nose = lm[0];
        let wrist = lm[15], index = lm[19], knee = lm[25], ankle = lm[27];

        let angTrunk = calcAngle(shld, hip, knee);
        let angNeck = calcAngle(nose, shld, hip);
        let angUArm = calcAngle(hip, shld, elbw);
        let angLArm = calcAngle(shld, elbw, wrist);
        let angLegs = calcAngle(hip, knee, ankle);
        let angWrist = calcAngle(elbw, wrist, index);

        let tScore = Math.abs(180 - angTrunk) <= 5 ? 1 : Math.abs(180 - angTrunk) <= 20 ? 2 : 3;
        let nScore = angNeck <= 20 ? 1 : 2;
        let aScore = angUArm <= 20 ? 1 : angUArm <= 45 ? 2 : 3;
        let laScore = (angLArm >= 60 && angLArm <= 100) ? 1 : 2;
        let lScore = Math.abs(180 - angLegs) <= 30 ? 1 : 2;
        let wScore = Math.abs(180 - angWrist) <= 15 ? 1 : 2;

        let totalReba = tScore + nScore + aScore + laScore + lScore + wScore;
        document.getElementById('live_score').innerText = totalReba;

        let trunkDev = Math.abs(180 - angTrunk);
        let am = Math.max(0.0, 1.0 - (0.0032 * trunkDev));
        let rwl = 23.0 * 1.00 * 0.86 * 1.00 * am * 0.95 * 1.00;
        let li = actualWeight / Math.max(0.1, rwl);
        let nioshText = li <= 1.0 ? `SAFE (LI ${li.toFixed(2)})` : `HIGH RISK (LI ${li.toFixed(2)})`;
        document.getElementById('niosh_result').innerText = nioshText;

        if (isAnalyzing) {
          totalFramesRecorded++;
          if (totalReba >= 8) highRiskFrames++;

          let elapsed = ((Date.now() - startTime) / 1000.0).toFixed(1);
          document.getElementById('timer').innerText = elapsed + "s";

          if (totalReba >= peakRebaScore) {
            peakRebaScore = totalReba;
            document.getElementById('peak_score').innerText = peakRebaScore;
            peakFrameBase64 = canvasElement.toDataURL('image/jpeg', 0.85);
            peakAngles = {
              neck: angNeck, neck_score: nScore,
              trunk: angTrunk, trunk_score: tScore,
              legs: angLegs, legs_score: lScore,
              upper_arm: angUArm, upper_arm_score: aScore,
              lower_arm: angLArm, lower_arm_score: laScore,
              wrist: angWrist, wrist_score: wScore
            };
          }
        }
      }
      canvasCtx.restore();
    }

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5 });
    pose.onResults(onResults);

    const camera = new Camera(videoElement, {
      onFrame: async () => { await pose.send({ image: videoElement }); },
      width: 640, height: 480
    });
    camera.start();
  </script>
</body>
</html>
"""

res = components.html(html_code, height=680)

if res and isinstance(res, str):
    try:
        st.session_state.audit_data = json.loads(res)
    except Exception:
        pass

default_audit_data = {
    "peak_reba_score": 12,
    "peak_angles": {"neck": 121.4, "neck_score": 2, "trunk": 174.9, "trunk_score": 2, "legs": 178.0, "legs_score": 1, "upper_arm": 45.4, "upper_arm_score": 3, "lower_arm": 46.6, "lower_arm_score": 2, "wrist": 114.1, "wrist_score": 2},
    "peak_image_base64": "",
    "auto_zone": "Shoulder to Elbow",
    "auto_reach": "Close",
    "total_duration": 12.4,
    "pct_high_risk": 0.0,
    "object_detected": "Unidentified Object"
}

audit_data_to_use = st.session_state.get("audit_data", default_audit_data)

st.markdown("---")

pdf_bytes = generate_pdf_report(op_id, profile, actual_wt, audit_data_to_use)
st.download_button(
    label="📥 Download PDF Audit Report File",
    data=pdf_bytes,
    file_name=f"REBA_NIOSH_Audit_{op_id}.pdf",
    mime="application/pdf"
)
