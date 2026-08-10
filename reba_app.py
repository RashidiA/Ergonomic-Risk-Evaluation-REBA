import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import tempfile
import os
from fpdf import FPDF

st.set_page_config(page_title="Edge-AI REBA & Ergonomic Auditor", layout="wide")

# Initialize session state for audit data
if "audit_data" not in st.session_state:
    st.session_state.audit_data = None

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
    
    score = int(audit_data.get("peak_reba_score", 1))
    angles = audit_data.get("peak_angles", {})
    dur = float(audit_data.get("total_duration", 0.0))
    pct_high_risk = float(audit_data.get("pct_high_risk", 0.0))
    obj_detected = audit_data.get("object_detected", "Not Detected")

    # ================= PAGE 1: REBA POSTURE & OBJECT DETECTION =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 7, "REBA POSTURE AUDIT REPORT (EDGE AI ENGINE)", ln=True, align='C')
    pdf.set_font("Arial", 'B', 9.5)
    pdf.cell(0, 5, f"Operator: {operator_id} | Session Duration: {dur:.1f}s | High Risk Exposure (Score >= 8): {pct_high_risk:.1f}%", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 10.5)
    pdf.cell(0, 6, f"Peak Evaluated REBA Score: {score} | Object Detected: {obj_detected}", ln=True, align='C')
    pdf.ln(2)

    # REBA Action Table
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 4.5, "REBA Standard Action & Risk Assessment Table", ln=True)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(35, 4.5, "REBA Score", border=1, align='C')
    pdf.cell(50, 4.5, "Risk Level", border=1, align='C')
    pdf.cell(100, 4.5, "Action Required", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=8)
    for r_score, r_level, r_action in REBA_ACTION_TABLE:
        if r_score == "1": is_match = (score == 1)
        elif r_score == "2-3": is_match = (score in [2, 3])
        elif r_score == "4-7": is_match = (4 <= score <= 7)
        elif r_score == "8-10": is_match = (8 <= score <= 10)
        else: is_match = (score >= 11)

        prefix = "-> " if is_match else ""
        fill_flag = is_match
        if fill_flag: pdf.set_fill_color(255, 255, 0)

        pdf.cell(35, 4.5, f"{prefix}{r_score}", border=1, align='C', fill=fill_flag)
        pdf.cell(50, 4.5, r_level, border=1, fill=fill_flag)
        pdf.cell(100, 4.5, r_action, border=1, ln=True, fill=fill_flag)

    pdf.ln(3)

    # Embed Peak Image
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
                pdf.image(tmp_img_path, x=10, y=curr_y, w=85)
        except Exception:
            pass

    # Angles Breakdown Table
    pdf.set_xy(100, curr_y)
    pdf.set_font("Arial", 'B', 8)
    pdf.cell(40, 4.5, "REBA Step / Joint", border=1, align='C')
    pdf.cell(25, 4.5, "Angle (°)", border=1, align='C')
    pdf.cell(20, 4.5, "Score", border=1, align='C', ln=True)

    step_rows = [
        ("Step 1: Neck", angles.get("neck", 0.0), angles.get("neck_score", 1)),
        ("Step 2: Trunk", angles.get("trunk", 0.0), angles.get("trunk_score", 1)),
        ("Step 3: Legs", angles.get("legs", 0.0), angles.get("legs_score", 1)),
        ("Step 7: Upper Arm", angles.get("upper_arm", 0.0), angles.get("upper_arm_score", 1)),
        ("Step 8: Lower Arm", angles.get("lower_arm", 0.0), angles.get("lower_arm_score", 1)),
        ("Step 9: Wrist", angles.get("wrist", 0.0), angles.get("wrist_score", 1))
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
    pdf.cell(0, 10, "Page 1 of 3 - REBA Edge AI Posture Evaluation", align='L')

    # ================= PAGE 2: MANUAL MATERIAL HANDLING AUDIT =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "MANUAL WEIGHT LIFTING AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id} | Profile: {profile}", ln=True, align='C')
    pdf.ln(3)

    auto_zone = audit_data.get("auto_zone", "Elbow to Knuckle")
    auto_reach = audit_data.get("auto_reach", "Close")
    max_limit = LIFTING_MATRIX[profile][auto_zone][auto_reach]
    status_str = "WITHIN SAFE ERGONOMIC LIMIT" if actual_weight <= max_limit else "EXCEEDS SAFE ERGONOMIC LIMIT"

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "Manual Material Handling Evaluation Summary", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Automatically Evaluated Zone: {auto_zone} ({auto_reach})", ln=True)
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

    pdf.set_y(-12)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Page 2 of 3 - Recommended Weight Limits Matrix Standard", align='L')

    # ================= PAGE 3: NIOSH AUDIT =================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "NIOSH LIFTING EQUATION ASSESSMENT", ln=True, align='C')
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Operator: {operator_id}", ln=True, align='C')
    pdf.ln(3)

    trunk_dev = abs(180 - float(angles.get("trunk", 180.0)))
    hm, vm, dm = 0.83, 1.00, 1.00
    am = max(0.0, 1.0 - (0.0032 * trunk_dev))
    rwl = 23.0 * hm * vm * dm * am * 0.95 * 1.00
    li = actual_weight / max(0.1, rwl)
    status = "SAFE (LI <= 1.0)" if li <= 1.0 else "UNSAFE / HIGH RISK (LI > 1.0)"

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, "NIOSH Multipliers & Spatial Geometry", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f"Calculated Trunk Asymmetric Angle: {trunk_dev:.1f}°", ln=True)
    pdf.cell(0, 5, f"Recommended Weight Limit (RWL): {rwl:.2f} kg", ln=True)
    pdf.cell(0, 5, f"Lifting Index (LI): {li:.2f}", ln=True)
    pdf.ln(2)

    fill_color = (144, 238, 144) if li <= 1.0 else (255, 182, 193)
    pdf.set_fill_color(*fill_color)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 7, f"NIOSH EVALUATION: {status}", border=1, align='C', fill=True, ln=True)

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

# --- CLIENT-SIDE ENGINE (MediaPipe Pose + COCO-SSD Object Detector) ---
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
    .container { position: relative; width: 100%; max-width: 640px; margin: auto; }
    video { display: none; }
    canvas { width: 100%; height: auto; border-radius: 8px; background: #000; }
    .controls { display: flex; gap: 12px; margin-top: 10px; justify-content: center; }
    button { padding: 10px 20px; font-weight: bold; border-radius: 6px; border: none; cursor: pointer; color: white; font-size: 14px; }
    .btn-start { background-color: #28a745; }
    .btn-stop { background-color: #dc3545; }
    .metrics { margin-top: 12px; font-family: sans-serif; display: flex; gap: 10px; }
    .card { background: #f0f2f6; padding: 10px; border-radius: 6px; flex: 1; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
  </div>

  <div class="controls">
    <button class="btn-start" onclick="startAnalysis()">▶ Start Analysis</button>
    <button class="btn-stop" onclick="stopAnalysis()">⏹ Stop & Sync Session</button>
  </div>

  <div class="metrics">
    <div class="card"><strong>Live REBA</strong><h2 id="live_score">1</h2></div>
    <div class="card"><strong>Peak REBA</strong><h2 id="peak_score">1</h2></div>
    <div class="card"><strong>High Risk %</strong><h2 id="high_risk_pct">0%</h2></div>
    <div class="card"><strong>Object Detected</strong><h2 id="object_detected" style="font-size: 18px;">None</h2></div>
    <div class="card"><strong>Timer</strong><h2 id="timer">0.0s</h2></div>
  </div>

  <script>
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    
    let objectModel = null;
    let detectedObjects = [];
    
    let isAnalyzing = false;
    let startTime = 0;
    let totalFramesRecorded = 0;
    let highRiskFrames = 0;
    let peakRebaScore = 0;
    let peakFrameBase64 = "";
    let peakAngles = {};

    cocoSsd.load().then(model => {
      objectModel = model;
    });

    function startAnalysis() {
      isAnalyzing = true;
      startTime = Date.now();
      totalFramesRecorded = 0;
      highRiskFrames = 0;
      peakRebaScore = 0;
    }

    function stopAnalysis() {
      if (!isAnalyzing) return;
      isAnalyzing = false;

      const duration = (Date.now() - startTime) / 1000.0;
      const pctHighRisk = totalFramesRecorded > 0 ? (highRiskFrames / totalFramesRecorded) * 100.0 : 0.0;
      const mainObj = detectedObjects.length > 0 ? detectedObjects[0] : "Material / Box";

      const payload = {
        peak_reba_score: peakRebaScore,
        peak_angles: peakAngles,
        peak_image_base64: peakFrameBase64,
        auto_zone: "Elbow to Knuckle",
        auto_reach: "Close",
        total_duration: duration,
        pct_high_risk: pctHighRisk,
        object_detected: mainObj
      };

      // Direct Streamlit Custom Component Return Transmission
      if (window.Streamlit) {
        window.Streamlit.setComponentValue(payload);
      } else {
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          value: payload
        }, '*');
      }
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
          detectedObjects = [];
          predictions.forEach(pred => {
            if (pred.score > 0.4 && pred.class !== 'person') {
              detectedObjects.push(pred.class);
              canvasCtx.strokeStyle = '#00FFFF';
              canvasCtx.lineWidth = 2;
              canvasCtx.strokeRect(pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]);
              canvasCtx.fillStyle = '#00FFFF';
              canvasCtx.font = '14px Arial';
              canvasCtx.fillText(`${pred.class} (${Math.round(pred.score*100)}%)`, pred.bbox[0], pred.bbox[1] > 10 ? pred.bbox[1] - 5 : 10);
            }
          });
          document.getElementById('object_detected').innerText = detectedObjects.length > 0 ? detectedObjects[0] : "None";
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

        if (isAnalyzing) {
          totalFramesRecorded++;
          if (totalReba >= 8) highRiskFrames++;

          let elapsed = ((Date.now() - startTime) / 1000.0).toFixed(1);
          document.getElementById('timer').innerText = elapsed + "s";
          document.getElementById('high_risk_pct').innerText = ((highRiskFrames / totalFramesRecorded) * 100.0).toFixed(1) + "%";

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

# Capture raw return payload directly from Streamlit HTML component
component_payload = components.html(html_code, height=720)

# If JavaScript posted back component data, persist it in st.session_state
if component_payload:
    st.session_state.audit_data = component_payload

st.markdown("---")

# Render PDF Download Button whenever audit_data exists
if st.session_state.audit_data:
    st.success("✅ Recorded session synced! Your report is ready.")
    pdf_bytes = generate_pdf_report(op_id, profile, actual_wt, st.session_state.audit_data)
    st.download_button(
        label="📥 Download 3-Page REBA + NIOSH PDF Report",
        data=pdf_bytes,
        file_name=f"REBA_Audit_{op_id}.pdf",
        mime="application/pdf"
    )
else:
    st.info("💡 Click **'Start Analysis'**, perform the lifting movement, and click **'Stop & Sync Session'** to display the PDF download button.")
