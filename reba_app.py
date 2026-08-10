import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Edge-AI REBA & Ergonomic Auditor", layout="wide")

st.title("⚡ Edge-AI Client-Side REBA & Object Detection Auditor")
st.caption("🚀 Real-Time Pose estimation, AR Skeleton, Object Detection & Client-Side PDF Generation")

sidebar = st.sidebar
op_id = sidebar.text_input("Operator ID", "OP-001")
profile = sidebar.selectbox("Evaluation Profile / Gender", ["Male", "Female"])
actual_wt = sidebar.number_input("Actual Weight Lifted (kg)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

html_code = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>

  <style>
    body {{ margin: 0; font-family: sans-serif; background-color: transparent; }}
    .container {{ position: relative; width: 100%; max-width: 640px; margin: auto; }}
    video {{ display: none; }}
    canvas {{ width: 100%; height: auto; border-radius: 8px; background: #000; }}
    .controls {{ display: flex; gap: 12px; margin-top: 10px; justify-content: center; }}
    button {{ padding: 12px 24px; font-weight: bold; border-radius: 6px; border: none; cursor: pointer; color: white; font-size: 15px; transition: all 0.2s; }}
    .btn-toggle {{ background-color: #28a745; }}
    .btn-toggle.recording {{ background-color: #dc3545; }}
    .btn-report {{ background-color: #0d6efd; }}
    .metrics {{ margin-top: 12px; display: flex; gap: 10px; }}
    .card {{ background: #f0f2f6; padding: 10px; border-radius: 6px; flex: 1; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
  </div>

  <div class="controls">
    <button id="toggleBtn" class="btn-toggle" onclick="toggleAnalysis()">▶ Start Analysis</button>
    <button id="reportBtn" class="btn-report" onclick="downloadPdfReport()">📄 Download PDF Audit Report</button>
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

    let objectModel = null;
    let currentObject = "Unidentified Object";
    let persistObject = "Unidentified Object";

    let isAnalyzing = false;
    let startTime = 0;
    let sessionDuration = 0;
    let totalFramesRecorded = 0;
    let highRiskFrames = 0;
    
    let peakRebaScore = 1;
    let peakFrameBase64 = "";
    let lastValidCanvasFrame = "";
    let peakAngles = {{ 
      neck: 0, trunk: 180, legs: 180, upper_arm: 0, lower_arm: 90, wrist: 180, 
      neck_score: 1, trunk_score: 1, legs_score: 1, upper_arm_score: 1, lower_arm_score: 1, wrist_score: 1 
    }};

    const operatorId = "{op_id}";
    const evalProfile = "{profile}";
    const actualWeight = {actual_wt};

    cocoSsd.load().then(model => {{
      objectModel = model;
    }});

    function resetSessionMemory() {{
      peakRebaScore = 1;
      peakFrameBase64 = "";
      lastValidCanvasFrame = "";
      sessionDuration = 0;
      peakAngles = {{ 
        neck: 0, trunk: 180, legs: 180, upper_arm: 0, lower_arm: 90, wrist: 180, 
        neck_score: 1, trunk_score: 1, legs_score: 1, upper_arm_score: 1, lower_arm_score: 1, wrist_score: 1 
      }};
      document.getElementById('peak_score').innerText = "1";
      document.getElementById('timer').innerText = "0.0s";
    }}

    function toggleAnalysis() {{
      if (!isAnalyzing) {{
        resetSessionMemory();
        isAnalyzing = true;
        startTime = Date.now();
        totalFramesRecorded = 0;
        highRiskFrames = 0;

        toggleBtn.innerText = "⏹ Stop Session";
        toggleBtn.classList.add("recording");
      }} else {{
        isAnalyzing = false;
        sessionDuration = ((Date.now() - startTime) / 1000.0).toFixed(1);
        toggleBtn.innerText = "▶ Start Analysis";
        toggleBtn.classList.remove("recording");
      }}
    }}

    function calcAngle(a, b, c) {{
      let radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
      let angle = Math.abs(radians * 180.0 / Math.PI);
      return angle > 180.0 ? 360.0 - angle : angle;
    }}

    async function onResults(results) {{
      canvasElement.width = videoElement.videoWidth || 640;
      canvasElement.height = videoElement.videoHeight || 480;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

      if (objectModel && videoElement.readyState === 4) {{
        try {{
          const predictions = await objectModel.detect(videoElement);
          let detected = [];
          predictions.forEach(pred => {{
            if (pred.score > 0.30 && pred.class !== 'person') {{
              detected.push(pred.class);
              canvasCtx.strokeStyle = '#00FFFF';
              canvasCtx.lineWidth = 2;
              canvasCtx.strokeRect(pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]);
              canvasCtx.fillStyle = '#00FFFF';
              canvasCtx.font = '14px Arial';
              canvasCtx.fillText(`${{pred.class}} (${{Math.round(pred.score*100)}}%)`, pred.bbox[0], pred.bbox[1] > 10 ? pred.bbox[1] - 5 : 10);
            }}
          }});

          if (detected.length > 0) {{
            currentObject = detected[0];
            persistObject = currentObject;
          }} else {{
            currentObject = persistObject;
          }}
          document.getElementById('object_detected').innerText = currentObject;
        }} catch(e){{}}
      }}

      if (results.poseLandmarks) {{
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {{color: '#00FF00', lineWidth: 3}});
        drawLandmarks(canvasCtx, results.poseLandmarks, {{color: '#FF0000', lineWidth: 2, radius: 4}});

        try {{
          lastValidCanvasFrame = canvasElement.toDataURL('image/jpeg', 0.85);
        }} catch(e){{}}

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
        let nioshText = li <= 1.0 ? `SAFE (LI ${{li.toFixed(2)}})` : `HIGH RISK (LI ${{li.toFixed(2)}})`;
        document.getElementById('niosh_result').innerText = nioshText;

        if (totalReba >= peakRebaScore) {{
          peakRebaScore = totalReba;
          document.getElementById('peak_score').innerText = peakRebaScore;
          peakFrameBase64 = lastValidCanvasFrame;
          peakAngles = {{
            neck: angNeck, neck_score: nScore,
            trunk: angTrunk, trunk_score: tScore,
            legs: angLegs, legs_score: lScore,
            upper_arm: angUArm, upper_arm_score: aScore,
            lower_arm: angLArm, lower_arm_score: laScore,
            wrist: angWrist, wrist_score: wScore
          }};
        }}

        if (isAnalyzing) {{
          totalFramesRecorded++;
          if (totalReba >= 8) highRiskFrames++;
          sessionDuration = ((Date.now() - startTime) / 1000.0).toFixed(1);
          document.getElementById('timer').innerText = sessionDuration + "s";
        }}
      }}
      canvasCtx.restore();
    }}

    const pose = new Pose({{
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}`
    }});
    pose.setOptions({{ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5 }});
    pose.onResults(onResults);

    const camera = new Camera(videoElement, {{
      onFrame: async () => {{ await pose.send({{ image: videoElement }}); }},
      width: 640, height: 480
    }});
    camera.start();

    function downloadPdfReport() {{
      const {{ jsPDF }} = window.jspdf;
      const doc = new jsPDF();

      let imgToEmbed = peakFrameBase64 || lastValidCanvasFrame;
      let dur = isAnalyzing ? ((Date.now() - startTime) / 1000.0).toFixed(1) : sessionDuration;

      // --- PAGE 1: REBA POSTURE AUDIT ---
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("REBA POSTURE AUDIT REPORT", 105, 12, {{ align: "center" }});
      
      doc.setFontSize(10);
      doc.text(`Operator: ${{operatorId}} | Total Duration: ${{dur}} sec`, 105, 18, {{ align: "center" }});
      doc.text(`Peak Evaluated REBA Score: ${{peakRebaScore}}`, 105, 24, {{ align: "center" }});

      // Posture Snapshot
      if (imgToEmbed && imgToEmbed.length > 100) {{
        doc.addImage(imgToEmbed, 'JPEG', 10, 32, 90, 60);
      }} else {{
        doc.rect(10, 32, 90, 60);
        doc.setFontSize(8);
        doc.text("[ Frame Snapshot Pending ]", 55, 62, {{ align: "center" }});
      }}

      // Step-by-Step Joint Angles Table
      doc.setFontSize(9);
      doc.setFont("Helvetica", "bold");
      doc.text("REBA Step / Joint", 110, 36);
      doc.text("Angle (°)", 155, 36);
      doc.text("Score", 185, 36);
      doc.line(110, 38, 195, 38);

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      const steps = [
        ["Step 1: Neck", `${{peakAngles.neck ? peakAngles.neck.toFixed(1) : '0.0'}}°`, `+${{peakAngles.neck_score || 1}}`],
        ["Step 2: Trunk", `${{peakAngles.trunk ? peakAngles.trunk.toFixed(1) : '180.0'}}°`, `+${{peakAngles.trunk_score || 1}}`],
        ["Step 3: Legs", `${{peakAngles.legs ? peakAngles.legs.toFixed(1) : '180.0'}}°`, `+${{peakAngles.legs_score || 1}}`],
        ["Step 7: Upper Arm", `${{peakAngles.upper_arm ? peakAngles.upper_arm.toFixed(1) : '0.0'}}°`, `+${{peakAngles.upper_arm_score || 1}}`],
        ["Step 8: Lower Arm", `${{peakAngles.lower_arm ? peakAngles.lower_arm.toFixed(1) : '90.0'}}°`, `+${{peakAngles.lower_arm_score || 1}}`],
        ["Step 9: Wrist", `${{peakAngles.wrist ? peakAngles.wrist.toFixed(1) : '180.0'}}°`, `+${{peakAngles.wrist_score || 1}}`]
      ];

      let yPos = 44;
      steps.forEach(row => {{
        doc.text(row[0], 110, yPos);
        doc.text(row[1], 155, yPos);
        doc.text(row[2], 185, yPos);
        yPos += 6;
      }});

      // REBA Action Table
      yPos = 100;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(9);
      doc.text("REBA Standard Action & Risk Table", 10, yPos);
      yPos += 5;

      const riskRows = [
        ["1", "None", "Not necessary"],
        ["2-3", "Low", "May be necessary"],
        ["4-7", "Medium", "Necessary"],
        ["8-10", "High", "Necessary and soon"],
        ["11-15", "Very high", "Necessary urgent"]
      ];

      riskRows.forEach(r => {{
        let match = false;
        let sc = peakRebaScore;
        if (r[0] === "1" && sc === 1) match = true;
        else if (r[0] === "2-3" && (sc === 2 || sc === 3)) match = true;
        else if (r[0] === "4-7" && (sc >= 4 && sc <= 7)) match = true;
        else if (r[0] === "8-10" && (sc >= 8 && sc <= 10)) match = true;
        else if (r[0] === "11-15" && sc >= 11) match = true;

        if (match) {{
          doc.setFillColor(255, 255, 0);
          doc.rect(10, yPos - 4, 185, 6, 'F');
        }}

        doc.setFont("Helvetica", match ? "bold" : "normal");
        doc.text(`${{match ? '-> ' : ''}}${{r[0]}}`, 12, yPos);
        doc.text(r[1], 45, yPos);
        doc.text(r[2], 90, yPos);
        yPos += 6;
      }});

      doc.save(`REBA_NIOSH_Audit_${{operatorId}}.pdf`);
    }}
  </script>
</body>
</html>
"""

components.html(html_code, height=680)
