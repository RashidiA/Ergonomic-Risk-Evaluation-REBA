import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Edge-AI REBA & Ergonomic Auditor", layout="wide")

st.title("⚡ Edge-AI Client-Side REBA & Object Detection Auditor")
st.caption("🚀 Real-Time Pose estimation, AR Skeleton, Object Detection & 3-Page Client-Side PDF Generation")

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
    <button id="reportBtn" class="btn-report" onclick="downloadPdfReport()">📄 Download 3-Page PDF Report</button>
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
    
    // Frame distribution stats for % time breakdown
    let totalFramesRecorded = 0;
    let lowRiskFrames = 0;    // REBA 1-3
    let mediumRiskFrames = 0; // REBA 4-7
    let highRiskFrames = 0;   // REBA 8+

    let peakRebaScore = 1;
    let peakFrameBase64 = "";
    let lastValidCanvasFrame = "";
    let peakAngles = {{ 
      neck: 0, trunk: 180, legs: 180, upper_arm: 0, lower_arm: 90, wrist: 180, 
      neck_score: 1, trunk_score: 1, legs_score: 1, upper_arm_score: 1, lower_arm_score: 1, wrist_score: 1 
    }};

    let latestNiosh = {{ rwl: 18.8, li: 0.43, status: "SAFE", am: 1.0, hm: 1.0, vm: 0.86, dm: 1.0, fm: 0.95, cm: 1.0 }};

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
      totalFramesRecorded = 0;
      lowRiskFrames = 0;
      mediumRiskFrames = 0;
      highRiskFrames = 0;
      
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

        // Track frame counts for time percentage breakdown
        if (isAnalyzing) {{
          totalFramesRecorded++;
          if (totalReba <= 3) lowRiskFrames++;
          else if (totalReba <= 7) mediumRiskFrames++;
          else highRiskFrames++;
          
          sessionDuration = ((Date.now() - startTime) / 1000.0).toFixed(1);
          document.getElementById('timer').innerText = sessionDuration + "s";
        }}

        // NIOSH Calculation
        let trunkDev = Math.abs(180 - angTrunk);
        let am = Math.max(0.0, 1.0 - (0.0032 * trunkDev));
        let rwl = 23.0 * 1.00 * 0.86 * 1.00 * am * 0.95 * 1.00;
        let li = actualWeight / Math.max(0.1, rwl);
        let status = li <= 1.0 ? "SAFE" : "HIGH RISK";
        
        latestNiosh = {{ rwl: rwl, li: li, status: status, am: am, hm: 1.0, vm: 0.86, dm: 1.0, fm: 0.95, cm: 1.0 }};
        document.getElementById('niosh_result').innerText = `${{status}} (LI ${{li.toFixed(2)}})`;

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

      // ==========================================
      // --- PAGE 1: REBA POSTURE AUDIT ---
      // ==========================================
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("REBA POSTURE AUDIT REPORT", 105, 12, {{ align: "center" }});
      
      doc.setFontSize(10);
      doc.text(`Operator: ${{operatorId}} | Total Duration: ${{dur}} sec`, 105, 18, {{ align: "center" }});
      doc.text(`Peak Evaluated REBA Score: ${{peakRebaScore}}`, 105, 24, {{ align: "center" }});

      // Posture Snapshot
      if (imgToEmbed && imgToEmbed.length > 100) {{
        doc.addImage(imgToEmbed, 'JPEG', 10, 32, 95, 65);
      }} else {{
        doc.rect(10, 32, 95, 65);
        doc.setFontSize(8);
        doc.text("[ Frame Snapshot Pending ]", 57, 65, {{ align: "center" }});
      }}

      // Joint Angles Table
      doc.setFontSize(9);
      doc.setFont("Helvetica", "bold");
      doc.text("REBA Step / Joint", 112, 36);
      doc.text("Angle (°)", 158, 36);
      doc.text("Score", 188, 36);
      doc.line(112, 38, 198, 38);

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
        doc.text(row[0], 112, yPos);
        doc.text(row[1], 158, yPos);
        doc.text(row[2], 188, yPos);
        yPos += 6;
      }});

      // --- REBA ANALYSIS SCORE BY % (TIME) SECTION ---
      yPos = 104;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("REBA Posture Risk Time Exposure Breakdown (% Time)", 10, yPos);
      yPos += 4;
      doc.line(10, yPos, 198, yPos);
      yPos += 5;

      let totalF = totalFramesRecorded > 0 ? totalFramesRecorded : 1;
      let lowPct = ((lowRiskFrames / totalF) * 100).toFixed(1);
      let medPct = ((mediumRiskFrames / totalF) * 100).toFixed(1);
      let highPct = ((highRiskFrames / totalF) * 100).toFixed(1);

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text(`• Low Risk Duration (REBA 1 - 3): ${{lowPct}}% of total cycle time`, 12, yPos); yPos += 5;
      doc.text(`• Medium Risk Duration (REBA 4 - 7): ${{medPct}}% of total cycle time`, 12, yPos); yPos += 5;
      doc.text(`• High / Very High Risk Duration (REBA 8+): ${{highPct}}% of total cycle time`, 12, yPos); yPos += 8;

      // REBA Action Table
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
          doc.rect(10, yPos - 4, 188, 6, 'F');
        }}

        doc.setFont("Helvetica", match ? "bold" : "normal");
        doc.text(`${{match ? '-> ' : ''}}${{r[0]}}`, 12, yPos);
        doc.text(r[1], 45, yPos);
        doc.text(r[2], 95, yPos);
        yPos += 6;
      }});

      // ==========================================
      // --- PAGE 2: MANUAL MATERIAL HANDLING (MMH) ---
      // ==========================================
      doc.addPage();
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("MANUAL MATERIAL HANDLING (MMH) AUDIT REPORT", 105, 12, {{ align: "center" }});

      doc.setFontSize(10);
      doc.text(`Evaluation Profile: ${{evalProfile}} | Actual Load Lifted: ${{actualWeight}} kg`, 105, 18, {{ align: "center" }});

      // Lifting Matrix Table
      doc.setFontSize(9);
      doc.text("Standard Recommended Weight Limit Matrix (kg)", 10, 30);

      doc.line(10, 32, 198, 32);
      doc.text("Lifting Zone / Location", 12, 37);
      doc.text("Close Distance (kg)", 110, 37);
      doc.text("Far Distance (kg)", 160, 37);
      doc.line(10, 39, 198, 39);

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      const matrixData = evalProfile === "Male" ? [
        ["Above Shoulder", "10.0 kg", "5.0 kg"],
        ["Shoulder to Elbow", "20.0 kg", "10.0 kg"],
        ["Elbow to Knuckle", "25.0 kg", "15.0 kg"],
        ["Knuckle to Mid-Leg", "20.0 kg", "10.0 kg"],
        ["Below Mid-Leg", "10.0 kg", "5.0 kg"]
      ] : [
        ["Above Shoulder", "7.0 kg", "3.0 kg"],
        ["Shoulder to Elbow", "13.0 kg", "7.0 kg"],
        ["Elbow to Knuckle", "16.0 kg", "10.0 kg"],
        ["Knuckle to Mid-Leg", "13.0 kg", "7.0 kg"],
        ["Below Mid-Leg", "10.0 kg", "5.0 kg"]
      ];

      yPos = 45;
      matrixData.forEach(row => {{
        doc.text(row[0], 12, yPos);
        doc.text(row[1], 110, yPos);
        doc.text(row[2], 160, yPos);
        yPos += 6;
      }});

      // MMH Assessment Summary
      yPos += 10;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("MMH Load Limit Compliance Summary", 10, yPos);
      yPos += 6;
      doc.setFont("Helvetica", "normal");
      doc.setFontSize(9);

      let maxAllowable = evalProfile === "Male" ? 25.0 : 16.0;
      let mmhExceeded = actualWeight > maxAllowable;

      doc.text(`* Maximum Allowable Recommended Weight Limit (Ideal Zone): ${{maxAllowable}} kg`, 12, yPos); yPos += 6;
      doc.text(`* Actual Lifted Weight Input: ${{actualWeight}} kg`, 12, yPos); yPos += 6;
      doc.text(`* MMH Safety Status: ${{mmhExceeded ? 'EXCEEDED RECOMMENDED MAXIMUM LIMIT' : 'WITHIN RECOMMENDED SAFETY LIMITS'}}`, 12, yPos); yPos += 10;

      doc.setFont("Helvetica", "bold");
      doc.text("Ergonomic Control Measures & Recommendations:", 10, yPos); yPos += 6;
      doc.setFont("Helvetica", "normal");
      doc.text("1. Re-position material containers within the primary elbow-to-knuckle zone to minimize vertical reach.", 12, yPos); yPos += 6;
      doc.text("2. Utilize mechanical lift assistance or team lifting if handling loads over standard recommended thresholds.", 12, yPos); yPos += 6;
      doc.text("3. Eliminate excessive twisting or trunk lateral bending during manual transfer steps.", 12, yPos);

      // ==========================================
      // --- PAGE 3: NIOSH LIFTING EQUATION ---
      // ==========================================
      doc.addPage();
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("NIOSH LIFTING EQUATION AUDIT REPORT", 105, 12, {{ align: "center" }});

      doc.setFontSize(10);
      doc.text(`Calculated Load Weight: ${{actualWeight}} kg | Lifting Index (LI): ${{latestNiosh.li.toFixed(2)}}`, 105, 18, {{ align: "center" }});

      // NIOSH Summary Box
      doc.setFillColor(latestNiosh.li <= 1.0 ? 230 : 255, latestNiosh.li <= 1.0 ? 245 : 230, latestNiosh.li <= 1.0 ? 230 : 230);
      doc.rect(10, 26, 188, 16, 'F');
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text(`NIOSH Lifting Status: ${{latestNiosh.status}}`, 15, 33);
      doc.text(`Recommended Weight Limit (RWL): ${{latestNiosh.rwl.toFixed(2)}} kg`, 15, 39);

      // NIOSH Multipliers Breakdown Table
      yPos = 52;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(9);
      doc.text("NIOSH Multiplier Breakdown", 10, yPos); yPos += 4;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.text("Multiplier Component", 12, yPos);
      doc.text("Symbol", 100, yPos);
      doc.text("Evaluated Value", 150, yPos);
      yPos += 2;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      const nioshRows = [
        ["Load Constant", "LC", "23.0 kg"],
        ["Horizontal Multiplier", "HM", latestNiosh.hm.toFixed(2)],
        ["Vertical Multiplier", "VM", latestNiosh.vm.toFixed(2)],
        ["Distance Multiplier", "DM", latestNiosh.dm.toFixed(2)],
        ["Asymmetric Multiplier", "AM", latestNiosh.am.toFixed(2)],
        ["Frequency Multiplier", "FM", latestNiosh.fm.toFixed(2)],
        ["Coupling Multiplier", "CM", latestNiosh.cm.toFixed(2)]
      ];

      nioshRows.forEach(row => {{
        doc.text(row[0], 12, yPos);
        doc.text(row[1], 100, yPos);
        doc.text(row[2], 150, yPos);
        yPos += 6;
      }});

      yPos += 8;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(9);
      doc.text("NIOSH Risk Interpretation Guide:", 10, yPos); yPos += 5;
      doc.setFont("Helvetica", "normal");
      doc.text("• LI <= 1.0: Nominal risk to healthy industrial working populations.", 12, yPos); yPos += 5;
      doc.text("• LI > 1.0: Elevated risk of lower back strain and musculoskeletal fatigue.", 12, yPos); yPos += 5;
      doc.text("• LI > 3.0: High hazard task; immediate engineering intervention required.", 12, yPos);

      doc.save(`REBA_NIOSH_MMH_Audit_${{operatorId}}.pdf`);
    }}
  </script>
</body>
</html>
"""

components.html(html_code, height=680)
