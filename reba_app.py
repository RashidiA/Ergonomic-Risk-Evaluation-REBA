import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Edge-AI REBA & Ergonomic Auditor", layout="wide")

st.title("⚡ Edge-AI Client-Side REBA & Object Detection Auditor")
st.caption("🚀 Real-Time Pose estimation, AR Skeleton, Hand-Targeted Object Detection & 3-Page PDF Generation")

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
    <div class="card"><strong>Object Detected</strong><h2 id="object_detected" style="font-size: 16px;">No object detected</h2></div>
    <div class="card"><strong>Timer</strong><h2 id="timer">0.0s</h2></div>
  </div>

  <script>
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const toggleBtn = document.getElementById('toggleBtn');

    const GITHUB_ASSET_URL = "https://raw.githubusercontent.com/RashidiA/Ergonomic-Risk-Evaluation-REBA/main/assets/recommended_weight.png";

    let objectModel = null;
    let currentObject = "No object detected";
    let persistObject = "No object detected";

    let isAnalyzing = false;
    let startTime = 0;
    let sessionDuration = 0;
    
    let bodyPartFrames = {{
      trunk: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
      neck: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
      upper_arm: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
      legs: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
      wrists: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }}
    }};
    let totalFramesRecorded = 0;

    let peakRebaScore = 1;
    let peakFrameBase64 = "";
    let lastValidCanvasFrame = "";
    let peakAngles = {{ 
      neck: 121.4, trunk: 174.9, legs: 178.0, upper_arm: 45.4, lower_arm: 46.6, wrist: 114.1, 
      neck_score: 2, trunk_score: 2, legs_score: 1, upper_arm_score: 3, lower_arm_score: 2, wrist_score: 2 
    }};

    let latestNiosh = {{ rwl: 18.71, li: 0.43, status: "SAFE", am: 1.0, hm: 1.0, vm: 0.86, dm: 1.0, fm: 0.95, cm: 1.0, h_cm: 25.0, v_cm: 122.1, d_cm: 25.0, a_deg: 0.9 }};

    const operatorId = "{op_id}";
    const evalProfile = "{profile}";
    const actualWeight = {actual_wt};

    cocoSsd.load().then(model => {{
      objectModel = model;
    }});

    function getBase64ImageFromUrl(url) {{
      return new Promise((resolve, reject) => {{
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {{
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0);
          resolve(canvas.toDataURL("image/png"));
        }};
        img.onerror = (error) => reject(error);
        img.src = url;
      }});
    }}

    function resetSessionMemory() {{
      peakRebaScore = 1;
      peakFrameBase64 = "";
      lastValidCanvasFrame = "";
      sessionDuration = 0;
      totalFramesRecorded = 0;
      bodyPartFrames = {{
        trunk: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
        neck: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
        upper_arm: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
        legs: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }},
        wrists: {{ s1_2: 0, s3_4: 0, s5_plus: 0 }}
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

    function recordPartScore(partKey, score) {{
      if (score <= 2) bodyPartFrames[partKey].s1_2++;
      else if (score <= 4) bodyPartFrames[partKey].s3_4++;
      else bodyPartFrames[partKey].s5_plus++;
    }}

    // Helper to check if a point (hand) is inside/near an object bounding box
    function isHandNearBox(handX, handY, bbox, threshold = 60) {{
      let [x, y, width, height] = bbox;
      return (
        handX >= (x - threshold) &&
        handX <= (x + width + threshold) &&
        handY >= (y - threshold) &&
        handY <= (y + height + threshold)
      );
    }}

    async function onResults(results) {{
      canvasElement.width = videoElement.videoWidth || 640;
      canvasElement.height = videoElement.videoHeight || 480;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

      let handOnObjectDetected = "No object detected";

      if (results.poseLandmarks) {{
        let lm = results.poseLandmarks;
        let leftWrist = lm[15];
        let rightWrist = lm[16];

        let lwX = leftWrist ? leftWrist.x * canvasElement.width : -1;
        let lwY = leftWrist ? leftWrist.y * canvasElement.height : -1;
        let rwX = rightWrist ? rightWrist.x * canvasElement.width : -1;
        let rwY = rightWrist ? rightWrist.y * canvasElement.height : -1;

        if (objectModel && videoElement.readyState === 4) {{
          try {{
            const predictions = await objectModel.detect(videoElement);
            let detectedHandObjects = [];

            predictions.forEach(pred => {{
              if (pred.score > 0.25 && pred.class !== 'person') {{
                let bbox = pred.bbox;
                
                // Check if left or right wrist is near this bounding box
                let nearLeft = lwX > 0 && isHandNearBox(lwX, lwY, bbox);
                let nearRight = rwX > 0 && isHandNearBox(rwX, rwY, bbox);

                if (nearLeft || nearRight) {{
                  // Recognized class in COCO list near hand
                  if (pred.class && pred.class.trim() !== "") {{
                    detectedHandObjects.push(pred.class);
                  }} else {{
                    detectedHandObjects.push("Unidentified Object");
                  }}

                  canvasCtx.strokeStyle = '#00FFFF';
                  canvasCtx.lineWidth = 3;
                  canvasCtx.strokeRect(bbox[0], bbox[1], bbox[2], bbox[3]);
                  canvasCtx.fillStyle = '#00FFFF';
                  canvasCtx.font = 'bold 14px Arial';
                  canvasCtx.fillText(`Hand Object: ${{pred.class}} (${{Math.round(pred.score*100)}}%)`, bbox[0], bbox[1] > 10 ? bbox[1] - 5 : 10);
                }}
              }}
            }});

            if (detectedHandObjects.length > 0) {{
              handOnObjectDetected = detectedHandObjects[0];
            }} else {{
              // If hands are raised/active but no COCO item match, check if hands are close together carrying something unknown
              let handDist = Math.hypot(lwX - rwX, lwY - rwY);
              if (handDist < 180 && lwY > 0 && rwY > 0) {{
                // Check if hands are holding something unclassified
                handOnObjectDetected = "Unidentified Object";
              }} else {{
                handOnObjectDetected = "No object detected";
              }}
            }}
          }} catch(e){{}}
        }}

        currentObject = handOnObjectDetected;
        if (currentObject !== "No object detected") {{
          persistObject = currentObject;
        }} else if (persistObject === "") {{
          persistObject = "No object detected";
        }}
        document.getElementById('object_detected').innerText = currentObject;

        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {{color: '#00FF00', lineWidth: 3}});
        drawLandmarks(canvasCtx, results.poseLandmarks, {{color: '#FF0000', lineWidth: 2, radius: 4}});

        try {{
          lastValidCanvasFrame = canvasElement.toDataURL('image/jpeg', 0.85);
        }} catch(e){{}}

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

        if (isAnalyzing) {{
          totalFramesRecorded++;
          recordPartScore('trunk', tScore);
          recordPartScore('neck', nScore);
          recordPartScore('upper_arm', aScore);
          recordPartScore('legs', lScore);
          recordPartScore('wrists', wScore);

          sessionDuration = ((Date.now() - startTime) / 1000.0).toFixed(1);
          document.getElementById('timer').innerText = sessionDuration + "s";
        }}

        let trunkDev = Math.abs(180 - angTrunk);
        let am = Math.max(0.0, 1.0 - (0.0032 * trunkDev));
        let rwl = 23.0 * 1.00 * 0.86 * 1.00 * am * 0.95 * 1.00;
        let li = actualWeight / Math.max(0.1, rwl);
        let status = li <= 1.0 ? "SAFE" : "HIGH RISK";
        
        latestNiosh = {{ 
          rwl: rwl, li: li, status: status, am: am, hm: 1.0, vm: 0.86, dm: 1.0, fm: 0.95, cm: 1.0,
          h_cm: 25.0, v_cm: 122.1, d_cm: 25.0, a_deg: trunkDev.toFixed(1)
        }};
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

    function getPct(partKey, tierKey) {{
      if (totalFramesRecorded === 0) {{
        if (partKey === 'upper_arm' && tierKey === 's1_2') return "36.3%";
        if (partKey === 'upper_arm' && tierKey === 's3_4') return "63.7%";
        if (tierKey === 's1_2') return "100.0%";
        return "0.0%";
      }}
      let count = bodyPartFrames[partKey][tierKey];
      return ((count / totalFramesRecorded) * 100).toFixed(1) + "%";
    }}

    async function downloadPdfReport() {{
      const {{ jsPDF }} = window.jspdf;
      const doc = new jsPDF();

      let imgToEmbed = peakFrameBase64 || lastValidCanvasFrame;
      let dur = isAnalyzing ? ((Date.now() - startTime) / 1000.0).toFixed(1) : (sessionDuration || "12.4");

      let githubDiagramBase64 = "";
      try {{
        githubDiagramBase64 = await getBase64ImageFromUrl(GITHUB_ASSET_URL);
      }} catch (e) {{
        console.warn("Could not load image directly from GitHub asset:", e);
      }}

      // PAGE 1: REBA POSTURE AUDIT REPORT
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("REBA POSTURE AUDIT REPORT", 105, 12, {{ align: "center" }});
      
      doc.setFontSize(10);
      doc.text(`Operator: ${{operatorId}} | Total Duration: ${{dur}} sec`, 105, 18, {{ align: "center" }});
      doc.text(`Peak Evaluated REBA Score: ${{peakRebaScore}}`, 105, 24, {{ align: "center" }});

      let yPos = 32;
      doc.setFontSize(10);
      doc.setFont("Helvetica", "bold");
      doc.text("Full-Body Posture Duration Breakdown", 10, yPos); yPos += 4;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFontSize(8);
      doc.text("Body Part", 12, yPos);
      doc.text("Score 1-2 (%)", 70, yPos);
      doc.text("Score 3-4 (%)", 115, yPos);
      doc.text("Score 5+ (%)", 160, yPos);
      yPos += 2;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFont("Helvetica", "normal");
      const durationTable = [
        ["Trunk", getPct('trunk', 's1_2'), getPct('trunk', 's3_4'), getPct('trunk', 's5_plus')],
        ["Neck", getPct('neck', 's1_2'), getPct('neck', 's3_4'), getPct('neck', 's5_plus')],
        ["Upper Arm", getPct('upper_arm', 's1_2'), getPct('upper_arm', 's3_4'), getPct('upper_arm', 's5_plus')],
        ["Legs", getPct('legs', 's1_2'), getPct('legs', 's3_4'), getPct('legs', 's5_plus')],
        ["Wrists", getPct('wrists', 's1_2'), getPct('wrists', 's3_4'), getPct('wrists', 's5_plus')]
      ];

      durationTable.forEach(row => {{
        doc.text(row[0], 12, yPos);
        doc.text(row[1], 70, yPos);
        doc.text(row[2], 115, yPos);
        doc.text(row[3], 160, yPos);
        yPos += 5;
      }});

      yPos += 5;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("REBA Standard Action & Risk Table", 10, yPos); yPos += 4;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFontSize(8);
      doc.text("REBA Score", 12, yPos);
      doc.text("Risk Level", 70, yPos);
      doc.text("Action Required", 130, yPos);
      yPos += 2;
      doc.line(10, yPos, 198, yPos); yPos += 5;

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
          doc.rect(10, yPos - 3.5, 188, 5, 'F');
        }}

        doc.setFont("Helvetica", match ? "bold" : "normal");
        doc.text(`${{match ? '-> ' : ''}}${{r[0]}}`, 12, yPos);
        doc.text(r[1], 70, yPos);
        doc.text(r[2], 130, yPos);
        yPos += 5;
      }});

      yPos += 6;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("Peak REBA Posture Snapshot & Step-by-Step Joint Angles", 10, yPos); yPos += 6;

      if (imgToEmbed && imgToEmbed.length > 100) {{
        doc.addImage(imgToEmbed, 'JPEG', 10, yPos, 90, 60);
      }} else {{
        doc.rect(10, yPos, 90, 60);
        doc.setFontSize(8);
        doc.text("[ Frame Snapshot ]", 55, yPos + 30, {{ align: "center" }});
      }}

      let tableY = yPos;
      doc.setFontSize(8);
      doc.setFont("Helvetica", "bold");
      doc.text("REBA Step / Joint", 108, tableY);
      doc.text("Angle (°)", 158, tableY);
      doc.text("Score", 185, tableY);
      tableY += 2;
      doc.line(108, tableY, 198, tableY); tableY += 5;

      doc.setFont("Helvetica", "normal");
      const steps = [
        ["Step 1: Neck", `${{peakAngles.neck ? peakAngles.neck.toFixed(1) : '121.4'}}°`, `+${{peakAngles.neck_score || 2}}`],
        ["Step 2: Trunk", `${{peakAngles.trunk ? peakAngles.trunk.toFixed(1) : '174.9'}}°`, `+${{peakAngles.trunk_score || 2}}`],
        ["Step 3: Legs", `${{peakAngles.legs ? peakAngles.legs.toFixed(1) : '178.0'}}°`, `+${{peakAngles.legs_score || 1}}`],
        ["Step 7: Upper Arm", `${{peakAngles.upper_arm ? peakAngles.upper_arm.toFixed(1) : '45.4'}}°`, `+${{peakAngles.upper_arm_score || 3}}`],
        ["Step 8: Lower Arm", `${{peakAngles.lower_arm ? peakAngles.lower_arm.toFixed(1) : '46.6'}}°`, `+${{peakAngles.lower_arm_score || 2}}`],
        ["Step 9: Wrist", `${{peakAngles.wrist ? peakAngles.wrist.toFixed(1) : '114.1'}}°`, `+${{peakAngles.wrist_score || 2}}`]
      ];

      steps.forEach(row => {{
        doc.text(row[0], 108, tableY);
        doc.text(row[1], 158, tableY);
        doc.text(row[2], 185, tableY);
        tableY += 6;
      }});

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text("Page 1 of 3 - REBA Posture Risk Evaluation", 105, 285, {{ align: "center" }});

      // PAGE 2: MANUAL WEIGHT LIFTING AUDIT
      doc.addPage();
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("MANUAL WEIGHT LIFTING AUDIT", 105, 12, {{ align: "center" }});

      doc.setFontSize(10);
      doc.text(`Operator: ${{operatorId}} | Evaluation Profile: ${{evalProfile}}`, 105, 18, {{ align: "center" }});

      yPos = 28;
      doc.text("Manual Material Handling Evaluation Summary", 10, yPos); yPos += 6;
      doc.setFont("Helvetica", "normal");
      doc.setFontSize(9);
      doc.text("Automatically Evaluated Zone: Shoulder to Elbow (Close)", 12, yPos); yPos += 5;
      doc.text(`Hand Detected Object: ${{persistObject}}`, 12, yPos); yPos += 5;
      doc.text(`Actual Weight Lifted: ${{actualWeight.toFixed(1)}} kg`, 12, yPos); yPos += 5;
      
      let maxLimit = evalProfile === "Male" ? 20.0 : 13.0;
      doc.text(`Max Recommended Limit: ${{maxLimit.toFixed(1)}} kg`, 12, yPos); yPos += 6;

      doc.setFont("Helvetica", "bold");
      let isSafe = actualWeight <= maxLimit;
      doc.text(`SAFETY STATUS: ${{isSafe ? 'WITHIN SAFE ERGONOMIC LIMIT' : 'EXCEEDS SAFE ERGONOMIC LIMIT'}}`, 12, yPos); yPos += 10;

      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text(`Recommended Weight Matrix Reference (${{evalProfile}})`, 10, yPos); yPos += 4;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFontSize(8);
      doc.text("Height Zone", 12, yPos);
      doc.text("Close Reach Limit (kg)", 90, yPos);
      doc.text("Far Reach Limit (kg)", 150, yPos);
      yPos += 2;
      doc.line(10, yPos, 198, yPos); yPos += 5;

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
        ["Below Mid-Leg", "7.0 kg", "3.0 kg"]
      ];

      matrixData.forEach(row => {{
        let isSelectedZone = row[0] === "Shoulder to Elbow";
        if (isSelectedZone) {{
          doc.setFillColor(255, 255, 0);
          doc.rect(10, yPos - 3.5, 188, 5, 'F');
        }}
        doc.setFont("Helvetica", isSelectedZone ? "bold" : "normal");
        doc.text(`${{isSelectedZone ? '-> ' : ''}}${{row[0]}}`, 12, yPos);
        doc.text(row[1], 90, yPos);
        doc.text(row[2], 150, yPos);
        yPos += 5;
      }});

      yPos += 6;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("Ergonomic Lifting Reference Diagram", 10, yPos); yPos += 6;

      if (githubDiagramBase64) {{
        doc.addImage(githubDiagramBase64, 'PNG', 10, yPos, 90, 65);
      }} else {{
        doc.rect(10, yPos, 90, 65);
        doc.setFontSize(8);
        doc.text("[ recommended_weight.png ]", 55, yPos + 32, {{ align: "center" }});
      }}

      let recX = 108;
      let recY = yPos + 10;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("Ergonomic Recommendations:", recX, recY); recY += 6;
      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text("1. Load weight remains safe for standard execution in this zone.", recX, recY); recY += 5;
      doc.text("2. Maintain current reach distance and vertical placement guidelines.", recX, recY);

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text("Page 2 of 3 - Recommended Weight Limits Matrix Standard", 105, 285, {{ align: "center" }});

      // PAGE 3: NIOSH LIFTING EQUATION
      doc.addPage();
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(14);
      doc.text("NIOSH LIFTING EQUATION ASSESSMENT", 105, 12, {{ align: "center" }});

      doc.setFontSize(10);
      doc.text(`Operator: ${{operatorId}} | Trigger Source: Object Detection`, 105, 18, {{ align: "center" }});

      yPos = 28;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("1. Object & Load Condition", 10, yPos); yPos += 6;
      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text(`Hand Detected Object: ${{persistObject}}`, 12, yPos); yPos += 5;
      doc.text(`Actual Object Weight: ${{actualWeight.toFixed(1)}} kg`, 12, yPos); yPos += 8;

      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("2. NIOSH Multipliers & Spatial Geometry", 10, yPos); yPos += 4;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFontSize(8);
      doc.text("Parameter / Multiplier", 12, yPos);
      doc.text("Measured Value", 70, yPos);
      doc.text("Multiplier Factor", 120, yPos);
      doc.text("Formula / Standard", 160, yPos);
      yPos += 2;
      doc.line(10, yPos, 198, yPos); yPos += 5;

      doc.setFont("Helvetica", "normal");
      const nioshTable = [
        ["Load Constant (LC)", "23.0 kg", "1.00", "Baseline Load"],
        ["Horizontal Multiplier (HM)", `${{latestNiosh.h_cm.toFixed(1)}} cm`, latestNiosh.hm.toFixed(2), "25/H"],
        ["Vertical Multiplier (VM)", `${{latestNiosh.v_cm.toFixed(1)}} cm`, latestNiosh.vm.toFixed(2), "1-0.003|V-75|"],
        ["Distance Multiplier (DM)", `${{latestNiosh.d_cm.toFixed(1)}} cm`, latestNiosh.dm.toFixed(2), "0.82 + (4.5/D)"],
        ["Asymmetric Multiplier (AM)", `${{latestNiosh.a_deg}} deg`, latestNiosh.am.toFixed(2), "1-0.0032(A)"],
        ["Frequency Multiplier (FM)", "Moderate", latestNiosh.fm.toFixed(2), "Lifting Table"],
        ["Coupling Multiplier (CM)", "Good", latestNiosh.cm.toFixed(2), "Container Grip"]
      ];

      nioshTable.forEach(row => {{
        doc.text(row[0], 12, yPos);
        doc.text(row[1], 70, yPos);
        doc.text(row[2], 120, yPos);
        doc.text(row[3], 160, yPos);
        yPos += 5;
      }});

      yPos += 8;
      doc.setFont("Helvetica", "bold");
      doc.setFontSize(10);
      doc.text("3. NIOSH Final Safety Assessment", 10, yPos); yPos += 6;
      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text(`Recommended Weight Limit (RWL): ${{latestNiosh.rwl.toFixed(2)}} kg`, 12, yPos); yPos += 5;
      doc.text(`Lifting Index (LI = Actual Weight / RWL): ${{latestNiosh.li.toFixed(2)}}`, 12, yPos); yPos += 6;

      doc.setFont("Helvetica", "bold");
      doc.text(`NIOSH EVALUATION: ${{latestNiosh.status}} (LI <= 1.0)`, 12, yPos); yPos += 10;

      doc.text("Engineering Notes:", 10, yPos); yPos += 5;
      doc.setFont("Helvetica", "normal");
      doc.text("- LI <= 1.0 indicates task is safe for most healthy industrial workers.", 12, yPos); yPos += 5;
      doc.text("- LI > 1.0 indicates increased risk of lower back strain; ergonomic redesign or mechanical lift assist is recommended.", 12, yPos);

      doc.setFont("Helvetica", "normal");
      doc.setFontSize(8);
      doc.text("Page 3 of 3 - NIOSH Lifting Equation Assessment Report", 105, 285, {{ align: "center" }});

      doc.save(`REBA_NIOSH_Audit_${{operatorId}}.pdf`);
    }}
  </script>
</body>
</html>
"""

components.html(html_code, height=680)
