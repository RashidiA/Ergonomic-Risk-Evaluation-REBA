# 🛡️ AI REBA & Object-Aware Ergonomic Auditor

An AI-powered ergonomic assessment and Manual Material Handling (MMH) audit tool built to automate the **Rapid Entire Body Assessment (REBA)** process for industrial and automotive manufacturing. 

By combining real-time pose estimation with object detection, the web app evaluates both postural risk and weight lifting limits dynamically in a single web interface.

---

## 🚀 Features & Key Enhancements

* **Real-time Posture Tracking:** Uses MediaPipe Pose to track 33 skeletal keypoints without specialized hardware—just a standard smartphone or laptop webcam.
* **Automated REBA Scoring:** Instant angle and risk calculations for:
  * **Trunk:** Flexion, extension, and alignment.
  * **Neck:** Head tilt and positioning.
  * **Upper Arms:** Elevation and reaching metrics.
* **YOLO Object & Hand Interaction Tracking:**
  * Powered by **YOLOv8 Nano** (`yolov8n.pt`) to detect carried objects, tools, and industrial containers.
  * **Dynamic Hand Spatial Tracking:** Integrates MediaPipe wrist coordinates to define an active hand region, detecting whether an object is actively held or lifted in real time.
* **Automated MMH Evaluation:**
  * Automatically calculates **Vertical Height Zones** (*Above Shoulder, Shoulder to Elbow, Elbow to Knuckle, Knuckle to Mid-Leg, Below Mid-Leg*) and **Horizontal Reach** (*Close vs. Far*) based on live joint geometry.
  * Dynamically evaluates recommended weight limits against standard ergonomic matrices for **Male/Female** operators.
* **Strict 2-Page Audit PDF Reports:**
  * **Page 1:** Executive REBA posture analysis, percentage time breakdown per body part score, and highlighted REBA Action Level table.
  * **Page 2:** Manual Material Handling audit, automatically evaluated lifting zone, active YOLO hand-detected object, safe weight limits, and embedded ergonomic lifting diagram.
  * **Visual Matrix Highlighting:** Highlights exact evaluated posture and weight limit cells in **yellow** inside exported PDF tables for immediate visual clarity.
* **Cloud & Firewall Optimization:**
  * Built using **Frame Skipping (10-frame intervals)** for YOLO inference to maintain low latency and stay within Streamlit Cloud’s ~1 GB memory limit.
  * Integrated with **Metered.ca TURN/STUN servers** to bypass corporate network firewalls and proxy restrictions seamlessly.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **Computer Vision & AI:** MediaPipe Pose, Ultralytics YOLOv8 Nano
* **Video Streaming:** `streamlit-webrtc` + PyAV + OpenCV
* **Document Generation:** `fpdf2`
* **Networking / Firewall Bypass:** STUN/TURN via Metered.ca (Open Relay)

---

## 📦 Installation & Local Setup

### 1. Clone the repository
```bash
git clone [https://github.com/RashidiA/Ergonomic-Risk-Evaluation-REBA.git](https://github.com/RashidiA/Ergonomic-Risk-Evaluation-REBA.git)
cd Ergonomic-Risk-Evaluation-REBA

Create a virtual environment (Python 3.11 recommended):
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py
🔐 Deployment Configuration
When deploying to Streamlit Community Cloud, you must add your Metered.ca API Key to your Secrets:

Go to your App Settings on Streamlit Cloud.

Navigate to Secrets.

Add the following:

Ini, TOML

METERED_API_KEY = "your_pk_key_here"

📖 How to Use
Positioning: Stand 2-3 meters away from the camera, showing your profile (side view).

Metadata: Enter the Operator ID and Workstation in the sidebar.

Audit: Watch the live "Risk Score" metrics. If the score turns red, the posture requires immediate intervention.

Export: Click "Generate Audit Report" to save the findings as a PDF.

🤝 Contributing
Contributions to expand leg scoring, wrist flexion angles, or custom fine-tuned YOLO weights for specialized automotive parts are welcome!

Disclaimer: This tool is for educational and preliminary audit purposes. It should not replace professional medical or ergonomic advice.

Ergonomics and AI: Automating the REBA assessment

📖 Citation

Mohd Rashidi Asari. (2026). RashidiA/Ergonomic-Risk-Evaluation-REBA: Initial public release (v1.0.0). Zenodo. 
https://doi.org/10.5281/zenodo.18707034
