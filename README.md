🛡️ AI REBA Auditor
An AI-powered ergonomic assessment tool built for Automotive to automate the Rapid Entire Body Assessment (REBA) process. Using real-time computer vision, the app identifies skeletal landmarks to calculate postural risk scores instantly.

🚀 Features
Real-time Tracking: Uses MediaPipe Pose to track 33 body landmarks without special hardware—just a standard webcam.

Automated REBA Scoring: Instant angle calculation for:

Trunk: Flexion, extension, and upright positioning.

Neck: Head tilt and alignment.

Upper Arms: Reaching and elevation metrics.

PDF Audit Reports: Capture a snapshot of high-risk postures and download a professional audit report for safety records.

Firewall Bypass: Integrated with Metered.ca TURN servers to ensure connectivity on strict factory/corporate Wi-Fi networks.

🛠️ Tech Stack
Frontend: Streamlit

AI Engine: MediaPipe

Video Streaming: streamlit-webrtc

Networking: STUN/TURN via Metered.ca (Open Relay)

📦 Installation & Local Setup
Clone the repository:
git clone https://github.com/your-username/geely-reba-auditor.git
cd reba-auditor

Create a virtual environment (Python 3.11 recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
This is an open-source project created for non-commercial ergonomic safety. Contributions to include Leg or Wrist scoring logic are welcome!

Disclaimer: This tool is for educational and preliminary audit purposes. It should not replace professional medical or ergonomic advice.

Ergonomics and AI: Automating the REBA assessment

📖 Citation
Mohd Rashidi Asari. (2026). RashidiA/Ergonomic-Risk-Evaluation-REBA: Initial public release (v1.0.0). Zenodo. 
https://doi.org/10.5281/zenodo.18707034
