# 🛡️ Real-Time REBA & NIOSH Ergonomic AI Auditor

An AI-powered computer vision application built with **Streamlit**, **MediaPipe**, **YOLOv8**, and **FPDF**. The application performs live ergonomic posture risk assessment using the **REBA (Rapid Entire Body Assessment)** framework and automatically triggers a **NIOSH Lifting Equation Assessment** whenever a lifted object is detected on the operator's hand.

---

## ✨ Key Features

1. **Real-Time REBA Posture Evaluation:**
   - Tracks 3D full-body pose landmarks using MediaPipe.
   - Dynamic joint angle calculations for Trunk, Neck, Upper Arms, Legs, and Wrists.
   - Real-time time-series percentage breakdown across REBA score tiers ($1\text{--}2$, $3\text{--}4$, $5+$).

2. **YOLOv8-Driven Object Detection & Hand Intersection:**
   - Detects objects held or manipulated by the operator using YOLOv8 Nano.
   - Calculates dynamic bounding-box intersections between hand landmarks and objects to automatically trigger ergonomic lifting checks.

3. **Automated NIOSH Lifting Equation (NLE) Engine:**
   - Dynamically calculates spatial parameters ($H, V, D, A$) in real time from skeletal pixel-to-cm calibrations.
   - Computes all six NIOSH multipliers ($\text{HM}, \text{VM}, \text{DM}, \text{AM}, \text{FM}, \text{CM}$) to yield the **Recommended Weight Limit (RWL)** and **Lifting Index (LI)**.

4. **Comprehensive 3-Page PDF Audit Report:**
   - **Page 1:** Full-Body REBA Posture Breakdown & Action/Risk Level Reference.
   - **Page 2:** Standard Manual Material Handling (MMH) Weight Matrix Assessment.
   - **Page 3:** Dedicated NIOSH Lifting Equation Audit with Multiplier Table & Safety Status ($\text{LI} \le 1.0$).

5. **Firewall / WebRTC Bypass:**
   - Integrated Metered STUN/TURN server support for stable video streaming across corporate firewalls.

---

## 📐 Mathematical & Ergonomic Frameworks

### 1. REBA Joint Scoring
Joint angles ($\theta$) are calculated via vector dot products across skeletal landmark triplets:
$$\theta = \arccos\left( \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{}} \right)$$

### 2. NIOSH Lifting Equation
$$\text{RWL} = \text{LC} \times \text{HM} \times \text{VM} \times \text{DM} \times \text{AM} \times \text{FM} \times \text{CM}$$

Where:
* **Load Constant ($\text{LC}$):** $23 \text{ kg}$
* **Horizontal Multiplier ($\text{HM}$):** $\frac{25}{H}$ ($H$ in cm, bounded between $25$ and $63\text{ cm}$)
* **Vertical Multiplier ($\text{VM}$):** $1 - 0.003 \vert{}V - 75\vert{}$ ($V$ in cm)
* **Distance Multiplier ($\text{DM}$):** $0.82 + \frac{4.5}{D}$ ($D$ in cm)
* **Asymmetric Multiplier ($\text{AM}$):** $1 - 0.0032(\alpha)$ ($\alpha$ in degrees)
* **Lifting Index ($\text{LI}$):** $\frac{\text{Actual Weight}}{\text{RWL}}$ (Safe if $\text{LI} \le 1.0$)

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9, 3.10, or 3.11
* Web Camera (Local or External)

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
Contributions to custom fine-tuned YOLO weights for specialized automotive parts are welcome!

Disclaimer: This tool is for educational and preliminary audit purposes. It should not replace professional medical or ergonomic advice.

Ergonomics and AI: Automating the REBA assessment

📖 Citation

Mohd Rashidi Asari. (2026). RashidiA/Ergonomic-Risk-Evaluation-REBA: Initial public release (v1.0.0). Zenodo. 
https://doi.org/10.5281/zenodo.18707034
