<img width="1192" height="672" alt="Screenshot 2025-10-24 140703" src="https://github.com/user-attachments/assets/6401badc-e616-4ec0-ae50-70cff943c412" />
🚗 Intelligent License Plate Fraud Detection System (ILPFDS)

Fraudulent license plates pose a growing security and financial risk. Criminals often alter or forge plates to hide vehicle identities, enabling crimes like smuggling, theft, and toll evasion.
ILPFDS is an AI-powered solution that not only detects license plates but also checks their authenticity and structure using deep learning and OCR.

🎯 Objectives

Detect fraudulent or tampered plates under real-world conditions.
Use YOLOv8 for fast and accurate plate detection.
Apply CNN + OCR (EasyOCR/Tesseract) for text extraction and validation.
Identify anomalies in font, spacing, or layout.
Generate simple reports marking Valid ✅ or Fraudulent ❌ plates.

🧱 System Overview
Input Acquisition – capture image/video.
Plate Detection (YOLOv8) – locate license plates.
Preprocessing – enhance clarity and contrast.
Character Validation (CNN) – analyze authenticity.
Report Generation – display results with color-coded labels.

💻 Tech Stack
Task	Technology
Detection	YOLOv8
OCR	EasyOCR / Tesseract
Deep Learning	PyTorch
Image Processing	OpenCV
Visualization	Matplotlib / Streamlit
Language	Python
🚀 Run the Project
# Clone repository
git clone https://github.com/siya-06/ILPFDS.git
cd ILPFDS

# Install dependencies
pip install -r requirements.txt

# Run main script
python main.py
<img width="977" height="728" alt="Screenshot 2025-10-13 003628" src="https://github.com/user-attachments/assets/efa30dcc-9c78-4fe5-ad91-c12f8d87fe07" />

