# 🧠 NeuroScan AI — Brain Tumor Diagnostics Console

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neuroscan-diagnostics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NeuroScan AI is a state-of-the-art diagnostic assistant that leverages Deep Learning to detect brain tumors from MRI scans. It provides medical professionals with interpretable AI insights, including tumor classification, visual attention maps (Grad-CAM), and AI-generated clinical reports.

> **⚠️ Disclaimer:** This application is for educational and demonstrational purposes only. It is not a substitute for professional medical diagnosis.

---

## 🚀 Live Demo
**Try the live application here:** 👉 **[NeuroScan AI Console](https://neuroscan-diagnostics.streamlit.app)**

---

## ✨ Key Features
- **🔬 Deep Learning Classification**: Utilizes a highly optimized `MobileNetV2` Convolutional Neural Network (CNN) to classify MRI scans into four categories (Glioma, Meningioma, Pituitary, or No Tumor).
- **🔥 Explainable AI (Grad-CAM)**: Generates a heat map overlaid on the original MRI to show exactly *where* the model is looking to make its prediction, building trust and interpretability.
- **🤖 AI Medical Summaries**: Integrates with cutting-edge LLMs (via `g4f` / Claude AI) to automatically generate comprehensive clinical summaries based on the detected tumor type.
- **📄 Instant PDF Export**: Compiles the MRI, Grad-CAM heatmap, classification results, and AI clinical summary into a highly professional, downloadable medical PDF report.
- **🌐 Multilingual Support**: Generates AI reports in both **English** and **Hinglish**.

---

## 🏗️ Architecture & Workflow

The application follows a streamlined, modern AI workflow from image upload to clinical report generation:

```mermaid
graph TD
    %% Styling
    classDef user fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff,rx:10px,ry:10px
    classDef frontend fill:#3182ce,stroke:#2b6cb0,stroke-width:2px,color:#fff,rx:10px,ry:10px
    classDef ai fill:#805ad5,stroke:#6b46c1,stroke-width:2px,color:#fff,rx:10px,ry:10px
    classDef output fill:#38a169,stroke:#2f855a,stroke-width:2px,color:#fff,rx:10px,ry:10px
    classDef llm fill:#dd6b20,stroke:#c05621,stroke-width:2px,color:#fff,rx:10px,ry:10px

    %% Nodes
    A[👨‍⚕️ User]:::user
    B[💻 Streamlit UI]:::frontend
    C{🧠 MobileNetV2 CNN}:::ai
    D[📊 Classification Result]:::output
    E[🔥 Grad-CAM Heatmap]:::output
    F{🤖 Claude AI / g4f}:::llm
    G[📝 Clinical Summary]:::output
    H[📄 PDF Generator]:::frontend
    I[📥 Downloadable Report]:::output

    %% Flow
    A -->|Uploads MRI Scan| B
    B -->|Preprocesses Image| C
    C -->|Predicts Tumor Type| D
    C -->|Extracts Features| E
    D -->|Context Prompt| F
    F -->|Generates| G
    B --> H
    D --> H
    E --> H
    G --> H
    H -->|Exports| I
    I -->|Delivers to| A
```

---

## 🛠️ Technology Stack
- **Frontend & Deployment**: Streamlit, Streamlit Cloud
- **Deep Learning**: TensorFlow / Keras (MobileNetV2)
- **Computer Vision**: OpenCV, Pillow, NumPy
- **Generative AI**: `g4f` (Free GPT-4 / Claude API)
- **Document Generation**: ReportLab

---

## 💻 Local Installation

If you want to run this project locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/ajay160380/NeuroScan-AI-Brain-Tumor-Diagnostics-Console.git
cd NeuroScan-AI-Brain-Tumor-Diagnostics-Console
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ajay160380/NeuroScan-AI-Brain-Tumor-Diagnostics-Console/issues).

## 📝 License
This project is [MIT](https://opensource.org/licenses/MIT) licensed.
