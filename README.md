# 🚗 Edge-to-Cloud AI Vehicle Damage Assessment System

An intelligent Edge-to-Cloud AI architecture designed to automate vehicle inspection using computer vision and deep learning.

This system validates image quality at the edge before sending it to a cloud-based damage detection pipeline.

---

## 📌 Problem Statement

Manual vehicle inspections for insurance, resale, and rental use cases are:

- Time-consuming  
- Subjective  
- Error-prone  
- Operationally inefficient  

This project introduces an AI-driven pipeline that standardizes and automates vehicle damage assessment.

---

## 🧠 System Architecture

User Upload  
↓  
Edge Quality Model (TFLite - MobileNetV2)  
↓  
If Accepted  
↓  
Cloud AI Core (YOLO - Planned)  
↓  
Structured JSON Output  

---

## ✅ Current Implementation

- ✔ Edge Image Quality Classifier (MobileNetV2)
- ✔ Synthetic Data Generation Pipeline
- ✔ TFLite Model Conversion
- ✔ FastAPI Backend Integration
- ✔ Automatic Image Acceptance / Rejection

---

## 🚧 In Progress

- Damage Detection using YOLO
- Part Segmentation using Mask R-CNN
- IoU-based Fusion Engine
- Severity Assessment (Area-to-Part Ratio)
- Explainable AI (Grad-CAM)

---

## 📂 Project Structure

project_root/
│
├── database/
│   └── raw_image_test/
│
├── edge_training/
│   ├── edge_dataset/
│   ├── generate_edge_data.ipynb
│   ├── train_edge_model.ipynb
│   ├── edge_model/
│   └── edge_model.tflite
│
├── backend/
│   ├── main.py
│   ├── uploads/
│   └── requirements.txt
│
└── README.md

---

## 🧠 Edge Model Details

- Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224
- Output: Binary Classification (Good / Bad)
- Deployment Format: TensorFlow Lite (TFLite)

The edge model rejects images that are:

- Blurry  
- Overexposed  
- Too dark  
- Poor quality  

This prevents bad inputs from reaching the cloud AI core.

---

## ⚙️ Backend (FastAPI)

The FastAPI server:

1. Receives uploaded image  
2. Runs TFLite edge model  
3. Returns structured JSON response  

### Example Response (Accepted)

```json
{
  "status": "accepted",
  "confidence": 0.83
}
```

### Example Response (Rejected)

```json
{
  "status": "rejected",
  "reason": "Image quality insufficient",
  "confidence": 0.27
}
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
cd project_root/backend
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install fastapi uvicorn tensorflow opencv-python python-multipart
```

### 4️⃣ Run FastAPI Server

```bash
python3 -m uvicorn main:app --reload
```

Open:

http://127.0.0.1:8000/docs

Use Swagger UI to upload and test images.

---

## 📊 Model Training Workflow

- Generate synthetic distortions (blur, glare, darkness)
- Train MobileNetV2 using transfer learning
- Fine-tune last layers
- Convert model to TFLite
- Integrate with FastAPI backend

---

## 🔮 Future Improvements

- YOLO-based multi-class damage detection
- Mask R-CNN vehicle part segmentation
- IoU-based damage-to-part mapping
- Severity grading engine
- Grad-CAM explainability layer
- Mobile deployment (Flutter + TFLite)

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- TensorFlow Lite
- OpenCV
- FastAPI
- Uvicorn

---

## 👥 Authors

- **[Yashkrit Singh](https://github.com/your-username)** — B.Tech Mechanical Engineering, IIT Patna  
- **[Sumit Rajpoot](https://github.com/sumit-username)** — B.Tech Engineering Physics  
- **[Chhavi Bamoriya](https://github.com/chhavi-username)** — B.Tech Chemical Engineering  
---

## 📜 License

This project is developed for research and hackathon purposes.
