# Synthetic Image Detector

A full-stack AI microservice that distinguishes between authentic human photographs and AI-generated imagery (Midjourney, DALL-E, Stable Diffusion). This system combines a custom-architected Convolutional Neural Network (CNN) with a decoupled FastAPI backend and Gradio frontend, allowing for high-accuracy forensic analysis of digital media.

It features a production-ready inference engine optimized for CPU usage via ONNX, capable of detecting microscopic diffusion artifacts invisible to the human eye.

## Demo

<img width="1918" height="960" alt="image" src="https://github.com/user-attachments/assets/26f28195-863b-49a8-906b-0a5df402e0e1" />

<img width="1919" height="959" alt="image" src="https://github.com/user-attachments/assets/9850dd68-61b6-4dc0-8891-6d4aa5b5f2a2" />



## 🚀 Key Features

* **⚡ Custom Architecture:** Architected a specialized CNN from scratch (rather than using pre-trained Transfer Learning) to specifically target high-frequency diffusion noise patterns.
* **📷 Robust Preprocessing:** Implements exact "CenterCrop" logic to preserve microscopic pixel artifacts. Unlike standard resizing (which blurs AI noise), this acts as a "forensic lens" for the model.
* **🧠 ONNX Inference:** The PyTorch model is exported to ONNX format, allowing the system to run highly efficient CPU-based inference without heavy GPU dependencies.
* **📝 Microservice Design:** Fully decoupled architecture with a **FastAPI** backend for processing and a **Gradio** frontend for user interaction.

---

## 🛠️ Tech Stack

* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Gradio
* **Machine Learning:** PyTorch, ONNX Runtime
* **Image Processing:** PIL (Pillow), NumPy
* **Environment:** Python 3.11+

---

## 📂 Project Structure

```text
├── models/
│   └── artifacts/             # Saved Model Files (.onnx)
├── src/
│   ├── __init__.py
│   ├── classify.py            # ONNX Inference Logic
│   ├── preprocess.py          # Exact CenterCrop Transformation Pipeline
│   └── logger_config.py       # Centralized Logging
├── services/
│   ├── app.py                 # FastAPI Backend Endpoints
├── interface/
│   └── app.py                 # Gradio Frontend Client
├── notebooks/
│   ├── 1.0-Data_Exploration.ipynb    # Data exploration notebook
│   ├── 2.0-Model_Training.ipynb      # Load model and train
│   └── 3.0-Model-Testing.ipynb       # Test model and backend inference
├── pyproject.toml           # Dependencies
├── poetry.lock              
└── README.md                # Documentation
```

## ⚡ Installation & Setup
### 1. Prerequisites
* Python 3.11+ installed.

* Poetry installed for dependency management.

### 2. Clone and Install
```bash
git clone [https://github.com/yourusername/synthetic-image-detector.git](https://github.com/yourusername/synthetic-image-detector.git)
cd synthetic-image-detector
poetry install
```

### 3. Setup Models
Place your trained .onnx model file in the artifacts directory:

models/artifacts/your_custom_model.onnx

(The system automatically detects any model placed in this folder).

## 🏃‍♂️ How to Run
1. Start the Backend (API)
Open a terminal and launch the FastAPI server.
```bash
uvicorn src.api:app --reload
```
API will run at http://127.0.0.1:8000

2. 2. Start the Frontend (UI)
Open a second terminal and launch the Gradio interface.
```bash
python -m interface.app
```
Gradio Interface will run on http://127.0.0.1:7860

### 🧠 Model Performance
The model utilizes a custom CNN architecture optimized for 256x256 RGB images.
* It achieved 91% accuracy on the validation test set.
* It successfully distinguishes between high-fidelity photorealism and diffusion-based generation.

### 🔮 Future Improvements
* Vision Transformers (ViT): Implement a ViT architecture to compare attention-based detection vs. CNN feature extraction.
* Dockerization: Containerize both the API and UI for cloud deployment (AWS/GCP).
* Explainability: Add Grad-CAM heatmaps to visualize exactly which pixels the AI identified as "synthetic."
