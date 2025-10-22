# 🚀 OrbitEye – Real-Time Intelligent Safety Object Detection System

**OrbitEye** is a high-precision real-time object detection system built on the **YOLOv8** framework.
It identifies and classifies multiple **safety-critical objects**, ensuring enhanced situational awareness in mission environments like **space stations**, **industrial plants**, and **emergency zones**.
The model achieves a **94% mAP@0.5** accuracy on the test dataset, demonstrating strong detection consistency across multiple object classes.

---

## 🛰 Problem Statement

Traditional safety monitoring systems struggle to detect and classify safety equipment across diverse image types and lighting conditions.
Our solution — **OrbitEye** — bridges this gap using **deep learning and real-time vision inference** to automatically detect safety objects with high accuracy, even in challenging environments.

---

## 💡 Proposed Solution

**OrbitEye** uses a custom-trained **YOLOv8** model that can:

* Detect and classify **7 safety-critical objects**:

  * Oxygen Tank
  * Nitrogen Tank
  * First Aid Box
  * Fire Alarm
  * Safety Switch Panel
  * Emergency Phone
  * Fire Extinguisher
* Work seamlessly across **images, videos, and live camera feeds**
* Run as both a **web application** and a **Progressive Web App (PWA)** for offline accessibility
* Provide **real-time feedback** for proactive safety monitoring

---

## ⚙ Tech Stack

### 🧠 Model & Training

* Python
* YOLOv8 (Ultralytics)
* OpenCV, NumPy, PyTorch
* Custom dataset with YOLO-format annotations
* Trained using `train.py` with advanced augmentation and optimization

### 🔧 Backend

* Flask (for inference API & real-time streaming)
* RESTful endpoints for image/video upload and live detection

### 💻 Frontend

* HTML, CSS, JavaScript
* Flask templating + responsive UI
* **Progressive Web App (PWA)** for mobile installation & offline usage

### ☁ Deployment & Integration

* Local/Edge deployment with `.pt` model
* Configurable via `yolo_params.yaml`

---

## 🧩 Model Training Pipeline

### 📂 Dataset Preparation

* Collected and labeled a **custom dataset (7 classes)**
* Split into **train**, **validation**, and **test** sets

### ⚙ Training Configuration

* **Base Model:** YOLOv8s
* **Epochs:** 100
* **Optimizer:** AdamW (adaptive weight optimization for stable convergence)
* **Image Size:** 1024 × 1024
* **Batch Size:** 4
* **Augmentations:** Mosaic (0.9), MixUp, perspective warp, brightness/hue shifts
* **Learning Rate:** 0.00015 → 0.000005 (cosine decay)
* **Early Stopping:** Patience = 60 epochs

### 📈 Evaluation Metrics

* **Highest mAP@0.5:** 0.94
* **Robust detection and classification** on unseen test data

### ⚡ Performance Optimization

* Fine-tuned **loss weights** (box, class, DFL)
* **Label smoothing & regularization** to improve generalization
* **Cosine learning rate schedule** for smoother convergence

---

## 🧠 Core Features

✅ Real-time detection via webcam or IP camera
✅ Supports static image and video uploads
✅ Visualizes bounding boxes with class labels
✅ Adjustable confidence threshold
✅ PWA – installable, app-like experience
✅ Responsive, mobile-friendly web design
✅ Lightweight `.pt` model suitable for on-device inference

---

## 🧮 How It Works

1. User uploads an **image/video** or enables **real-time camera mode**
2. Flask backend processes the frame using **OpenCV**
3. The **YOLOv8s model (fine-tuned)** predicts object locations & classes
4. Results are displayed with bounding boxes, class names, and confidence scores
5. For videos, frames are extracted and processed sequentially for **stable predictions**

---

## 📊 Results

| Metric                      | Value            |
| --------------------------- | ---------------- |
| **mAP@0.5**                 | **0.94**         |
| **Classes Detected**        | 7                |
| **Model Size**              | ~50 MB           |
| **Average Inference Speed** | < 50ms per frame |

---

## 🌐 Live Interface (Web + PWA)

* **Frontend:** Clean, responsive UI built with HTML, CSS, and JS
* **Progressive Web App (PWA):** Installable on desktop/mobile
* **Backend:** Flask server for live and uploaded inference

Users can upload images or videos — or activate real-time camera mode — for instant object detection and visualization.

---

## 🔮 Future Enhancements

* Integration with **Raspberry Pi / Jetson Nano** for edge safety monitoring
* Real-time **alert system** for missing or misplaced safety gear
* **Multi-object tracking (MOT)** in videos
* **Cloud dashboard** for analytics and historical logs

---

## 🏁 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/OrbitEye.git
cd OrbitEye

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

Then open your browser and go to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🧾 License

This project is licensed under the **MIT License**.
© 2025 **Dive Into Infinity**
