#  Real-Time ASL Temporal Translator 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> An end-to-end edge AI pipeline that translates American Sign Language (ASL) into text in real-time using spatial-temporal neural networks.

##  Live Demo

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Live+Webcam+Inference+GIF+Goes+Here" alt="ASL Live Demo">
</p>

##  Overview
Standard computer vision models struggle with sign language because signs are not static images—they are movements over time. This project solves that by bridging **Computer Vision (MediaPipe/OpenCV)** for spatial feature extraction with **Sequential Deep Learning (LSTMs)** to map temporal patterns to English words.

The entire pipeline runs locally on edge compute (CPU/Standard GPU) without relying on paid APIs or cloud inference.

##  Architecture & Data Pipeline
1. **Feature Extraction (Spatial):** OpenCV captures the live video feed. Google MediaPipe extracts 21 3D hand landmarks (x, y, z coordinates) per frame.
2. **Sequence Windowing (Temporal):** The data pipeline captures a rolling buffer of 30 consecutive frames, converting the physical movement into a mathematical tensor of shape `(30, 126)` for two hands.
3. **Deep Learning Brain (LSTM):** A Long Short-Term Memory (LSTM) network processes the sequence, retaining memory of the movement arc to predict the correct sign.
4. **Real-Time Inference:** The highest-confidence prediction is overlaid directly onto the live OpenCV video feed.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow / Keras (LSTM architecture)
* **Data Processing:** NumPy, Scikit-Learn, Matplotlib

##  How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/real-time-asl.git
cd real-time-asl
