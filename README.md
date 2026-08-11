# Real-Time ASL Translator

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> An edge AI pipeline that recognizes American Sign Language (ASL) signs in real time from a webcam feed, using MediaPipe hand tracking and an LSTM neural network.

## Live Demo

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Live+Webcam+Inference+GIF+Goes+Here" alt="ASL Live Demo">
</p>

## Overview

Sign language recognition is a *temporal* problem, not a static image classification one — the same handshape can mean different things depending on how it moves over time. This project bridges **computer vision** (MediaPipe hand landmark tracking) with **sequential deep learning** (an LSTM network) to recognize signs from motion, not single frames.

The entire pipeline runs locally on CPU, with no paid APIs or cloud inference required.

## Current Status

This is a working prototype covering **5 signs**: Hello, Thank You, Please, Yes, No.

- Trained on a small dataset (~41 real examples from the WLASL dataset, expanded to ~205 via data augmentation)
- Validation accuracy around 90% on held-out data — though with such a small dataset, this number should be taken as a promising signal rather than a guarantee of real-world accuracy
- Tracks a single hand at a time
- Not yet tested extensively across different lighting conditions, backgrounds, or signers beyond the original developer

Expanding vocabulary and dataset size is the clear next step (see **Future Improvements** below).

## Architecture & Data Pipeline

1. **Feature Extraction (Spatial):** OpenCV captures the live video feed. MediaPipe's HandLandmarker (Tasks API) extracts 21 3D hand landmarks (x, y, z) per frame from a single hand.
2. **Sequence Windowing (Temporal):** A rolling buffer holds the most recent 30 frames, forming a tensor of shape `(30, 63)` — 30 frames × 21 landmarks × 3 coordinates.
3. **Deep Learning Brain (LSTM):** A stacked LSTM network processes the sequence, learning the motion pattern rather than just a single pose, to classify which sign it matches.
4. **Real-Time Inference:** The highest-confidence prediction is overlaid on the live OpenCV video feed, with a confidence threshold to avoid flickering between guesses.

## Tech Stack

* **Language:** Python 3.13
* **Computer Vision:** OpenCV, MediaPipe (Tasks API)
* **Deep Learning:** TensorFlow / Keras (LSTM architecture), trained in Google Colab
* **Data Processing:** NumPy, Scikit-learn (train/test split)
* **Dataset:** [WLASL (Word-Level American Sign Language)](https://github.com/dxli94/WLASL)

## Project Structure

```
real-time-asl/
├── RetrievingWebcam.py           # Webcam capture test (Phase 1)
├── RetrievingWebcam..py          # Webcam + hand landmark overlay (Phase 1)
├─  ExtractionOfvalues.py         # Extracts landmarks from WLASL videos into training data
├── Live_Inference.py             # Live webcam sign recognition (the main app)
├── Requirements.txt
├── hand_landmarker.task          # MediaPipe hand detection model (see setup below)
├── asl_model.h5                  # Trained LSTM model (see setup below)
└── notebooks/
    └── Train_Model.ipynb         # Colab notebook: data prep + LSTM training
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/real-time-asl.git
cd real-time-asl
```

### 2. Set up a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r Requirements.txt
```

### 3. Download the required model files

These aren't included in the repo (kept out via `.gitignore` since they're binary/regenerable):

- **`hand_landmarker.task`** — MediaPipe's hand detection model. Download from:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
  Place it in the project root.
- **`asl_model.h5`** — the trained sign-classification model. Either train it yourself using `notebooks/Train_Model.ipynb`, or place your own trained copy in the project root.

### 4. Run the live inference app

```bash
python LiveInference.py
```

Press **`q`** (with the video window focused) to quit.

*(Make sure your webcam isn't already in use by another app, like Zoom or Teams.)*

## Dataset & Preprocessing (WLASL)

Rather than recording a small set of self-made webcam clips, this project sources training data from **WLASL**, a large real-world video dataset for ASL recognition. Working with it required a proper data pipeline rather than a simple recording script:

- **JSON parsing:** matched target words ("glosses") to their video examples using `WLASL_v0_3.json`.
- **Missing-data handling:** cross-checked against `missing.txt` to skip broken/unavailable video links rather than crashing on them.
- **Batch feature extraction:** ran MediaPipe hand tracking across every valid video file, frame by frame.
- **Sequence standardization:** videos vary in length, so each one is resampled to a fixed 30-frame sequence (via evenly spaced sampling) to keep tensor shapes consistent for the LSTM.
- **Data augmentation:** given the small number of real examples per word, additional synthetic training examples were generated using coordinate jitter and slight time-warping.

## Future Improvements

- [ ] Expand vocabulary beyond the current 5 words
- [ ] Collect a larger, more diverse set of real (non-augmented) examples per word
- [ ] Add dynamic thresholding to reduce false positives when hands are simply resting
- [ ] Support two-hand signs (currently single-hand only)
- [ ] Port the model to TensorFlow Lite for deployment on a Raspberry Pi or other edge device
- [ ] Evaluate performance across different signers, lighting, and backgrounds

## Connect

Chukwuneta Emmanuel Chidubem
Aspiring Machine Learning Engineer | emzy45cool@gmail.com
