<div align="center">
<p align="center">
  <img src="https://github.com/user-attachments/assets/690d246d-a83b-4788-9996-0d14556322c4" alt="Trinetra Logo" width="100%">
</p>
<h1><b>TRINETRA (त्रिनेत्र)</b></h1><br>
Persistent Visual Tracking & Target Re Identification System in Cluttered Environments For Fighter Jets!
<br><br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-000000?style=for-the-badge&logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Blender](https://img.shields.io/badge/Blender-4.0+-E87D0D?style=for-the-badge&logo=blender&logoColor=white)](https://www.blender.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)](LICENSE)

<p align="center">
  <b>Maintaining target identity across occlusion, high speed maneuvers, and severe visual clutter.</b>
</p>

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Dual Demonstration System](#-dual-demonstration-system) •
[Quick Start](#-quick-start) •
[Tech Stack](#-tech-stack)

---

</div>

## Overview

In modern aerial defense, autonomous robotics, and search and rescue operations, visual clutter from buildings, terrain, vegetation, and other vehicles poses a severe challenge. Traditional visual tracking pipelines frequently experience **ID switches** or complete **target loss** during temporary occlusion.

Inspired by low-altitude dogfight maneuvers, **Trinetra** (*The Three-Eyed Engine*) is an AI driven, persistent visual tracking framework. It unifies high speed object detection, multi-object tracking, state estimation via Kalman filtering, and feature-based visual re-identification (Re-ID) to predict trajectories behind obstacles and seamlessly reacquire targets upon reappearance.

---

## Key Features

* **Real-Time Detection & Multi-Object Tracking:** Powered by YOLOv11 and ByteTrack for robust multi-target frame-by-frame associations.
* **Occlusion State Estimation:** Employs Kalman Motion Prediction to project target trajectories during complete visual loss.
* **Visual Re-Identification (Re-ID):** Maintains visual memory signatures (color embeddings and appearance features) to eliminate identity switches when targets re-emerge alongside distractors.
* **Dual Environment Validation Engine:** 
  1. **Physical Hardware Arena:** Live webcam tracking with ESP32/IMU-enabled physical jet models.
  2. **Procedural Blender Simulation:** 3D synthetic environment generator to simulate urban canyons, forest canopy, and mountain pass occlusions.
* **Interactive Live Dashboard:** Real-time stream visualizing bounding boxes, track confidence scores, state metrics (`VISIBLE` vs `OCCLUDED`), and predicted flight paths.

---

## System Architecture

Trinetra operates as a unified visual intelligence pipeline receiving input from either live video feeds or dynamic simulation camera streams:

              ┌────────────────────────────────────────┐
              │            INPUT SOURCES               │
              │   Physical Camera  │   Blender Stream  │
              └───────────────────┬────────────────────┘
                                  │
                                  ▼
              ┌────────────────────────────────────────┐
              │            TRINETRA AI ENGINE          │
              │                                        │
              │  1. Object Detection (YOLOv11)         │
              │  2. Multi-Object Tracking (ByteTrack)  │
              │  3. Motion Prediction (Kalman Filter)  │
              │  4. Appearance Re-ID (Feature Match)   │
              └───────────────────┬────────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
        ┌─────────────────────────┐ ┌─────────────────────────┐
        │   Physical Arena UI     │ │  3D Blender Simulator   │
        │ • Real-time overlay     │ │ • Dynamic camera path   │
        │ • Prediction trajectories│ │ • Synthetic dataset gen│
        └─────────────────────────┘ └─────────────────────────┘

---

## Dual Demonstration System

### 1. Hardware Arena (Physical Gamified Demo)
* **Setup:** A 1.5m x 1m tabletop arena representing a low-altitude battlefield with physical occlusion barriers.
* **Mechanics:** Players maneuver physical aircraft tokens (equipped with optional ESP32/IMU telemetry).
* **Objective:** As aircraft pass behind obstacles, Trinetra highlights the predicted point of re-emergence and locks back onto the correct target ID without swapping identities.


### 2. Blender 3D Synthetic Simulation
* **Environment Generation:** Procedural generation of mountain passes, forests, and building clusters.
* **Synthetic Data Pipeline:** Automatically exports annotated frames (`images` & `labels`) under varying lighting, camera angles, and weather conditions to train custom YOLO detection models.

---
<p align="center">
  <img src="https://github.com/user-attachments/assets/ce6f2537-f4b2-4261-98f9-5b6f77ee1d91" alt="Trinetra Logo" width="100%">
  <img src="https://github.com/user-attachments/assets/aca0b394-2862-47b9-83b3-75bec4012884" alt="Trinetra Logo" width="100%">
</p>

## Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Computer Vision & AI** | YOLOv11 (Ultralytics), OpenCV, ByteTrack, NumPy, SciPy |
| **3D Simulation** | Blender 4.0+ (Python API), Synthetic Data Generator |
| **Hardware & Telemetry** | ESP32, MPU6050 IMU, WebSockets / HTTP REST API |
| **Web Server & UI** | Python `aiohttp`, HTML5/JS Dashboard |

---

## Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* Virtual Environment manager (`venv` or `conda`)

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/trinetra.git](https://github.com/your-username/trinetra.git)
cd trinetra

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Running the Server & Dashboard

# Install dependencies
pip install -r requirements.txt
Start the main asynchronous tracking server:

```bash
python3 -u server.py
```
Once running, access the web interfaces:

* **Dashboard:** `http://localhost:8080/dashboard`
* **Phone / Camera Stream:** `http://localhost:8080/phone`

---

## Evaluation Metrics

Trinetra is evaluated using industry-standard multi-object tracking metrics:

* **MOTA (Multi-Object Tracking Accuracy):** Overall tracking precision across complex scenes.
* **ID Switches (IDSW):** Frequency of identity swaps during cross-overs and re-emergence.
* **Reacquisition Time:** Latency in milliseconds to correctly identify targets post-occlusion.
* **Occlusion Trajectory RMSE:** Accuracy of Kalman-predicted path vs. ground-truth location.

---

## Future Applications

While aerial defense serves as the primary scenario, Trinetra's visual persistence framework naturally extends to:

* **Autonomous Drones:** Navigation and target retention in dense forest/urban environments.
* **Search & Rescue:** Continuous tracking of survivors through smoke, debris, or foliage.
* **Intelligent Transportation:** Persistent vehicle tracking across tunnels and overpasses.

---

<div align="center">

Developed with ❤️ during the Hackathon.

</div>
