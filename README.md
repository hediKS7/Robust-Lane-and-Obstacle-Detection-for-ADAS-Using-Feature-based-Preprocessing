Here's a comprehensive, ready-to-paste `README.md` file with detailed explanations and system architecture:

---

# 🚗 Robust Lane and Obstacle Detection for ADAS Using Feature-based Preprocessing

<div align="center">

![ADAS System](https://img.shields.io/badge/ADAS-Computer%20Vision-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Advanced Driver Assistance System for robust lane and obstacle detection under challenging conditions**

</div>

## 🎯 Project Overview

This project implements a **robust computer vision system** for Advanced Driver Assistance Systems (ADAS) that reliably detects lane markings and road obstacles under various challenging conditions. Traditional ADAS systems often struggle with degraded visual inputs caused by weather, lighting variations, and poor road markings. Our approach combines **classical computer vision techniques** with **deep learning models** to create a hybrid system that maintains high detection accuracy even in suboptimal conditions.

### 🔍 Problem Statement
- **Image degradation** in real driving conditions (rain, fog, shadows, motion blur)
- **Weak edge and feature visibility** of lane markings
- **High sensitivity** of detection algorithms to noisy inputs
- **Variable lighting conditions** affecting detection consistency

### 💡 Proposed Solution
- **Feature-based preprocessing** using classical filters (Sobel, Prewitt, Laplacian, Gaussian)
- **Hybrid detection pipelines** combining traditional CV and CNN approaches
- **Multi-condition robustness** through adaptive preprocessing strategies
- **Real-time processing** capabilities for vehicle integration

---

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Multi-Condition Robustness** | Works under low light, rain, fog, shadows, and motion blur | Reliable performance in real-world scenarios |
| **Dual Detection Pipelines** | Separate optimized pipelines for lanes and obstacles | Specialized processing for different detection tasks |
| **Classical Filter Preprocessing** | Sobel, Canny, Gaussian, and morphological operations | Enhanced edge visibility and noise reduction |
| **Hybrid Approach** | Combines filter-based methods with CNN architectures | Balances speed and accuracy |
| **Real-time Processing** | Optimized for vehicle-mounted hardware | Suitable for ADAS applications |
| **Interactive UI Demo** | Graphical interface for testing and visualization | Easy system evaluation and demonstration |

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    VEHICLE-MOUNTED CAMERA                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    IMAGE ACQUISITION MODULE                  │
│  • Frame capture (1280×720 @ 30fps)                         │
│  • Initial normalization and resizing                        │
│  • Buffering for continuous processing                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                PREPROCESSING PIPELINE (CLASSICAL)           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Gaussian│  │  Sobel  │  │  Canny  │  │  ROI    │        │
│  │  Blur   │→│  Filter  │→│  Edge   │→│ Selection│        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└────────────────────────┬─────────────────┬──────────────────┘
                         │                 │
                         ▼                 ▼
┌─────────────────┐    ┌─────────────────────────────────────┐
│  LANE DETECTION │    │       OBSTACLE DETECTION            │
│    PIPELINE     │    │                                     │
├─────────────────┤    ├─────────────────────────────────────┤
│ • Hough Transform│   │ • YOLO-based Detection             │
│ • Sliding Window │   │ • Instance Segmentation            │
│ • Polynomial Fit │   │ • Multi-class Classification       │
│ • Curved/Straight│   └─────────────────────────────────────┘
└─────────────────┘                    │
                         │             │
                         ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FUSION & DECISION LAYER                   │
│  • Lane position validation                                 │
│  • Obstacle distance estimation                             │
│  • Risk assessment and alert generation                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT VISUALIZATION                      │
│  • Overlay lanes on original video                          │
│  • Bounding boxes for obstacles                             │
│  • Warning indicators and alerts                            │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Components

#### 1. **Image Acquisition Module**
- **Input**: Continuous stream from front-facing vehicle camera
- **Resolution**: 1280×720 pixels
- **Frame Rate**: 30 FPS (adjustable)
- **Pre-processing**: Automatic white balance, exposure adjustment

#### 2. **Preprocessing Module**
```
Raw Image → Grayscale Conversion → Noise Reduction → Edge Enhancement → ROI Extraction
    │              │                   │                  │                  │
    │              │              Gaussian Blur      Sobel/Canny        Polygon Mask
    │              │              (σ=1.5)         (Thresholds: 50,150)  (Trapezoidal)
    └──────────────┴──────────────────┴──────────────────┴──────────────────┘
```

#### 3. **Lane Detection Pipeline**
```
Two Parallel Approaches:
A. Classical Approach (Straight Lanes):
   Preprocessed Image → Canny Edge → Hough Transform → Line Clustering → Lane Marking
  
B. Advanced Approach (Curved Lanes):
   Preprocessed Image → Sliding Window Search → Polynomial Fitting → Lane Tracking
   
   Sliding Window Process:
   1. Histogram peak detection for lane base
   2. 9 vertical windows (height: 80px)
   3. Window recentering based on detected pixels
   4. 2nd-order polynomial fitting: x = Ay² + By + C
```

#### 4. **Obstacle Detection Pipeline**
```
Input Frame → YOLO Network → Bounding Boxes → Non-Max Suppression → Classification
                    │
                    ├──→ Segmentation Mask (optional)
                    │
                    └──→ Distance Estimation (using camera calibration)
```


---

## 📄 License

This project is developed for academic purposes as part of the final year project at Ecole Polytechnique de Tunisie. The code is available for educational and research purposes. For commercial use, please contact the authors.

```
Copyright (c) 2026 Hedi Ksentini

Permission is hereby granted for academic and research use only, subject to the following conditions:
1. The above copyright notice and this permission notice shall be included in all copies.
2. Appropriate credit must be given to the original authors.
3. Commercial use requires explicit written permission from the authors.
```

---



<div align="center">

### 🚀 Ready to enhance your ADAS capabilities?

**Star this repo if you find it useful!**

[Report Bug](https://github.com/yourusername/robust-lane-obstacle-detection/issues) · 
[Request Feature](https://github.com/yourusername/robust-lane-obstacle-detection/issues) · 
[Contact Authors](mailto:contact@example.com)

</div>
