
---

# Automated Vehicle Counting Using Deep Learning

### Submitted by: Lindon Falconer
### Chosen and completed by: Nathan Myrie 

## Project Overview

**Engineering Programme:**
- Electrical Power Engineering
- Electronics Engineering
- Biomedical Engineering

**Academic Year:** 2024/25  
**No. of Students:** 1 / 1

**Project Title:** Automated Vehicle Counting Using Machine Learning  
**Project Proposer(s):** Lindon Falconer  
**Project Supervisor(s):** Lindon Falconer  
**Project Category:** IV: Investigation with System Development and/or Design

---

## 1. Background

A traffic counter is an electronic device used to count, classify, and/or measure the speed of vehicular traffic passing along a given roadway. Typically deployed near the roadway, it uses on-road media such as pneumatic road tubes, piezo-electric sensors, or inductive loops to detect passing vehicles. Pneumatic road tubes are generally used for temporary studies, while piezo-electric sensors and inductive loops are employed for permanent studies to analyze seasonal traffic trends and monitor congestion on major roads.

With advancements in technology, video processing can now be used to count and classify motor vehicles.

---

## 2. Objectives

1. **Research deep learning methods:**  
   Explore various deep learning approaches for counting pedestrians and vehicles using cameras. Compare and select the most suitable method.

2. **Object detection datasets:**  
   Search for vehicle (car, truck, bus, bike) and pedestrian object detection datasets. Optionally, create your own dataset. Develop and train a deep learning model capable of detecting vehicles and pedestrians from live or recorded video. Transfer learning may be used to enhance the model's performance in terms of accuracy, precision, and recall.

3. **System implementation at UWI main gate:**  
   Deploy the system to count pedestrians and vehicles entering or leaving the campus. The constraints and limitations of the setup will be determined in collaboration with the supervisor. Analyze the model using metrics such as accuracy, precision, recall, F1 score, ROC, and AUC curves.

4. **Communication system design:**  
   Implement a communication system to store and display counting data on a website. Additionally, design a solar PV system, communication system, and enclosure to support the device in remote locations (this will not be implemented during the project).

---

## 3. Implementation and Methodology

The design will use customized or existing deep learning models. Transfer learning can also be leveraged in this project. The student and supervisor will define the required hardware to train the model and run the application. A basic website will be used for storing and viewing counting data.

---

## 4. Summary of Requirements

**Prerequisite Skills and Knowledge:**  
- Machine learning and deep learning  
- Python programming  
- Web development  
- Linux operating systems

The student is expected to learn these technologies during the first semester.

**General Hardware Requirements:**  
- Small computer  
- Camera  
- Wireless communication modules  
- Solar panel  
- Batteries

**General Software Requirements:**  
- Embedded system software development  
- Deep learning and image recognition libraries (TensorFlow, PyTorch, OpenCV, etc.)  
- Python  
- Linux/Unix

---

## 5. Selection Details

- **Selection Opens:** Monday, September 16, 2024 - 2:00 AM  
- **Selection Closes:** Saturday, September 28, 2024 - 12:23 AM  

---

# Automated Vehicle Counting Using Deep Learning

**Authors:** Nathan Myrie
**Supervisor:** Lindon Falconer
**Academic Year:** 2024/25

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Background](#background)
* [System Architecture](#system-architecture)
* [Implementation & Methodology](#implementation--methodology)
* [Hardware & Software Requirements](#hardware--software-requirements)
* [Data Collection & Preparation](#data-collection--preparation)
* [Model Training](#model-training)
* [Deployment](#deployment)
* [Results & Evaluation](#results--evaluation)
* [Budget & Costs](#budget--costs)
* [Future Work](#future-work)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

## Project Overview

This project implements an automated traffic counting system using deep learning-based object detection and tracking. It captures live video feeds at the University of the West Indies main gate, identifies and counts vehicles and pedestrians in real time, and streams data to a web dashboard for visualization and analysis.

## Objectives

* Research and compare deep learning approaches for multi-class counting (cars, trucks, buses, bikes, pedestrians)
* Develop and train a robust object detection model with transfer learning
* Deploy the system on a resource-constrained edge device
* Implement a wireless communication pipeline and web-based dashboard
* Design a solar-powered enclosure for remote, off-grid deployment (design only)

## Background

Traditional traffic counters (pneumatic road tubes, piezo-electric sensors, inductive loops) are often intrusive, single-purpose, and require manual data retrieval. Modern video-based systems offer non-intrusive installation, real-time analytics, and multi-class classification, but are seldom optimized for low-power, remote deployments.

## System Architecture

```
Camera --> Edge Device (Raspberry Pi 4B + Coral USB Accelerator)
      --> Object Detection & Tracking (YOLOv11 + SORT)
      --> Counting Logic
      --> FastAPI Server
      --> AWS PostgreSQL Database
      --> Web Dashboard (React.js)
```

## Implementation & Methodology

1. **Model Selection:** YOLOv11 and MobileNet-SSD evaluated for mAP and latency.
2. **Transfer Learning:** Pre-trained on COCO dataset, fine-tuned with custom campus footage.
3. **Tracking & Counting:** SORT algorithm assigns unique IDs and enforces virtual line logic.
4. **Data Pipeline:** Counts POSTed via FastAPI to a cloud database; front-end polls API.

## Hardware & Software Requirements

* **Hardware:**

  * Raspberry Pi 4B
  * Coral USB Accelerator
  * Outdoor USB camera
  * MiFi hotspot (4G)
  * Solar PV panels & battery (design)
* **Software:**

  * Python 3.x
  * TensorFlow / PyTorch / OpenCV
  * FastAPI / PostgreSQL
  * React.js / Chart.js for dashboard

## Data Collection & Preparation

* **Sources:** Open Images, custom campus video (approx. 12,000 frames)
* **Annotation:** Pascal VOC format, classes: car, truck, bus, bike, pedestrian
* **Splits:** 70% training, 20% validation, 10% test

## Model Training

* **Environment:** Google Colab + Coral TPU
* **Parameters:** 50 epochs, batch size 16, initial learning rate 1e-3
* **Metrics:** mAP\@0.5, precision, recall, F1-score

## Deployment

* **Inference:** Process every 3rd frame at \~8 FPS on Pi+Coral
* **Communication:** Counts sent every minute via HTTPS POST
* **Dashboard:** Live count graphs, historical data filters, class breakdowns

## Results & Evaluation

* **Detection:** YOLOv11 achieved 82% mAP; SSD achieved 75% mAP.
* **Throughput:** YOLOv11 \~7 FPS; SSD \~10 FPS on edge device.
* **Counting Accuracy:** ±5% error vs. manual ground truth across test intervals.

## Budget & Costs

| Item                   | Cost (JMD)     |
| ---------------------- | -------------- |
| Initial Budget         | 100,000        |
| Estimated Cost         | 35,000         |
| **Total Expenditure**  | **110,879.48** |
| Monthly Recurring Cost | 6,585.17       |

> **Note:** Actual costs exceeded budget by \~10% and estimates by >200%, indicating the need for more accurate forecasting.

## Future Work

* Automate region-of-interest detection and dynamic virtual lines
* Integrate additional sensors (e.g., LiDAR, radar) for robustness
* Expand dataset with diverse lighting and weather conditions
* Upgrade edge hardware (Jetson Nano, additional TPUs)

## Acknowledgements

* Supervisor: Lindon Falconer
* Project Proposer: Lindon Falconer
* Funding & Support: UWI Department of Engineering

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

