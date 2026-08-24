# Real-Time Automated Quality Control (QC) & Telemetry Pipeline

An end-to-end computer vision and web-streaming pipeline designed for automated quality control inspections. The system processes continuous video feeds using PyTorch and YOLOv8, computes real-time detection metrics and frame latency, and streams both annotated video feeds and JSON telemetry over high-speed FastAPI endpoints.

## Features
* **Real-Time Inference Engine:** Powered by YOLOv8 and PyTorch for bounding box object detection and tracking.
* **Low-Latency REST & Stream Endpoints:** Asynchronous FastAPI server serving live JSON telemetry and Multipart JPEG video feeds.
* **Integrated Web Dashboard:** Embedded side-by-side frontend interface for simultaneous visual monitoring and data telemetry.

## Tech Stack
* **Language:** Python 3.10+
* **Computer Vision & ML:** OpenCV, PyTorch, Ultralytics YOLOv8
* **Backend Framework:** FastAPI, Uvicorn

## Quickstart Guide

### 1. Clone the Repository & Setup Environment
```bash
git clone [https://github.com/your-username/qc-vision-pipeline.git](https://github.com/your-username/qc-vision-pipeline.git)
cd qc-vision-pipeline