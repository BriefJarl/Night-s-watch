# Night's Watch — Intelligent Border Video Analytics Platform (IBVAP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-yellow.svg)](https://ultralytics.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![SIH26187](https://img.shields.io/badge/SIH-26187-orange.svg)]()

> **Smart Automation (Edge-Compatible AI Vision Engine) developed for the Ministry of Home Affairs / Sashastra Seema Bal (SSB) under SIH26187.**

## 📖 Overview

Continuous human observation of standard-definition CCTV infrastructure at remote border outposts causes severe cognitive fatigue. Furthermore, heavy GPU clusters cannot be deployed at these remote borders due to extreme costs, thermal limits, power, and network constraints.

**Night's Watch** is a software-defined intelligence layer designed specifically to solve this problem. It is a highly optimised, edge-compatible Computer Vision pipeline that uses **software-defined triage (detection-gated pipelines)** and **Semantic Compression (JSON over raw video)** to operate efficiently on low-power edge nodes while guaranteeing data resiliency over unreliable network links.

## ✨ Core Features

*   **3-Stage Detection-Gated Pipeline:** Conserves compute by triggering heavy neural networks (YOLOv8/RT-DETR) only when zero-cost motion (MOG2) is detected. Objects are then handed off to a mathematical tracker (DeepSORT) to eliminate per-frame AI overhead.
*   **4-Layer False Alarm Suppression Stack:** Employs static masking, track confirmation, homography geometric gating (size-vs-depth), and an operator hard-negative loop to filter out 98% of nuisance alerts.
*   **Semantic Compression & Store-and-Forward:** Converts events into 2-8KB JSON payloads instead of streaming heavy H.264 video. Integrates a local SQLite edge queue for 100% data retention during satellite link outages, syncing asynchronously with the central cloud.
*   **ANPR & Analytics:** Bypasses edge hardware limits with software perspective-warping and multi-frame statistical voting for Indian standard license plates.
*   **GenAI RAG Copilot:** Embeds JSON alert payloads as semantic text into a PostgreSQL database with `pgvector`. Operators can query historical intrusions using natural language (e.g., *"Show me all vehicles near the red zone last night"*).
*   **Ranked Operator Dashboard:** A Streamlit front-end featuring a dynamic Folium MapLibre view, priority-ranked alert queue, and zero-latency MJPEG live CCTV walls.

## 🏗️ System Architecture & Hardware Tiers

### Hardware Deployment Tiers (Optimised for Tiers 1 & 2)

| Operational Tier | Target Deployment Location | Hardware Specification | Processing Capacity |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Tactical Edge)** | Remote Patrol Posts | Raspberry Pi 5 + Hailo-8L AI NPU | 2-4 RTSP Streams |
| **Tier 2 (Standard BOP)** | Standard Border Out Posts | NVIDIA Jetson Orin Nano Super | 8-12 RTSP Streams |
| **Tier 3 (Check Post Hub)** | Strategic Highways | Mini-PC with iGPU + OpenVINO | 16-24 RTSP Streams |
| **Tier 4 (Central Command)**| Regional Headquarters | Server with T4/A10 GPU Cluster | 100+ Streams |

### Directory Structure

```text
Night's Watch/
├── backend/                  # Cloud infrastructure (FastAPI)
│   ├── main.py               # REST API & async routing via Uvicorn
│   ├── database.py           # PostgreSQL + pgvector schema
│   └── genai_copilot.py      # LLM embedding and semantic search
├── edge_node/                # Remote node AI analytics (Python/OpenCV)
│   ├── vision_engine.py      # 3-stage gated pipeline & RTSP ingestion
│   ├── false_alarm_filter.py # Suppression stack & Homography math
│   ├── anpr_engine.py        # Perspective warp & OCR voting
│   └── edge_queue.py         # SQLite store-and-forward worker
├── frontend/                 # Operator Dashboard
│   └── app.py                # Streamlit UI with Folium/MapLibre mapping
├── tests/                    # Evaluation & Quality Assurance
├── .agents/                  # Autonomous system rules and knowledge
├── ARCHITECTURE.md           # Deep-dive system documentation
├── docker-compose.yml        # Container orchestration
└── requirements.txt          # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

*   **Python:** Version 3.10 or higher.
*   **Docker & Docker Compose:** Required to run the PostgreSQL database with `pgvector` locally.

### 1. Database Setup (Cloud Server)

Ensure Docker is running, then spin up the `pgvector` database:

```bash
docker-compose up -d
```

### 2. Environment & Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

*(Note: YOLOv8 model weights will be downloaded automatically on the first run).*

### 3. Running the Stack

You will need to open three separate terminal windows (ensure the virtual environment is activated in each).

**Terminal 1: Start the Central Backend (FastAPI)**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start the Edge Node Simulator**
```bash
# This will launch the vision_engine and begin processing sample-videos/
python -c "from backend.main import _launch_vision_engines; _launch_vision_engines()"
```
*(In a real production environment, edge nodes run physically detached from the backend server).*

**Terminal 3: Start the Operator Dashboard (Streamlit)**
```bash
streamlit run frontend/app.py
```

## 📡 API Reference (Backend)

The Central Backend provides the following core endpoints (view full Swagger UI at `http://localhost:8000/docs`):

*   `POST /api/v1/alerts`: Ingests JSON alert payloads from edge nodes (Store-and-Forward sync).
*   `GET /api/v1/stream/{camera_id}`: Provides MJPEG video streaming for the live CCTV dashboard wall.
*   `POST /api/v1/investigate`: GenAI RAG copilot endpoint accepting natural language queries to search the `pgvector` DB.
*   `POST /api/v1/alerts/{alert_id}/feedback`: Hard-negative feedback loop endpoint.
*   `GET /api/v1/cameras/{camera_id}/zones`: Syncs operator-drawn geofence polygons back to the edge node.

## 👥 Meet the Team (Prashant_Sahyog)

*   **Nitish (AI/ML Lead):** Architect of the 3-stage detection-gated pipeline, YOLOv8/RT-DETR schemas, DeepSORT tracking, and the RAG architecture.
*   **Devansh (Optimization):** Compiles PyTorch models into ONNX/OpenVINO to minimize edge latency; manages memory constraints for the GenAI module.
*   **Prashant (Frontend Lead):** Developer of the Streamlit dashboard, Folium mapping, and ranked-alert queue to minimize cognitive overload.
*   **Bhumika (Backend Lead):** Designer of the end-to-end FastAPI infrastructure, async routing, and legacy system integrations.
*   **Reddy (Quality Engineering):** Architect of PostgreSQL/pgvector schemas, SQLite edge queues, and validates GPU/CPU frame drop rates.
*   **Garima (DevOps):** Manages RTSP ingestion, evaluation datasets (LLVIP, VisDrone, UA-DETRAC), and Docker containerization.

---
*Developed with purpose for a safer, smarter border.*
