<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/128/2092/2092661.png" width="80" alt="IBVAP Logo"/>
</p>

<h1 align="center">Night's Watch — IBVAP</h1>
<h3 align="center">Intelligent Border Video Analytics Platform</h3>
<p align="center">
  <strong>SIH26187 · Ministry of Home Affairs / SSB · Smart Automation</strong><br>
  Edge-compatible AI vision engine for real-time border surveillance on standard CCTV infrastructure.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/FastAPI-0.103+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-00BFFF?logo=yolo&logoColor=white" alt="YOLOv8"/>
</p>

---

## Overview

**Night's Watch (IBVAP)** is a software-defined intelligence layer for border surveillance that runs on existing standard-definition CCTV infrastructure. It uses a **3-stage detection-gated pipeline** (MOG2 → YOLOv8n → DeepSORT) to minimize compute on edge hardware while maximizing threat detection accuracy.

### Key Features

| Feature | Description |
|---------|-------------|
| **3-Stage Gated Pipeline** | MOG2 motion gate → YOLOv8n classification → DeepSORT persistent tracking |
| **4-Layer False Alarm Suppression** | Static masks, track confirmation (≥8 frames + ≥2m), homography geometric gate, hard-negative feedback loop |
| **Ground Homography** | Pixel → real-world GPS coordinate mapping via calibrated 3×3 homography matrix |
| **MJPEG Streaming Wall** | Zero-latency live feeds with AI-annotated bounding boxes in browser |
| **ANPR Engine** | Perspective warp + EasyOCR + Indian plate regex + multi-frame character voting |
| **Ranked Alert Dashboard** | Priority-scored threat queue with glassmorphism UI (Streamlit) |
| **Tactical Map** | Folium-based geospatial plotting of incursions |
| **AI Copilot (RAG)** | Natural language querying of historical alerts via sentence-transformers embeddings |
| **Store-and-Forward** | SQLite edge queue with async sync — zero data loss on network failure |
| **Zone Configuration** | Per-camera surveillance modes: Alert, Civilian, No Civilian, No Vehicle, Emergency |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      CENTRAL COMMAND (Tier 4)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   FastAPI     │  │  PostgreSQL  │  │   Streamlit        │   │
│  │   Backend     │  │  + pgvector  │  │   Dashboard        │   │
│  │  (main.py)    │  │ (database.py)│  │   (app.py)         │   │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘   │
│         │ REST API + MJPEG Streaming                           │
├─────────┼──────────────────────────────────────────────────────┤
│         │              EDGE NODE (Tier 1-3)                    │
│  ┌──────┴───────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ VisionEngine │  │ FalseAlarm   │  │   EdgeQueue        │   │
│  │ (3-stage AI) │→ │ Filter       │→ │   (SQLite S&F)     │   │
│  │ MOG2→YOLO→   │  │ (Homography) │  │                    │   │
│  │ DeepSORT     │  │              │  │ ┌────────────────┐ │   │
│  └──────────────┘  └──────────────┘  │ │ ANPR Engine    │ │   │
│                                       │ │ (EasyOCR)      │ │   │
│                                       │ └────────────────┘ │   │
│                                       └────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Night-s-watch/
├── backend/
│   ├── main.py              # FastAPI central command server
│   ├── database.py          # PostgreSQL/pgvector schema (with SQLite fallback)
│   └── genai_copilot.py     # RAG embedding & prompt generation
├── edge_node/
│   ├── vision_engine.py     # 3-stage detection-gated pipeline
│   ├── false_alarm_filter.py # Homography + 4-layer suppression stack
│   ├── rule_engine.py       # Zone geofencing + behavioral rules + scoring
│   ├── anpr_engine.py       # License plate recognition with multi-frame voting
│   └── edge_queue.py        # SQLite store-and-forward sync worker
├── frontend/
│   └── app.py               # Streamlit tactical command dashboard
├── sample-videos/           # Demo surveillance clips (Git LFS)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Quick Start (Clone & Run)

### Prerequisites

- **Python 3.10+** (tested on 3.12 and 3.14)
- **Git LFS** (for pulling sample videos)
- ~2 GB disk space (videos + model weights)
- **Optional:** PostgreSQL with pgvector (falls back to in-memory storage)
- **Optional:** [Ollama](https://ollama.ai) with `llama3` for AI Reports

### 1. Clone the Repository

```bash
# Install Git LFS if you haven't
git lfs install

# Clone with LFS — this pulls the sample videos automatically
git clone https://github.com/BriefJarl/Night-s-watch.git
cd Night-s-watch
```

> **Note:** The first clone will download ~830 MB of sample surveillance videos via Git LFS. If you want to skip them initially, use `GIT_LFS_SKIP_SMUDGE=1 git clone ...` and run `git lfs pull` later.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Users (Optional):** If you have an NVIDIA GPU, install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Start the Backend

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for the startup to complete. You'll see:
```
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
Embedding model loaded successfully.
[Backend] Launching engine CAM-BOP-01 on '...'...
[Backend] Phase 5: 6 camera engine(s) active.
INFO:     Application startup complete.
```

> **First Run:** YOLOv8 will automatically download `yolov8n.pt` (~6 MB) on first launch. The sentence-transformers model (~90 MB) is also auto-downloaded.

### 4. Start the Frontend (New Terminal)

```bash
# From the project root
python -m streamlit run frontend/app.py
```

### 5. Open the Dashboard

Navigate to **http://localhost:8501** in your browser.

1. You'll see the **System Initialization** wizard
2. Assign a surveillance zone to each camera (defaults to "Civilian zone")
3. Click **Save Configuration & Start System**
4. The dashboard will start showing live MJPEG feeds, alerts, and threat detections

---

## Surveillance Modes

| Mode | Behavior |
|------|----------|
| **Civilian zone** | Lenient — ignores all person/vehicle detections |
| **Alert zone** | Standard — reports all confirmed detections as medium-priority |
| **No Civilian zone** | Restricted — flags any **person** as HIGH ALERT intrusion |
| **No vehicle zone** | Restricted — flags any **vehicle** as HIGH ALERT intrusion |
| **Emergency/sensitive zone** | Maximum security — flags **everything** as HIGH ALERT |

Change modes anytime in the **⚙️ Camera Config** tab — takes effect immediately.

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📹 **Live Feeds** | MJPEG surveillance wall (1×1 to 10×10 grid) with AI-annotated bounding boxes |
| 🗺️ **Tactical Map** | Folium dark-mode map with threat markers plotted from homography coordinates |
| 🛡️ **Alert Queue** | Priority-ranked alert cards with thumbnails, telemetry, and Dispatch/False Alarm buttons |
| ⚙️ **Camera Config** | Per-camera surveillance mode selector |
| 🤖 **AI Reports** | Natural language querying of historical alerts via RAG (requires Ollama for full LLM responses) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/cameras` | List active camera streams |
| `GET` | `/api/v1/stream/{camera_id}` | MJPEG live stream |
| `GET` | `/api/v1/stats` | Dashboard KPI statistics |
| `GET` | `/api/v1/alerts` | Filtered alert list (priority, camera, status) |
| `POST` | `/api/v1/alerts` | Ingest alert from edge node |
| `POST` | `/api/v1/alerts/{id}/feedback` | Operator feedback (Confirm/False Alarm) |
| `GET` | `/api/v1/cameras/{id}/zones` | Get camera zone config |
| `POST` | `/api/v1/cameras/{id}/zones` | Set camera zone config |
| `GET` | `/api/v1/config/status` | Check if system is configured |
| `POST` | `/api/v1/config/init` | Initial system configuration |
| `POST` | `/api/v1/investigate` | AI Copilot RAG query |

---

## Optional: AI Reports with Ollama

To enable full AI-generated reports:

```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3
ollama serve
```

The AI Reports tab will automatically detect Ollama at `localhost:11434` and generate incident summaries.

Without Ollama, the tab still works — it shows the RAG context and prompt that would be sent to an LLM.

---

## Optional: PostgreSQL with pgvector

For persistent alert storage and semantic search:

```bash
# Using Docker
docker run -d --name ibvap-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ibvap \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Set the connection string
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/ibvap"
```

Without PostgreSQL, the system uses in-memory storage (resets on restart).

---

## Adding Your Own Videos

Place any **landscape-oriented** MP4 files in the `sample-videos/` directory. The backend automatically:
1. Scans for `.mp4`, `.avi`, `.mkv`, `.mov` files
2. Filters for landscape orientation (width > height)
3. Launches up to 6 camera engines
4. Assigns camera IDs: `CAM-BOP-01`, `CAM-BOP-02`, etc.

---

## Team

| Member | Role |
|--------|------|
| **Nitish** | AI/ML Lead — 3-stage pipeline, YOLOv8/DeepSORT, RAG |
| **Devansh** | Optimization — ONNX/OpenVINO, edge latency, GenAI module |
| **Prashant** | Frontend Lead — Streamlit dashboard, Folium mapping |
| **Bhumika** | Backend Lead — FastAPI infrastructure, async routing |
| **Arnav** | Quality Engineering — PostgreSQL/pgvector, SQLite edge queues |
| **Garima** | DevOps — RTSP ingestion, Docker containerization |

---

## 📄 License & Attribution

This project is developed as a prototype for **Smart India Hackathon (SIH26187)** under the **Ministry of Home Affairs / Sashastra Seema Bal (SSB)**.

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details and permissions.

---

<p align="center">
  <strong>Built for SIH26187 · Sashastra Seema Bal (SSB) · Ministry of Home Affairs</strong><br>
  <em>"The Night gathers, and now my watch begins."</em>
</p>
