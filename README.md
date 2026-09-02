# 🛡️ NIGHT'S WATCH

### Intelligent Border Video Analytics Platform (IBVAP)

> **AI-powered surveillance platform for intelligent video analytics, object detection, and tactical monitoring.**

---

## 🚀 Overview

**NIGHT'S WATCH (IBVAP Link)** transforms CCTV and video feeds into actionable security intelligence using AI-powered video analytics.

### Key Capabilities

- 🎥 Multi-camera surveillance
- 🤖 YOLO-based object detection
- 🧠 AI video analytics
- 🚨 Detection and alert monitoring
- 📊 Surveillance analytics dashboard
- 🎬 Annotated video processing
- 🔗 FastAPI REST APIs

---

## 🏗️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, FastAPI, YOLO, OpenCV |
| **Frontend** | Streamlit |
| **AI / Vision** | YOLO, OpenCV |
| **Tools** | Git, GitHub, Postman, VS Code |

---

## 📂 Project Structure

```text
Trinetra/
├── backend/              # FastAPI + AI Detection
│   ├── app/
│   └── media/processed/
│
├── frontend/             # Streamlit Dashboard
│   └── assets/videos/
│
├── .env.example
├── .gitignore
└── requirements.txt
```

---

# ⚡ Quick Start

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/BriefJarl/Night-s-watch.git
cd Night-s-watch
```

---

# 🔧 Run the Backend

Open **PowerShell Terminal 1**:

```powershell
cd C:\Trinetra
```

### Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Configure environment variables

```powershell
Copy-Item .env.example .env
```

> ⚠️ Never commit `.env`, API keys, passwords, or tokens.

### Start FastAPI

```powershell
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Backend Running 🚀

```text
API:     http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
ReDoc:   http://127.0.0.1:8000/redoc
```

---

# 🖥️ Run the Frontend

Open **PowerShell Terminal 2**:

```powershell
cd C:\Trinetra
.\.venv\Scripts\Activate.ps1
cd frontend
```

Start Streamlit:

```powershell
streamlit run app.py
```

### Frontend Running 🚀

```text
http://localhost:8501
```

---

# ▶️ Run the Complete System

You need **two terminals running simultaneously**:

### Terminal 1 — Backend

```powershell
cd C:\Trinetra
.\.venv\Scripts\Activate.ps1
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Terminal 2 — Frontend

```powershell
cd C:\Trinetra
.\.venv\Scripts\Activate.ps1
cd frontend
streamlit run app.py
```

### System Flow

```text
🎥 Video / CCTV
       ↓
🤖 YOLO AI Detection
       ↓
⚙️ FastAPI Backend
       ↓
🎬 Processed Video + Detection Results
       ↓
🖥️ Streamlit Surveillance Dashboard
```

---

# 🧠 Video Detection API

Example request:

```powershell
curl.exe -X POST `
"http://127.0.0.1:8000/api/v1/ai/detect/video/annotated?confidence_threshold=0.5" `
-F "file=@C:\Trinetra\frontend\assets\videos\camera2.mp4"
```

Processed videos are generated in:

```text
backend/media/processed/
```

---

# 🧪 Health Check

```powershell
curl.exe http://127.0.0.1:8000/api/v1/health
```

---

# 🔐 Security

The repository excludes sensitive files such as:

```text
.env
.venv/
API keys
Passwords
Tokens
Temporary files
Logs
```

Use `.env.example` for configuration templates.

---

## 🔮 Future Improvements

- Real-time CCTV / RTSP streaming
- Advanced threat detection
- Authentication & role-based access
- Real-time notifications
- Cloud deployment
- Docker support

---

<div align="center">

# 🛡️ NIGHT'S WATCH

### Intelligent Surveillance. AI Detection. Tactical Intelligence.

**Built with FastAPI • YOLO • OpenCV • Streamlit**

</div>
