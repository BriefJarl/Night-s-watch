# 🛡️ NIGHT'S WATCH
## Intelligent Border Video Analytics Platform (IBVAP)

> **AI-powered surveillance and border intelligence platform for real-time video analytics, threat detection, and tactical monitoring.**

---

## 🚀 Overview

**NIGHT'S WATCH (IBVAP Link)** is an intelligent surveillance platform designed to transform CCTV and video feeds into actionable security intelligence.

The platform combines:

- 🎥 Multi-camera surveillance
- 🤖 YOLO-based object detection
- 🧠 AI-powered video analytics
- 🚨 Threat and alert monitoring
- 📊 Detection analytics
- 🔗 REST API integration
- 🖥️ Interactive surveillance dashboard

The system processes uploaded video footage, performs AI detection, generates annotated videos, and displays surveillance insights through a centralized frontend dashboard.

---

# 🏗️ Architecture

```text
                 ┌──────────────────┐
                 │   CCTV / Videos  │
                 └────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   FastAPI Backend   │
              │                     │
              │  • Video Upload     │
              │  • YOLO Detection   │
              │  • AI Processing    │
              │  • REST APIs        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Processed Videos    │
              │ Detection Results   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Streamlit Frontend  │
              │                     │
              │ • Live Feeds        │
              │ • AI Detection      │
              │ • Alerts            │
              │ • Analytics         │
              └─────────────────────┘
