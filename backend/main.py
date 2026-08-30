import os
import sys
import time
import asyncio
import base64
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import init_db, get_db, AlertEvent, CameraZone
import genai_copilot

# ---------------------------------------------------------------------------
# Phase 5: Engine Registry — maps camera_id -> VisionEngine instance
# ---------------------------------------------------------------------------
# Import is deferred so the backend can start even if edge deps are missing.
ACTIVE_ENGINES: Dict[str, Any] = {}   # camera_id -> VisionEngine
CAMERA_SOURCES: Dict[str, str] = {}   # camera_id -> video file path

# In-memory zone store fallback (used when PostgreSQL is unavailable).
ZONE_STORE: Dict[str, Dict[str, Any]] = {}

MAX_CAMERAS = 6          # cap for demo performance
SAMPLE_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "..", "sample-videos")


# Pydantic Schemas for API validation
class WorldCoords(BaseModel):
    x: float = 0.0
    y: float = 0.0


class AlertPayload(BaseModel):
    alert_id: str
    timestamp: str
    camera_id: str = "CAM-BOP-01"
    track_id: int
    object_class: str
    class_id: int
    confidence: float
    world_coords: WorldCoords
    velocity_mps: float
    heading_deg: float
    zone: str
    primary_rule: str
    active_rules: List[str] = Field(default_factory=list)
    priority_score: float
    priority_level: str
    license_plate: Optional[str] = None
    thumbnail_b64: Optional[str] = None
    synced_status: Optional[bool] = True
    is_threat: Optional[bool] = None


class FeedbackPayload(BaseModel):
    action: str = Field(..., description="'CONFIRMED_BREACH' or 'FALSE_ALARM'")
    operator_id: str = "OPERATOR-01"
    notes: Optional[str] = ""


class InvestigateQuery(BaseModel):
    query: str


class ZonePayload(BaseModel):
    """Phase 5: Payload for saving operator-drawn polygon zone."""
    polygon: List[List[float]] = Field(
        ...,
        description="List of [x, y] pixel coordinate pairs defining the restricted zone polygon."
    )
    zone_label: str = Field(default="RESTRICTED", description="Display label for the zone.")


class AlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    camera_id: str
    track_id: int
    object_class: str
    class_id: int
    confidence: float
    world_coords: WorldCoords
    velocity_mps: float
    heading_deg: float
    zone: str
    primary_rule: str
    active_rules: List[str]
    priority_score: float
    priority_level: str
    license_plate: Optional[str]
    thumbnail_b64: Optional[str]
    feedback_status: Optional[str] = "PENDING_REVIEW"
    operator_notes: Optional[str] = ""


# FastAPI App Initialization
app = FastAPI(
    title="IBVAP - Central Command Backend",
    description="Intelligent Border Video Analytics Platform API (SIH26187)",
    version="1.0.0",
)

# Enable CORS for Streamlit frontend and local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage: Thread-safe in-memory store with file persistence
ALERTS_DB: Dict[str, Dict[str, Any]] = {}
HARD_NEGATIVE_DIR = os.path.join(os.path.dirname(__file__), "data", "hard_negatives")
os.makedirs(HARD_NEGATIVE_DIR, exist_ok=True)
START_TIME = time.time()

@app.on_event("startup")
def on_startup():
    init_db()
    _launch_vision_engines()


def _probe_landscape(video_path: str) -> bool:
    """
    Phase 5: Probes a video file using cv2 to check if it is landscape orientation.
    Returns True only if width > height (landscape clip suitable for surveillance wall).
    """
    cap = None
    try:
        cap = __import__('cv2').VideoCapture(video_path)
        if not cap.isOpened():
            return False
        w = cap.get(__import__('cv2').CAP_PROP_FRAME_WIDTH)
        h = cap.get(__import__('cv2').CAP_PROP_FRAME_HEIGHT)
        return w > h
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()


def _launch_vision_engines() -> None:
    """
    Phase 5: Scans sample-videos/, filters for landscape MP4s, launches up to
    MAX_CAMERAS VisionEngine daemon threads, and registers them in ACTIVE_ENGINES.
    """
    global ACTIVE_ENGINES, CAMERA_SOURCES

    try:
        # Ensure edge_node is importable when running from backend/
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from edge_node.vision_engine import VisionEngine
    except ImportError as e:
        print(f"[Backend] WARNING: Could not import VisionEngine: {e}. MJPEG streams unavailable.")
        return

    video_dir = os.path.abspath(SAMPLE_VIDEO_DIR)
    if not os.path.isdir(video_dir):
        print(f"[Backend] WARNING: sample-videos/ directory not found at {video_dir}")
        return

    candidates = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
    ])

    launched = 0
    for idx, filename in enumerate(candidates):
        if launched >= MAX_CAMERAS:
            break
        full_path = os.path.join(video_dir, filename)

        # Landscape orientation filter (width > height)
        if not _probe_landscape(full_path):
            print(f"[Backend] Skipping '{filename}' — not landscape orientation.")
            continue

        camera_id = f"CAM-BOP-{idx + 1:02d}"
        print(f"[Backend] Launching engine {camera_id} on '{filename}'...")

        try:
            engine = VisionEngine(
                source=full_path,
                camera_id=camera_id,
                backend_url="http://127.0.0.1:8000",
                headless=True,
                enable_sync=False,   # sync handled by edge_queue separately
            )
            engine.run_background()
            ACTIVE_ENGINES[camera_id] = engine
            CAMERA_SOURCES[camera_id] = filename
            launched += 1
        except Exception as e:
            print(f"[Backend] Failed to start engine for '{filename}': {e}")

    print(f"[Backend] Phase 5: {launched} camera engine(s) active.")


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint for edge node connectivity and heartbeat polling.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ibvap_central_backend",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "uptime_seconds": int(time.time() - START_TIME),
            "active_cameras": list(ACTIVE_ENGINES.keys()),
        },
    )


# ---------------------------------------------------------------------------
# Phase 5: Camera Registry Endpoint
# ---------------------------------------------------------------------------

@app.get("/api/v1/cameras")
async def list_cameras():
    """
    Phase 5: Returns the list of active camera IDs and their video sources.
    Used by the Streamlit frontend to populate the CCTV wall and config tab.
    """
    cameras = [
        {"camera_id": cid, "source": CAMERA_SOURCES.get(cid, "unknown")}
        for cid in ACTIVE_ENGINES
    ]
    return {"total": len(cameras), "cameras": cameras}


# ---------------------------------------------------------------------------
# Phase 5: MJPEG Streaming Endpoint
# ---------------------------------------------------------------------------

async def _mjpeg_generator(camera_id: str):
    """
    Async generator that yields MJPEG multipart frames from the engine's JPEG buffer.
    Polls the buffer at ~25 FPS ceiling using asyncio.sleep to stay non-blocking.
    """
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    engine = ACTIVE_ENGINES.get(camera_id)
    if engine is None:
        return

    while True:
        frame_bytes = engine.get_latest_frame_jpeg()
        if frame_bytes:
            yield boundary + frame_bytes + b"\r\n"
        await asyncio.sleep(0.04)  # ~25 FPS ceiling


@app.get("/api/v1/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """
    Phase 5: MJPEG streaming endpoint — returns a continuous multipart/x-mixed-replace
    response of AI-processed JPEG frames from the active VisionEngine buffer.
    Rendered as a zero-control live feed using a plain <img> tag in the frontend.
    """
    if camera_id not in ACTIVE_ENGINES:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' is not active. Check /api/v1/cameras for active IDs."
        )
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Phase 5: Zone Configuration Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/cameras/{camera_id}/zones", status_code=status.HTTP_200_OK)
async def save_camera_zone(camera_id: str, payload: ZonePayload, db: Session = Depends(get_db)):
    """
    Phase 5: Saves operator-drawn pixel polygon for a camera.
    Persists to PostgreSQL (CameraZone table) with in-memory fallback for edge nodes.
    Also immediately propagates the new zone to the active engine's RuleEngine.
    """
    polygon = payload.polygon
    if len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Zone polygon must have at least 3 points.")

    # Persist to DB if available
    if db:
        try:
            existing = db.query(CameraZone).filter(CameraZone.camera_id == camera_id).first()
            if existing:
                existing.polygon_json = polygon
                existing.zone_label = payload.zone_label
                existing.updated_at = datetime.now(timezone.utc)
            else:
                db.add(CameraZone(
                    camera_id=camera_id,
                    zone_label=payload.zone_label,
                    polygon_json=polygon,
                ))
            db.commit()
        except Exception as e:
            print(f"[Backend] Zone DB write failed: {e}")
            db.rollback()

    # Always update in-memory fallback store
    ZONE_STORE[camera_id] = {"polygon": polygon, "zone_label": payload.zone_label}

    # Immediately propagate to active engine's RuleEngine (no 30-s poll lag)
    if camera_id in ACTIVE_ENGINES:
        ACTIVE_ENGINES[camera_id].rule_engine.set_user_zone(polygon)

    return {
        "status": "zone_saved",
        "camera_id": camera_id,
        "zone_label": payload.zone_label,
        "point_count": len(polygon),
    }


@app.get("/api/v1/cameras/{camera_id}/zones")
async def get_camera_zone(camera_id: str, db: Session = Depends(get_db)):
    """
    Phase 5: Retrieves the current zone polygon for a camera.
    Queried by VisionEngine zone-polling worker every 30 seconds.
    """
    # Try DB first
    if db:
        try:
            row = db.query(CameraZone).filter(CameraZone.camera_id == camera_id).first()
            if row:
                return {
                    "camera_id": camera_id,
                    "zone_label": row.zone_label,
                    "polygon": row.polygon_json,
                }
        except Exception:
            pass

    # Fallback: in-memory store
    if camera_id in ZONE_STORE:
        return {"camera_id": camera_id, **ZONE_STORE[camera_id]}

    return {"camera_id": camera_id, "zone_label": None, "polygon": []}


@app.post("/api/v1/alerts", status_code=status.HTTP_201_CREATED)
async def ingest_alert(payload: AlertPayload, db: Session = Depends(get_db)):
    """
    Ingests compressed JSON alert payloads from Edge Nodes.
    """
    alert_dict = payload.model_dump()
    alert_id = alert_dict["alert_id"]

    # Determine is_threat if not explicitly provided
    if alert_dict["is_threat"] is None:
        alert_dict["is_threat"] = alert_dict["priority_level"] in ["CRITICAL", "HIGH"]

    # Default operator feedback state
    if alert_id not in ALERTS_DB:
        alert_dict["feedback_status"] = "PENDING_REVIEW"
        alert_dict["operator_notes"] = ""
        alert_dict["received_at"] = time.time()
    else:
        # Preserve existing feedback state if already reviewed
        alert_dict["feedback_status"] = ALERTS_DB[alert_id].get("feedback_status", "PENDING_REVIEW")
        alert_dict["operator_notes"] = ALERTS_DB[alert_id].get("operator_notes", "")
        alert_dict["received_at"] = ALERTS_DB[alert_id].get("received_at", time.time())

    ALERTS_DB[alert_id] = alert_dict
    
    # Phase 4 RAG Pipeline: Vectorize and store in PostgreSQL
    if db:
        try:
            existing = db.query(AlertEvent).filter(AlertEvent.alert_id == alert_id).first()
            if not existing:
                semantic_text = genai_copilot.translate_alert_to_text(alert_dict)
                embedding = genai_copilot.embed_text(semantic_text)
                
                db_event = AlertEvent(
                    alert_id=alert_id,
                    camera_id=alert_dict.get("camera_id", "CAM-BOP-01"),
                    raw_payload=alert_dict,
                    semantic_text=semantic_text,
                    embedding=embedding
                )
                db.add(db_event)
                db.commit()
        except Exception as e:
            print(f"Failed to index alert in pgvector database: {e}")
            db.rollback()

    return {"status": "ingested", "alert_id": alert_id}


@app.get("/api/v1/alerts")
async def get_alerts(
    priority: Optional[str] = Query(None, description="Filter by priority level: CRITICAL, HIGH, MEDIUM, LOW"),
    camera_id: Optional[str] = Query(None, description="Filter by Camera ID"),
    is_threat: Optional[bool] = Query(None, description="Filter by threat status"),
    feedback_status: Optional[str] = Query(None, description="Filter by feedback status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Retrieves ranked alert events sorted by Priority Score descending (Critical -> High -> Medium).
    """
    results = list(ALERTS_DB.values())

    if priority:
        results = [a for a in results if a.get("priority_level", "").upper() == priority.upper()]
    if camera_id:
        results = [a for a in results if a.get("camera_id", "").upper() == camera_id.upper()]
    if is_threat is not None:
        results = [a for a in results if a.get("is_threat") == is_threat]
    if feedback_status:
        results = [a for a in results if a.get("feedback_status", "").upper() == feedback_status.upper()]

    # Sort strictly by priority score descending, then newest first
    results.sort(key=lambda x: (x.get("priority_score", 0.0), x.get("received_at", 0.0)), reverse=True)

    paginated = results[offset : offset + limit]
    return {
        "total_count": len(results),
        "returned_count": len(paginated),
        "alerts": paginated,
    }


@app.post("/api/v1/alerts/{alert_id}/feedback")
async def record_feedback(alert_id: str, feedback: FeedbackPayload):
    """
    Hard-Negative Operator Feedback Loop:
    Logs operator audit decisions ('CONFIRMED_BREACH' or 'FALSE_ALARM').
    If flagged as FALSE_ALARM, persists the base64 thumbnail to hard-negatives directory for localized model retraining.
    """
    if alert_id not in ALERTS_DB:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")

    alert = ALERTS_DB[alert_id]
    action = feedback.action.upper()

    if action not in ["CONFIRMED_BREACH", "FALSE_ALARM"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid feedback action. Must be 'CONFIRMED_BREACH' or 'FALSE_ALARM'.",
        )

    alert["feedback_status"] = action
    alert["operator_notes"] = feedback.notes or ""
    alert["feedback_by"] = feedback.operator_id
    alert["feedback_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Hard-Negative Data Harvest: save crop to disk for active learning
    if action == "FALSE_ALARM" and alert.get("thumbnail_b64"):
        try:
            b64_data = alert["thumbnail_b64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            save_path = os.path.join(HARD_NEGATIVE_DIR, f"false_alarm_{alert_id}.jpg")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"[Backend] Error saving hard negative sample: {e}")

    return {
        "status": "feedback_recorded",
        "alert_id": alert_id,
        "action": action,
        "timestamp": alert["feedback_timestamp"],
    }


@app.get("/api/v1/stats")
async def get_stats():
    """
    Returns high-level tactical statistics for command dashboard.
    """
    alerts = list(ALERTS_DB.values())
    total_alerts = len(alerts)
    critical_count = sum(1 for a in alerts if a.get("priority_level") == "CRITICAL")
    high_count = sum(1 for a in alerts if a.get("priority_level") == "HIGH")
    medium_count = sum(1 for a in alerts if a.get("priority_level") == "MEDIUM")
    low_count = sum(1 for a in alerts if a.get("priority_level") == "LOW")

    false_alarms = sum(1 for a in alerts if a.get("feedback_status") == "FALSE_ALARM")
    confirmed_breaches = sum(1 for a in alerts if a.get("feedback_status") == "CONFIRMED_BREACH")
    pending_reviews = sum(1 for a in alerts if a.get("feedback_status") == "PENDING_REVIEW")

    active_cameras = list(set(a.get("camera_id", "CAM-BOP-01") for a in alerts))

    return {
        "total_alerts": total_alerts,
        "priority_breakdown": {
            "CRITICAL": critical_count,
            "HIGH": high_count,
            "MEDIUM": medium_count,
            "LOW": low_count,
        },
        "review_breakdown": {
            "CONFIRMED_BREACH": confirmed_breaches,
            "FALSE_ALARM": false_alarms,
            "PENDING_REVIEW": pending_reviews,
        },
        "active_cameras": active_cameras,
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.post("/api/v1/investigate")
async def investigate_alerts(payload: InvestigateQuery, db: Session = Depends(get_db)):
    """
    Phase 4 GenAI Copilot endpoint. 
    Embeds natural language query and searches pgvector database for historical analysis.
    """
    if not db:
        raise HTTPException(status_code=503, detail="PostgreSQL database not connected or pgvector unavailable.")
        
    try:
        # Step 1: Embed the query
        query_embedding = genai_copilot.embed_text(payload.query)
        
        # Step 2: Query pgvector using cosine distance, limit 5
        results = db.query(AlertEvent).order_by(
            AlertEvent.embedding.cosine_distance(query_embedding)
        ).limit(5).all()
        
        # Step 3: Generate the RAG LLM Prompt
        prompt = genai_copilot.generate_rag_prompt(payload.query, results)
        
        # Return structured data that a frontend or LLM client can use
        context_docs = [{"alert_id": r.alert_id, "text": r.semantic_text} for r in results]
        
        return {
            "query": payload.query,
            "generated_prompt": prompt,
            "context_documents": context_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
