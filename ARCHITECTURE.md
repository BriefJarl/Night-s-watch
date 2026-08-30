# IBVAP — Technical Architecture Deep-Dive

> **Project:** Night's Watch — Intelligent Border Video Analytics Platform  
> **SIH Problem Statement:** 26187  
> **Organisation:** Ministry of Home Affairs / Sashastra Seema Bal (SSB)  
> **Version:** 1.0.0 (Phase 5 complete)  
> **Date:** August 2026

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy)
2. [Development Phases](#2-development-phases)
3. [Edge Node Architecture](#3-edge-node-architecture)
   - [Stage 1: Lightweight Monitor](#31-stage-1-lightweight-monitor-mog2)
   - [Stage 2: Gated Verification](#32-stage-2-gated-verification-yolov8n)
   - [Stage 3: Persistent Tracking](#33-stage-3-persistent-tracking-deepsort)
4. [False Alarm Suppression Stack](#4-false-alarm-suppression-stack)
   - [Homography Calibration](#41-ground-homography-calibration)
   - [Size-vs-Depth Gate](#42-size-vs-depth-homography-gate)
   - [Track Confirmation Gate](#43-track-confirmation-gate)
   - [Tamper Detection](#44-camera-tamper-detection)
   - [Hard-Negative Loop](#45-hard-negative-operator-feedback-loop)
5. [Rule Engine & Priority Scoring](#5-rule-engine--priority-scoring)
6. [ANPR Engine](#6-anpr-engine)
7. [Semantic Compression & Store-and-Forward](#7-semantic-compression--store-and-forward)
8. [Central Backend (FastAPI)](#8-central-backend-fastapi)
9. [GenAI RAG Copilot](#9-genai-rag-copilot)
10. [Streamlit Dashboard](#10-streamlit-dashboard)
11. [Database Schemas](#11-database-schemas)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Performance Budgets](#13-performance-budgets)
14. [Deployment Notes](#14-deployment-notes)

---

## 1. System Philosophy

### Core Design Principles

**1. Compute Triage (Detection-Gated Pipeline)**  
Never run neural networks on every frame. The system implements a 3-stage cascade where each stage acts as a gating function. Only frames that pass all prior gates reach the most expensive operations.

**2. Semantic Compression over Raw Streaming**  
A border deployment may have satellite links with 10 kbps uplink. Raw H.264 video requires 500 kbps+. IBVAP converts alerts into JSON payloads of 2–8 KB each, making the system link-agnostic.

**3. Store-and-Forward Resilience**  
All alert data is persisted to local SQLite before any attempt to sync with the cloud backend. This guarantees zero data loss during link outages.

**4. Software-Defined Fixes**  
Hardware limitations (no GPU, no high-resolution camera, poor lighting) are compensated with algorithmic solutions: homography for GPS mapping, MOG2 for motion pre-screening, EasyOCR multi-frame voting for motion blur compensation.

**5. False Alarm Budget**  
Every confirmed alert must pass a 4-layer validation stack before being displayed to an operator. The goal is to keep false alarm rate below 2% to prevent alert fatigue.

---

## 2. Development Phases

| Phase | Description | Modules |
|-------|-------------|---------|
| **Phase 1** | 3-Stage Detection-Gated Pipeline | `vision_engine.py` (Stages 1-3) |
| **Phase 2** | False Alarm Suppression Stack + Homography + Rule Engine | `false_alarm_filter.py`, `rule_engine.py` |
| **Phase 3** | ANPR Engine + Semantic Compression + Store-and-Forward | `anpr_engine.py`, `edge_queue.py` |
| **Phase 4** | GenAI RAG Copilot + PostgreSQL pgvector | `genai_copilot.py`, `database.py` |
| **Phase 5** | MJPEG Streaming + Zone Configurator + Multi-Camera Dashboard | `main.py` (streaming endpoints), `vision_engine.py` (MJPEG buffer), `rule_engine.py` (user zones), `frontend/app.py` |

---

## 3. Edge Node Architecture

### Module: `edge_node/vision_engine.py`

The `VisionEngine` class is the central orchestrator for all edge-side processing.

```
VisionEngine.__init__()
├── cv2.VideoCapture(source)            # RTSP / file / webcam
├── cv2.createBackgroundSubtractorMOG2  # Stage 1
├── YOLO("yolov8n.pt")                  # Stage 2
├── DeepSort(embedder="mobilenet")      # Stage 3
├── FalseAlarmFilter()                  # Phase 2
├── RuleEngine()                        # Phase 2
├── ANPREngine(use_gpu=False)           # Phase 3
└── EdgeQueue(db_path, backend_url)     # Phase 3 + 5
```

#### 3.1 Stage 1: Lightweight Monitor (MOG2)

```python
self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Frame history for background model
    varThreshold=50,    # Sensitivity to foreground/background variance
    detectShadows=False # Disable shadow detection to save CPU
)
```

**Processing rate:** Max 2 FPS (`fps_limit=2.0`, `process_interval=0.5s`)

**Motion gate condition:**
```python
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
motion_detected = any(cv2.contourArea(c) > self.motion_threshold_area for c in contours)
# self.motion_threshold_area = 500 pixels^2
```

Only if `motion_detected = True` does Stage 2 run.

#### 3.2 Stage 2: Gated Verification (YOLOv8n)

```python
results = self.model(frame, classes=[0, 2, 3, 5, 7], verbose=False)
# COCO class filter: person(0), car(2), motorcycle(3), bus(5), truck(7)
```

Detections are formatted for DeepSORT:
```python
detections_for_tracker.append(([x1, y1, bw, bh], conf, class_id))
# Format: ([left, top, width, height], confidence, class_id)
```

#### 3.3 Stage 3: Persistent Tracking (DeepSORT)

```python
self.tracker = DeepSort(
    max_age=30,          # Frames to keep a track alive without a detection
    n_init=2,            # Frames before a track is confirmed
    nms_max_overlap=1.0, # Allow overlapping bounding boxes
    embedder="mobilenet" # Appearance embedding for re-identification
)
tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
```

After Stage 3, only **confirmed tracks** (`track.is_confirmed()`) proceed to the False Alarm Filter.

#### Phase 5: MJPEG Frame Buffer

```python
# Thread-safe JPEG buffer — written by VisionEngine, read by FastAPI endpoint
self._latest_jpeg: Optional[bytes] = None
self._frame_lock = threading.Lock()

# FastAPI polls this at ~25 FPS
def get_latest_frame_jpeg(self) -> Optional[bytes]:
    with self._frame_lock:
        return self._latest_jpeg
```

**Resolution pipeline:** All frames are immediately downscaled to **1280×720** after `cap.read()`:
```python
frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
```
This cuts pixel count ~9× for 4K sources before any downstream AI operation.

#### Phase 5: Zone Polling

Every 30 seconds, the engine polls the backend REST API for operator-drawn zones:
```python
url = f"{self.backend_url}/api/v1/cameras/{self.camera_id}/zones"
resp = requests.get(url, timeout=2.0)
polygon = resp.json().get("polygon", [])
self.rule_engine.set_user_zone(polygon)
```

---

## 4. False Alarm Suppression Stack

### Module: `edge_node/false_alarm_filter.py`

#### 4.1 Ground Homography Calibration

A 3×3 homography matrix **H** is computed from 4 pixel-to-world point correspondences:

```python
self.src_pts = np.array([
    [400.0,  1080.0],  # Bottom Left (image)
    [1520.0, 1080.0],  # Bottom Right (image)
    [800.0,  500.0],   # Distant Left (image)
    [1120.0, 500.0],   # Distant Right (image)
], dtype=np.float32)

self.dst_pts = np.array([
    [-5.0, 2.0],   # Bottom Left (2m depth in world)
    [5.0,  2.0],   # Bottom Right
    [-5.0, 15.0],  # Distant Left (15m depth)
    [5.0,  15.0],  # Distant Right
], dtype=np.float32)

self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
self.H_inv = np.linalg.inv(self.H)
```

**Pixel → World transform:**
```python
def pixel_to_world(self, u, v):
    pt_img = np.array([[[u, v]]], dtype=np.float32)
    pt_world = cv2.perspectiveTransform(pt_img, self.H)
    return float(pt_world[0][0][0]), float(pt_world[0][0][1])
```

**World → Pixel transform (for overlay rendering):**
```python
def world_to_pixel(self, x_w, y_w):
    pt_world = np.array([[[x_w, y_w]]], dtype=np.float32)
    pt_img = cv2.perspectiveTransform(pt_world, self.H_inv)
    return int(round(pt_img[0][0][0])), int(round(pt_img[0][0][1]))
```

#### 4.2 Size-vs-Depth Homography Gate

Rejects forced-perspective detections using the **pinhole camera model**:

```
H_estimated = (h_px * D) / f_eff
```

Where `h_px` = bounding box height in pixels, `D` = projected ground distance in metres, `f_eff` = focal length in pixels (default: 1000).

**Physical dimension bounds per class:**

| Object Class | Min Height (m) | Max Height (m) |
|--------------|---------------|----------------|
| Person (0) | 0.40 | 2.60 |
| Vehicle (2,3,5,7) | 0.60 | 5.20 |

**Extreme outlier filters:**
- `distance > 20m` AND `bbox_height > 450px` → bug on lens
- `distance < 3m` AND `bbox_height < 15px` → micro noise

#### 4.3 Track Confirmation Gate

```python
# Gate 1: Persistence (>= 8 frames)
if persistence_frames < self.min_persistence_frames:  # 8
    return {"is_valid": False, "filter_reason": "PERSISTENCE_GATE"}

# Gate 2: Spatial Displacement (>= 2.0 metres)
displacement_m = sqrt((x_w - init_x)**2 + (y_w - init_y)**2)
if displacement_m < self.min_displacement_meters:  # 2.0
    return {"is_valid": False, "filter_reason": "DISPLACEMENT_GATE"}
```

#### 4.4 Camera Tamper Detection

```python
@staticmethod
def check_tampering(gray_frame, blur_threshold=45.0):
    # 1. Occlusion: std deviation collapse
    mean_val, std_val = cv2.meanStdDev(gray_frame)
    if std_val[0][0] < 12.0:
        return True, "OCCLUSION_UNIFORM"

    # 2. Spray paint: histogram peak > 85%
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    if np.max(hist / hist.sum()) > 0.85:
        return True, "LENS_SPRAY_OCCLUSION"

    # 3. Defocus: Laplacian variance < threshold
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return True, "DEFOCUS_BLUR"

    return False, "OK"
```

#### 4.5 Hard-Negative Operator Feedback Loop

When an operator marks an alert as `FALSE_ALARM` via the dashboard:

1. The FastAPI `/api/v1/alerts/{alert_id}/feedback` endpoint receives the action.
2. The base64-encoded thumbnail crop is decoded and saved to `backend/data/hard_negatives/false_alarm_{alert_id}.jpg`.
3. These crops form a localised retraining dataset that can be used for fine-tuning or ONNX calibration on the specific camera's environment.

---

## 5. Rule Engine & Priority Scoring

### Module: `edge_node/rule_engine.py`

#### Zone Geofencing

Two concentric polygonal zones in world coordinates:

```python
# Red Zone: Zero Line / High Security strip
self.red_zone = np.array([[-12, 0], [12, 0], [12, 8], [-12, 8]], dtype=np.float32)

# Amber Zone: Buffer Warning zone
self.amber_zone = np.array([[-15, 8], [15, 8], [15, 16], [-15, 16]], dtype=np.float32)

# Zone assignment via cv2.pointPolygonTest
def get_zone_for_point(self, x_w, y_w):
    if cv2.pointPolygonTest(self.red_zone, (x_w, y_w), False) >= 0:
        return "RED_ZONE"
    if cv2.pointPolygonTest(self.amber_zone, (x_w, y_w), False) >= 0:
        return "AMBER_ZONE"
    return "GREEN_ZONE"
```

#### Behavioural Rules

| Rule | Trigger Condition |
|------|-------------------|
| `TRIPWIRE_INBOUND` | Track trajectory crosses virtual fence with inward direction |
| `CRAWLING_INTRUSION` | Person class, aspect ratio (H/W) < 0.85, velocity 0.05–1.2 m/s |
| `LOITERING` | Dwell time in RED/AMBER zone ≥ 10 seconds |
| `ZONE_INTRUSION` | Bounding box centroid inside RED_ZONE world polygon |
| `SPEEDING` | Person > 4.5 m/s OR Vehicle > 12.0 m/s |
| `RESTRICTED_ZONE_INTRUSION` | Bounding box bottom-center pixel inside operator-drawn polygon |
| `TAMPER_OCCLUSION/DEFOCUS` | Camera tamper check returns True |

#### Priority Scoring Function

```python
score = W_rule * C_zone * C_class * C_time * confidence
score = clamp(score, 0.0, 100.0)
```

**Multipliers:**
- `C_zone`: RED_ZONE=1.25, AMBER_ZONE=1.0, GREEN_ZONE=0.7
- `C_class`: Person=1.10, Bus/Truck=1.08, Car/Motorcycle=1.05
- `C_time`: Night (20:00–05:00)=1.25, Day=1.0

**Priority Levels:**
- CRITICAL: score ≥ 75.0
- HIGH: score ≥ 50.0
- MEDIUM: score ≥ 25.0
- LOW: score < 25.0

---

## 6. ANPR Engine

### Module: `edge_node/anpr_engine.py`

#### Processing Pipeline

```
Vehicle Crop
    │
    ▼
[Heuristic ROI] → bottom 40% center of vehicle bounding box
    │
    ▼
[Perspective Warp] → cv2.getPerspectiveTransform to 240×80 px
    │
    ▼
[Preprocessing]
    ├── Grayscale conversion
    ├── CLAHE (clipLimit=2.0, tileGridSize=8×8)
    ├── Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
    └── Otsu thresholding
    │
    ▼
[EasyOCR] → readtext(preprocessed)
    │
    ▼
[Indian Plate Validation]
    ├── REGEX_INDIAN_STANDARD: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
    └── REGEX_BHARAT_SERIES: ^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$
    │
    ▼
[Multi-Frame Buffer] → plate_history[track_id].append(candidate)
    │
    ▼
[Character Majority Vote] → per-position Counter across last 15 frames
    │
    ▼
Final Plate String + Confidence
```

#### Character Ambiguity Correction

```python
NUM_TO_ALPHA = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
ALPHA_TO_NUM = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "D": "0", "Q": "0"}
```

Applied positionally based on Indian plate syntax (state code must be alphabetic, district code numeric, etc.).

---

## 7. Semantic Compression & Store-and-Forward

### Module: `edge_node/edge_queue.py`

#### SQLite Schema

```sql
CREATE TABLE alerts (
    alert_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    object_class    TEXT NOT NULL,
    class_id        INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    world_coords_x  REAL NOT NULL,
    world_coords_y  REAL NOT NULL,
    velocity_mps    REAL NOT NULL,
    heading_deg     REAL NOT NULL,
    zone            TEXT NOT NULL,
    primary_rule    TEXT NOT NULL,
    active_rules_json TEXT NOT NULL,
    priority_score  REAL NOT NULL,
    priority_level  TEXT NOT NULL,
    license_plate   TEXT,
    thumbnail_b64   TEXT,
    synced_status   INTEGER DEFAULT 0,  -- 0=pending, 1=synced
    created_at      REAL NOT NULL
);

CREATE INDEX idx_alerts_sync_priority
ON alerts (synced_status, priority_score DESC, created_at ASC);
```

#### Alert Payload (JSON)

```json
{
  "alert_id": "ALT-1724880300000-42",
  "timestamp": "2026-08-28T23:45:00.000000Z",
  "camera_id": "CAM-BOP-01",
  "track_id": 42,
  "object_class": "person",
  "class_id": 0,
  "confidence": 0.87,
  "world_coords": {"x": 2.45, "y": 7.12},
  "velocity_mps": 2.5,
  "heading_deg": 180.0,
  "zone": "RED_ZONE",
  "primary_rule": "ZONE_INTRUSION",
  "active_rules": ["ZONE_INTRUSION", "LOITERING"],
  "priority_score": 81.3,
  "priority_level": "CRITICAL",
  "is_threat": true,
  "license_plate": null,
  "thumbnail_b64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

#### Sync Worker State Machine

```
[STOPPED] ──start_sync_worker()──▶ [RUNNING]
                                         │
                                    every 2s
                                         │
                                         ▼
                              GET /api/v1/health
                             ┌───────────────────┐
                         200 │                   │ Error/Non-200
                             ▼                   ▼
                      [ONLINE]           [OFFLINE - buffer]
                             │
                    GET pending alerts
                    (priority DESC)
                             │
                    POST /api/v1/alerts
                    (batch of 25)
                             │
                    mark_as_synced()
```

---

## 8. Central Backend (FastAPI)

### Module: `backend/main.py`

#### Startup Sequence

```python
@app.on_event("startup")
def on_startup():
    init_db()          # Create PostgreSQL tables + pgvector extension
    _launch_vision_engines()  # Scan sample-videos/, launch VisionEngine daemons
```

#### Camera Engine Registry

```python
ACTIVE_ENGINES: Dict[str, VisionEngine] = {}  # camera_id -> VisionEngine
CAMERA_SOURCES: Dict[str, str] = {}           # camera_id -> filename
MAX_CAMERAS = 6                                # Performance cap for demo
```

Videos are filtered for **landscape orientation** (`width > height`) before launching:
```python
def _probe_landscape(video_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return w > h
```

#### MJPEG Streaming

```python
async def _mjpeg_generator(camera_id: str):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    engine = ACTIVE_ENGINES.get(camera_id)
    while True:
        frame_bytes = engine.get_latest_frame_jpeg()
        if frame_bytes:
            yield boundary + frame_bytes + b"\r\n"
        await asyncio.sleep(0.04)  # ~25 FPS ceiling

@app.get("/api/v1/stream/{camera_id}")
async def stream_camera(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

#### Alert Ingestion with RAG Indexing

```python
@app.post("/api/v1/alerts", status_code=201)
async def ingest_alert(payload: AlertPayload, db: Session = Depends(get_db)):
    # 1. Store in ALERTS_DB (in-memory)
    ALERTS_DB[alert_id] = alert_dict

    # 2. Index in pgvector (Phase 4 RAG pipeline)
    semantic_text = genai_copilot.translate_alert_to_text(alert_dict)
    embedding = genai_copilot.embed_text(semantic_text)
    db.add(AlertEvent(
        alert_id=alert_id,
        semantic_text=semantic_text,
        embedding=embedding,  # Vector(384)
        raw_payload=alert_dict
    ))
    db.commit()
```

---

## 9. GenAI RAG Copilot

### Module: `backend/genai_copilot.py`

#### Semantic Translation

```python
def translate_alert_to_text(alert: dict) -> str:
    return (
        f"A {priority} priority {object_class} was detected by {camera} "
        f"at coordinates {x:.5f}, {y:.5f} on {timestamp} "
        f"moving at {velocity:.1f} m/s with a heading of {heading:.1f} degrees."
        # + "It was flagged as a potential threat." if is_threat
    )
```

#### Embedding

```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 384-dimensional dense vector representation

def embed_text(text: str) -> list[float]:
    return model.encode(text).tolist()
```

#### Cosine Similarity Search

```python
# PostgreSQL pgvector cosine distance operator: <=>
results = db.query(AlertEvent).order_by(
    AlertEvent.embedding.cosine_distance(query_embedding)
).limit(5).all()
```

#### RAG Prompt Template

```python
prompt = f"""You are a tactical assistant for the IBVAP system.
Based on the following historical alerts retrieved from the database, answer the officer's query.
Keep your answer concise, factual, and strictly based on the provided context.

--- Context ---
{context_text}

--- Query ---
{query}

--- Response ---
"""
```

---

## 10. Streamlit Dashboard

### Module: `frontend/app.py`

#### Tab Structure

| Tab | Content |
|-----|---------|
| Command Overview | KPI cards (total alerts, priority breakdown, active cameras) |
| Alert Queue | Ranked alert cards sorted by priority_score DESC |
| Live CCTV Wall | MJPEG feeds via `<img src="/api/v1/stream/{id}">` HTML |
| Geospatial Map | Folium map with colour-coded alert pins |
| GenAI Copilot | Natural language investigation form |
| Camera Config | Zone drawing canvas + zone save/retrieve |

#### CCTV Wall Implementation

The dashboard renders camera feeds as HTML `<img>` tags pointing to the FastAPI MJPEG endpoint. This bypasses Streamlit's file-upload mechanism entirely for zero-overhead streaming:

```python
# Rendered as unsafe HTML for native browser MJPEG rendering
stream_url = f"{BACKEND_URL}/api/v1/stream/{camera_id}"
st.markdown(f'<img src="{stream_url}" style="width:100%;border-radius:8px;" />', unsafe_allow_html=True)
```

#### Alert Card Priority Scoring Display

```python
priority_colors = {
    "CRITICAL": ("#ef4444", "card-critical"),
    "HIGH": ("#f97316", "card-high"),
    "MEDIUM": ("#eab308", "card-medium"),
    "LOW": ("#22c55e", "card-low"),
}
```

---

## 11. Database Schemas

### PostgreSQL (Central Command)

#### `alert_events` table

```sql
CREATE TABLE alert_events (
    id            SERIAL PRIMARY KEY,
    alert_id      VARCHAR UNIQUE NOT NULL,
    timestamp     TIMESTAMP DEFAULT NOW(),
    camera_id     VARCHAR,
    raw_payload   JSONB NOT NULL,
    semantic_text TEXT NOT NULL,
    embedding     VECTOR(384)   -- pgvector column
);

CREATE INDEX ON alert_events USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

#### `camera_zones` table

```sql
CREATE TABLE camera_zones (
    id           SERIAL PRIMARY KEY,
    camera_id    VARCHAR UNIQUE NOT NULL,
    zone_label   VARCHAR DEFAULT 'RESTRICTED',
    polygon_json JSONB NOT NULL,  -- [[x,y], [x,y], ...]
    updated_at   TIMESTAMP DEFAULT NOW()
);
```

### SQLite (Edge Node)

```sql
-- See Section 7 for full schema
-- Key index for efficient priority-ordered sync:
CREATE INDEX idx_alerts_sync_priority
ON alerts (synced_status, priority_score DESC, created_at ASC);
```

---

## 12. Data Flow Diagrams

### Alert Lifecycle

```
[Camera Frame]
     │
     ▼
[MOG2 Motion Check] ─── NO ──▶ [Skip Frame, write raw JPEG to buffer]
     │ YES
     ▼
[YOLOv8n Detection] ─── NO targets ──▶ [DeepSORT update (empty)]
     │
     ▼
[DeepSORT Tracking]
     │
     ▼ For each confirmed track:
[FalseAlarmFilter.validate_track()]
     │ INVALID ──▶ [Draw grey box, skip alert]
     │ VALID
     ▼
[ANPREngine.process_vehicle_frame()] (vehicles only)
     │
     ▼
[RuleEngine.evaluate_track()]
     │ Returns alert_payload with priority_score
     ▼
[EdgeQueue.enqueue_alert()] ─── if priority >= 35.0 or is_threat
     │
     ▼ (background thread)
[EdgeQueue.sync_once()]
     │ POST /api/v1/alerts ──▶ [FastAPI ingest_alert()]
                                      │
                                      ▼
                            [genai_copilot.embed_text()]
                                      │
                                      ▼
                            [PostgreSQL AlertEvent INSERT]
                                      │
                                      ▼
                            [ALERTS_DB dict update]
```

---

## 13. Performance Budgets

### Tier 1 (Raspberry Pi 5 + Hailo-8L)

| Stage | Target Latency | Frequency |
|-------|----------------|-----------|
| MOG2 motion check | < 5ms | Every frame |
| YOLOv8n (Hailo NPU) | < 25ms | On motion trigger only |
| DeepSORT update | < 10ms | Every AI frame |
| FalseAlarmFilter | < 2ms | Per confirmed track |
| RuleEngine | < 1ms | Per valid track |
| ANPR (per vehicle) | < 50ms | Per vehicle track |
| Total pipeline (AI frame) | < 100ms | Max 2 FPS = 500ms budget |

### Frame Budget at 2 FPS

```
500ms frame budget (2 FPS):
├── Frame capture + resize: ~5ms
├── MOG2 application: ~3ms
├── [IF MOTION] YOLOv8n: ~25ms (Hailo NPU)
├── DeepSORT update: ~10ms
├── FalseAlarmFilter (per track × N): ~2ms × N
├── RuleEngine (per track × N): ~1ms × N
├── JPEG encode + buffer write: ~3ms
└── Remaining headroom: ~451ms
```

### Memory Budget (Tier 1)

| Component | RAM Usage |
|-----------|-----------|
| YOLOv8n model weights | ~6.5 MB |
| DeepSORT (MobileNet) | ~8 MB |
| OpenCV frame buffer (720p) | ~2.8 MB |
| JPEG frame buffer | ~50-150 KB |
| SQLite edge_alerts.db | < 50 MB (typical) |
| Total target | < 200 MB |

---

## 14. Deployment Notes

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@localhost:5432/ibvap` | PostgreSQL connection string |

### Docker Compose Services

```yaml
# docker-compose.yml (expected)
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ibvap
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Running Without PostgreSQL

The system gracefully degrades when PostgreSQL is unavailable:
- `database.py` catches the connection error and sets `engine = None`
- All DB-dependent FastAPI endpoints return `503` or work in fallback mode
- The edge queue continues to SQLite locally
- Only the `/api/v1/investigate` endpoint is unavailable without pgvector

### Security Notes for Production

1. Replace `allow_origins=["*"]` CORS policy with specific domain allowlist.
2. Add FastAPI authentication (OAuth2/API key) on all endpoints.
3. Use HTTPS (TLS termination via nginx or Caddy) for the MJPEG streaming endpoint.
4. Store PostgreSQL credentials in a secrets manager (not environment variables).
5. Restrict `backend/data/hard_negatives/` directory to read-write by the backend service user only.

---

*Document maintained by the Night's Watch development team — SIH26187*
