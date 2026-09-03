import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cameras import router as cameras_router
from app.api.detection_events import router as detection_events_router
from app.api.dashboard import router as dashboard_router
from app.api.alerts import router as alerts_router
from app.api.health import router as health_router

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.database import Base, engine, SessionLocal

# Import models before create_all so SQLAlchemy registers all tables
from app.models.camera import Camera
from app.models.detection_event import DetectionEvent
from app.models.alert import Alert


# NOTE: the AI router (app.api.routes.ai) is intentionally NOT registered.
# It imports ultralytics/torch, which exceeds the 512 MB memory limit of the
# free hosting tier. Live YOLO inference runs from the local development
# environment; the hosted deployment serves the dashboard, alerting and
# event pipeline.


configure_logging()


DEMO_OBJECTS = [
    "person", "person", "car", "truck",
    "motorcycle", "bus", "person", "car",
]

DEMO_CAMERAS = [
    (1, "CAM-01", "BOP Ajnala Sector Fence"),
    (2, "CAM-02", "Attari Approach Road"),
]


def seed_demo_data() -> None:
    """
    Populate a small, deterministic demo dataset.

    The free hosting tier has no persistent disk, so the SQLite file is
    recreated whenever the service restarts. Without seeding, an evaluator
    opening the link after a restart would see an empty dashboard. This runs
    on every boot but only inserts when the tables are empty.
    """
    db = SessionLocal()

    try:
        if db.query(Camera).count() == 0:
            for cam_id, name, location in DEMO_CAMERAS:
                db.add(
                    Camera(
                        id=cam_id,
                        name=name,
                        location=location,
                        stream_url=f"rtsp://demo.local/{name.lower()}",
                        is_active=True,
                    )
                )
            db.commit()

        if db.query(DetectionEvent).count() == 0:
            rng = random.Random(42)  # deterministic across restarts
            now = datetime.utcnow()

            for i in range(24):
                event = DetectionEvent(
                    camera_id=rng.choice([1, 2]),
                    object_type=rng.choice(DEMO_OBJECTS),
                    confidence=round(rng.uniform(0.62, 0.97), 2),
                    detected_at=now - timedelta(minutes=i * 17),
                )
                db.add(event)
                db.flush()

                if event.confidence >= 0.90:
                    level = "CRITICAL"
                elif event.confidence >= 0.75:
                    level = "HIGH"
                else:
                    level = "MEDIUM"

                db.add(
                    Alert(
                        detection_event_id=event.id,
                        camera_id=event.camera_id,
                        alert_level=level,
                        message=(
                            f"{level} alert: {event.object_type} detected "
                            f"with confidence {event.confidence:.2f}"
                        ),
                        is_resolved=(i % 4 == 0),
                        created_at=event.detected_at,
                    )
                )

            db.commit()

    except Exception as error:
        db.rollback()
        print(f"Demo seeding skipped: {error}")

    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Trinetra Backend...")
    print(f"Environment: {settings.APP_ENV}")

    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

    seed_demo_data()
    print("Demo data ready.")

    yield

    print("Shutting down Trinetra Backend...")


app = FastAPI(
    title="Trinetra API",
    description="AI-powered Intelligent Border Video Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# Routers
app.include_router(health_router)
app.include_router(cameras_router)
app.include_router(detection_events_router)
app.include_router(dashboard_router)
app.include_router(alerts_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to Trinetra API",
        "status": "online",
        "docs": "/docs",
    }
