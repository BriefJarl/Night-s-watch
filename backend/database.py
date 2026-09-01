import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector


# ---------------------------------------------------------
# LOAD BACKEND .env SAFELY
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None

if DATABASE_URL:

    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
        )

        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )

        print("[Database] Database engine configured.")

    except Exception as e:
        print(f"[Database] Warning: Engine initialization failed: {e}")

else:
    print(
        "[Database] DATABASE_URL not configured. "
        "Backend will run in demo/in-memory mode."
    )


# ---------------------------------------------------------
# SQLALCHEMY BASE
# ---------------------------------------------------------

Base = declarative_base()


# ---------------------------------------------------------
# CAMERA ZONES TABLE
# ---------------------------------------------------------

class CameraZone(Base):

    __tablename__ = "camera_zones"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    camera_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    zone_label = Column(
        String,
        default="RESTRICTED",
        nullable=False,
    )

    polygon_json = Column(
        JSON,
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<CameraZone "
            f"camera_id={self.camera_id} "
            f"zone_label={self.zone_label}>"
        )


# ---------------------------------------------------------
# ALERT EVENTS TABLE
# ---------------------------------------------------------

class AlertEvent(Base):

    __tablename__ = "alert_events"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    alert_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    camera_id = Column(
        String,
        index=True,
        nullable=False,
    )

    raw_payload = Column(
        JSON,
        nullable=False,
    )

    semantic_text = Column(
        String,
        nullable=False,
    )

    embedding = Column(
        Vector(384),
        nullable=True,
    )

    def __repr__(self):
        return (
            f"<AlertEvent "
            f"alert_id={self.alert_id} "
            f"camera_id={self.camera_id}>"
        )


# ---------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------

def init_db():

    if engine is None:
        print(
            "[Database] Database engine unavailable. "
            "Skipping database initialization."
        )
        return False

    try:

        Base.metadata.create_all(bind=engine)

        print(
            "[Database] Tables initialized successfully."
        )

        return True

    except Exception as e:

        print(
            f"[Database] Warning: Failed to initialize tables: {e}"
        )

        return False


# ---------------------------------------------------------
# DATABASE SESSION DEPENDENCY
# ---------------------------------------------------------

def get_db():

    if SessionLocal is None:

        yield None
        return

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()