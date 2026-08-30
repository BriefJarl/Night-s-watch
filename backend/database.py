import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone

# Use DATABASE_URL from environment or default to a local test postgres instance
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://postgres:postgres@localhost:5432/ibvap"
)

# Connect to database
try:
    engine = create_engine(DATABASE_URL)
    # Ensure vector extension is installed on the database
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
except Exception as e:
    print(f"Warning: Could not connect to PostgreSQL or create vector extension. Error: {e}")
    engine = None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

class CameraZone(Base):
    """
    Stores operator-defined restricted zone polygons (pixel coordinates) per camera.
    Drawn interactively in the Camera Configuration tab and enforced by the RuleEngine.
    """
    __tablename__ = "camera_zones"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True, nullable=False)
    zone_label = Column(String, default="RESTRICTED")
    # list of [x, y] pixel coordinate pairs: [[x1,y1],[x2,y2],...]
    polygon_json = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<CameraZone {self.camera_id} ({self.zone_label})>"


class AlertEvent(Base):
    """
    SQLAlchemy model for storing alert events and their vector embeddings.
    """
    __tablename__ = "alert_events"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    camera_id = Column(String, index=True)
    
    # Store the entire raw JSON for reference or UI
    raw_payload = Column(JSON, nullable=False)
    
    # The generated natural language description
    semantic_text = Column(String, nullable=False)
    
    # The embedding vector for semantic search (384 dims for all-MiniLM-L6-v2)
    embedding = Column(Vector(384))

    def __repr__(self):
        return f"<AlertEvent {self.alert_id} ({self.camera_id})>"

def init_db():
    """Create all tables if they don't exist."""
    if engine:
        try:
            Base.metadata.create_all(bind=engine)
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to create tables. Error: {e}")
    else:
        print("Database engine not available. Skipping init_db.")

def get_db():
    """Dependency for FastAPI endpoints to get a DB session."""
    if not SessionLocal:
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
