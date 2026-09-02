from app.core.database import Base, engine

# Import all models before create_all()
from app.models.camera import Camera
from app.models.detection_event import DetectionEvent


def init_database():
    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")


if __name__ == "__main__":
    init_database()