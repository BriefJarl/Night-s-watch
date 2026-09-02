from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from app.api.cameras import router as cameras_router

from app.api.detection_events import router as detection_events_router

from app.api.dashboard import router as dashboard_router

from app.api.alerts import router as alerts_router

from app.api.routes import ai

from app.api.health import router as health_router

from app.core.config import settings

from app.core.logging_config import configure_logging

from app.core.database import Base, engine

# Import models before create_all so SQLAlchemy registers all tables

from app.models.camera import Camera

from app.models.detection_event import DetectionEvent

from app.models.alert import Alert





configure_logging()





@asynccontextmanager

async def lifespan(app: FastAPI):

    print("Starting Trinetra Backend...")

    print(f"Environment: {settings.APP_ENV}")

    Base.metadata.create_all(bind=engine)

    print("Database tables initialized successfully.")

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

    allow_credentials=True,

    allow_methods=[

        "GET",

        "POST",

        "PUT",

        "PATCH",

        "DELETE",

    ],

    allow_headers=[

        "Content-Type",

        "Authorization",

    ],

)





# Routers

app.include_router(health_router)

app.include_router(cameras_router)

app.include_router(detection_events_router)

app.include_router(dashboard_router)

app.include_router(alerts_router)

app.include_router(ai.router)

@app.get(

    "/",

    tags=["Root"],

)

def root():

    return {

        "message": "Welcome to Trinetra API",

        "status": "online",

        "docs": "/docs",

    }
