from functools import lru_cache

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict



PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATABASE_PATH = PROJECT_ROOT / "backend" / "data" / "trinetra.db"

class Settings(BaseSettings):

    APP_NAME: str = "Trinetra Backend"

    APP_ENV: str = "development"

    DEBUG: bool = False

    API_HOST: str = "127.0.0.1"

    API_PORT: int = 8000

    SECRET_KEY: str

    ALLOWED_ORIGINS: str = "*"

    DATABASE_URL: str = f"sqlite:///{DATABASE_PATH.as_posix()}"

    YOLO_MODEL: str = "yolo11n.pt"

    DETECTION_CONFIDENCE: float = 0.50

    MAX_CAMERAS: int = 4

    FRAME_SKIP: int = 2

    DEMO_MODE: bool = True

    model_config = SettingsConfigDict(

        env_file=PROJECT_ROOT / ".env",

        env_file_encoding="utf-8",

        case_sensitive=True,

        extra="ignore",

    )

    @property

    def allowed_origins_list(self) -> list[str]:

        return [

            origin.strip()

            for origin in self.ALLOWED_ORIGINS.split(",")

            if origin.strip()

        ]



@lru_cache

def get_settings() -> Settings:

    return Settings()

settings = get_settings()
