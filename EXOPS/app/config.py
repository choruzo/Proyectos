from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # App
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    # En DEV_MODE=true el login omite la conexión real al vCenter
    DEV_MODE: bool = False

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/exops.db"

    # SSL
    SSL_CERT_PATH: str = "certs/cert.pem"
    SSL_KEY_PATH: str = "certs/key.pem"

    # vCenter cuenta de servicio (colección en background — opcional)
    VCENTER_SERVICE_HOST: Optional[str] = None
    VCENTER_SERVICE_USER: Optional[str] = None
    VCENTER_SERVICE_PASS: Optional[str] = None
    VCENTER_METRICS_INTERVAL_MINUTES: int = 15


settings = Settings()
