from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Integer, String

from app.database import Base


# --- SQLAlchemy Model ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="operator")
    is_active = Column(Boolean, default=True)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)


# --- Pydantic Schemas ---

class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None
    password: str
    role: str = "operator"


class UserRead(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    app_username: str
    app_password: str
    vcenter_host: str
    vcenter_username: str
    vcenter_password: str
