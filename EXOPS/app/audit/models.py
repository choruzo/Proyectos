from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.database import Base


# --- SQLAlchemy Model ---

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String, nullable=False)
    action = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    resource_name = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    result = Column(String, nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    vcenter_host = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


# --- Pydantic Schema ---

class AuditLogRead(BaseModel):
    id: int
    user_id: Optional[int] = None
    username: str
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    details: Optional[str] = None
    result: str
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    vcenter_host: Optional[str] = None
    timestamp: datetime

    model_config = {"from_attributes": True}
