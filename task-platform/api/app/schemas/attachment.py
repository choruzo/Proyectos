from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AttachmentOut(BaseModel):
    id: str
    task_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    created_at: datetime

    class Config:
        from_attributes = True
