from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SavedViewCreate(BaseModel):
    name: str
    mode: str = "kanban"
    config: dict


class SavedViewUpdate(BaseModel):
    name: str | None = None
    mode: str | None = None
    config: dict | None = None


class SavedViewOut(BaseModel):
    id: str
    project_id: str
    name: str
    mode: str
    config: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
