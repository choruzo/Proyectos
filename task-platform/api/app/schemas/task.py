from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class TaskCreate(BaseModel):
    project_id: str
    release_id: str | None = None
    title: str
    description: str | None = None
    status: str = "Backlog"
    priority: str = "Medium"
    due_date: date | None = None
    estimate_hours: float | None = None
    tags: list[str] = []


class TaskUpdate(BaseModel):
    release_id: str | None = None
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: date | None = None
    estimate_hours: float | None = None
    tags: list[str] | None = None


class TaskOut(BaseModel):
    id: str
    project_id: str
    release_id: str | None = None
    title: str
    description: str | None = None
    status: str
    priority: str
    due_date: date | None = None
    estimate_hours: float | None = None
    tags: list[str] = []

    class Config:
        from_attributes = True
