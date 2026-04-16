from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ReleaseCreate(BaseModel):
    name: str
    note: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str = "planned"


class ReleaseUpdate(BaseModel):
    name: str | None = None
    note: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str | None = None


class ReleaseOut(BaseModel):
    id: str
    project_id: str
    name: str
    note: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str

    class Config:
        from_attributes = True
