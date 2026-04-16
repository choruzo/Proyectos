from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ProjectStatsOut(BaseModel):
    project_id: str
    project_name: str

    tasks_total: int
    backlog_total: int
    tasks_by_status: dict[str, int]
    tasks_by_priority: dict[str, int]

    attachments_total: int
    attachments_bytes: int

    audit_total: int
    audit_last_7d: int


class AdminStatsOut(BaseModel):
    generated_at: datetime
    projects: list[ProjectStatsOut]
