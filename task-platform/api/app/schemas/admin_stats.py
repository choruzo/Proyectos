from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ReleaseTasksOut(BaseModel):
    release_id: str | None
    release_name: str
    tasks_total: int


class ProjectStatsOut(BaseModel):
    project_id: str
    project_name: str

    tasks_total: int
    backlog_total: int
    overdue_total: int
    missing_due_date_total: int

    tasks_by_status: dict[str, int]
    tasks_by_priority: dict[str, int]
    tasks_by_release: list[ReleaseTasksOut]

    attachments_total: int
    attachments_bytes: int

    audit_total: int
    audit_last_7d: int
    audit_last_window: int


class AdminStatsOut(BaseModel):
    generated_at: datetime
    audit_window_days: int
    projects: list[ProjectStatsOut]


class ProjectStatsResponseOut(BaseModel):
    generated_at: datetime
    audit_window_days: int
    project: ProjectStatsOut
