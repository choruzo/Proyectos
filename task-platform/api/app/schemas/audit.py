from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AuditLogOut(BaseModel):
    id: str
    actor_user_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None

    entity_type: str
    entity_id: str
    action: str

    before: dict | None = None
    after: dict | None = None
    meta: dict | None = None

    created_at: datetime
