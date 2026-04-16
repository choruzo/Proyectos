from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog


def _dump(obj: Any | None) -> str | None:
    if obj is None:
        return None
    return json.dumps(jsonable_encoder(obj), ensure_ascii=False)


def log_event(
    db: Session,
    *,
    actor_user_id: str | None,
    project_id: str | None,
    task_id: str | None,
    entity_type: str,
    entity_id: str,
    action: str,
    before: Any | None = None,
    after: Any | None = None,
    meta: Any | None = None,
) -> AuditLog:
    row = AuditLog(
        actor_user_id=actor_user_id,
        project_id=project_id,
        task_id=task_id,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        before_json=_dump(before),
        after_json=_dump(after),
        meta_json=_dump(meta),
    )
    db.add(row)
    db.flush()
    return row
