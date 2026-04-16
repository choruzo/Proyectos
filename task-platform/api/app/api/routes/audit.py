from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_project_access, require_task_access
from app.db.session import get_db
from app.models.audit_log import AuditLog
from app.schemas.audit import AuditLogOut

router = APIRouter(tags=["audit"])


def _to_out(row: AuditLog) -> AuditLogOut:
    return AuditLogOut(
        id=row.id,
        actor_user_id=row.actor_user_id,
        project_id=row.project_id,
        task_id=row.task_id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        action=row.action,
        before=json.loads(row.before_json) if row.before_json else None,
        after=json.loads(row.after_json) if row.after_json else None,
        meta=json.loads(row.meta_json) if row.meta_json else None,
        created_at=row.created_at,
    )


@router.get("/tasks/{task_id}/audit", response_model=list[AuditLogOut])
def list_task_audit(
    task_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> list[AuditLogOut]:
    require_task_access(db=db, user=user, task_id=task_id)
    stmt = select(AuditLog).where(AuditLog.task_id == task_id).order_by(AuditLog.created_at.desc()).limit(limit)
    return [_to_out(r) for r in db.scalars(stmt)]


@router.get("/projects/{project_id}/audit", response_model=list[AuditLogOut])
def list_project_audit(
    project_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> list[AuditLogOut]:
    require_project_access(db=db, user=user, project_id=project_id)
    stmt = select(AuditLog).where(AuditLog.project_id == project_id).order_by(AuditLog.created_at.desc()).limit(limit)
    return [_to_out(r) for r in db.scalars(stmt)]
