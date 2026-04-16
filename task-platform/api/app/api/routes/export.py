from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.audit_log import AuditLog
from app.models.project import Project
from app.models.release import Release
from app.models.saved_view import SavedView
from app.models.tag import Tag
from app.models.task import Task
from app.models.task_tag import TaskTag

router = APIRouter(prefix="/export", tags=["export"])


@router.get("")
def export_json(db: Session = Depends(get_db), _user=Depends(get_current_user)) -> dict:
    projects = list(db.scalars(select(Project)))
    releases = list(db.scalars(select(Release)))
    tasks = list(db.scalars(select(Task)))
    tags = list(db.scalars(select(Tag)))
    task_tags = list(db.scalars(select(TaskTag)))
    attachments = list(db.scalars(select(Attachment)))
    views = list(db.scalars(select(SavedView)))
    audit_logs = list(db.scalars(select(AuditLog)))

    return jsonable_encoder(
        {
            "generated_at": datetime.now(UTC),
            "projects": projects,
            "releases": releases,
            "tasks": tasks,
            "tags": tags,
            "task_tags": task_tags,
            "attachments": attachments,
            "views": views,
            "audit_logs": audit_logs,
        }
    )
