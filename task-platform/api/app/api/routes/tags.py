from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.tag import Tag
from app.models.task import Task
from app.models.task_tag import TaskTag

router = APIRouter(prefix="/projects/{project_id}/tags", tags=["tags"])


@router.get("")
def suggest_tags(
    project_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
) -> list[str]:
    stmt = (
        select(Tag.name, func.count(Tag.id).label("cnt"))
        .join(TaskTag, TaskTag.tag_id == Tag.id)
        .join(Task, Task.id == TaskTag.task_id)
        .where(Task.project_id == project_id)
        .group_by(Tag.name)
        .order_by(func.count(Tag.id).desc(), Tag.name)
        .limit(limit)
    )
    return [name for (name, _cnt) in db.execute(stmt).all()]
