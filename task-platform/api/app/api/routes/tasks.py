from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, delete, or_, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_project_access
from app.core.constants import TASK_PRIORITIES, TASK_STATUSES
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.release import Release
from app.models.tag import Tag
from app.models.task import Task
from app.models.task_tag import TaskTag
from app.schemas.task import TaskCreate, TaskOut, TaskUpdate
from app.services.audit import log_event
from app.services.tags import get_or_create_tag

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _priority_order_expr(col):
    return case(
        (col == "High", 0),
        (col == "Medium", 1),
        (col == "Low", 2),
        else_=3,
    )


def _load_tags_map(db: Session, task_ids: list[str]) -> dict[str, list[str]]:
    if not task_ids:
        return {}

    rows = db.execute(
        select(TaskTag.task_id, Tag.name)
        .join(Tag, Tag.id == TaskTag.tag_id)
        .where(TaskTag.task_id.in_(task_ids))
        .order_by(Tag.name)
    ).all()

    out: dict[str, list[str]] = defaultdict(list)
    for task_id, name in rows:
        out[task_id].append(name)
    return out


def _task_snapshot(db: Session, task: Task) -> dict:
    dto = TaskOut.model_validate(task)
    dto.tags = _load_tags_map(db, [task.id]).get(task.id, [])
    return dto.model_dump()


@router.get("", response_model=list[TaskOut])
def list_tasks(
    project_id: str | None = None,
    release_id: str | None = None,
    backlog: bool = False,
    status: str | None = None,
    priority: str | None = None,
    tag: str | None = None,
    q: str | None = Query(default=None, min_length=1),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> list[TaskOut]:
    stmt = select(Task)

    if not user.is_admin:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id is required")
        require_project_access(db=db, user=user, project_id=project_id)

    if project_id:
        stmt = stmt.where(Task.project_id == project_id)
    if backlog:
        stmt = stmt.where(Task.release_id.is_(None))
    elif release_id:
        stmt = stmt.where(Task.release_id == release_id)
    if status:
        stmt = stmt.where(Task.status == status)
    if priority:
        stmt = stmt.where(Task.priority == priority)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Task.title.ilike(like), Task.description.ilike(like)))
    if tag:
        stmt = (
            stmt.join(TaskTag, TaskTag.task_id == Task.id)
            .join(Tag, Tag.id == TaskTag.tag_id)
            .where(Tag.name == tag)
        )

    stmt = stmt.order_by(_priority_order_expr(Task.priority), Task.due_date.asc().nulls_last(), Task.created_at.desc())

    tasks = list(db.scalars(stmt))
    tags_map = _load_tags_map(db, [t.id for t in tasks])

    out: list[TaskOut] = []
    for t in tasks:
        dto = TaskOut.model_validate(t)
        dto.tags = tags_map.get(t.id, [])
        out.append(dto)
    return out


@router.post("", response_model=TaskOut)
def create_task(
    payload: TaskCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> TaskOut:
    if payload.status not in TASK_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if payload.priority not in TASK_PRIORITIES:
        raise HTTPException(status_code=400, detail="Invalid priority")

    require_project_access(db=db, user=user, project_id=payload.project_id)

    if payload.release_id is not None:
        rel = db.get(Release, payload.release_id)
        if rel is None or rel.project_id != payload.project_id:
            raise HTTPException(status_code=400, detail="Invalid release_id")

    data = payload.model_dump(exclude={"tags"})
    data["created_by"] = user.id
    data["updated_by"] = user.id
    task = Task(**data)
    db.add(task)
    db.flush()

    for name in payload.tags:
        if not name.strip():
            continue
        tag_obj = get_or_create_tag(db, name)
        db.add(TaskTag(task_id=task.id, tag_id=tag_obj.id))

    after = TaskOut.model_validate(task)
    after.tags = [n.strip() for n in payload.tags if n.strip()]

    log_event(
        db,
        actor_user_id=user.id,
        project_id=task.project_id,
        task_id=task.id,
        entity_type="task",
        entity_id=task.id,
        action="task.create",
        after=after,
        meta={"tags": after.tags},
    )

    db.commit()
    db.refresh(task)

    out = TaskOut.model_validate(task)
    out.tags = after.tags
    return out


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(
    task_id: str,
    payload: TaskUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> TaskOut:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    require_project_access(db=db, user=user, project_id=task.project_id)

    before = _task_snapshot(db, task)

    data = payload.model_dump(exclude_unset=True)

    # Never allow clients to overwrite ownership/audit fields.
    for k in ("created_by", "updated_by"):
        data.pop(k, None)

    if "release_id" in data and data["release_id"] is not None:
        rel = db.get(Release, data["release_id"])
        if rel is None or rel.project_id != task.project_id:
            raise HTTPException(status_code=400, detail="Invalid release_id")

    if "status" in data and data["status"] not in TASK_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if "priority" in data and data["priority"] not in TASK_PRIORITIES:
        raise HTTPException(status_code=400, detail="Invalid priority")

    tags = data.pop("tags", None)

    changed_fields = list(data.keys())
    if tags is not None:
        changed_fields.append("tags")

    for k, v in data.items():
        setattr(task, k, v)

    task.updated_by = user.id

    if tags is not None:
        db.execute(delete(TaskTag).where(TaskTag.task_id == task.id))
        for name in tags:
            if not name.strip():
                continue
            tag_obj = get_or_create_tag(db, name)
            db.add(TaskTag(task_id=task.id, tag_id=tag_obj.id))

    db.flush()
    after = _task_snapshot(db, task)

    log_event(
        db,
        actor_user_id=user.id,
        project_id=task.project_id,
        task_id=task.id,
        entity_type="task",
        entity_id=task.id,
        action="task.update",
        before=before,
        after=after,
        meta={"changed_fields": changed_fields},
    )

    db.commit()
    db.refresh(task)

    out = TaskOut.model_validate(task)
    out.tags = _load_tags_map(db, [task.id]).get(task.id, [])
    return out


@router.delete("/{task_id}")
def delete_task(
    task_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> dict:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    require_project_access(db=db, user=user, project_id=task.project_id)

    before = _task_snapshot(db, task)

    # Delete attachment files + rows
    attachments = list(db.scalars(select(Attachment).where(Attachment.task_id == task_id)))

    log_event(
        db,
        actor_user_id=user.id,
        project_id=task.project_id,
        task_id=task.id,
        entity_type="task",
        entity_id=task.id,
        action="task.delete",
        before=before,
        meta={
            "attachments": [
                {
                    "id": a.id,
                    "original_name": a.original_name,
                    "storage_key": a.storage_key,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ]
        },
    )

    from app.core.config import settings
    import os

    for a in attachments:
        abs_path = os.path.join(settings.uploads_dir, a.storage_key)
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass

    db.execute(delete(Attachment).where(Attachment.task_id == task_id))
    db.delete(task)
    db.commit()
    return {"ok": True}
