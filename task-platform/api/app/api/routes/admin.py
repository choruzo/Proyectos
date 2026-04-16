from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.core.security import hash_password
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.audit_log import AuditLog
from app.models.project import Project
from app.models.release import Release
from app.models.task import Task
from app.models.user import User
from app.models.user_project import UserProject
from app.schemas.admin_stats import AdminStatsOut, ProjectStatsOut
from app.schemas.user import (
    AdminUserCreate,
    AdminUserOut,
    AdminUserUpdate,
    ProjectUsersOut,
    ProjectUsersUpdate,
    ResetPasswordRequest,
    UserProjectsOut,
    UserProjectsUpdate,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(user: User) -> User:
    return require_admin(user)


@router.get("/users", response_model=list[AdminUserOut])
def list_users(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[User]:
    _require_admin(user)
    return list(db.scalars(select(User).order_by(User.email)))


@router.post("/users", response_model=AdminUserOut)
def create_user(
    payload: AdminUserCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> User:
    _require_admin(user)

    exists = db.scalar(select(User).where(User.email == payload.email))
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    u = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        is_admin=payload.is_admin,
        is_active=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@router.patch("/users/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: str,
    payload: AdminUserUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> User:
    _require_admin(user)

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(target, k, v)

    db.commit()
    db.refresh(target)
    return target


@router.post("/users/{user_id}/reset-password")
def reset_password(
    user_id: str,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    _require_admin(user)

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    target.password_hash = hash_password(payload.new_password)
    db.commit()
    return {"ok": True}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    _require_admin(user)

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Soft-delete to avoid breaking audit references / future ownership links.
    target.is_active = False
    db.commit()
    return {"ok": True}


@router.get("/users/{user_id}/projects", response_model=UserProjectsOut)
def get_user_projects(
    user_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UserProjectsOut:
    _require_admin(user)

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    rows = db.execute(select(UserProject.project_id).where(UserProject.user_id == user_id)).all()
    return UserProjectsOut(project_ids=[pid for (pid,) in rows])


@router.put("/users/{user_id}/projects", response_model=UserProjectsOut)
def replace_user_projects(
    user_id: str,
    payload: UserProjectsUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UserProjectsOut:
    _require_admin(user)

    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    unique_ids = []
    seen = set()
    for pid in payload.project_ids:
        if pid in seen:
            continue
        seen.add(pid)
        unique_ids.append(pid)

    if unique_ids:
        exists = set(pid for (pid,) in db.execute(select(Project.id).where(Project.id.in_(unique_ids))).all())
        missing = [pid for pid in unique_ids if pid not in exists]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_project_ids": missing})

    db.execute(delete(UserProject).where(UserProject.user_id == user_id))
    for pid in unique_ids:
        db.add(UserProject(user_id=user_id, project_id=pid))

    db.commit()
    return UserProjectsOut(project_ids=unique_ids)


@router.get("/projects/{project_id}/users", response_model=ProjectUsersOut)
def get_project_users(
    project_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectUsersOut:
    _require_admin(user)

    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = db.execute(select(UserProject.user_id).where(UserProject.project_id == project_id)).all()
    return ProjectUsersOut(user_ids=[uid for (uid,) in rows])


@router.put("/projects/{project_id}/users", response_model=ProjectUsersOut)
def replace_project_users(
    project_id: str,
    payload: ProjectUsersUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProjectUsersOut:
    _require_admin(user)

    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    unique_ids: list[str] = []
    seen = set()
    for uid in payload.user_ids:
        if uid in seen:
            continue
        seen.add(uid)
        unique_ids.append(uid)

    if unique_ids:
        exists = set(uid for (uid,) in db.execute(select(User.id).where(User.id.in_(unique_ids))).all())
        missing = [uid for uid in unique_ids if uid not in exists]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_user_ids": missing})

    db.execute(delete(UserProject).where(UserProject.project_id == project_id))
    for uid in unique_ids:
        db.add(UserProject(user_id=uid, project_id=project_id))

    db.commit()
    return ProjectUsersOut(user_ids=unique_ids)


@router.get("/stats", response_model=AdminStatsOut)
def admin_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AdminStatsOut:
    _require_admin(user)

    now = datetime.now(UTC)
    since_7d = now - timedelta(days=7)

    projects = list(db.scalars(select(Project).order_by(Project.name)))

    tasks_by_status: dict[str, dict[str, int]] = {}
    for pid, status_name, cnt in db.execute(
        select(Task.project_id, Task.status, func.count(Task.id))
        .group_by(Task.project_id, Task.status)
    ).all():
        tasks_by_status.setdefault(pid, {})[status_name] = int(cnt)

    tasks_by_priority: dict[str, dict[str, int]] = {}
    for pid, prio, cnt in db.execute(
        select(Task.project_id, Task.priority, func.count(Task.id))
        .group_by(Task.project_id, Task.priority)
    ).all():
        tasks_by_priority.setdefault(pid, {})[prio] = int(cnt)

    backlog_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(Task.project_id, func.count(Task.id))
            .where(Task.release_id.is_(None))
            .group_by(Task.project_id)
        ).all()
    )

    attachment_rows = db.execute(
        select(Task.project_id, func.count(Attachment.id), func.coalesce(func.sum(Attachment.size_bytes), 0))
        .join(Task, Task.id == Attachment.task_id)
        .group_by(Task.project_id)
    ).all()
    attachments_totals: dict[str, tuple[int, int]] = {pid: (int(cnt), int(sz)) for pid, cnt, sz in attachment_rows}

    audit_totals = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(AuditLog.project_id, func.count(AuditLog.id))
            .where(AuditLog.project_id.is_not(None))
            .group_by(AuditLog.project_id)
        ).all()
    )

    audit_last_7d = dict(
        (pid, int(cnt))
        for pid, cnt in db.execute(
            select(AuditLog.project_id, func.count(AuditLog.id))
            .where(AuditLog.project_id.is_not(None), AuditLog.created_at >= since_7d)
            .group_by(AuditLog.project_id)
        ).all()
    )

    out_projects: list[ProjectStatsOut] = []
    for p in projects:
        by_status = tasks_by_status.get(p.id, {})
        by_priority = tasks_by_priority.get(p.id, {})
        tasks_total = int(sum(by_status.values()))
        backlog_total = int(backlog_totals.get(p.id, 0))

        att_cnt, att_sz = attachments_totals.get(p.id, (0, 0))
        audit_cnt = int(audit_totals.get(p.id, 0))
        audit_7d = int(audit_last_7d.get(p.id, 0))

        out_projects.append(
            ProjectStatsOut(
                project_id=p.id,
                project_name=p.name,
                tasks_total=tasks_total,
                backlog_total=backlog_total,
                tasks_by_status=by_status,
                tasks_by_priority=by_priority,
                attachments_total=att_cnt,
                attachments_bytes=att_sz,
                audit_total=audit_cnt,
                audit_last_7d=audit_7d,
            )
        )

    return AdminStatsOut(generated_at=now, projects=out_projects)
