from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.core.security import hash_password
from app.db.session import get_db
from app.models.project import Project
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
from app.services.stats import compute_project_stats

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
    audit_days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AdminStatsOut:
    _require_admin(user)

    projects = list(db.scalars(select(Project).order_by(Project.name)))
    now, stats_by_project = compute_project_stats(db, [p.id for p in projects], audit_days=audit_days)

    out_projects: list[ProjectStatsOut] = []
    for p in projects:
        stats = stats_by_project.get(p.id)
        if stats is None:
            stats = {
                "tasks_total": 0,
                "backlog_total": 0,
                "overdue_total": 0,
                "missing_due_date_total": 0,
                "tasks_by_status": {},
                "tasks_by_priority": {},
                "tasks_by_release": [{"release_id": None, "release_name": "Backlog", "tasks_total": 0}],
                "attachments_total": 0,
                "attachments_bytes": 0,
                "audit_total": 0,
                "audit_last_7d": 0,
                "audit_last_window": 0,
            }

        out_projects.append(ProjectStatsOut(project_id=p.id, project_name=p.name, **stats))

    return AdminStatsOut(generated_at=now, audit_window_days=audit_days, projects=out_projects)
