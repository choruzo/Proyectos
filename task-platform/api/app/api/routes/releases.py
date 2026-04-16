from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin, require_project_access
from app.db.session import get_db
from app.models.project import Project
from app.models.release import Release
from app.schemas.release import ReleaseCreate, ReleaseOut, ReleaseUpdate
from app.services.audit import log_event

router = APIRouter(prefix="/projects/{project_id}/releases", tags=["releases"])


@router.get("", response_model=list[ReleaseOut])
def list_releases(
    project_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> list[Release]:
    require_project_access(db=db, user=user, project_id=project_id)
    return list(db.scalars(select(Release).where(Release.project_id == project_id).order_by(Release.name)))


@router.post("", response_model=ReleaseOut)
def create_release(
    project_id: str,
    payload: ReleaseCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> Release:
    require_admin(user)
    require_project_access(db=db, user=user, project_id=project_id)
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rel = Release(project_id=project_id, **payload.model_dump())
    db.add(rel)
    db.flush()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="release",
        entity_id=rel.id,
        action="release.create",
        after=ReleaseOut.model_validate(rel).model_dump(),
    )

    db.commit()
    db.refresh(rel)
    return rel


@router.patch("/{release_id}", response_model=ReleaseOut)
def update_release(
    project_id: str,
    release_id: str,
    payload: ReleaseUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> Release:
    require_admin(user)
    require_project_access(db=db, user=user, project_id=project_id)
    rel = db.get(Release, release_id)
    if rel is None or rel.project_id != project_id:
        raise HTTPException(status_code=404, detail="Release not found")

    before = ReleaseOut.model_validate(rel).model_dump()

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(rel, k, v)

    after = ReleaseOut.model_validate(rel).model_dump()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="release",
        entity_id=rel.id,
        action="release.update",
        before=before,
        after=after,
        meta={"changed_fields": list(data.keys())},
    )

    db.commit()
    db.refresh(rel)
    return rel


@router.delete("/{release_id}")
def delete_release(
    project_id: str,
    release_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> dict:
    require_admin(user)
    require_project_access(db=db, user=user, project_id=project_id)
    rel = db.get(Release, release_id)
    if rel is None or rel.project_id != project_id:
        raise HTTPException(status_code=404, detail="Release not found")

    before = ReleaseOut.model_validate(rel).model_dump()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="release",
        entity_id=rel.id,
        action="release.delete",
        before=before,
    )

    db.delete(rel)
    db.commit()
    return {"ok": True}
