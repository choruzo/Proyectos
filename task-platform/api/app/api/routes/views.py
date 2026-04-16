from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.saved_view import SavedView
from app.schemas.view import SavedViewCreate, SavedViewOut, SavedViewUpdate
from app.services.audit import log_event

router = APIRouter(prefix="/projects/{project_id}/views", tags=["views"])


def _to_out(row: SavedView) -> SavedViewOut:
    return SavedViewOut(
        id=row.id,
        project_id=row.project_id,
        name=row.name,
        mode=row.mode,
        config=json.loads(row.config_json),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=list[SavedViewOut])
def list_views(
    project_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
) -> list[SavedViewOut]:
    stmt = select(SavedView).where(SavedView.project_id == project_id).order_by(SavedView.name)
    return [_to_out(v) for v in db.scalars(stmt)]


@router.post("", response_model=SavedViewOut)
def create_view(
    project_id: str,
    payload: SavedViewCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> SavedViewOut:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    exists = db.scalar(select(SavedView).where(SavedView.project_id == project_id, SavedView.name == payload.name))
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="View name already exists")

    view = SavedView(
        project_id=project_id,
        name=payload.name,
        mode=payload.mode,
        config_json=json.dumps(payload.config, ensure_ascii=False),
    )
    db.add(view)
    db.flush()

    created = _to_out(view)
    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="view",
        entity_id=view.id,
        action="view.create",
        after=created.model_dump(),
    )

    db.commit()
    db.refresh(view)
    return _to_out(view)


@router.patch("/{view_id}", response_model=SavedViewOut)
def update_view(
    project_id: str,
    view_id: str,
    payload: SavedViewUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> SavedViewOut:
    view = db.get(SavedView, view_id)
    if view is None or view.project_id != project_id:
        raise HTTPException(status_code=404, detail="View not found")

    before = _to_out(view).model_dump()

    data = payload.model_dump(exclude_unset=True)

    if "name" in data and data["name"] is not None:
        exists = db.scalar(
            select(SavedView).where(
                SavedView.project_id == project_id,
                SavedView.name == data["name"],
                SavedView.id != view_id,
            )
        )
        if exists is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="View name already exists")

    if "config" in data:
        view.config_json = json.dumps(data.pop("config") or {}, ensure_ascii=False)

    changed_fields = list(data.keys())

    for k, v in data.items():
        setattr(view, k, v)

    after = _to_out(view).model_dump()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="view",
        entity_id=view.id,
        action="view.update",
        before=before,
        after=after,
        meta={"changed_fields": changed_fields},
    )

    db.commit()
    db.refresh(view)
    return _to_out(view)


@router.delete("/{view_id}")
def delete_view(
    project_id: str,
    view_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> dict:
    view = db.get(SavedView, view_id)
    if view is None or view.project_id != project_id:
        raise HTTPException(status_code=404, detail="View not found")

    before = _to_out(view).model_dump()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=None,
        entity_type="view",
        entity_id=view.id,
        action="view.delete",
        before=before,
    )

    db.execute(delete(SavedView).where(SavedView.id == view_id))
    db.commit()
    return {"ok": True}
