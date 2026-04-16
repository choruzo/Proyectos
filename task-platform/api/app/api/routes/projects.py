from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.project import Project
from app.models.task import Task
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate
from app.services.audit import log_event

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db), _user=Depends(get_current_user)) -> list[Project]:
    return list(db.scalars(select(Project).order_by(Project.name)))


@router.post("", response_model=ProjectOut)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> Project:
    exists = db.scalar(select(Project).where(Project.name == payload.name))
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Project name already exists")

    project = Project(name=payload.name, description=payload.description)
    db.add(project)
    db.flush()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project.id,
        task_id=None,
        entity_type="project",
        entity_id=project.id,
        action="project.create",
        after=ProjectOut.model_validate(project).model_dump(),
    )

    db.commit()
    db.refresh(project)
    return project


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    before = ProjectOut.model_validate(project).model_dump()

    data = payload.model_dump(exclude_unset=True)

    if "name" in data and data["name"] is not None:
        exists = db.scalar(select(Project).where(Project.name == data["name"], Project.id != project_id))
        if exists is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Project name already exists")

    for k, v in data.items():
        setattr(project, k, v)

    after = ProjectOut.model_validate(project).model_dump()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project.id,
        task_id=None,
        entity_type="project",
        entity_id=project.id,
        action="project.update",
        before=before,
        after=after,
        meta={"changed_fields": list(data.keys())},
    )

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> dict:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    before = ProjectOut.model_validate(project).model_dump()

    # Collect uploads before cascading deletes remove attachment rows.
    task_ids = [tid for (tid,) in db.execute(select(Task.id).where(Task.project_id == project_id)).all()]
    storage_keys = [
        sk
        for (sk,) in db.execute(
            select(Attachment.storage_key)
            .join(Task, Task.id == Attachment.task_id)
            .where(Task.project_id == project_id)
        ).all()
    ]

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project.id,
        task_id=None,
        entity_type="project",
        entity_id=project.id,
        action="project.delete",
        before=before,
        meta={"task_ids": task_ids, "storage_keys": storage_keys},
    )

    for sk in storage_keys:
        abs_path = os.path.join(settings.uploads_dir, sk)
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass

    for tid in task_ids:
        shutil.rmtree(os.path.join(settings.uploads_dir, tid), ignore_errors=True)

    db.delete(project)
    db.commit()
    return {"ok": True}
