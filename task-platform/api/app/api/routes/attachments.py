from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.task import Task
from app.schemas.attachment import AttachmentOut
from app.services.audit import log_event

router = APIRouter(tags=["attachments"])

MAX_BYTES = 100 * 1024 * 1024


@router.get("/tasks/{task_id}/attachments", response_model=list[AttachmentOut])
def list_attachments(
    task_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
) -> list[Attachment]:
    return list(db.scalars(select(Attachment).where(Attachment.task_id == task_id).order_by(Attachment.created_at.desc())))


@router.post("/tasks/{task_id}/attachments", response_model=AttachmentOut)
def upload_attachment(
    task_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> Attachment:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    storage_key = f"{task_id}/{uuid.uuid4()}"
    abs_dir = os.path.join(settings.uploads_dir, task_id)
    os.makedirs(abs_dir, exist_ok=True)
    abs_path = os.path.join(settings.uploads_dir, storage_key)

    total = 0
    with open(abs_path, "wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                f.close()
                try:
                    os.remove(abs_path)
                except FileNotFoundError:
                    pass
                raise HTTPException(status_code=413, detail="File too large (max 100MB)")
            f.write(chunk)

    att = Attachment(
        task_id=task_id,
        original_name=file.filename or "upload",
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=total,
        storage_key=storage_key,
        uploaded_by=user.id,
    )
    db.add(att)
    db.flush()

    log_event(
        db,
        actor_user_id=user.id,
        project_id=task.project_id,
        task_id=task_id,
        entity_type="attachment",
        entity_id=att.id,
        action="attachment.upload",
        after=AttachmentOut.model_validate(att).model_dump(),
    )

    db.commit()
    db.refresh(att)
    return att


@router.get("/attachments/{attachment_id}/download")
def download_attachment(
    attachment_id: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
) -> FileResponse:
    att = db.get(Attachment, attachment_id)
    if att is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    abs_path = os.path.join(settings.uploads_dir, att.storage_key)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File missing")

    return FileResponse(
        abs_path,
        media_type=att.mime_type,
        filename=att.original_name,
    )


@router.delete("/attachments/{attachment_id}")
def delete_attachment(
    attachment_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
) -> dict:
    att = db.get(Attachment, attachment_id)
    if att is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    task = db.get(Task, att.task_id)
    project_id = task.project_id if task else None

    log_event(
        db,
        actor_user_id=user.id,
        project_id=project_id,
        task_id=att.task_id,
        entity_type="attachment",
        entity_id=att.id,
        action="attachment.delete",
        before=AttachmentOut.model_validate(att).model_dump(),
    )

    abs_path = os.path.join(settings.uploads_dir, att.storage_key)
    try:
        os.remove(abs_path)
    except FileNotFoundError:
        pass

    db.execute(delete(Attachment).where(Attachment.id == attachment_id))
    db.commit()
    return {"ok": True}
