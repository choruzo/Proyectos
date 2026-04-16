from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import decode_token
from app.db.session import get_db
from app.models.attachment import Attachment
from app.models.task import Task
from app.models.user import User
from app.models.user_project import UserProject

security = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = decode_token(creds.credentials)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.scalar(select(User).where(User.id == user_id))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


def require_admin(user: User) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return user


def require_project_access(*, db: Session, user: User, project_id: str) -> None:
    if user.is_admin:
        return

    membership = db.scalar(
        select(UserProject).where(UserProject.user_id == user.id, UserProject.project_id == project_id)
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Project access denied")


def require_task_access(*, db: Session, user: User, task_id: str) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    require_project_access(db=db, user=user, project_id=task.project_id)
    return task


def require_attachment_access(*, db: Session, user: User, attachment_id: str) -> Attachment:
    att = db.get(Attachment, attachment_id)
    if att is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    task = db.get(Task, att.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    require_project_access(db=db, user=user, project_id=task.project_id)
    return att
