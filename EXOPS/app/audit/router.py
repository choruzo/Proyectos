from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.models import AuditLog, AuditLogRead
from app.auth.service import get_current_user
from app.database import get_db

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/", response_model=list[AuditLogRead])
async def list_audit_logs(
    user: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    from_date: Optional[datetime] = Query(default=None),
    to_date: Optional[datetime] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AuditLog)

    # Operators solo ven sus propios logs
    if current_user.role != "admin":
        stmt = stmt.where(AuditLog.user_id == current_user.id)
    elif user:
        stmt = stmt.where(AuditLog.username == user)

    if action:
        stmt = stmt.where(AuditLog.action == action)
    if from_date:
        stmt = stmt.where(AuditLog.timestamp >= from_date)
    if to_date:
        stmt = stmt.where(AuditLog.timestamp <= to_date)

    stmt = stmt.order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()
