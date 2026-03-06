from typing import Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.models import AuditLog


def get_client_ip(request: Request) -> str:
    """Extrae la IP real del cliente, considera X-Forwarded-For."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def log_action(
    db: AsyncSession,
    user_id: Optional[int],
    username: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    details: Optional[str] = None,
    result: str = "success",
    error_message: Optional[str] = None,
    ip_address: Optional[str] = None,
    vcenter_host: Optional[str] = None,
) -> AuditLog:
    entry = AuditLog(
        user_id=user_id,
        username=username,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        details=details,
        result=result,
        error_message=error_message,
        ip_address=ip_address,
        vcenter_host=vcenter_host,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry
