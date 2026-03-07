import asyncio
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import log_action
from app.auth.service import get_current_user
from app.database import get_db
from app.vcenter import hosts as host_service
from app.vcenter.connection import get_vcenter_session

router = APIRouter(prefix="/api/v1/hosts", tags=["hosts"])


class MaintenanceAction(str, Enum):
    enter = "enter"
    exit = "exit"


class MaintenanceRequest(BaseModel):
    action: MaintenanceAction


@router.get("/")
async def list_hosts(current_user=Depends(get_current_user)):
    si = await get_vcenter_session(current_user)
    try:
        hosts = await asyncio.to_thread(host_service.list_hosts, si)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return hosts


@router.post("/{host_id}/maintenance")
async def maintenance_action(
    host_id: str,
    body: MaintenanceRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    si = await get_vcenter_session(current_user)

    fn = host_service.enter_maintenance if body.action == MaintenanceAction.enter else host_service.exit_maintenance

    try:
        await asyncio.to_thread(fn, si, host_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    await log_action(
        db=db,
        user_id=current_user.id,
        username=current_user.username,
        action=f"host_maintenance_{body.action}",
        resource_type="host",
        resource_id=host_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"status": "ok", "action": body.action, "host_id": host_id}
