import asyncio
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import log_action
from app.auth.service import get_current_user
from app.database import get_db
from app.vcenter import vms as vm_service
from app.vcenter.connection import get_vcenter_session

router = APIRouter(prefix="/api/v1/vms", tags=["vms"])


class PowerAction(str, Enum):
    on = "on"
    off = "off"
    reboot = "reboot"


class PowerRequest(BaseModel):
    action: PowerAction


@router.get("/")
async def list_vms(
    current_user=Depends(get_current_user),
):
    si = await get_vcenter_session(current_user)
    try:
        vms = await asyncio.to_thread(vm_service.list_vms, si)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return vms


@router.post("/{vm_id}/power")
async def power_action(
    vm_id: str,
    body: PowerRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    si = await get_vcenter_session(current_user)

    action_map = {
        PowerAction.on: vm_service.power_on,
        PowerAction.off: vm_service.power_off,
        PowerAction.reboot: vm_service.reboot,
    }
    fn = action_map[body.action]

    try:
        await asyncio.to_thread(fn, si, vm_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    await log_action(
        db=db,
        user_id=current_user.id,
        username=current_user.username,
        action=f"vm_power_{body.action}",
        resource_type="vm",
        resource_id=vm_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"status": "ok", "action": body.action, "vm_id": vm_id}
