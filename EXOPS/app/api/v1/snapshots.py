import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import log_action
from app.auth.service import get_current_user
from app.database import get_db
from app.vcenter import snapshots as snapshot_service
from app.vcenter.connection import get_vcenter_session

router = APIRouter(prefix="/api/v1/snapshots", tags=["snapshots"])


class CreateSnapshotRequest(BaseModel):
    name: str
    description: str = ""


@router.get("/{vm_id}")
async def list_snapshots(
    vm_id: str,
    current_user=Depends(get_current_user),
):
    si = await get_vcenter_session(current_user)
    try:
        result = await asyncio.to_thread(snapshot_service.list_snapshots, si, vm_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return result


@router.post("/{vm_id}")
async def create_snapshot(
    vm_id: str,
    body: CreateSnapshotRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    si = await get_vcenter_session(current_user)
    try:
        result = await asyncio.to_thread(
            snapshot_service.create_snapshot, si, vm_id, body.name, body.description
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    await log_action(
        db=db,
        user_id=current_user.id,
        username=current_user.username,
        action="snapshot_create",
        resource_type="snapshot",
        resource_id=vm_id,
        ip_address=request.client.host if request.client else None,
    )
    return result


@router.post("/{vm_id}/{snapshot_id}/restore")
async def restore_snapshot(
    vm_id: str,
    snapshot_id: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    si = await get_vcenter_session(current_user)
    try:
        result = await asyncio.to_thread(
            snapshot_service.restore_snapshot, si, vm_id, snapshot_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    await log_action(
        db=db,
        user_id=current_user.id,
        username=current_user.username,
        action="snapshot_restore",
        resource_type="snapshot",
        resource_id=snapshot_id,
        ip_address=request.client.host if request.client else None,
    )
    return result


@router.delete("/{vm_id}/{snapshot_id}")
async def delete_snapshot(
    vm_id: str,
    snapshot_id: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    si = await get_vcenter_session(current_user)
    try:
        result = await asyncio.to_thread(
            snapshot_service.delete_snapshot, si, vm_id, snapshot_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    await log_action(
        db=db,
        user_id=current_user.id,
        username=current_user.username,
        action="snapshot_delete",
        resource_type="snapshot",
        resource_id=snapshot_id,
        ip_address=request.client.host if request.client else None,
    )
    return result
