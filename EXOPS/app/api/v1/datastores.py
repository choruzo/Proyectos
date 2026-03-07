import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.service import get_current_user
from app.vcenter import datastores as datastore_service
from app.vcenter.connection import get_vcenter_session

router = APIRouter(prefix="/api/v1/datastores", tags=["datastores"])


@router.get("/")
async def list_datastores(current_user=Depends(get_current_user)):
    si = await get_vcenter_session(current_user)
    try:
        datastores = await asyncio.to_thread(datastore_service.list_datastores, si)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return datastores
