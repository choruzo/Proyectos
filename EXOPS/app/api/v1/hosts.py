from fastapi import APIRouter, Depends

from app.auth.service import get_current_user

router = APIRouter(prefix="/api/v1/hosts", tags=["hosts"])


@router.get("/")
async def list_hosts(current_user=Depends(get_current_user)):
    return {"status": "not implemented"}


@router.get("/{host_id}")
async def get_host(host_id: str, current_user=Depends(get_current_user)):
    return {"status": "not implemented"}
