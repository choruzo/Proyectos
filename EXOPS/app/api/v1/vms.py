from fastapi import APIRouter, Depends

from app.auth.service import get_current_user

router = APIRouter(prefix="/api/v1/vms", tags=["vms"])


@router.get("/")
async def list_vms(current_user=Depends(get_current_user)):
    return {"status": "not implemented"}


@router.get("/{vm_id}")
async def get_vm(vm_id: str, current_user=Depends(get_current_user)):
    return {"status": "not implemented"}
