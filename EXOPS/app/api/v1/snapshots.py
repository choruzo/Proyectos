from fastapi import APIRouter, Depends

from app.auth.service import get_current_user

router = APIRouter(prefix="/api/v1/snapshots", tags=["snapshots"])


@router.get("/")
async def list_snapshots(current_user=Depends(get_current_user)):
    return {"status": "not implemented"}
