from fastapi import APIRouter, Depends

from app.auth.service import get_current_user

router = APIRouter(prefix="/api/v1/datastores", tags=["datastores"])


@router.get("/")
async def list_datastores(current_user=Depends(get_current_user)):
    return {"status": "not implemented"}


@router.get("/{datastore_id}")
async def get_datastore(datastore_id: str, current_user=Depends(get_current_user)):
    return {"status": "not implemented"}
