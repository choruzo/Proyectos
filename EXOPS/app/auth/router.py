from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import LoginRequest, UserRead
from app.auth.service import authenticate_user, create_access_token, get_current_user
from app.audit.service import get_client_ip, log_action
from app.config import settings
from app.database import get_db
from app.vcenter.connection import connect, disconnect

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login")
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    # 1. Validar credenciales de la app
    user = await authenticate_user(db, payload.app_username, payload.app_password)
    if user is None:
        await log_action(
            db,
            user_id=None,
            username=payload.app_username,
            action="login",
            result="error",
            error_message="Credenciales de aplicación inválidas",
            ip_address=get_client_ip(request),
            vcenter_host=payload.vcenter_host,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas",
        )

    # 2. Conectar al vCenter (saltear en DEV_MODE)
    if settings.DEV_MODE:
        logger.warning(f"DEV_MODE: saltando conexión vCenter para user_id={user.id}")
    else:
        connect(user.id, payload.vcenter_host, payload.vcenter_username, payload.vcenter_password)

    # 3. Emitir JWT como cookie httpOnly
    token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role}
    )
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=3600,
    )

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="login",
        result="success",
        ip_address=get_client_ip(request),
        vcenter_host=payload.vcenter_host,
    )

    return {"message": "Login exitoso", "username": user.username, "role": user.role}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    disconnect(current_user.id)
    response.delete_cookie("access_token")

    await log_action(
        db,
        user_id=current_user.id,
        username=current_user.username,
        action="logout",
        result="success",
        ip_address=get_client_ip(request),
    )

    return {"message": "Sesión cerrada"}


@router.get("/me", response_model=UserRead)
async def get_me(current_user=Depends(get_current_user)):
    return current_user
