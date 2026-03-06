from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Cookie, Depends, HTTPException, status
from jose import JWTError, jwt
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db

LOCK_THRESHOLD = 5       # intentos fallidos antes de bloquear
LOCK_MINUTES = 15        # minutos de bloqueo


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str):
    from app.auth.models import TokenData

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        username: Optional[str] = payload.get("username")
        role: Optional[str] = payload.get("role")
        if user_id is None:
            raise credentials_exception
        return TokenData(user_id=int(user_id), username=username, role=role)
    except JWTError:
        raise credentials_exception


async def authenticate_user(db: AsyncSession, username: str, password: str):
    from app.auth.models import User

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if user is None:
        return None

    # Comprobar bloqueo
    if user.locked_until and datetime.now(timezone.utc) < user.locked_until.replace(tzinfo=timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cuenta bloqueada hasta {user.locked_until.strftime('%H:%M:%S UTC')}",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta desactivada",
        )

    if not verify_password(password, user.hashed_password):
        user.failed_attempts += 1
        if user.failed_attempts >= LOCK_THRESHOLD:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCK_MINUTES)
            logger.warning(f"Cuenta '{username}' bloqueada por {LOCK_MINUTES} minutos")
        await db.commit()
        return None

    # Login OK: resetear contadores
    user.failed_attempts = 0
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return user


async def get_current_user(
    access_token: Optional[str] = Cookie(default=None),
    db: AsyncSession = Depends(get_db),
):
    from app.auth.models import User

    if access_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autenticado",
        )

    token_data = decode_token(access_token)
    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado o inactivo",
        )
    return user


async def require_admin(current_user=Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requiere rol admin",
        )
    return current_user
