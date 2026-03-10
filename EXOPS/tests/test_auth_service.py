"""Tests unitarios directos para las funciones del servicio de autenticación."""
from datetime import timedelta, datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.auth.models import User
from app.auth.service import (
    authenticate_user,
    create_access_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database import Base


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


async def _make_session():
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False), engine


# --- hash / verify ---

def test_hash_and_verify_password():
    hashed = hash_password("mysecret")
    assert verify_password("mysecret", hashed) is True
    assert verify_password("wrong", hashed) is False


# --- create / decode token ---

def test_create_and_decode_token():
    token = create_access_token({"sub": "42", "username": "alice", "role": "admin"})
    data = decode_token(token)
    assert data.user_id == 42
    assert data.username == "alice"
    assert data.role == "admin"


def test_decode_token_invalid_raises():
    with pytest.raises(HTTPException) as exc:
        decode_token("not.a.valid.token")
    assert exc.value.status_code == 401


def test_decode_token_missing_sub_raises():
    # Token sin campo 'sub'
    token = create_access_token({"username": "alice"})
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


# --- authenticate_user ---

async def test_authenticate_user_success(db):
    user = User(
        username="alice",
        hashed_password=hash_password("pass123"),
        role="operator",
        is_active=True,
    )
    db.add(user)
    await db.commit()

    result = await authenticate_user(db, "alice", "pass123")
    assert result is not None
    assert result.username == "alice"
    assert result.failed_attempts == 0


async def test_authenticate_user_wrong_password(db):
    user = User(
        username="bob",
        hashed_password=hash_password("correct"),
        role="operator",
        is_active=True,
    )
    db.add(user)
    await db.commit()

    result = await authenticate_user(db, "bob", "wrong")
    assert result is None
    await db.refresh(user)
    assert user.failed_attempts == 1


async def test_authenticate_user_not_found(db):
    result = await authenticate_user(db, "noexiste", "pass")
    assert result is None


async def test_authenticate_user_inactive_raises(db):
    user = User(
        username="charlie",
        hashed_password=hash_password("pass"),
        role="operator",
        is_active=False,
    )
    db.add(user)
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await authenticate_user(db, "charlie", "pass")
    assert exc.value.status_code == 403


async def test_authenticate_user_locked_raises(db):
    from datetime import timedelta, timezone, datetime
    user = User(
        username="locked",
        hashed_password=hash_password("pass"),
        role="operator",
        is_active=True,
        locked_until=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    db.add(user)
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await authenticate_user(db, "locked", "pass")
    assert exc.value.status_code == 403


# --- get_current_user ---

async def test_get_current_user_no_token(db):
    with pytest.raises(HTTPException) as exc:
        await get_current_user(access_token=None, db=db)
    assert exc.value.status_code == 401


async def test_get_current_user_invalid_token(db):
    with pytest.raises(HTTPException) as exc:
        await get_current_user(access_token="bad.token.here", db=db)
    assert exc.value.status_code == 401


async def test_get_current_user_user_not_in_db(db):
    token = create_access_token({"sub": "9999", "username": "ghost", "role": "admin"})
    with pytest.raises(HTTPException) as exc:
        await get_current_user(access_token=token, db=db)
    assert exc.value.status_code == 401


async def test_get_current_user_success(db):
    user = User(
        username="diana",
        hashed_password=hash_password("pass"),
        role="admin",
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": str(user.id), "username": "diana", "role": "admin"})
    result = await get_current_user(access_token=token, db=db)
    assert result.username == "diana"
