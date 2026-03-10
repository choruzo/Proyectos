"""Fixtures compartidos para todos los tests de integración."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.auth.models import User
from app.auth.service import create_access_token, hash_password
from app.database import Base, get_db
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSession = async_sessionmaker(engine, expire_on_commit=False)
    async with TestSession() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db):
    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def admin_user(db):
    user = User(
        username="admin",
        hashed_password=hash_password("adminpass"),
        role="admin",
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def operator_user(db):
    user = User(
        username="operator",
        hashed_password=hash_password("operpass"),
        role="operator",
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def admin_client(client, admin_user):
    token = create_access_token(
        {
            "sub": str(admin_user.id),
            "username": admin_user.username,
            "role": admin_user.role,
        }
    )
    client.cookies.set("access_token", token)
    return client


@pytest_asyncio.fixture
async def operator_client(client, operator_user):
    token = create_access_token(
        {
            "sub": str(operator_user.id),
            "username": operator_user.username,
            "role": operator_user.role,
        }
    )
    client.cookies.set("access_token", token)
    return client


@pytest_asyncio.fixture
async def mock_vcenter():
    """Parchea get_vcenter_session en cada router para evitar llamadas reales a vCenter."""
    si = MagicMock()
    vcenter_mock = AsyncMock(return_value=si)
    with (
        patch("app.api.v1.vms.get_vcenter_session", new=vcenter_mock),
        patch("app.api.v1.hosts.get_vcenter_session", new=vcenter_mock),
        patch("app.api.v1.datastores.get_vcenter_session", new=vcenter_mock),
        patch("app.api.v1.snapshots.get_vcenter_session", new=vcenter_mock),
    ):
        yield si


@pytest.fixture(autouse=True)
def dev_mode(monkeypatch):
    """Activa DEV_MODE para que todos los tests salten SmartConnect."""
    monkeypatch.setattr("app.auth.router.settings.DEV_MODE", True)
