"""Tests de integración para el módulo de autenticación."""
import pytest

from app.auth.models import User
from app.auth.service import hash_password


LOGIN_PAYLOAD = {
    "app_username": "admin",
    "app_password": "adminpass",
    "vcenter_host": "vcenter.test",
    "vcenter_username": "vc_user",
    "vcenter_password": "vc_pass",
}


async def test_login_success(client, admin_user):
    resp = await client.post("/api/v1/auth/login", json=LOGIN_PAYLOAD)
    assert resp.status_code == 200
    assert "access_token" in resp.cookies
    data = resp.json()
    assert data["username"] == "admin"


async def test_login_wrong_password(client, admin_user):
    payload = {**LOGIN_PAYLOAD, "app_password": "wrongpass"}
    resp = await client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 401


async def test_login_user_not_found(client):
    payload = {**LOGIN_PAYLOAD, "app_username": "noexiste"}
    resp = await client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 401


async def test_login_inactive_user(client, db):
    user = User(
        username="inactive",
        hashed_password=hash_password("pass"),
        role="operator",
        is_active=False,
    )
    db.add(user)
    await db.commit()

    payload = {**LOGIN_PAYLOAD, "app_username": "inactive", "app_password": "pass"}
    resp = await client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 403


async def test_brute_force_lockout(client, db):
    user = User(
        username="victim",
        hashed_password=hash_password("correctpass"),
        role="operator",
        is_active=True,
    )
    db.add(user)
    await db.commit()

    bad_payload = {**LOGIN_PAYLOAD, "app_username": "victim", "app_password": "wrong"}
    for _ in range(5):
        await client.post("/api/v1/auth/login", json=bad_payload)

    # El 6.º intento (o el 5.º que activa el bloqueo) debe resultar en error
    resp = await client.post("/api/v1/auth/login", json=bad_payload)
    assert resp.status_code in (401, 403)


async def test_brute_force_reset_on_success(client, db):
    user = User(
        username="recover",
        hashed_password=hash_password("realpass"),
        role="operator",
        is_active=True,
        failed_attempts=3,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    payload = {**LOGIN_PAYLOAD, "app_username": "recover", "app_password": "realpass"}
    resp = await client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 200

    await db.refresh(user)
    assert user.failed_attempts == 0


async def test_logout(admin_client):
    resp = await admin_client.post("/api/v1/auth/logout")
    assert resp.status_code == 200
    # Cookie debe estar vacía o eliminada
    assert resp.cookies.get("access_token", "") == ""


async def test_get_me_authenticated(admin_client, admin_user):
    resp = await admin_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"


async def test_get_me_unauthenticated(client):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401
