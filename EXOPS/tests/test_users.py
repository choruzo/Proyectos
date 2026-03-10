"""Tests de integración para el módulo de usuarios."""
import pytest

from app.auth.models import User
from app.auth.service import hash_password


async def test_list_users_admin(admin_client, admin_user):
    resp = await admin_client.get("/api/v1/users/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(u["username"] == "admin" for u in data)


async def test_list_users_operator_forbidden(operator_client):
    resp = await operator_client.get("/api/v1/users/")
    assert resp.status_code == 403


async def test_create_user(admin_client):
    payload = {
        "username": "newuser",
        "password": "newpass123",
        "role": "operator",
    }
    resp = await admin_client.post("/api/v1/users/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "newuser"
    assert data["role"] == "operator"


async def test_create_user_duplicate(admin_client, admin_user):
    payload = {
        "username": "admin",
        "password": "cualquier",
        "role": "operator",
    }
    resp = await admin_client.post("/api/v1/users/", json=payload)
    assert resp.status_code == 409


async def test_get_user(admin_client, operator_user):
    resp = await admin_client.get(f"/api/v1/users/{operator_user.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "operator"


async def test_update_user(admin_client, operator_user):
    payload = {"email": "op@example.com", "role": "admin"}
    resp = await admin_client.put(f"/api/v1/users/{operator_user.id}", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "op@example.com"
    assert data["role"] == "admin"


async def test_deactivate_user(admin_client, operator_user):
    resp = await admin_client.patch(f"/api/v1/users/{operator_user.id}/deactivate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_active"] is False


async def test_cannot_deactivate_self(admin_client, admin_user):
    resp = await admin_client.patch(f"/api/v1/users/{admin_user.id}/deactivate")
    assert resp.status_code == 400


async def test_delete_user(admin_client, operator_user):
    resp = await admin_client.delete(f"/api/v1/users/{operator_user.id}")
    assert resp.status_code == 204
