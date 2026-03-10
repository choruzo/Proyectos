"""Tests de integración para el módulo de auditoría."""
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import select

from app.audit.models import AuditLog
from app.audit.service import log_action


async def test_log_action_creates_record(db, admin_user):
    entry = await log_action(
        db=db,
        user_id=admin_user.id,
        username=admin_user.username,
        action="test_action",
        result="success",
    )
    assert entry.id is not None
    assert entry.action == "test_action"
    assert entry.username == "admin"


async def test_list_audit_admin_sees_all(admin_client, db, admin_user, operator_user):
    # Crear logs para ambos usuarios directamente en DB
    await log_action(db, user_id=admin_user.id, username="admin", action="login", result="success")
    await log_action(db, user_id=operator_user.id, username="operator", action="login", result="success")

    resp = await admin_client.get("/api/v1/audit/")
    assert resp.status_code == 200
    data = resp.json()
    usernames = {entry["username"] for entry in data}
    assert "admin" in usernames
    assert "operator" in usernames


async def test_list_audit_operator_sees_own(operator_client, db, admin_user, operator_user):
    await log_action(db, user_id=admin_user.id, username="admin", action="login", result="success")
    await log_action(db, user_id=operator_user.id, username="operator", action="login", result="success")

    resp = await operator_client.get("/api/v1/audit/")
    assert resp.status_code == 200
    data = resp.json()
    # El operator solo debe ver sus propios logs
    for entry in data:
        assert entry["user_id"] == operator_user.id


async def test_filter_by_action(admin_client, db, admin_user):
    await log_action(db, user_id=admin_user.id, username="admin", action="login", result="success")
    await log_action(db, user_id=admin_user.id, username="admin", action="logout", result="success")

    resp = await admin_client.get("/api/v1/audit/?action=login")
    assert resp.status_code == 200
    data = resp.json()
    assert all(entry["action"] == "login" for entry in data)


async def test_filter_by_date_range(admin_client, db, admin_user):
    await log_action(db, user_id=admin_user.id, username="admin", action="test", result="success")

    from_date = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
    to_date = (datetime.now(timezone.utc) + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")

    resp = await admin_client.get(f"/api/v1/audit/?from_date={from_date}&to_date={to_date}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1


async def test_pagination_skip_limit(admin_client, db, admin_user):
    for i in range(5):
        await log_action(db, user_id=admin_user.id, username="admin", action=f"action_{i}", result="success")

    resp = await admin_client.get("/api/v1/audit/?skip=0&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) <= 2


async def test_unauthenticated_rejected(client):
    resp = await client.get("/api/v1/audit/")
    assert resp.status_code == 401
