"""Tests de integración para el endpoint de snapshots (vCenter mockeado)."""
from unittest.mock import patch

import pytest


FAKE_SNAPSHOTS = [
    {"id": "snap-1", "name": "antes-del-parche", "description": "", "created": "2026-01-01T10:00:00"},
]

FAKE_SNAP_RESULT = {"id": "snap-2", "name": "nuevo-snap", "description": "test"}


async def test_list_snapshots(admin_client, mock_vcenter):
    with patch("app.api.v1.snapshots.snapshot_service.list_snapshots", return_value=FAKE_SNAPSHOTS):
        resp = await admin_client.get("/api/v1/snapshots/vm-1")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["id"] == "snap-1"


async def test_create_snapshot(admin_client, mock_vcenter):
    with patch("app.api.v1.snapshots.snapshot_service.create_snapshot", return_value=FAKE_SNAP_RESULT):
        resp = await admin_client.post("/api/v1/snapshots/vm-1", json={"name": "nuevo-snap", "description": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "nuevo-snap"


async def test_create_snapshot_missing_name(admin_client, mock_vcenter):
    resp = await admin_client.post("/api/v1/snapshots/vm-1", json={"description": "sin nombre"})
    assert resp.status_code == 422


async def test_restore_snapshot(admin_client, mock_vcenter):
    with patch("app.api.v1.snapshots.snapshot_service.restore_snapshot", return_value={"status": "ok"}):
        resp = await admin_client.post("/api/v1/snapshots/vm-1/snap-1/restore")
    assert resp.status_code == 200


async def test_delete_snapshot(admin_client, mock_vcenter):
    with patch("app.api.v1.snapshots.snapshot_service.delete_snapshot", return_value={"status": "deleted"}):
        resp = await admin_client.delete("/api/v1/snapshots/vm-1/snap-1")
    assert resp.status_code == 200
