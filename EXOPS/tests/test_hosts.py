"""Tests de integración para el endpoint de hosts (vCenter mockeado)."""
from unittest.mock import patch

import pytest


FAKE_HOSTS = [
    {"id": "host-1", "name": "esx01.local", "state": "connected", "cpu_usage_pct": 20.5, "memory_usage_pct": 60.0},
]


async def test_list_hosts(admin_client, mock_vcenter):
    with patch("app.api.v1.hosts.host_service.list_hosts", return_value=FAKE_HOSTS):
        resp = await admin_client.get("/api/v1/hosts/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["name"] == "esx01.local"


async def test_enter_maintenance(admin_client, mock_vcenter):
    with patch("app.api.v1.hosts.host_service.enter_maintenance", return_value=None):
        resp = await admin_client.post("/api/v1/hosts/host-1/maintenance", json={"action": "enter"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["action"] == "enter"


async def test_exit_maintenance(admin_client, mock_vcenter):
    with patch("app.api.v1.hosts.host_service.exit_maintenance", return_value=None):
        resp = await admin_client.post("/api/v1/hosts/host-1/maintenance", json={"action": "exit"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "exit"
