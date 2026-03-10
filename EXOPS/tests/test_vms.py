"""Tests de integración para el endpoint de VMs (vCenter mockeado)."""
from unittest.mock import patch, MagicMock

import pytest


FAKE_VMS = [
    {"id": "vm-1", "name": "VM-Test-1", "power_state": "poweredOn", "cpu": 2, "memory_mb": 4096},
    {"id": "vm-2", "name": "VM-Test-2", "power_state": "poweredOff", "cpu": 1, "memory_mb": 2048},
]


async def test_list_vms(admin_client, mock_vcenter):
    with patch("app.api.v1.vms.vm_service.list_vms", return_value=FAKE_VMS):
        resp = await admin_client.get("/api/v1/vms/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "vm-1"


async def test_power_on(admin_client, mock_vcenter):
    with patch("app.api.v1.vms.vm_service.power_on", return_value=None):
        resp = await admin_client.post("/api/v1/vms/vm-1/power", json={"action": "on"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["action"] == "on"


async def test_power_off(admin_client, mock_vcenter):
    with patch("app.api.v1.vms.vm_service.power_off", return_value=None):
        resp = await admin_client.post("/api/v1/vms/vm-1/power", json={"action": "off"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "off"


async def test_power_invalid_action(admin_client, mock_vcenter):
    resp = await admin_client.post("/api/v1/vms/vm-1/power", json={"action": "explode"})
    assert resp.status_code == 422
