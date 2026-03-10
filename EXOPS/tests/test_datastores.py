"""Tests de integración para el endpoint de datastores (vCenter mockeado)."""
from unittest.mock import patch

import pytest


FAKE_DATASTORES = [
    {"id": "ds-1", "name": "datastore-A", "capacity_gb": 2000.0, "free_gb": 500.0, "used_pct": 75.0},
    {"id": "ds-2", "name": "datastore-B", "capacity_gb": 1000.0, "free_gb": 800.0, "used_pct": 20.0},
]


async def test_list_datastores(admin_client, mock_vcenter):
    with patch("app.api.v1.datastores.datastore_service.list_datastores", return_value=FAKE_DATASTORES):
        resp = await admin_client.get("/api/v1/datastores/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2


async def test_list_datastores_unauthenticated(client):
    resp = await client.get("/api/v1/datastores/")
    assert resp.status_code == 401
