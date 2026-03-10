"""Tests unitarios para el colector de métricas en background."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.metrics.collector import ServiceAccountCollector


@pytest.fixture
def collector():
    return ServiceAccountCollector(
        host="vcenter.test",
        username="svc-user@vsphere.local",
        password="secret",
        interval_minutes=15,
    )


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------

async def test_connect_success(collector):
    mock_si = MagicMock()
    with patch("app.metrics.collector.SmartConnect", return_value=mock_si) as mock_sc:
        result = await collector.connect()

    assert result is True
    assert collector._si is mock_si
    assert collector._needs_reconnect is False


async def test_connect_failure(collector):
    with patch("app.metrics.collector.SmartConnect", side_effect=Exception("timeout")):
        result = await collector.connect()

    assert result is False
    assert collector._si is None


# ---------------------------------------------------------------------------
# collect_once()
# ---------------------------------------------------------------------------

async def test_collect_once_no_si(collector):
    """Sin ServiceInstance activa debe retornar False sin explotar."""
    result = await collector.collect_once()
    assert result is False


async def test_collect_once_saves_snapshot(collector):
    collector._si = MagicMock()

    fake_vms = [
        {"power_state": "poweredOn"},
        {"power_state": "poweredOff"},
    ]
    fake_hosts = [
        {
            "connection_state": "connected",
            "in_maintenance": False,
            "cpu_total_mhz": 10000,
            "cpu_usage_mhz": 3000,
            "mem_total_mb": 65536,
            "mem_usage_mb": 32768,
        }
    ]
    fake_datastores = [
        {"capacity_gb": 1000.0, "free_gb": 400.0},
    ]

    with (
        patch("app.metrics.collector.vm_service.list_vms", return_value=fake_vms),
        patch("app.metrics.collector.host_service.list_hosts", return_value=fake_hosts),
        patch("app.metrics.collector.datastore_service.list_datastores", return_value=fake_datastores),
        patch("app.metrics.collector.metrics_service.save_snapshot", new_callable=AsyncMock, return_value=True) as mock_save,
    ):
        result = await collector.collect_once()

    assert result is True
    mock_save.assert_awaited_once()
    saved = mock_save.call_args[0][0]
    assert saved["vms_on"] == 1
    assert saved["vms_total"] == 2
    assert saved["hosts_connected"] == 1
    assert saved["cpu_usage_pct"] == 30.0
    assert saved["mem_usage_pct"] == 50.0
    assert saved["datastores_used_pct"] == 60.0


async def test_collect_once_vcenter_error_sets_reconnect(collector):
    collector._si = MagicMock()

    with patch("app.metrics.collector.vm_service.list_vms", side_effect=Exception("connection reset")):
        result = await collector.collect_once()

    assert result is False
    assert collector._needs_reconnect is True


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------

async def test_start_stop_cycle(collector):
    """start() crea la tarea, stop() la cancela y llama Disconnect."""
    mock_si = MagicMock()

    async def fake_run():
        await asyncio.sleep(9999)

    with (
        patch.object(collector, "_run", side_effect=fake_run),
        patch("app.metrics.collector.Disconnect") as mock_disconnect,
    ):
        collector._si = mock_si
        collector.start()

        assert collector._task is not None
        assert not collector._task.done()

        await collector.stop()

        assert collector._task.done()
        assert collector._si is None
        mock_disconnect.assert_called_once_with(mock_si)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_singleton_none_when_no_env():
    """El singleton debe ser None cuando las vars de entorno no están configuradas."""
    with patch("app.metrics.collector.settings") as mock_settings:
        mock_settings.VCENTER_SERVICE_HOST = None
        mock_settings.VCENTER_SERVICE_USER = None
        mock_settings.VCENTER_SERVICE_PASS = None

        # Reimportar el módulo para evaluar el bloque de singleton
        import importlib
        import app.metrics.collector as mod
        importlib.reload(mod)

        assert mod.collector is None
