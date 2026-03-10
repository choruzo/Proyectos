"""Colector de métricas en background con cuenta de servicio vCenter."""
import asyncio
from typing import Optional

from loguru import logger
from pyVim.connect import Disconnect, SmartConnect

from app.config import settings
from app.metrics import service as metrics_service
from app.vcenter import datastores as datastore_service
from app.vcenter import hosts as host_service
from app.vcenter import vms as vm_service


class ServiceAccountCollector:
    def __init__(self, host: str, username: str, password: str, interval_minutes: int = 15):
        self._host = host
        self._username = username
        self._password = password
        self._interval_seconds = interval_minutes * 60
        self._si = None
        self._task: Optional[asyncio.Task] = None
        self._needs_reconnect = False

    async def connect(self) -> bool:
        """Establece conexión pyVmomi con la cuenta de servicio. Nunca lanza excepciones."""
        try:
            si = await asyncio.to_thread(
                SmartConnect,
                host=self._host,
                user=self._username,
                pwd=self._password,
                disableSslCertValidation=True,
            )
            self._si = si
            self._needs_reconnect = False
            logger.info(f"[Collector] Conectado a vCenter {self._host} como {self._username}")
            return True
        except Exception as exc:
            logger.warning(f"[Collector] Error al conectar a vCenter {self._host}: {exc}")
            return False

    async def collect_once(self) -> bool:
        """Recopila métricas del vCenter y guarda un snapshot. Nunca lanza excepciones."""
        if self._si is None:
            logger.warning("[Collector] collect_once llamado sin ServiceInstance activa.")
            return False
        try:
            vms, hosts, datastores = await asyncio.gather(
                asyncio.to_thread(vm_service.list_vms, self._si),
                asyncio.to_thread(host_service.list_hosts, self._si),
                asyncio.to_thread(datastore_service.list_datastores, self._si),
            )
        except Exception as exc:
            logger.warning(f"[Collector] Error al recopilar datos de vCenter: {exc}")
            self._needs_reconnect = True
            return False

        vms_on = sum(1 for v in vms if v["power_state"] == "poweredOn")
        hosts_connected = sum(
            1 for h in hosts
            if h["connection_state"] == "connected" and not h["in_maintenance"]
        )
        hosts_maint = sum(1 for h in hosts if h["in_maintenance"])

        total_cap = sum(d["capacity_gb"] for d in datastores)
        total_free = sum(d["free_gb"] for d in datastores)
        ds_used_pct = round((1 - total_free / total_cap) * 100, 1) if total_cap > 0 else 0.0

        cpu_total = sum(h["cpu_total_mhz"] for h in hosts)
        cpu_used = sum(h["cpu_usage_mhz"] for h in hosts)
        cpu_pct = round(cpu_used / cpu_total * 100, 1) if cpu_total > 0 else 0.0

        mem_total = sum(h["mem_total_mb"] for h in hosts)
        mem_used = sum(h["mem_usage_mb"] for h in hosts)
        mem_pct = round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0.0

        snapshot = {
            "vms_total": len(vms),
            "vms_on": vms_on,
            "hosts_total": len(hosts),
            "hosts_connected": hosts_connected,
            "hosts_in_maintenance": hosts_maint,
            "datastores_total_gb": round(total_cap, 1),
            "datastores_free_gb": round(total_free, 1),
            "datastores_used_pct": ds_used_pct,
            "cpu_usage_pct": cpu_pct,
            "mem_usage_pct": mem_pct,
        }

        saved = await metrics_service.save_snapshot(snapshot)
        if saved:
            logger.info(
                f"[Collector] Snapshot guardado — VMs encendidas: {vms_on}, "
                f"hosts conectados: {hosts_connected}"
            )
        return saved

    async def _run(self) -> None:
        ok = await self.connect()
        if not ok:
            logger.warning("[Collector] Conexión inicial fallida — colector desactivado.")
            return

        logger.info(f"[Collector] Activo. Intervalo: {self._interval_seconds // 60} min.")
        while True:
            if self._needs_reconnect:
                if not await self.connect():
                    await asyncio.sleep(self._interval_seconds)
                    continue
            await self.collect_once()
            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                logger.info("[Collector] Tarea cancelada.")
                raise

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="metrics_collector")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._si:
            try:
                await asyncio.to_thread(Disconnect, self._si)
            except Exception as exc:
                logger.warning(f"[Collector] Error al desconectar: {exc}")
            finally:
                self._si = None


collector: Optional[ServiceAccountCollector] = None

if (
    settings.VCENTER_SERVICE_HOST
    and settings.VCENTER_SERVICE_USER
    and settings.VCENTER_SERVICE_PASS
):
    collector = ServiceAccountCollector(
        host=settings.VCENTER_SERVICE_HOST,
        username=settings.VCENTER_SERVICE_USER,
        password=settings.VCENTER_SERVICE_PASS,
        interval_minutes=settings.VCENTER_METRICS_INTERVAL_MINUTES,
    )
    logger.info(
        f"[Collector] Configurado: {settings.VCENTER_SERVICE_HOST} "
        f"@ {settings.VCENTER_METRICS_INTERVAL_MINUTES} min."
    )
else:
    logger.info("[Collector] Vars de servicio no configuradas — colección en background desactivada.")
