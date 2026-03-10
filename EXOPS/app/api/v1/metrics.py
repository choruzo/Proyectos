"""Endpoints de métricas: KPIs en tiempo real + historial de la caché semanal."""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.service import get_current_user
from app.metrics import service as metrics_service
from app.vcenter import datastores as datastore_service
from app.vcenter import hosts as host_service
from app.vcenter import vms as vm_service
from app.vcenter.connection import get_vcenter_session

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("/kpis")
async def get_kpis(current_user=Depends(get_current_user)):
    """Obtiene KPIs actuales del vCenter y guarda un snapshot en la caché."""
    si = await get_vcenter_session(current_user)
    try:
        vms, hosts, datastores = await asyncio.gather(
            asyncio.to_thread(vm_service.list_vms, si),
            asyncio.to_thread(host_service.list_hosts, si),
            asyncio.to_thread(datastore_service.list_datastores, si),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    # Métricas agregadas
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

    # Guardar en caché en background (sin bloquear la respuesta)
    asyncio.ensure_future(metrics_service.save_snapshot(snapshot.copy()))

    return {
        "current": snapshot,
        "vms": vms,
        "hosts": hosts,
        "datastores": datastores,
    }


@router.get("/history")
async def get_history(
    hours: int = 24,
    current_user=Depends(get_current_user),
):
    """Devuelve snapshots históricos del cache JSON. Máximo 7 días (168 h)."""
    snapshots = await metrics_service.get_history(hours)
    return {"snapshots": snapshots, "hours": hours, "count": len(snapshots)}
