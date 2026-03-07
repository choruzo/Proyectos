"""Servicio de operaciones sobre datastores en vCenter."""

from pyVmomi import vim


def list_datastores(si) -> list[dict]:
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.Datastore], True
    )
    result = []
    for ds in container.view:
        summary = ds.summary
        capacity = summary.capacity or 0
        free = summary.freeSpace or 0

        capacity_gb = round(capacity / 1024**3, 1) if capacity else 0
        free_gb = round(free / 1024**3, 1) if free else 0
        used_pct = round((1 - free / capacity) * 100, 1) if capacity > 0 else 0

        result.append({
            "id": ds._moId,
            "name": summary.name,
            "type": summary.type,
            "capacity_gb": capacity_gb,
            "free_gb": free_gb,
            "used_pct": used_pct,
            "accessible": summary.accessible,
            "vm_count": len(ds.vm or []),
        })
    container.Destroy()
    result.sort(key=lambda x: x["used_pct"], reverse=True)
    return result
