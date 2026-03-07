"""Servicio de operaciones sobre hosts ESXi en vCenter."""

from pyVmomi import vim


def _get_host_by_moid(si, moid: str):
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.HostSystem], True
    )
    host = next((h for h in container.view if h._moId == moid), None)
    container.Destroy()
    return host


def list_hosts(si) -> list[dict]:
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.HostSystem], True
    )
    result = []
    for host in container.view:
        summary = host.summary
        hw = summary.hardware
        qs = summary.quickStats
        runtime = summary.runtime

        cpu_total_mhz = (hw.cpuMhz * hw.numCpuCores) if hw else 0
        cpu_usage_mhz = qs.overallCpuUsage or 0
        mem_total_mb = round(hw.memorySize / 1024 / 1024) if hw else 0
        mem_usage_mb = qs.overallMemoryUsage or 0

        vms_on = sum(
            1 for vm in (host.vm or [])
            if vm.runtime.powerState == vim.VirtualMachine.PowerState.poweredOn
        )

        result.append({
            "id": host._moId,
            "name": summary.config.name,
            "connection_state": str(runtime.connectionState),
            "in_maintenance": runtime.inMaintenanceMode,
            "esxi_version": summary.config.product.version if summary.config.product else None,
            "num_cpu_cores": hw.numCpuCores if hw else None,
            "cpu_total_mhz": cpu_total_mhz,
            "cpu_usage_mhz": cpu_usage_mhz,
            "cpu_usage_pct": round(cpu_usage_mhz / cpu_total_mhz * 100, 1) if cpu_total_mhz else 0,
            "mem_total_mb": mem_total_mb,
            "mem_usage_mb": mem_usage_mb,
            "mem_usage_pct": round(mem_usage_mb / mem_total_mb * 100, 1) if mem_total_mb else 0,
            "vms_on": vms_on,
            "vms_total": len(host.vm or []),
        })
    container.Destroy()
    return result


def enter_maintenance(si, host_id: str) -> None:
    host = _get_host_by_moid(si, host_id)
    if host is None:
        raise ValueError(f"Host {host_id} no encontrado")
    host.EnterMaintenanceMode_Task(timeout=0, evacuatePoweredOffVms=True)


def exit_maintenance(si, host_id: str) -> None:
    host = _get_host_by_moid(si, host_id)
    if host is None:
        raise ValueError(f"Host {host_id} no encontrado")
    host.ExitMaintenanceMode_Task(timeout=0)
