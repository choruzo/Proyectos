"""Servicio de operaciones sobre VMs en vCenter (pyVmomi síncrono)."""
from pyVmomi import vim


def _get_vm_by_moid(si, moid: str):
    """Encuentra una VM por su Managed Object ID."""
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.VirtualMachine], True
    )
    vm = next((v for v in container.view if v._moId == moid), None)
    container.Destroy()
    return vm


def list_vms(si) -> list[dict]:
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.VirtualMachine], True
    )
    result = []
    for vm in container.view:
        summary = vm.summary
        config = summary.config
        runtime = summary.runtime
        guest = summary.guest
        result.append({
            "id": vm._moId,
            "name": config.name,
            "power_state": runtime.powerState,
            "num_cpu": config.numCpu,
            "memory_mb": config.memorySizeMB,
            "ip_address": guest.ipAddress if guest else None,
            "host_name": runtime.host.name if runtime.host else None,
            "guest_os": config.guestFullName,
        })
    container.Destroy()
    return result


def power_on(si, vm_id: str) -> None:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    vm.PowerOnVM_Task()


def power_off(si, vm_id: str) -> None:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    vm.PowerOffVM_Task()


def reboot(si, vm_id: str) -> None:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    try:
        vm.RebootGuest()
    except Exception:
        vm.ResetVM_Task()
