"""Servicio de operaciones sobre snapshots de VMs en vCenter (pyVmomi síncrono)."""
from pyVmomi import vim
from pyVim.task import WaitForTask


def _get_vm_by_moid(si, moid: str):
    container = si.content.viewManager.CreateContainerView(
        si.content.rootFolder, [vim.VirtualMachine], True
    )
    vm = next((v for v in container.view if v._moId == moid), None)
    container.Destroy()
    return vm


def _snapshot_tree_to_dict(snap_list, depth=0) -> list[dict]:
    result = []
    for snap in snap_list:
        result.append({
            "id": snap.snapshot._moId,
            "name": snap.name,
            "description": snap.description,
            "created_at": snap.createTime.isoformat() if snap.createTime else None,
            "state": snap.state,
            "depth": depth,
            "children": _snapshot_tree_to_dict(snap.childSnapshotList, depth + 1),
        })
    return result


def _find_snapshot_by_moid(snap_list, snapshot_id: str):
    for snap in snap_list:
        if snap.snapshot._moId == snapshot_id:
            return snap.snapshot
        found = _find_snapshot_by_moid(snap.childSnapshotList, snapshot_id)
        if found:
            return found
    return None


def list_snapshots(si, vm_id: str) -> dict:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    current_snapshot_id = None
    snapshots = []
    if vm.snapshot:
        snapshots = _snapshot_tree_to_dict(vm.snapshot.rootSnapshotList)
        if vm.snapshot.currentSnapshot:
            current_snapshot_id = vm.snapshot.currentSnapshot._moId
    return {
        "vm_name": vm.name,
        "snapshots": snapshots,
        "current_snapshot_id": current_snapshot_id,
    }


def create_snapshot(si, vm_id: str, name: str, description: str = "") -> dict:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    task = vm.CreateSnapshot_Task(name, description, memory=False, quiesce=False)
    WaitForTask(task)
    return {"status": "ok", "vm_id": vm_id, "name": name}


def restore_snapshot(si, vm_id: str, snapshot_id: str) -> dict:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    if not vm.snapshot:
        raise ValueError(f"La VM {vm_id} no tiene snapshots")
    snap = _find_snapshot_by_moid(vm.snapshot.rootSnapshotList, snapshot_id)
    if snap is None:
        raise ValueError(f"Snapshot {snapshot_id} no encontrado")
    task = snap.RevertToSnapshot_Task()
    WaitForTask(task)
    return {"status": "ok", "vm_id": vm_id, "snapshot_id": snapshot_id}


def delete_snapshot(si, vm_id: str, snapshot_id: str) -> dict:
    vm = _get_vm_by_moid(si, vm_id)
    if vm is None:
        raise ValueError(f"VM {vm_id} no encontrada")
    if not vm.snapshot:
        raise ValueError(f"La VM {vm_id} no tiene snapshots")
    snap = _find_snapshot_by_moid(vm.snapshot.rootSnapshotList, snapshot_id)
    if snap is None:
        raise ValueError(f"Snapshot {snapshot_id} no encontrado")
    task = snap.RemoveSnapshot_Task(removeChildren=False)
    WaitForTask(task)
    return {"status": "ok", "vm_id": vm_id, "snapshot_id": snapshot_id}
