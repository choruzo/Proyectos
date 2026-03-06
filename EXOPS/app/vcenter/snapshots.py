"""Servicio de operaciones sobre snapshots de VMs en vCenter."""


async def list_snapshots(si, vm_id: str):
    pass


async def create_snapshot(si, vm_id: str, name: str, description: str = ""):
    pass


async def restore_snapshot(si, vm_id: str, snapshot_id: str):
    pass


async def delete_snapshot(si, vm_id: str, snapshot_id: str):
    pass
