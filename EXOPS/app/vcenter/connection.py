from typing import Optional

from fastapi import Depends, HTTPException, status
from loguru import logger
from pyVim.connect import Disconnect, SmartConnect

# Pool en memoria: user_id → ServiceInstance
_pool: dict[int, object] = {}


def connect(user_id: int, host: str, username: str, password: str) -> object:
    """Crea conexión pyVmomi y la guarda en el pool."""
    try:
        si = SmartConnect(host=host, user=username, pwd=password, disableSslCertValidation=True)
        _pool[user_id] = si
        logger.info(f"vCenter conectado para user_id={user_id} host={host}")
        return si
    except Exception as exc:
        logger.warning(f"Error conectando a vCenter host={host}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No se pudo conectar al vCenter: {exc}",
        )


def get_connection(user_id: int) -> Optional[object]:
    """Devuelve la ServiceInstance del pool o None."""
    return _pool.get(user_id)


def disconnect(user_id: int) -> None:
    """Desconecta del vCenter y elimina del pool."""
    si = _pool.pop(user_id, None)
    if si is not None:
        try:
            Disconnect(si)
            logger.info(f"vCenter desconectado para user_id={user_id}")
        except Exception as exc:
            logger.warning(f"Error al desconectar vCenter user_id={user_id}: {exc}")


async def get_vcenter_session(current_user=None):
    """
    Dependencia FastAPI: obtiene la ServiceInstance del pool.
    Lanza 401 si no hay sesión activa.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autenticado",
        )
    si = get_connection(current_user.id)
    if si is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión vCenter no activa. Vuelve a hacer login.",
        )
    return si
