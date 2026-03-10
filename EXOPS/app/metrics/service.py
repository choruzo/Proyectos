"""Servicio de caché de métricas históricas (JSON, máximo 7 días)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_FILE = Path("data/metrics_cache.json")
MAX_AGE_DAYS = 7
MIN_INTERVAL_MINUTES = 5

_lock = asyncio.Lock()


def _load_cache_sync() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    try:
        raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return raw.get("snapshots", [])
    except Exception:
        return []


def _save_cache_sync(snapshots: list[dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"snapshots": snapshots}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _prune(snapshots: list[dict]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    result = []
    for s in snapshots:
        try:
            ts = datetime.fromisoformat(s["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                result.append(s)
        except (KeyError, ValueError):
            pass
    return result


async def save_snapshot(snapshot: dict) -> bool:
    """Persiste un snapshot si han pasado al menos MIN_INTERVAL_MINUTES desde el último.

    Devuelve True si se guardó, False si se saltó por proximidad temporal.
    """
    async with _lock:
        snapshots = _load_cache_sync()

        if snapshots:
            try:
                last_ts = datetime.fromisoformat(snapshots[-1]["timestamp"])
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_ts) < timedelta(minutes=MIN_INTERVAL_MINUTES):
                    return False
            except (KeyError, ValueError):
                pass

        snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
        snapshots.append(snapshot)
        snapshots = _prune(snapshots)
        _save_cache_sync(snapshots)
        return True


async def get_history(hours: int = 24) -> list[dict]:
    """Devuelve los snapshots de las últimas N horas (máx 168 = 7 días)."""
    hours = min(hours, 168)
    async with _lock:
        snapshots = _load_cache_sync()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []
    for s in snapshots:
        try:
            ts = datetime.fromisoformat(s["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                result.append(s)
        except (KeyError, ValueError):
            pass
    return result
