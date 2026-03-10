# Plan: Métricas por Host — Análisis de Rendimiento

## Objetivo

Crear un sistema de recolección de métricas **individuales por host ESXi** con datos de CPU, RAM, red y disco, manteniendo un histórico de 7 días. El sistema debe escalar automáticamente al añadir nuevos hosts sin ninguna configuración adicional.

---

## Estado actual

- `data/metrics_cache.json` — un único fichero con snapshots **agregados** de todo el clúster.
- El colector ya recorre todos los hosts via `host_service.list_hosts()` que devuelve CPU %, RAM %, VMs, estado.
- `PerformanceManager` de pyVmomi (disponible en `si.content.perfManager`) permite consultar contadores avanzados: CPU ready, balloon memory, IOPS, throughput de red y disco, latencia.

---

## Métricas objetivo por host

| Categoría | Métrica | Fuente pyVmomi |
|---|---|---|
| **CPU** | `cpu_usage_pct` | `quickStats.overallCpuUsage / (cpuMhz * numCpuCores)` |
| **CPU** | `cpu_ready_pct` | `PerformanceManager` — counter `cpu.ready.summation` |
| **RAM** | `mem_usage_pct` | `quickStats.overallMemoryUsage / memorySize` |
| **RAM** | `mem_balloon_mb` | `PerformanceManager` — counter `mem.vmmemctl.average` |
| **RAM** | `mem_swap_mb` | `PerformanceManager` — counter `mem.swapused.average` |
| **Red** | `net_rx_kbps` | `PerformanceManager` — counter `net.bytesRx.average` |
| **Red** | `net_tx_kbps` | `PerformanceManager` — counter `net.bytesTx.average` |
| **Red** | `net_dropped_rx` | `PerformanceManager` — counter `net.droppedRx.summation` |
| **Red** | `net_dropped_tx` | `PerformanceManager` — counter `net.droppedTx.summation` |
| **Disco** | `disk_read_kbps` | `PerformanceManager` — counter `disk.read.average` |
| **Disco** | `disk_write_kbps` | `PerformanceManager` — counter `disk.write.average` |
| **Disco** | `disk_read_iops` | `PerformanceManager` — counter `disk.numberReadAveraged.average` |
| **Disco** | `disk_write_iops` | `PerformanceManager` — counter `disk.numberWriteAveraged.average` |
| **Disco** | `disk_latency_ms` | `PerformanceManager` — counter `disk.totalLatency.average` |
| **Sistema** | `vms_on`, `vms_total` | `host.vm` |
| **Sistema** | `uptime_seconds` | `quickStats.uptime` |

---

## Estructura de ficheros resultante

```
data/
├── metrics_cache.json              ← ya existe (métricas agregadas del clúster)
└── host_metrics/
    ├── host-1.json                 ← un fichero por host (clave: MOID del host)
    ├── host-2.json
    └── host-3.json                 ← se crea automáticamente al detectar nuevos hosts
```

Cada fichero `host-{moid}.json`:
```json
{
  "host_id": "host-1",
  "host_name": "esxi01.local",
  "snapshots": [
    {
      "timestamp": "2026-03-10T14:30:00+00:00",
      "cpu_usage_pct": 21.7,
      "cpu_ready_pct": 0.3,
      "mem_usage_pct": 34.0,
      "mem_balloon_mb": 0,
      "mem_swap_mb": 0,
      "net_rx_kbps": 1240,
      "net_tx_kbps": 890,
      "net_dropped_rx": 0,
      "net_dropped_tx": 0,
      "disk_read_kbps": 512,
      "disk_write_kbps": 256,
      "disk_read_iops": 45,
      "disk_write_iops": 22,
      "disk_latency_ms": 3.2,
      "vms_on": 18,
      "vms_total": 32,
      "uptime_seconds": 1209600
    }
  ]
}
```

---

## Plan por fases

### Fase 1 — Infraestructura de caché por host
**Archivos nuevos:** `app/metrics/host_cache_service.py`

- Misma estructura que `service.py` pero orientado a ficheros por host.
- Funciones:
  - `save_host_snapshot(host_id, host_name, snapshot)` — crea/actualiza `data/host_metrics/{host_id}.json`
  - `get_host_history(host_id, hours)` — devuelve snapshots del fichero del host
  - `list_monitored_hosts()` — escanea `data/host_metrics/` y devuelve lista de `{host_id, host_name}`
- Mismas garantías: `asyncio.Lock` por host, ventana 7 días, mínimo 5 min entre snapshots.
- El lock debe ser **por fichero** (no un lock global) para no bloquear hosts independientes.
- **Escalado automático:** si el fichero no existe, se crea. Si aparece un nuevo host, se detecta en la siguiente ejecución del colector.

**Tests:** `tests/test_host_cache_service.py`

---

### Fase 2 — Recolección básica (quickStats)
**Archivos modificados:** `app/vcenter/hosts.py`, `app/metrics/collector.py`

- Añadir `uptime_seconds` al dict que ya devuelve `list_hosts()` (`quickStats.uptime`).
- En `collector.py`, después de `collect_once()` (que guarda el snapshot global), iterar sobre cada host y llamar `host_cache_service.save_host_snapshot()` con los datos que ya se tienen: `cpu_usage_pct`, `mem_usage_pct`, `vms_on`, `vms_total`, `uptime_seconds`.
- Sin cambios en la frecuencia ni en el `ServiceInstance`.
- Esto da historial básico funcional antes de añadir PerformanceManager.

**Tests:** ampliar `tests/test_collector.py`

---

### Fase 3 — Métricas avanzadas con PerformanceManager
**Archivos nuevos:** `app/vcenter/host_perf.py`

- Nueva función `get_host_perf_metrics(si, host_moid) -> dict` que consulta `si.content.perfManager`.
- Proceso:
  1. Construir mapa `{counter_key -> counter_id}` una vez por sesión (cacheable).
  2. Llamar `QueryStats()` para el host con los contadores listados en la tabla de métricas objetivo.
  3. Devolver dict con los valores o `None` si el contador no está disponible.
- La función debe ser tolerante a fallos: si un contador no existe en el entorno, devuelve `None` para ese campo (no lanza excepción).
- Llamada desde `collector.py` con `asyncio.to_thread()`.

> **Nota:** QueryStats tiene coste. Se ejecuta en el mismo intervalo que el colector (15 min por defecto) para no sobrecargar vCenter.

**Tests:** `tests/test_host_perf.py` con mocks de pyVmomi

---

### Fase 4 — Endpoints API
**Archivos nuevos:** `app/api/v1/host_metrics.py`

Endpoints a crear:

```
GET /api/v1/metrics/hosts/
    → Lista de hosts monitorizados con último snapshot disponible
    → Útil para saber qué hosts tienen datos

GET /api/v1/metrics/hosts/{host_id}/history?hours=24
    → Historial de snapshots del host en las últimas N horas (máx 168)

GET /api/v1/metrics/hosts/compare?hours=24
    → Todos los hosts con sus series temporales para comparativa
    → Respuesta: { "hosts": [ { "host_id": ..., "snapshots": [...] } ] }
```

- Todos protegidos con `get_current_user`.
- Registrar en `main.py` el nuevo router.

**Tests:** `tests/test_host_metrics_api.py`

---

### Fase 5 — Frontend de análisis
**Archivos nuevos:** `templates/host_analytics.html`, `static/js/host_analytics.js`

- Nueva página accesible desde el menú lateral como "Rendimiento / Hosts".
- Selector de host (dropdown) + selector de rango temporal (24h / 48h / 7d).
- Gráficas con Chart.js (ya permitido por CSP):
  - CPU usage % + CPU ready % en el mismo eje.
  - RAM usage % + balloon MB.
  - Red: RX/TX kbps en un gráfico de área.
  - Disco: IOPS lectura/escritura + latencia media.
- Modo comparativa: mostrar todos los hosts en el mismo gráfico (líneas de colores distintos).
- Auto-refresh opcional (30 s).

---

## Consideraciones de escalado

- **Nuevo host detectado:** al siguiente ciclo del colector, `list_hosts()` lo devolverá y se creará su fichero JSON automáticamente. Cero configuración.
- **Host eliminado/desconectado:** su fichero JSON permanece con los datos históricos. Los snapshots caducan a los 7 días naturalmente. Se puede añadir un comando de limpieza manual en el futuro.
- **Múltiples hosts en paralelo:** `asyncio.gather()` para llamar `get_host_perf_metrics()` en paralelo por cada host, reduciendo el tiempo total de recolección.
- **Locks independientes por host:** un `dict[host_id -> asyncio.Lock]` en `host_cache_service.py` evita contención entre hosts.

---

## Orden de implementación recomendado

```
Fase 1 → Fase 2 → Tests básicos
                ↓
            Fase 3 → Tests con mocks
                ↓
            Fase 4 → Tests de API
                ↓
            Fase 5 (opcional, al final)
```

Las fases 1-4 son independientes del frontend y aportan valor desde el primer día como API.

---

## Estimación de tamaño de datos

Con 3 hosts y snapshots cada 15 min durante 7 días:
- Snapshots por host: `7 días × 96 snapshots/día = 672 snapshots`
- Tamaño por snapshot: ~500 bytes JSON
- Por host: ~336 KB
- Total 3 hosts: ~1 MB

Perfectamente manejable con JSON en disco. Si el número de hosts crece a 20+, evaluar migrar a SQLite con tabla `host_metrics(host_id, timestamp, ...)`.
