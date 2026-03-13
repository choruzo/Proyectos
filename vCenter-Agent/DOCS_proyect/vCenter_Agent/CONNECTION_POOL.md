# Sistema de Conexiones vCenter

Documentación técnica del `VCenterConnectionPool` y el sistema de fallback a vcsim.

**Archivo:** `src/utils/vcenter_tools.py`

---

## Arquitectura del Connection Pool

```mermaid
classDiagram
    class VCenterConnectionPool {
        -dict _pool
        -RLock _lock
        -int _max_connections
        -int _connection_timeout
        +get_connection(host, user, pwd, logger, config) ServiceInstance
        +release_connection(host, user)
        +cleanup_all()
        -_cleanup_expired_connections()
        -_check_reachable(host, port, timeout) bool
        -_connect_vcsim(fb_config, logger) ServiceInstance
        -_get_fallback_config(config) dict
    }

    class ConnectionEntry {
        +ServiceInstance si
        +float last_used
        +bool in_use
        +bool simulated
    }

    class VcsimManager {
        +ensure_running()
        +stop()
        +status() str
    }

    VCenterConnectionPool "1" --> "0..5" ConnectionEntry : _pool dict
    VCenterConnectionPool --> VcsimManager : usa si auto_start
```

---

## Estructura Interna del Pool

```python
_pool = {
    ("vcenter-host.local", "admin"): {
        "si":        ServiceInstance,  # objeto pyvmomi
        "last_used": 1709123456.789,   # timestamp Unix
        "in_use":    False,            # True = ocupado
        "simulated": False             # True = vcsim
    }
}
```

Clave del pool: tupla `(host, user)` — permite múltiples conexiones a distintos hosts o usuarios vCenter.

---

## Flujo de `get_connection()`

```mermaid
flowchart TD
    START([get_connection llamado]) --> FB_CONFIG[_get_fallback_config\nleer config o config.json]
    FB_CONFIG --> CHECK_MODE{fb_mode?}

    CHECK_MODE -->|force_vcsim| VCSIM_DIRECT[_connect_vcsim\nconexión directa al simulador]
    CHECK_MODE -->|auto o disabled| LOCK[Adquirir _lock RLock]

    LOCK --> CLEANUP[_cleanup_expired_connections\neliminar conexiones expiradas]
    CLEANUP --> EXIST{¿key en _pool?}

    EXIST -->|Sí| VALIDATE{¿sesión activa?\nsi.content.sessionManager\n.currentSession}
    VALIDATE -->|Válida| REUSE[Marcar in_use=True\nActualizar last_used\nRetornar si existente]
    VALIDATE -->|Inválida| REMOVE[Eliminar entrada\ndel pool]
    REMOVE --> AUTO_CHECK

    EXIST -->|No| AUTO_CHECK{fb_mode == auto\nY fb_enabled?}

    AUTO_CHECK -->|Sí| PING[_check_reachable\nTCP ping port 443\ntimeout 3s]
    PING --> REACH{¿Accesible?}
    REACH -->|No| VCSIM[_connect_vcsim\nfallback al simulador]
    REACH -->|Sí| CONNECT

    AUTO_CHECK -->|No| CONNECT[SmartConnect\nhost, user, pwd, port=443]

    CONNECT --> CONN_OK{¿Exitoso?}
    CONN_OK -->|Sí| ADD[Agregar al _pool\nin_use=True\nlast_used=now]
    CONN_OK -->|No| FALLBACK{¿fb_enabled?}
    FALLBACK -->|Sí| VCSIM
    FALLBACK -->|No| NONE[Retornar None]

    ADD --> RETURN([Retornar ServiceInstance])
    REUSE --> RETURN
    VCSIM_DIRECT --> RETURN
    VCSIM --> RETURN
    NONE --> RETURN
```

---

## Limpieza de Conexiones Expiradas

```mermaid
flowchart LR
    TRIGGER[_cleanup_expired_connections] --> ITER[Iterar _pool.items]
    ITER --> EACH{Para cada conexión}
    EACH --> COND{time.time - last_used\n> timeout_30s\nY NOT in_use?}
    COND -->|Sí| DISC[Disconnect si\nEliminar del _pool]
    COND -->|No| KEEP[Mantener conexión]
    DISC --> NEXT[Siguiente]
    KEEP --> NEXT
    NEXT --> EACH
```

**Cuándo se ejecuta:** Al inicio de cada llamada a `get_connection()`, con el lock ya adquirido.

---

## Modos de Fallback a vcsim

```mermaid
stateDiagram-v2
    state "Modo Fallback" as FB {
        [*] --> Deshabilitado: fb_enabled = false (default)
        [*] --> Auto: fb_enabled = true, fb_mode = "auto"
        [*] --> Forzado: fb_enabled = true, fb_mode = "force_vcsim"
    }

    Deshabilitado --> Real: SmartConnect siempre
    Auto --> Real: Si vCenter accesible
    Auto --> Simulador: Si TCP ping falla
    Forzado --> Simulador: Siempre vcsim

    Real: Conexión a vCenter real
    Simulador: Conexión a vcsim (127.0.0.1:8989)
```

### Configuración de fallback en `config.json`

```json
{
  "vcenter_fallback": {
    "enabled": false,
    "mode": "auto",
    "vcsim": {
      "host": "127.0.0.1",
      "port": 8989,
      "user": "user",
      "pwd": "pass",
      "auto_start": true
    }
  }
}
```

---

## Health Check TCP

```mermaid
sequenceDiagram
    participant Pool as ConnectionPool
    participant Socket as socket.create_connection
    participant vC as vCenter host:443

    Pool->>Socket: create_connection(host, 443, timeout=3.0)
    alt Conexión exitosa en <3s
        Socket-->>Pool: socket object
        Pool->>Socket: socket.close()
        Pool-->>Pool: return True (accesible)
    else Timeout o error
        Socket-->>Pool: TimeoutError / OSError
        Pool-->>Pool: return False (no accesible)
        Pool->>Pool: _connect_vcsim()
    end
```

---

## Ciclo de Vida de una Conexión

```mermaid
gantt
    title Ciclo de Vida de una Conexión en el Pool
    dateFormat X
    axisFormat %ss

    section Conexión A
    SmartConnect           :done, a1, 0, 2
    En uso (query 1)       :active, a2, 2, 5
    Libre (in_use=False)   :a3, 5, 35
    Expirada (>30s libre)  :crit, a4, 35, 36
    Disconnect + Eliminar  :crit, a5, 36, 37

    section Conexión B
    SmartConnect           :done, b1, 10, 12
    En uso (query 2)       :active, b2, 12, 15
    Libre                  :b3, 15, 42
```

---

## `release_connection()` — Liberar Conexión

Después de que una herramienta MCP termina su operación, debe liberar la conexión:

```python
def release_connection(self, host: str, user: str):
    key = (host, user)
    with self._lock:
        if key in self._pool:
            self._pool[key]["in_use"] = False
```

Esto permite que el pool gestione la conexión para futuras consultas sin crear una nueva sesión vCenter.

---

## Comparativa: Conexión Real vs vcsim

| Aspecto | vCenter Real | vcsim (simulador) |
|---------|-------------|-------------------|
| Host | `vcenter-host.local` | `127.0.0.1` |
| Puerto | 443 (HTTPS) | 8989 |
| VMs | Reales, producción | Inventario pre-grabado |
| SSL | Verificación deshabilitada | Sin SSL |
| Uso | Producción | Desarrollo / Testing CI |
| `simulated` flag | `False` | `True` |
| Fuente datos | `get_connection()` directo | `_connect_vcsim()` |

---

## Integración con MCPToolRegistry

```mermaid
flowchart TD
    REGISTRY["MCPToolRegistry.__init__\n(config, user_mapping)"] --> POOL_CREATE["VCenterConnectionPool\n(max=5, timeout=30s)"]

    subgraph Tool Execution
        TOOL["Tool closure ejecutada\nlist_vms_for_user()"]
        GET_SI["get_user_si(username)"]
        GET_CONN["_connection_pool.get_connection\n(host, user, pwd, config=self.config)"]
        TOOL --> GET_SI --> GET_CONN
    end

    POOL_CREATE --> GET_CONN
    GET_CONN --> SI[ServiceInstance]
    SI --> PYVM[pyvmomi calls\ncontainer_view, etc.]
```

La instancia del pool se crea una sola vez en `MCPToolRegistry.__init__` y se comparte entre todas las herramientas del mismo usuario.

---

## Parámetros del Pool

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `max_connections` | 5 | Conexiones máximas simultáneas |
| `connection_timeout` | 30s | Tiempo hasta marcar conexión como expirada |
| `tcp_ping_timeout` | 3.0s | Timeout del health check TCP |
| `tcp_ping_port` | 443 | Puerto HTTPS de vCenter |
| `vcsim_port` | 8989 | Puerto del simulador (configurable) |
| `lock_type` | `threading.RLock` | Re-entrant lock, thread-safe |
