# API Reference — vCenter Multi-Agent System

**Versión:** 3.0
**Fecha:** 2026-02-25
**Base URL:** `http://<host>:9100`

---

## Índice

1. [Autenticación y Sesiones](#1-autenticación-y-sesiones)
2. [Interfaces de Usuario](#2-interfaces-de-usuario)
3. [Chat (Core)](#3-chat-core)
4. [Gestión de Usuarios](#4-gestión-de-usuarios-superuser)
5. [Estadísticas del Sistema](#5-estadísticas-del-sistema-superuser)
6. [Descargas](#6-descargas)
7. [Infraestructura vCenter](#7-infraestructura-vcenter)
8. [Monitorización ESXi](#8-monitorización-esxi)
9. [Herramientas MCP (Agentes LLM)](#9-herramientas-mcp-agentes-llm)
10. [Roles y Permisos](#10-roles-y-permisos)
11. [Variables de Entorno](#11-variables-de-entorno)
12. [Códigos de Error Comunes](#12-códigos-de-error-comunes)

---

## 1. Autenticación y Sesiones

### 1.1 Login

```
POST /login
```

Autentica al usuario y crea sesión. No requiere autenticación previa.

**Body (form-data):**

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `username` | string | Sí | Nombre de usuario (case-insensitive) |
| `password` | string | Sí | Contraseña en texto plano |

**Respuesta:** Redirección `302` según rol:
- `superuser` → `/admin/stats`
- `admin` → `/admin`
- `user` → `/app`

**Errores:**
- `401` — Credenciales incorrectas
- `429` — Demasiados intentos fallidos (bloqueo 5 min)

---

### 1.2 Logout

```
GET /logout
POST /logout
```

Cierra sesión del usuario. Requiere sesión activa.

**Respuesta:** Redirección `302` a `/`.

---

### 1.3 Health Check

```
GET /health
```

Verifica el estado del orquestador. No requiere autenticación.

**Respuesta `200`:**
```json
{
  "status": "ok",
  "orchestrator": true,
  "agents": ["vcenter", "documentation"]
}
```

---

## 2. Interfaces de Usuario

Todos estos endpoints devuelven HTML. Requieren sesión activa (se redirige a `/` si no autenticado).

| Endpoint | Método | Rol mínimo | Descripción |
|----------|--------|------------|-------------|
| `/` | GET | — | Página de login |
| `/app` | GET | `user` | Chat principal |
| `/admin` | GET | `admin` | Dashboard administrativo |
| `/monitoring` | GET | `admin` | Dashboard de monitorización ESXi |
| `/admin/stats` | GET | `superuser` | Estadísticas del sistema (RAG, sesiones) |
| `/admin/users` | GET | `superuser` | Gestión de usuarios |

---

## 3. Chat (Core)

### 3.1 Chat con Streaming SSE *(Recomendado)*

```
POST /chat/stream
```

Envía un mensaje y recibe la respuesta como stream de eventos SSE. Requiere sesión activa.

**Headers:**
```
Content-Type: application/json
```

**Body JSON:**
```json
{
  "message": "¿Cuántas VMs tiene el host esxi8-135?"
}
```

**Respuesta:** `text/event-stream` con los siguientes eventos en orden:

#### Evento `routing`
Indica qué agente procesará la consulta.
```
event: routing
data: {"agent": "vcenter", "label": "Consultando agente vCenter..."}
```

Valores posibles de `agent`: `vcenter`, `documentation`, `general`

#### Evento `heartbeat`
Enviado cada 2 segundos mientras el agente procesa.
```
event: heartbeat
data: {"ts": 1740492000.5, "elapsed": 4.2}
```

#### Evento `token`
Un fragmento de la respuesta del LLM (texto plano, sin Markdown).
```
event: token
data: {"t": "El host esxi8-135 tiene ", "i": 0}
```

#### Evento `done`
Fin del stream. La respuesta completa ya fue enviada como tokens.
```
event: done
data: {"agent": "vcenter", "attachments": [], "total_tokens": 47}
```

Cuando hay archivos descargables:
```json
{
  "agent": "vcenter",
  "attachments": [
    {"filename": "reporte_recursos.zip", "url": "/api/download/abc123token"}
  ],
  "total_tokens": 89
}
```

#### Evento `error`
Enviado en lugar de `done` si ocurre un fallo.
```
event: error
data: {"error": "Sesión expirada", "code": 401}
```

**Errores HTTP:**
- `400` — Mensaje vacío
- `401` — Sesión expirada o inválida

**Notas de implementación:**
- Se usa `fetch()` + `response.body.getReader()` (NO `EventSource` nativo, ya que solo soporta GET)
- Los tokens llegan como texto plano; el Markdown se renderiza en el cliente solo al recibir `done`
- El **sticky routing** activa enrutamiento automático al mismo agente durante 180 s si el mensaje es corto (≤ 15 palabras) y contiene números/valores de seguimiento (p.ej. "30 días", "datastore_35")

---

### 3.2 Chat Legacy *(Fallback)*

```
POST /chat
```

Envía un mensaje y recibe la respuesta completa en JSON. El frontend usa este endpoint automáticamente si SSE falla. Requiere sesión activa.

**Body JSON:**
```json
{
  "message": "Lista mis VMs"
}
```

**Respuesta `200`:**
```json
{
  "agent": "vcenter",
  "username": "jamb",
  "response": "Tus VMs son:\n- MCU_P27_JaMB\n- EqSIM_P27_JaMB",
  "attachments": [
    {
      "filename": "reporte.csv",
      "url": "/api/download/tok_abc123"
    }
  ]
}
```

**Errores:**
- `400` — Mensaje vacío
- `401` — Sesión expirada

---

## 4. Gestión de Usuarios *(superuser)*

Todos los endpoints requieren rol `superuser`.

### 4.1 Listar Usuarios

```
GET /api/admin/users
```

**Respuesta `200`:**
```json
{
  "users": [
    {
      "id": 1,
      "username": "jamb",
      "abbr": "JaMB",
      "is_active": 1,
      "is_admin": 0,
      "role": "user",
      "created_at": "2025-10-24T10:00:00",
      "updated_at": "2025-10-24T10:00:00"
    }
  ]
}
```

---

### 4.2 Crear Usuario

```
POST /api/admin/users
```

**Body JSON:**
```json
{
  "username": "jdoe",
  "password": "secreto123",
  "abbr": "JDoe",
  "role": "user"
}
```

| Campo | Tipo | Requerido | Valores |
|-------|------|-----------|---------|
| `username` | string | Sí | — |
| `password` | string | Sí | — |
| `abbr` | string | Sí | Abreviatura para nombres de VM |
| `role` | string | No | `user` (default) / `admin` / `superuser` |

**Respuesta `201`:**
```json
{"ok": true, "username": "jdoe"}
```

**Errores:**
- `400` — Parámetros inválidos o rol no reconocido
- `409` — El usuario ya existe

---

### 4.3 Actualizar Usuario

```
PUT /api/admin/users/<target_username>
```

**Body JSON** (todos los campos son opcionales):
```json
{
  "role": "admin",
  "abbr": "JDoe2",
  "is_active": true
}
```

**Respuesta `200`:**
```json
{"ok": true}
```

**Errores:**
- `400` — Sin cambios o superuser intenta desactivarse a sí mismo
- `404` — Usuario no encontrado

---

### 4.4 Eliminar Usuario

```
DELETE /api/admin/users/<target_username>
```

**Query params:**

| Param | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `hard` | string | `false` | `true` para eliminación permanente; `false` para soft-delete |

**Respuesta `200`:**
```json
{"ok": true, "action": "soft_delete"}
```

**Errores:**
- `400` — Auto-eliminación / intento de eliminar el último superuser activo
- `404` — Usuario no encontrado

---

### 4.5 Resetear Contraseña

```
POST /api/admin/users/<target_username>/reset-password
```

**Body JSON:**
```json
{"password": "nuevaClave456"}
```

**Respuesta `200`:**
```json
{"ok": true}
```

**Errores:**
- `400` — Contraseña vacía
- `404` — Usuario no encontrado

---

## 5. Estadísticas del Sistema *(superuser)*

### 5.1 Estadísticas de Agentes

```
GET /api/stats/agents
```

**Query params opcionales:** `start_date`, `end_date` (ISO 8601, p.ej. `2025-10-01T00:00:00Z`)

**Respuesta `200`:**
```json
{
  "vcenter": {
    "total_queries": 42,
    "avg_response_time_ms": 1234,
    "error_rate": 0.05
  },
  "documentation": {
    "total_queries": 28,
    "cache_hit_rate": 0.35,
    "avg_response_time_ms": 456
  }
}
```

---

### 5.2 Estadísticas del Sistema

```
GET /api/stats/system
```

**Query params opcionales:** `start_date`, `end_date`

**Respuesta `200`:**
```json
{
  "cpu_usage_percent": 45.2,
  "memory_usage_percent": 62.1,
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 5.3 Estadísticas de Sesiones

```
GET /api/stats/sessions
```

**Respuesta `200`:**
```json
{
  "active_sessions": 5,
  "session_timeout_seconds": 3600,
  "connection_pool": {
    "active_connections": 2,
    "max_connections": 5,
    "idle_connections": 3
  }
}
```

---

## 6. Descargas

### 6.1 Descargar Archivo Temporal

```
GET /api/download/<token>
```

Descarga un archivo generado por el agente vCenter (CSV, ZIP). No requiere sesión, solo el token válido.

**Respuesta `200`:** Archivo adjunto (`Content-Disposition: attachment`)

**Errores:**
- `404` — Token inválido o expirado (tokens válidos ~30 min)
- `410` — Archivo no disponible

---

## 7. Infraestructura vCenter

Requieren sesión activa (rol mínimo `user`).

### 7.1 Estado de Hosts y Datastores

```
GET /api/hosts_status
```

**Respuesta `200`:**
```json
{
  "timestamp": "2025-10-24T12:00:00Z",
  "user": "jamb",
  "hosts": [
    {
      "name": "esxi8-135",
      "cpu_usage_pct": 35.2,
      "memory_gb": "256/512",
      "connection_state": "connected",
      "power_state": "poweredOn"
    }
  ],
  "datastores": [
    {
      "name": "datastore_01",
      "capacity_gb": 1024,
      "free_gb": 512,
      "type": "VMFS"
    }
  ],
  "summary": {
    "total_hosts": 2,
    "total_datastores": 3
  }
}
```

---

### 7.2 Estado de Hosts (Dashboard)

```
GET /api/host-status
```

Versión formateada para dashboards de monitorización.

**Respuesta `200`:**
```json
{
  "status": "success",
  "timestamp": "2025-10-24T12:00:00Z",
  "user": "jamb",
  "hosts": [
    {
      "name": "esxi8-135",
      "cpu_usage": "35.2%",
      "memory_usage": "256GB / 512GB",
      "connection_state": "connected",
      "status": "poweredOn"
    }
  ],
  "total_hosts": 2
}
```

---

### 7.3 Estadísticas del Orquestador

```
GET /api/system-stats
```

**Respuesta `200`:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-24T12:00:00Z",
  "vcenter_connection": "connected",
  "orchestrator": {
    "active_sessions": 3,
    "session_timeout_seconds": 3600,
    "agents_loaded": ["vcenter", "documentation"],
    "total_agents": 2
  },
  "security_features": {
    "password_hashing": "bcrypt",
    "session_management": "enabled",
    "authentication": "required",
    "session_validation": "enabled"
  },
  "configuration": {
    "llm_model": "gpt-oss:20b",
    "config_loaded": true
  }
}
```

---

### 7.4 Lista de Datastores

```
GET /api/datastores
```

**Respuesta `200`:**
```json
{
  "status": "success",
  "datastores": [
    {
      "name": "datastore_01",
      "capacity": 1099511627776,
      "freeSpace": 549755813888,
      "type": "VMFS",
      "accessible": true
    }
  ],
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 7.5 Listar Archivos en Datastore

```
GET /api/datastore/list
```

**Query params:**

| Param | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `datastore` | string | Sí | Nombre del datastore |
| `path` | string | No | Ruta a navegar (default `/`) |

**Respuesta `200`:**
```json
{
  "status": "success",
  "datastore": "datastore_01",
  "path": "/VMs/",
  "files": [
    {
      "name": "vm-001.vmx",
      "type": "file",
      "size": 2458,
      "modified": "2024-09-25T09:20:00",
      "path": "/VMs/vm-001.vmx"
    },
    {
      "name": "templates",
      "type": "folder",
      "size": "",
      "modified": "2024-09-20T14:30:00",
      "path": "/VMs/templates/"
    }
  ],
  "timestamp": "2025-10-24T12:00:00Z"
}
```

**Errores:**
- `400` — Parámetro `datastore` no proporcionado

---

## 8. Monitorización ESXi

Requieren sesión activa (rol mínimo `user`). `<host_id>` es el identificador del host (p.ej. `esxi8-135`).

### 8.1 Hosts Configurados

```
GET /api/esxi/hosts
```

**Respuesta `200`:**
```json
{
  "status": "success",
  "hosts": [
    {"id": "esxi8-135", "host": "192.168.1.135", "user": "root", "enabled": true}
  ],
  "total_hosts": 2,
  "enabled_hosts": 2,
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 8.2 Estado del Host

```
GET /api/esxi/<host_id>/status
```

**Respuesta `200`:**
```json
{
  "status": "success",
  "host_id": "esxi8-135",
  "online": true,
  "connection_state": "connected",
  "power_state": "poweredOn",
  "vmware_version": "7.0.3",
  "active_vms": 12,
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 8.3 Recursos del Host

```
GET /api/esxi/<host_id>/resources
```

**Respuesta `200`:**
```json
{
  "status": "success",
  "host_id": "esxi8-135",
  "cpu": {
    "total_mhz": 57600,
    "used_mhz": 28800,
    "cpu_usage_percent": 50.0,
    "cores": 24,
    "threads": 48
  },
  "memory": {
    "total_gb": 512,
    "used_gb": 256,
    "usage_percent": 50.0
  },
  "storage": {
    "local_storage_gb": 2048
  },
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 8.4 Rendimiento del Host

```
GET /api/esxi/<host_id>/performance
```

**Respuesta `200`:**
```json
{
  "status": "success",
  "host_id": "esxi8-135",
  "performance": {
    "cpu_usage_mhz": 28800,
    "memory_usage_gb": 256,
    "network_rx_mbps": 45.2,
    "network_tx_mbps": 32.1,
    "latency_ms": 2.5
  },
  "timestamp": "2025-10-24T12:00:00Z"
}
```

---

### 8.5 Información Completa del Host

```
GET /api/esxi/<host_id>/complete
```

Combina status + resources + performance en una sola llamada.

**Respuesta `200`:**
```json
{
  "status": "success",
  "host_id": "esxi8-135",
  "timestamp": "2025-10-24T12:00:00Z",
  "host_status": { "...": "ver /status" },
  "resources":   { "...": "ver /resources" },
  "performance": { "...": "ver /performance" }
}
```

**Nota:** Si alguna sub-llamada falla, retorna `206 Partial Content` con los datos disponibles.

---

### 8.6 Datos Históricos

```
GET /api/esxi/<host_id>/history/<metric>
```

Retorna datos históricos formateados para Chart.js (recogidos en intervalos de 10 min).

**Métricas disponibles:**

| Métrica | Descripción |
|---------|-------------|
| `cpu` | Uso de CPU (%) |
| `memory` | Uso de memoria (%) |
| `network_received` | Red recibida (Mbps) |
| `network_transmitted` | Red transmitida (Mbps) |
| `storage_read_latency` | Latencia de lectura (ms) |
| `storage_write_latency` | Latencia de escritura (ms) |
| `datastore_usage` | Uso de datastores (%) |
| `uptime` | Tiempo de actividad |
| `vmkernel_cpu` | CPU del VMkernel |
| `vmkernel_memory` | Memoria del VMkernel |

**Respuesta `200`:**
```json
{
  "status": "success",
  "host_id": "esxi8-135",
  "metric": "cpu",
  "chart_data": {
    "labels": ["12:00", "12:10", "12:20"],
    "datasets": [{
      "label": "CPU Usage %",
      "data": [45.2, 48.1, 50.3],
      "borderColor": "rgb(75, 192, 192)",
      "tension": 0.1
    }],
    "total_points": 3
  },
  "timestamp": "2025-10-24T12:00:00Z"
}
```

**Sin datos aún:**
```json
{
  "status": "warning",
  "message": "No hay datos históricos disponibles aún",
  "host_id": "esxi8-135",
  "metric": "cpu",
  "chart_data": {}
}
```

**Errores:**
- `400` — Métrica no reconocida

---

### 8.7 Métricas Avanzadas (requiere `advanced_esxi_collector.py`)

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/esxi/<host_id>/advanced-metrics` | Todas las métricas avanzadas combinadas |
| `GET /api/esxi/<host_id>/network` | Interfaces de red y tráfico I/O |
| `GET /api/esxi/<host_id>/storage` | Latencia de storage y capacidad de datastores |
| `GET /api/esxi/<host_id>/temperature` | Sensores de temperatura y estado del hardware |
| `GET /api/esxi/<host_id>/vmkernel` | Rendimiento del VMkernel |

**Error si el recolector no está disponible:**
```json
{"status": "error", "code": 503, "message": "Recolector avanzado no disponible"}
```

---

## 9. Herramientas MCP (Agentes LLM)

Las herramientas MCP **no son endpoints HTTP**. Son funciones invocadas internamente por el agente vCenter (LangChain) vía protocolo MCP. Están definidas en `server/mcp_tool_registry.py` y expuestas como `@mcp.tool()` en `server/mcp_vcenter_server.py`.

> Toda operación vCenter del chat pasa obligatoriamente por este registro (requisito de seguridad).

### Grupo: Core (15 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `get_templates` | `username` | Lista plantillas de VM disponibles |
| `get_hosts` | `username` | Lista hosts ESXi con CPU y RAM |
| `get_datastores` | `username` | Lista datastores con capacidad |
| `deploy_dev_env` | `target_username`, `session_username`, `mcu_template?`, `eqsim_template?` | ⚠️ Despliega entorno estándar: 1 MCU + 1 EqSIM |
| `deploy_dev_env_2mcu` | `target_username`, `session_username`, `mcu_template?`, `eqsim_template?` | ⚠️ Despliega entorno GTR: 2 MCUs + 1 EqSIM |
| `list_vms_for_user` | `session_username`, `target_username?` | Lista VMs del usuario (o de otro si se especifica) |
| `delete_vms` | `vm_names[]`, `username` | ⚠️ Elimina VMs permanentemente |
| `clone_mcu_template` | `target_username`, `session_username`, `template_name?`, `count?`, `host_name?`, `datastore_name?` | ⚠️ Clona N MCUs para un usuario |
| `list_vms_by_host` | `host_name`, `username` | Lista VMs en un host ESXi específico |
| `list_all_vms` | `username` | Lista TODAS las VMs del vCenter (lento) |
| `generate_resource_report` | `username`, `vm_name?` | Genera informe ZIP de recursos |
| `get_obsolete_vms` | `username`, `days_threshold?` | VMs sin encender en N días (default 30) |
| `export_vm_performance` | `vm_name`, `username` | Exporta métricas de rendimiento a CSV |
| `power_operations` | `vm_names[]`, `operation`, `username` | ⚠️ Encender/apagar/suspender/resetear VMs |
| `get_vm_details` | `vm_names[]`, `username` | Info detallada de una o varias VMs |

**Valores válidos para `template_name`:** `p24`, `p27`, `p28`
**Valores válidos para `operation`:** `poweron`, `poweroff`, `suspend`, `reset`

### Grupo: Snapshots (4 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `create_snapshot` | `vm_name`, `snapshot_name`, `username`, `description?` | Crea snapshot |
| `list_snapshots` | `vm_name`, `username` | Lista snapshots de una VM |
| `revert_snapshot` | `vm_name`, `snapshot_name`, `username` | ⚠️ Revierte VM al snapshot |
| `delete_snapshot` | `vm_name`, `snapshot_name`, `username` | ⚠️ Elimina snapshot permanentemente |

### Grupo: Reconfiguración de VM (3 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `reconfigure_vm` | `vm_name`, `username`, `cpu_count?`, `memory_mb?` | ⚠️ Modifica CPU y/o RAM (requiere VM apagada) |
| `rename_vm` | `vm_name`, `new_name`, `username` | Renombra una VM |
| `change_vm_network` | `vm_name`, `interface_index`, `network_name`, `username` | ⚠️ Cambia red/VLAN de una NIC |

### Grupo: ESXi Directo (3 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `get_esxi_status` | `host_id`, `username` | Estado general del host ESXi |
| `get_esxi_resources` | `host_id`, `username` | CPU, RAM y storage del host |
| `get_esxi_performance` | `host_id`, `username` | Métricas de rendimiento en tiempo real |

### Grupo: Datastore Avanzado (2 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `browse_datastore` | `datastore_name`, `path?`, `username` | Navega archivos del datastore |
| `get_all_datastores` | `username` | Lista completa de datastores |

### Grupo: Eventos y Alarmas (2 herramientas)

| Herramienta | Parámetros clave | Descripción |
|-------------|-----------------|-------------|
| `get_vm_events` | `vm_name`, `username` | Historial de eventos de una VM |
| `get_active_alarms` | `username` | Alarmas activas en el vCenter |

> **⚠️ Operaciones destructivas**: `delete_vms`, `power_operations`, `revert_snapshot`, `delete_snapshot`, `reconfigure_vm`, `change_vm_network` y los despliegues requieren confirmación explícita del usuario en el chat antes de ejecutarse.

---

## 10. Roles y Permisos

```
Jerarquía: user < admin < superuser
```

| Recurso | user | admin | superuser |
|---------|:----:|:-----:|:---------:|
| `GET /app` (chat) | ✓ | ✓ | ✓ |
| `POST /chat`, `/chat/stream` | ✓ | ✓ | ✓ |
| `GET /api/*` (hosts, datastores, system-stats) | ✓ | ✓ | ✓ |
| `GET /api/esxi/*` | ✓ | ✓ | ✓ |
| `GET /admin` (dashboard) | — | ✓ | ✓ |
| `GET /monitoring` | — | ✓ | ✓ |
| `GET /admin/stats` | — | — | ✓ |
| `GET /admin/users` | — | — | ✓ |
| `GET|POST|PUT|DELETE /api/admin/users` | — | — | ✓ |
| `GET /api/stats/*` | — | — | ✓ |

**Protecciones de seguridad:**
- Rate limiting: 5 intentos de login fallidos → bloqueo IP por 5 min
- Contraseñas: `bcrypt` con salt
- Sesiones: token hexadecimal seguro, timeout automático de 3600 s
- Logging de todos los accesos en `logs/security/` y `logs/audit/`

---

## 11. Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ORCH_EXECUTOR_MODEL` | `gpt-oss:20b` | Modelo LLM principal |
| `ORCH_FORMATTER_MODEL` | `gpt-oss:20b` | Modelo para formatear queries |
| `ENABLE_QUERY_FORMATTING` | `false` | Activa el pre-formateo de consultas vCenter |
| `FORMATTER_TIMEOUT` | `5` | Timeout del formateador (segundos) |
| `ORCH_SECRET` | *(auto)* | Clave secreta de sesión Flask |
| `ORCH_PORT` | `9100` | Puerto del servidor |
| `DATABASE_PATH` | `data/users.db` | Ruta a la base de datos SQLite |

---

## 12. Códigos de Error Comunes

| Código | Significado | Causa típica |
|--------|-------------|--------------|
| `400` | Bad Request | Parámetros faltantes o inválidos |
| `401` | Unauthorized | Sesión expirada o inexistente |
| `403` | Forbidden | Rol insuficiente para el recurso |
| `404` | Not Found | Token de descarga expirado / usuario no encontrado |
| `409` | Conflict | Usuario ya existe (al crear) |
| `410` | Gone | Archivo temporal no disponible |
| `429` | Too Many Requests | Demasiados intentos de login fallidos |
| `500` | Internal Server Error | Error del servidor (ver `logs/system/`) |
| `503` | Service Unavailable | Host ESXi o recolector avanzado no disponible |

---

*Generado desde el código fuente. Archivos de referencia principales: `src/api/main_agent.py`, `server/mcp_tool_registry.py`, `server/mcp_vcenter_server.py`.*
