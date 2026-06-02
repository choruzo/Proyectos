---
tipo: api
versión: 3.0
tags: [api, rest, endpoints, sse, flask, http]
última_actualización: 2026-03-24
relacionado:
  - "[[Endpoint-Chat]]"
  - "[[Autenticacion]]"
  - "[[Orquestador]]"
  - "[[Agente-vCenter]]"
---

# API Reference — vCenter Multi-Agent System

Referencia completa de la API REST del sistema multi-agente vCenter.

**Base URL:** `http://<host>:9100`  
**Versión:** 3.0

## Índice

1. [Autenticación](#autenticación)
2. [Chat (Core)](#chat-core)
3. [Gestión de Usuarios](#gestión-de-usuarios-superuser)
4. [Estadísticas del Sistema](#estadísticas-del-sistema-superuser)
5. [Descargas](#descargas)
6. [Operaciones vCenter Directas](#operaciones-vcenter-directas)
7. [Monitorización ESXi](#monitorización-esxi)
8. [Roles y Permisos](#roles-y-permisos)
9. [Códigos de Error](#códigos-de-error)

## Autenticación

### POST /login

Autentica usuario y crea sesión.

**Body (form-data):**
```
username=admin
password=secret123
```

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `username` | string | Sí | Case-insensitive |
| `password` | string | Sí | Texto plano |

**Respuesta:** Redirección `302` según rol:
- `superuser` → `/admin/stats`
- `admin` → `/admin`
- `user` → `/app`

**Errores:**
- `401` — Credenciales incorrectas
- `429` — Demasiados intentos (bloqueo 5 min)

### POST /logout / GET /logout

Cierra sesión. Requiere sesión activa.

**Respuesta:** Redirección `302` a `/`

### GET /health

Health check del sistema. No requiere autenticación.

**Respuesta `200`:**
```json
{
  "status": "ok",
  "orchestrator": true,
  "agents": ["vcenter", "documentation"]
}
```

## Chat (Core)

Ver también: [[Endpoint-Chat]] para detalles técnicos del protocolo SSE.

### POST /chat/stream *(Recomendado)*

Chat con streaming SSE. Requiere sesión activa.

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "message": "¿Cuántas VMs tiene el host esxi8-135?"
}
```

**Respuesta:** `text/event-stream` con eventos:

#### Evento `routing`
```
event: routing
data: {"agent": "vcenter", "label": "Consultando agente vCenter..."}
```

Valores: `vcenter`, `documentation`, `general`

#### Evento `heartbeat`
```
event: heartbeat
data: {"ts": 1740492000.5, "elapsed": 4.2}
```

#### Evento `token`
```
event: token
data: {"t": "El host esxi8-135 tiene ", "i": 0}
```

#### Evento `done`
```
event: done
data: {
  "agent": "vcenter",
  "attachments": [
    {"filename": "reporte.zip", "url": "/api/download/abc123"}
  ],
  "total_tokens": 47
}
```

#### Evento `error`
```
event: error
data: {"error": "Sesión expirada", "code": 401}
```

**Errores:**
- `400` — Mensaje vacío
- `401` — Sesión expirada

**Notas:**
- Usar `fetch()` + `getReader()` (no `EventSource` nativo)
- **Sticky routing**: Follow-up &lt;15 palabras con números → mismo agente 180s

### POST /chat *(Legacy)*

Chat no-streaming. Fallback automático si SSE falla.

**Body:**
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
    {"filename": "reporte.csv", "url": "/api/download/tok_abc123"}
  ]
}
```

## Gestión de Usuarios *(superuser)*

Todos requieren rol `superuser`.

### GET /api/admin/users

Lista todos los usuarios del sistema.

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

### POST /api/admin/users

Crea nuevo usuario.

**Body:**
```json
{
  "username": "jdoe",
  "password": "secreto123",
  "abbr": "JDoe",
  "role": "user"
}
```

**Respuesta `201`:**
```json
{"ok": true, "username": "jdoe"}
```

**Errores:**
- `400` — Parámetros inválidos
- `409` — Usuario ya existe

### PUT /api/admin/users/&lt;username&gt;

Actualiza usuario existente.

**Body (todos opcionales):**
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

### DELETE /api/admin/users/&lt;username&gt;

Elimina usuario.

**Respuesta `200`:**
```json
{"ok": true}
```

**Errores:**
- `403` — No puede eliminar usuario logueado

## Estadísticas del Sistema *(superuser)*

### GET /api/admin/stats

Métricas del sistema (RAG, sesiones, memoria).

**Respuesta `200`:**
```json
{
  "rag_metrics": {
    "avg_query_time_ms": 234.5,
    "cache_hit_rate": 0.42,
    "total_queries": 1523,
    "avg_docs_retrieved": 8.3
  },
  "active_sessions": 3,
  "memory_usage_mb": 1024
}
```

### GET /api/admin/vcenter_activity

Actividad de usuarios en vCenter (últimas 24h).

**Query params:**
- `hours=24` (default)

**Respuesta `200`:**
```json
{
  "users": [
    {
      "username": "jamb",
      "total_tasks": 47,
      "error_rate": 0.02,
      "actions": ["PowerOnVM_Task", "CreateVM_Task"]
    }
  ],
  "period_hours": 24
}
```

## Descargas

### GET /api/download/&lt;token&gt;

Descarga archivo generado (CSV, ZIP, JSON).

**Respuesta `200`:**
```
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="reporte.csv"
```

**Errores:**
- `404` — Token inválido o expirado

## Operaciones vCenter Directas

Todos requieren rol `admin` o `superuser`.

### GET /vcenter/vms

Lista VMs del usuario logueado.

**Respuesta `200`:**
```json
{
  "vms": [
    {
      "name": "MCU_P27_JaMB",
      "power_state": "poweredOn",
      "cpu": 4,
      "memory_gb": 16,
      "guest_os": "SUSE Linux Enterprise 15"
    }
  ]
}
```

### GET /vcenter/hosts

Lista hosts ESXi.

**Respuesta `200`:**
```json
{
  "hosts": [
    {
      "name": "esxi8-135.local",
      "status": "connected",
      "cpu_usage_pct": 34.2,
      "memory_usage_pct": 67.8
    }
  ]
}
```

### GET /vcenter/datastores

Lista datastores.

**Respuesta `200`:**
```json
{
  "datastores": [
    {
      "name": "datastore_44",
      "capacity_gb": 2048,
      "free_space_gb": 1024,
      "usage_pct": 50.0
    }
  ]
}
```

## Monitorización ESXi

### GET /api/esxi/&lt;hostname&gt;/metrics

Métricas en tiempo real de un host ESXi.

**Respuesta `200`:**
```json
{
  "host": "esxi8-135",
  "cpu": {
    "usage_pct": 34.2,
    "cores": 16
  },
  "memory": {
    "usage_pct": 67.8,
    "total_gb": 128,
    "used_gb": 86.8
  },
  "storage": {
    "datastores": [
      {"name": "datastore_44", "usage_pct": 50.0}
    ]
  },
  "network": {
    "tx_mbps": 123.4,
    "rx_mbps": 456.7
  }
}
```

**Errores:**
- `404` — Host no encontrado o deshabilitado

## Roles y Permisos

| Endpoint | `user` | `admin` | `superuser` |
|----------|--------|---------|-------------|
| `/login`, `/logout` | ✅ | ✅ | ✅ |
| `/app` (chat) | ✅ | ✅ | ✅ |
| `/chat`, `/chat/stream` | ✅ | ✅ | ✅ |
| `/vcenter/*` | ❌ | ✅ | ✅ |
| `/admin` (UI) | ❌ | ✅ | ✅ |
| `/admin/stats` | ❌ | ❌ | ✅ |
| `/api/admin/users` | ❌ | ❌ | ✅ |
| `/api/esxi/*` | ❌ | ✅ | ✅ |

## Códigos de Error

| Código | Nombre | Descripción |
|--------|--------|-------------|
| `400` | Bad Request | Parámetros inválidos o faltantes |
| `401` | Unauthorized | Sesión expirada o credenciales incorrectas |
| `403` | Forbidden | Rol insuficiente para la operación |
| `404` | Not Found | Recurso no encontrado |
| `409` | Conflict | Recurso ya existe (ej. username duplicado) |
| `429` | Too Many Requests | Límite de intentos excedido (bloqueo temporal) |
| `500` | Internal Server Error | Error del servidor |

## Variables de Entorno

El servidor lee las siguientes variables:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `ORCH_EXECUTOR_MODEL` | Modelo LLM ejecutor | `gpt-oss:20b` |
| `ORCH_FORMATTER_MODEL` | Modelo LLM formatter | `gpt-oss:20b` |
| `FLASK_PORT` | Puerto del servidor | `9100` |
| `VCENTER_HOST` | Host vCenter (override config.json) | — |
| `VCENTER_USER` | Usuario vCenter | — |
| `VCENTER_PASS` | Password vCenter | — |

## Autenticación con curl

```bash
# Login
curl -X POST http://localhost:9100/login \
  -d "username=admin&password=secret" \
  -c cookies.txt

# Chat (usando cookies de sesión)
curl -X POST http://localhost:9100/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Lista mis VMs"}' \
  -b cookies.txt

# Logout
curl -X POST http://localhost:9100/logout -b cookies.txt
```

## Autenticación con JavaScript

```javascript
// Login
const response = await fetch('/login', {
  method: 'POST',
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: 'username=admin&password=secret'
});

// Chat SSE
const response = await fetch('/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: '¿Cuántas VMs?'})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  // Procesar eventos SSE...
}
```

## Enlaces Relacionados

- [[Endpoint-Chat]] — Detalles técnicos del protocolo SSE
- [[Autenticacion]] — Flujo de login/logout, roles
- [[Orquestador]] — Routing de queries, clasificación 4-capas
- [[Agente-vCenter]] — Operaciones vCenter disponibles
- [[Structured-Logging]] — Logs de API en `logs/api/api.log`

***

**Versión del documento:** 3.0  
**Fuente original:** `vcenter_agent_system/DOCS_proyect/API/API_REFERENCE.md`
