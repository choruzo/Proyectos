---
tipo: componente
versión: 1.0
tags: [seguridad, autenticacion, autorizacion, aislamiento, mcp, audit]
última_actualización: 2026-03-24
relacionado:
  - "[[Autenticacion]]"
  - "[[Sistema-MCP]]"
  - "[[Herramientas-MCP]]"
  - "[[Connection-Pool]]"
  - "[[Structured-Logging]]"
---

# Modelo de Seguridad del Agente vCenter

Documentación del sistema de aislamiento por usuario, patrón MCP obligatorio y controles de seguridad del sistema multi-agente.

---

## Seguridad de Transporte (HTTPS / nginx)

El sistema utiliza **nginx como proxy inverso SSL** delante de Flask. Flask escucha únicamente en `127.0.0.1:5001` (inaccesible desde la red); nginx gestiona TLS en el puerto 5000.

```
Usuarios ── https://host:5000 ──► nginx:5000 (TLSv1.2+)
                                       │
                              proxy_pass ↓
                                       │
                            Flask:5001 (127.0.0.1 — no expuesto)
```

### Configuraciones nginx

| Plataforma | Archivo |
|------------|---------|
| Windows | `vcenter_agent_system/nginx/nginx_windows.conf` |
| Ubuntu | `vcenter_agent_system/nginx/nginx_ubuntu.conf` |

### Security Headers (`@app.after_request`)

Aplicados en todas las respuestas Flask desde `src/api/main_agent.py`:

| Header | Valor | Propósito |
|--------|-------|-----------|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Fuerza HTTPS (HSTS) |
| `X-Content-Type-Options` | `nosniff` | Evita MIME sniffing |
| `X-Frame-Options` | `DENY` | Anti-clickjacking |
| `X-XSS-Protection` | `1; mode=block` | Protección XSS legacy |
| `Content-Security-Policy` | `default-src 'self'; ...` | Restricción de recursos |

### Session Cookie Flags

```python
app.config.update(
    SESSION_COOKIE_SECURE=True,      # Solo por HTTPS
    SESSION_COOKIE_HTTPONLY=True,    # Inaccesible desde JS
    SESSION_COOKIE_SAMESITE='Strict', # Protección CSRF
    PREFERRED_URL_SCHEME='https',
)
```

---

## Principios de Seguridad

```mermaid
mindmap
  root((Seguridad\nvCenter Agent))
    Aislamiento por Usuario
      Namespace de VMs por abreviatura
      ConversationMemory separada
      Closures independientes
    MCP Obligatorio
      Todas las tools pasan por Registry
      Sin bypass directo a pyvmomi
      Audit log de cada operación
    Autenticación Doble
      Flask session tokens
      SQLite persistent sessions
      bcrypt password hashing
    Principio de Mínimo Privilegio
      Tools por usuario, no globales
      session_abbr filtra namespace
      Sin acceso cross-user
```

---

## Patrón MCP Obligatorio

**Todos los accesos a vCenter deben pasar obligatoriamente por `MCPToolRegistry`.** No existe ninguna excepción a esta regla.

```mermaid
flowchart TD
    LLM[LangChain LLM] -->|selecciona tool| TOOL[StructuredTool]

    TOOL -->|llama| REGISTRY["MCPToolRegistry\nclosure del usuario"]

    REGISTRY -->|get_user_si| POOL[VCenterConnectionPool]
    POOL -->|SmartConnect| VCENTER[(vCenter)]

    style BYPASS fill:#ff6b6b,color:#fff
    BYPASS[BYPASS DIRECTO\nSmartConnect sin Registry]

    note["NUNCA hacer esto:\nsi = SmartConnect(host, user, pwd)\n\nSiempre usar:\nsi = get_user_si(username)"]
    style note fill:#fff3cd
```

### Por qué es obligatorio

| Riesgo sin MCP Registry | Consecuencia |
|------------------------|--------------|
| LLM invoca SmartConnect con credenciales hardcoded | Exposición de credenciales en logs |
| Tool ejecuta operación en nombre de otro usuario | Violación de aislamiento por usuario |
| Sin audit log | Sin trazabilidad de operaciones |
| Sin filtro de namespace | Un usuario podría ver/modificar VMs de otros |

---

## Aislamiento por Usuario

Cada usuario opera dentro de su propio namespace de VMs, determinado por su `session_abbr`. El filtro se aplica en el closure de `MCPToolRegistry` y no puede ser modificado ni por el LLM ni por el usuario.

```mermaid
sequenceDiagram
    participant U1 as Usuario jmartinb
    participant U2 as Usuario pgarcia
    participant Agent as agent.py
    participant Registry as MCPToolRegistry

    U1->>Agent: "Listar mis VMs"
    Agent->>Registry: create_tool_functions("jmartinb", "JaMB")
    Note over Registry: Closure captura session_abbr="JaMB"

    U2->>Agent: "Listar mis VMs"
    Agent->>Registry: create_tool_functions("pgarcia", "PaGa")
    Note over Registry: Closure captura session_abbr="PaGa"

    Registry->>Registry: list_vms_for_user() de jmartinb
    Note over Registry: Filtra VMs con prefijo "JaMB_*"

    Registry->>Registry: list_vms_for_user() de pgarcia
    Note over Registry: Filtra VMs con prefijo "PaGa_*"

    Note over Registry: Sin acceso cross-user
```

### Mapeo Usuario → Abreviatura

El archivo `config/user_mapping.json` mapea cada usuario a su abreviatura de namespace:

```json
{
  "jmartinb": "JaMB",
  "pgarcia":  "PaGa",
  "admin":    "ADM"
}
```

Las VMs creadas para un usuario siempre tienen el prefijo de su abreviatura. El filtro se aplica en el closure, no puede ser modificado por el LLM.

---

## Sistema de Autenticación Dual

El sistema emplea dos mecanismos de sesión complementarios: uno en memoria para velocidad y otro persistente en SQLite para resiliencia ante reinicios.

```mermaid
flowchart TD
    subgraph Entrada["Autenticación de Entrada"]
        LOGIN[POST /login] --> BCRYPT{bcrypt.checkpw\npassword vs hash}
        BCRYPT -->|Válido| FLASK_SESSION[Flask session token\nIn-memory 3600s]
        BCRYPT -->|Inválido| DENY[401 Unauthorized]
        FLASK_SESSION --> SQLITE_SESSION[SQLite session\ndata/users.db\nPersistente]
    end

    subgraph Middleware["Middleware de Protección"]
        ROUTE[Ruta protegida] --> DECORATOR[@authenticated_action]
        DECORATOR --> CHECK{session en\nACTIVE_SESSIONS?}
        CHECK -->|Sí| ALLOW[Continuar]
        CHECK -->|No| REDIRECT[Redirigir a /login]
    end

    subgraph Roles["Control de Roles"]
        ADMIN[@admin_required] --> ROLE_CHECK{role == admin\no superuser?}
        SUPERUSER[@superuser_required] --> SU_CHECK{role == superuser?}
        ROLE_CHECK -->|No| FORBIDDEN[403 Forbidden]
        SU_CHECK -->|No| FORBIDDEN
    end
```

### Roles de usuario

| Rol | Permisos |
|-----|----------|
| `user` | Operaciones sobre sus propias VMs (namespace propio) |
| `admin` | Gestión de usuarios, acceso al panel de administración |
| `superuser` | Acceso completo, incluyendo operaciones cross-user |

### Decoradores de Seguridad

```python
from src.utils.context_middleware import authenticated_action, security_sensitive
from src.auth.decorators import admin_required, superuser_required

@app.route('/api/sensitive', methods=['POST'])
@authenticated_action      # Verifica sesión activa en ACTIVE_SESSIONS
@security_sensitive        # Registra en logs/security/
def sensitive_endpoint():
    username = session['username']
    ...

@app.route('/admin/users', methods=['POST'])
@admin_required            # Solo admin o superuser
def admin_endpoint():
    ...

@app.route('/admin/superuser-action', methods=['POST'])
@superuser_required        # Solo superuser
def superuser_endpoint():
    ...
```

---

## Flujo de Sesiones

```mermaid
stateDiagram-v2
    [*] --> Anónimo: Visita la aplicación

    Anónimo --> Autenticado: POST /login con credenciales válidas
    Autenticado --> Activo: Session en ACTIVE_SESSIONS (in-memory)
    Activo --> Activo: Cualquier request dentro de 3600s
    Activo --> Expirado: Sin actividad 3600s
    Expirado --> Anónimo: Sesión limpiada de ACTIVE_SESSIONS
    Autenticado --> Anónimo: POST /logout

    note right of Activo
        SQLite persiste sesión
        Flask session token en cookie
        ConversationBufferMemory activa
    end note
```

---

## Logging de Seguridad y Auditoría

Todas las operaciones de seguridad se registran en logs estructurados JSON categorizados. **Nunca usar `print()`.**

```mermaid
flowchart LR
    OP[Operación vCenter] --> LOG_CTX["log_context(operation, user, vm)"]

    LOG_CTX --> API["logs/api/\nHTTP requests"]
    LOG_CTX --> AUDIT["logs/audit/\nAcciones de usuario"]
    LOG_CTX --> SECURITY["logs/security/\nEventos de autenticación"]
    LOG_CTX --> PERF["logs/performance/\nTiempos de respuesta"]
    LOG_CTX --> BUSINESS["logs/business/\nOperaciones vCenter"]
    LOG_CTX --> SYSTEM["logs/system/\nErrores del sistema"]
```

### Ejemplo de log de auditoría

```json
{
  "timestamp": "2026-03-12T10:30:00Z",
  "level": "INFO",
  "category": "audit",
  "operation": "delete_vms",
  "user": "jmartinb",
  "session_abbr": "JaMB",
  "vms_deleted": ["JaMB_MCU_01", "JaMB_EqSIM_01"],
  "duration_ms": 4523
}
```

---

## Herramientas Destructivas — Controles

Existen 8 herramientas cuyas acciones son irreversibles o de alto impacto. El sistema prompt del agente indica al LLM que debe solicitar confirmación explícita antes de ejecutarlas.

```mermaid
flowchart TD
    USER[Usuario solicita acción] --> LLM[LLM evalúa intención]

    LLM --> CONFIRM{¿La operación es\ndestructiva?}

    CONFIRM -->|No: listar, detalles, etc.| EXEC_SAFE[Ejecutar directamente]

    CONFIRM -->|Sí: eliminar, revertir| SYSTEM_PROMPT["System prompt indica:\n'Confirma antes de acciones\ndestructivas'"]

    SYSTEM_PROMPT --> RESPONSE[LLM solicita confirmación al usuario]
    RESPONSE --> USER2[Usuario confirma]
    USER2 --> EXEC_DEST[Ejecutar herramienta destructiva]

    EXEC_DEST --> AUDIT_LOG[log_audit con detalles completos]
```

### Tabla de herramientas destructivas

| Herramienta | Acción irreversible |
|-------------|---------------------|
| `delete_vms_tool` | Elimina VMs permanentemente |
| `revert_snapshot_tool` | Pierde estado actual de la VM |
| `delete_snapshot_tool` | Elimina snapshot permanentemente |
| `reconfigure_vm_tool` | Modifica hardware de la VM |
| `change_vm_network_tool` | Cambia VLAN (puede perder conectividad) |
| `remove_vm_nic_tool` | Elimina adaptador de red |
| `power_operations_tool` | Apagar puede corromper datos sin shutdown |
| `deploy_dev_env` | Consume recursos de vCenter |

---

## Anti-Patrones Prohibidos

### Nunca crear conexiones directas a vCenter

```python
# MAL: bypasea el pool y los controles de seguridad
from pyVim.connect import SmartConnect
si = SmartConnect(host="vcenter", user="admin", pwd="secret")
```

```python
# BIEN: usa el connection pool con aislamiento por usuario
si = self.get_user_si(username)  # En MCPToolRegistry
```

### Nunca usar print() para logging

```python
# MAL: no queda en logs estructurados
print(f"Eliminando VM {vm_name}")
```

```python
# BIEN: queda en audit log con contexto completo
logger.log_business_operation("vm_delete", {"vm": vm_name, "user": username})
```

### Nunca exponer credenciales en respuestas

```python
# MAL: el LLM podría incluir esto en la respuesta al usuario
return f"Conectado a {host} con usuario {user} y contraseña {pwd}"
```

```python
# BIEN: solo retornar información operacional
return f"Conectado exitosamente a vCenter ({host})"
```

---

## Resumen del Modelo de Seguridad — 6 Capas

```mermaid
graph LR
    subgraph Capa0["Capa 0: Transporte"]
        TLS[nginx TLS + Security Headers<br/>Cookie Flags Secure/HttpOnly]
    end

    subgraph Capa1["Capa 1: Autenticación"]
        AUTH[bcrypt + Flask session]
    end

    subgraph Capa2["Capa 2: Autorización"]
        ROLE[Roles: user / admin / superuser]
    end

    subgraph Capa3["Capa 3: Aislamiento"]
        NAMESPACE[Namespace por session_abbr]
        CLOSURE[Closures per-user]
    end

    subgraph Capa4["Capa 4: Trazabilidad"]
        AUDIT[Audit logs JSON]
        SECURITY[Security logs JSON]
    end

    subgraph Capa5["Capa 5: MCP Obligatorio"]
        REGISTRY[MCPToolRegistry centralizado]
    end

    Capa0 --> Capa1 --> Capa2 --> Capa3 --> Capa4
    Capa3 --> Capa5
```

| Capa | Mecanismo | Archivo clave |
|------|-----------|---------------|
| 0 - Transporte | nginx TLS (TLSv1.2+), security headers, cookie flags | `nginx/nginx_windows.conf`, `nginx/nginx_ubuntu.conf`, `src/api/main_agent.py` |
| 1 - Autenticación | bcrypt + Flask session (3600s) + SQLite | `src/auth/auth_service.py`, `src/auth/database.py` |
| 2 - Autorización | Roles user/admin/superuser + decoradores | `src/auth/decorators.py`, `src/auth/user_manager.py` |
| 3 - Aislamiento | Namespace `session_abbr`, closures per-user | `server/mcp_tool_registry.py`, `config/user_mapping.json` |
| 4 - Trazabilidad | Logs JSON por categoría (audit, security) | `src/utils/structured_logger.py` |
| 5 - MCP Obligatorio | Todas las tools pasan por MCPToolRegistry | `server/mcp_tool_registry.py`, `server/mcp_tool_wrappers.py` |

---

## Enlaces relacionados

- [[Autenticacion]] — Sistema de autenticación y gestión de sesiones
- [[Sistema-MCP]] — Arquitectura MCP y tool registry centralizado
- [[Herramientas-MCP]] — Catálogo completo de las 36 herramientas vCenter
- [[Connection-Pool]] — Pool de conexiones VCenterConnectionPool
- [[Structured-Logging]] — Framework de logging estructurado JSON
