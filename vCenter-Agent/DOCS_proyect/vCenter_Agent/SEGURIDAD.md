# Modelo de Seguridad del Agente vCenter

Documentación del sistema de aislamiento por usuario, patrón MCP obligatorio y controles de seguridad.

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

```mermaid
flowchart TD
    LLM[LangChain LLM] -->|selecciona tool| TOOL[StructuredTool]

    TOOL -->|llama| REGISTRY["MCPToolRegistry\nclosure del usuario"]

    REGISTRY -->|get_user_si| POOL[VCenterConnectionPool]
    POOL -->|SmartConnect| VCENTER[(vCenter)]

    style BYPASS fill:#ff6b6b,color:#fff
    BYPASS[❌ BYPASS DIRECTO\nSmartConnect sin Registry]

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

    Note over Registry: ✓ Sin acceso cross-user
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

### Decoradores de Seguridad

```python
from src.utils.context_middleware import authenticated_action, security_sensitive
from src.auth.decorators import admin_required, superuser_required

@app.route('/api/sensitive', methods=['POST'])
@authenticated_action      # Verifica sesión activa
@security_sensitive        # Registra en logs/security/
def sensitive_endpoint():
    username = session['username']
    ...

@app.route('/admin/users', methods=['POST'])
@admin_required            # Solo admin o superuser
def admin_endpoint():
    ...
```

---

## Logging de Seguridad y Auditoría

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

**Las 8 herramientas destructivas son:**

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

## Anti-Patrones Prohibidos

### ❌ Nunca crear conexiones directas

```python
# MAL — bypasea el pool y los controles de seguridad
from pyVim.connect import SmartConnect
si = SmartConnect(host="vcenter", user="admin", pwd="secret")
```

```python
# BIEN — usa el connection pool con aislamiento
si = self.get_user_si(username)  # En MCPToolRegistry
```

### ❌ Nunca usar print() para logging

```python
# MAL — no queda en logs estructurados
print(f"Eliminando VM {vm_name}")
```

```python
# BIEN — queda en audit log con contexto
logger.log_business_operation("vm_delete", {"vm": vm_name, "user": username})
```

### ❌ Nunca exponer credenciales en respuestas

```python
# MAL — el LLM podría incluir esto en la respuesta
return f"Conectado a {host} con usuario {user} y contraseña {pwd}"
```

```python
# BIEN — solo retornar información operacional
return f"Conectado exitosamente a vCenter ({host})"
```

---

## Resumen del Modelo de Seguridad

```mermaid
graph LR
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

    Capa1 --> Capa2 --> Capa3 --> Capa4
    Capa3 --> Capa5
```
