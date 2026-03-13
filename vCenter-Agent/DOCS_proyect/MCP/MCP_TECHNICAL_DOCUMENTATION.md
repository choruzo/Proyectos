# Documentación Técnica — Sistema MCP (Model Context Protocol)

> **Capa de Seguridad y Herramientas para el Agente vCenter**
> Versión: 3.0 | Última actualización: 2026-03-12

---

## Índice

1. [Visión General](#1-visión-general)
2. [Arquitectura del Sistema MCP](#2-arquitectura-del-sistema-mcp)
3. [Flujo de una petición de herramienta](#3-flujo-de-una-petición-de-herramienta)
4. [Componentes](#4-componentes)
   - 4.1 [mcp_vcenter_server.py — Servidor FastMCP](#41-mcp_vcenter_serverpy--servidor-fastmcp)
   - 4.2 [mcp_tool_registry.py — Registro centralizado](#42-mcp_tool_registrypy--registro-centralizado)
   - 4.3 [mcp_tool_wrappers.py — Adaptadores LangChain](#43-mcp_tool_wrapperspy--adaptadores-langchain)
   - 4.4 [mcp_client.py — Cliente inactivo](#44-mcp_clientpy--cliente-inactivo)
5. [Catálogo de herramientas (32 tools)](#5-catálogo-de-herramientas-32-tools)
6. [Connection Pool y aislamiento por usuario](#6-connection-pool-y-aislamiento-por-usuario)
7. [Patrón de seguridad MCP-Only](#7-patrón-de-seguridad-mcp-only)
8. [Integración con el Agente vCenter](#8-integración-con-el-agente-vcenter)
9. [Añadir una nueva herramienta](#9-añadir-una-nueva-herramienta)
10. [Referencia de archivos](#10-referencia-de-archivos)

---

## 1. Visión General

El sistema MCP es la **capa de herramientas** del agente vCenter. Actúa como intermediario obligatorio entre el LLM (que razona sobre qué acción ejecutar) y pyvmomi (que ejecuta las operaciones en VMware).

```
LLM → MCPToolRegistry → pyvmomi → vCenter
```

**Por qué existe esta capa:**

| Sin MCP | Con MCP |
|---------|---------|
| Herramientas sueltas por archivo, difíciles de auditar | Todas las 32 herramientas en un único registro |
| Conexiones a vCenter creadas ad-hoc | Pool de conexiones centralizado con límite y timeout |
| Sin trazabilidad por usuario | Cada operación lleva `username` en el log |
| El LLM podría llamar funciones internas directamente | Todas las operaciones pasan por closures controlados |

---

## 2. Arquitectura del Sistema MCP

```mermaid
graph TB
    subgraph AGENT["src/core/agent.py"]
        LLM[ChatOllama\ngpt-oss:20b]
        AE[AgentExecutor\nLangChain]
        LLM --> AE
    end

    subgraph REGISTRY["server/mcp_tool_registry.py"]
        REG[MCPToolRegistry\nInstancia global]
        CTF[create_tool_functions\nclosures por usuario]
        POOL[VCenterConnectionPool\nmax 5 conexiones]
        REG --> CTF
        REG --> POOL
    end

    subgraph WRAPPERS["server/mcp_tool_wrappers.py"]
        WRAP[create_mcp_aware_tools\nStructuredTool LangChain]
    end

    subgraph TOOLS["32 herramientas como closures"]
        direction LR
        G0[Grupo 0\n15 originales]
        G1[Grupo 1\nSnapshots x4]
        G2[Grupo 2\nReconfig VM x3]
        G2B[Grupo 2b\nDatastore x2]
        G2C[Grupo 2c\nNICs x3]
        G3[Grupo 3\nESXi directo x3]
        G5[Grupo 5\nEventos x2]
        G6[Grupo 6\nFechas creación x1]
    end

    subgraph VCENTER["Infraestructura VMware"]
        PVMOMI[pyvmomi\nvcenter_tools.py]
        VC[(vCenter)]
        PVMOMI --> VC
    end

    subgraph SERVER["server/mcp_vcenter_server.py (opcional)"]
        FASTMCP[FastMCP\nstdio transport\n29 endpoints @mcp.tool]
    end

    AE -->|invoke herramienta| WRAP
    WRAP -->|StructuredTool| CTF
    CTF --> G0 & G1 & G2 & G2B & G2C & G3 & G5 & G6
    G0 & G1 & G2 & G2B & G2C & G3 & G5 & G6 --> PVMOMI
    POOL --> PVMOMI

    SERVER -.->|uso futuro\nClaude Desktop| PVMOMI
```

---

## 3. Flujo de una petición de herramienta

```mermaid
sequenceDiagram
    participant U as Usuario
    participant AE as AgentExecutor
    participant LLM as ChatOllama
    participant ST as StructuredTool
    participant REG as MCPToolRegistry
    participant POOL as ConnectionPool
    participant VC as vCenter

    U ->> AE: "Muéstrame las VMs del host esxi8-135"
    AE ->> LLM: Prompt + herramientas disponibles
    LLM -->> AE: tool_call: list_vms_by_host_tool(host_name="esxi8-135")

    AE ->> ST: invoke(args)
    Note over ST: StructuredTool.from_function(func)
    ST ->> REG: list_vms_by_host_tool("esxi8-135")
    Note over REG: Closure captura: si, config,\nlogger, username

    REG ->> POOL: get_user_si(username)
    POOL -->> REG: ServiceInstance (cacheado o nuevo)

    REG ->> VC: list_vms_by_host(si, host_name, ...)
    VC -->> REG: Lista de VMs

    REG -->> ST: str resultado
    ST -->> AE: observation
    AE ->> LLM: observation → respuesta final
    LLM -->> U: Respuesta formateada
```

---

## 4. Componentes

### 4.1 `mcp_vcenter_server.py` — Servidor FastMCP

Expone las **29 herramientas originales** como endpoints MCP estándar con `@mcp.tool()`.

```mermaid
flowchart LR
    INIT[Inicio del servidor] --> LOAD[Cargar config.json\nuser_mapping.json]
    LOAD --> POOL[VCenterConnectionPool\nglobal]
    POOL --> MCP[FastMCP vcenter-mcp-server\nstdio transport]

    MCP --> T1["@mcp.tool() get_templates"]
    MCP --> T2["@mcp.tool() get_hosts"]
    MCP --> T3["@mcp.tool() deploy_dev_env"]
    MCP --> TN["... 26 más ..."]

    EXEC["python mcp_vcenter_server.py"] --> MCP
```

**Propósito actual:** Disponible para integración con clientes MCP estándar (Claude Desktop, etc.) y para tests. **No se usa en producción directamente** — el agente usa `MCPToolRegistry` directamente.

**Diferencia con el Registry:**

| `mcp_vcenter_server.py` | `mcp_tool_registry.py` |
|-------------------------|------------------------|
| Herramientas con `@mcp.tool()` | Closures Python (dict) |
| Transporte stdio JSON-RPC | Llamada directa en proceso |
| 29 herramientas | 32 herramientas |
| Para clientes MCP externos | Para el AgentExecutor LangChain |

### 4.2 `mcp_tool_registry.py` — Registro centralizado

Es el **núcleo del sistema**. Instancia global única en `agent.py` que gestiona:

```mermaid
flowchart TD
    INST[MCPToolRegistry.__init__\nconfig · user_mapping · template_mapping]

    subgraph POOL_CONN["Conexión por usuario"]
        GSI[get_user_si\nusername]
        CACHE{username en\nuser_connections?}
        GSI --> CACHE
        CACHE -->|Sí| CACHED[Retornar SI cacheada]
        CACHE -->|No| NEWCONN[VCenterConnectionPool\nget_connection con fallback vcsim]
        NEWCONN --> STORE[Guardar en user_connections]
    end

    subgraph NORMALIZATION["Normalización de usuario"]
        NORM[normalize_to_abbr\nusername / name]
        NORM --> MAP{En user_mapping?}
        MAP -->|Clave| ABBR[Retornar abreviatura]
        MAP -->|Valor| NAME[Retornar tal cual]
        MAP -->|No| FALLBACK[Retornar name sin espacios]
    end

    subgraph FACTORY["Fábrica de herramientas"]
        CTF[create_tool_functions\nusername · session_abbr]
        CTF --> CLOSURE["Closure captura:
        si · config · logger
        username · session_abbr
        normalize_to_abbr"]
        CLOSURE --> DICT["dict con 32 funciones
        listas para StructuredTool"]
    end

    INST --> POOL_CONN
    INST --> NORMALIZATION
    INST --> FACTORY
```

**Patrón closure — por qué es importante:**

```python
# create_tool_functions crea closures que capturan el contexto del usuario
def create_tool_functions(self, username: str, session_abbr: str):
    si = self.get_user_si(username)      # Conexión específica del usuario
    config = self.config                  # Config compartida (solo lectura)

    def power_operations_tool(vm_names, operation: str) -> str:
        """..."""
        return power_operations_vm(si, vm_names, operation, config, logger, session_abbr)
        #                          ^^ capturado del scope externo

    return {'power_operations_tool': power_operations_tool, ...}
```

Esto garantiza que **cada usuario opera con su propia conexión** sin riesgo de mezcla de sesiones.

### 4.3 `mcp_tool_wrappers.py` — Adaptadores LangChain

Convierte el dict de funciones del Registry en herramientas `StructuredTool` que LangChain puede invocar.

```mermaid
flowchart LR
    DICT["Dict de funciones\n{nombre: callable}"]

    DICT --> LOOP["for tool_name, func in tool_functions.items()"]
    LOOP --> ST["StructuredTool.from_function\nfunc=func"]

    ST --> INFER["Inferencia automática:
    name  ← func.__name__
    description ← func.__doc__
    args_schema ← type hints"]

    INFER --> LIST["List[StructuredTool]\nentregado a AgentExecutor"]
```

**Fuente única de verdad:** El nombre y la descripción de cada herramienta vienen directamente de `__name__` y `__doc__` de las funciones del Registry. No hay duplicación.

### 4.4 `mcp_client.py` — Cliente inactivo

```mermaid
flowchart LR
    STATUS["ESTADO: INACTIVO\nNo se usa en producción"] --> DESIGN["Diseño original:
    MCPClient → subprocess
    mcp_vcenter_server.py (stdio)
    JSON-RPC 2.0"]

    DESIGN --> FUTURE["Uso futuro:
    Integración Claude Desktop
    Clientes MCP externos
    Arquitectura distribuida"]

    CURRENT["Arquitectura actual:
    MCPToolRegistry (in-process)
    Sin overhead de subprocess
    Sin serialización JSON"] --> STATUS
```

---

## 5. Catálogo de herramientas (32 tools)

### Mapa visual por grupo

```mermaid
mindmap
  root((32 Herramientas\nMCP vCenter))
    Grupo 0 - Originales 15
      get_templates
      get_hosts_tool
      get_datastores_tool
      deploy_dev_env
      deploy_dev_env_2mcu
      list_vms_for_user
      delete_vms_tool
      clone_mcu_template
      list_vms_by_host_tool
      list_all_vms_tool
      generate_resource_report_tool
      get_obsolete_vms_tool
      export_vm_performance_tool
      power_operations_tool
      get_vm_details_tool
    Grupo 1 - Snapshots 4
      create_snapshot_tool
      list_snapshots_tool
      revert_snapshot_tool
      delete_snapshot_tool
    Grupo 2 - Reconfig VM 3
      reconfigure_vm_tool
      rename_vm_tool
      change_vm_network_tool
    Grupo 2b - Datastore 2
      browse_datastore_tool
      get_all_datastores_tool
    Grupo 2c - NICs 3
      add_vm_nic_tool
      remove_vm_nic_tool
      list_available_networks_tool
    Grupo 3 - ESXi directo 3
      get_esxi_status_tool
      get_esxi_resources_tool
      get_esxi_performance_tool
    Grupo 5 - Eventos 2
      get_vm_events_tool
      get_active_alarms_tool
    Grupo 6 - Fechas 1
      get_vm_creation_dates_tool
```

### Tabla completa

| Herramienta | Grupo | Operación crítica | Parámetros principales |
|-------------|-------|:-----------------:|------------------------|
| `get_templates` | 0 | — | — |
| `get_hosts_tool` | 0 | — | — |
| `get_datastores_tool` | 0 | — | — |
| `deploy_dev_env` | 0 | — | `username`, `mcu_template`, `eqsim_template` |
| `deploy_dev_env_2mcu` | 0 | — | `username`, `mcu_template`, `eqsim_template` |
| `list_vms_for_user` | 0 | — | `username_` (opcional) |
| `delete_vms_tool` | 0 | **SÍ** | `vm_names: list[str]` |
| `clone_mcu_template` | 0 | — | `username_`, `template_name`, `count`, `host_name`, `datastore_name` |
| `list_vms_by_host_tool` | 0 | — | `host_name` |
| `list_all_vms_tool` | 0 | — | — |
| `generate_resource_report_tool` | 0 | — | `vm_name` (opcional) |
| `get_obsolete_vms_tool` | 0 | — | `days_threshold` (default 30) |
| `export_vm_performance_tool` | 0 | — | `vm_name` |
| `power_operations_tool` | 0 | **SÍ** | `vm_names`, `operation` (poweron/poweroff/suspend/reset) |
| `get_vm_details_tool` | 0 | — | `vm_names` |
| `create_snapshot_tool` | 1 | — | `vm_name`, `snapshot_name`, `description` |
| `list_snapshots_tool` | 1 | — | `vm_name` |
| `revert_snapshot_tool` | 1 | **SÍ** | `vm_name`, `snapshot_name` |
| `delete_snapshot_tool` | 1 | **SÍ** | `vm_name`, `snapshot_name` |
| `reconfigure_vm_tool` | 2 | **SÍ** | `vm_name`, `cpu_count`, `memory_mb`, `cores_per_socket` |
| `rename_vm_tool` | 2 | — | `vm_name`, `new_name` |
| `change_vm_network_tool` | 2 | **SÍ** | `vm_name`, `interface_index`, `network_name` |
| `browse_datastore_tool` | 2b | — | `datastore_name`, `path` |
| `get_all_datastores_tool` | 2b | — | — |
| `add_vm_nic_tool` | 2c | — | `vm_name`, `network_name`, `adapter_type` |
| `remove_vm_nic_tool` | 2c | **SÍ** | `vm_name`, `interface_index` |
| `list_available_networks_tool` | 2c | — | — |
| `get_esxi_status_tool` | 3 | — | `host_id` |
| `get_esxi_resources_tool` | 3 | — | `host_id` |
| `get_esxi_performance_tool` | 3 | — | `host_id` |
| `get_vm_events_tool` | 5 | — | `vm_name`, `max_events` (default 20) |
| `get_active_alarms_tool` | 5 | — | — |
| `get_vm_creation_dates_tool` | 6 | — | `vm_names` (separados por coma) |

> **Operación crítica**: La herramienta ejecuta inmediatamente sin confirmación adicional. El LLM debe advertir al usuario antes de invocarla.

---

## 6. Connection Pool y aislamiento por usuario

```mermaid
flowchart TD
    subgraph REQUESTS["Peticiones simultáneas"]
        U1[Usuario JaMB]
        U2[Usuario AnRG]
        U3[Usuario MaJB]
    end

    subgraph REGISTRY_CACHE["MCPToolRegistry.user_connections"]
        C1["JaMB → SI_1"]
        C2["AnRG → SI_2"]
        C3["MaJB → SI_3"]
    end

    subgraph POOL["VCenterConnectionPool\nmax_connections=5 · timeout=30s"]
        P1[Conexión 1]
        P2[Conexión 2]
        P3[Conexión 3]
        PX[... hasta 5]
    end

    subgraph FALLBACK["Fallback vcsim (si vCenter inaccesible)"]
        HEALTH[vcenter_health_check\nTCP socket check]
        VCSIM[vcsim Docker\nSimulador vCenter]
        HEALTH --> VCSIM
    end

    U1 --> C1 --> P1
    U2 --> C2 --> P2
    U3 --> C3 --> P3
    P1 & P2 & P3 --> VC[(vCenter)]
    POOL -->|inaccesible| FALLBACK
```

**Reglas del pool:**
- Máximo 5 conexiones simultáneas
- Timeout de conexión: 30 s
- `get_connection(config=self.config)` activa el fallback a vcsim si vCenter no responde
- Las conexiones se cachean en `user_connections` — una por usuario, reutilizada entre peticiones

---

## 7. Patrón de seguridad MCP-Only

```mermaid
flowchart TD
    QUERY[Consulta del usuario] --> AE[AgentExecutor]
    AE --> LLM[LLM decide herramienta]

    LLM --> CHECK{Herramienta\ndisponible en\nStructuredTools?}

    CHECK -->|No| REJECT[LLM no puede ejecutar\noperación no registrada]
    CHECK -->|Sí| ST[StructuredTool.invoke]

    ST --> CLOSURE[Closure del Registry\ncaptura si + username + config]
    CLOSURE --> AUDIT[audit_logger.audit\noperación + usuario]
    AUDIT --> PYVMOMI[pyvmomi → vCenter]

    DIRECT["Llamada directa\na vcenter_tools.py"] -.->|BLOQUEADO\npor diseño| PYVMOMI

    style DIRECT fill:#e74c3c,color:#fff
    style REJECT fill:#e67e22,color:#fff
    style AUDIT fill:#27ae60,color:#fff
```

**¿Por qué no llamar directamente a `vcenter_tools.py`?**

Si el agente pudiera llamar las funciones de `vcenter_tools.py` directamente:
- No habría trazabilidad de quién ejecutó qué operación
- Una conexión `si` podría reutilizarse entre usuarios sin control
- No se aplicaría la normalización de usuario (`normalize_to_abbr`)
- Las operaciones críticas no quedarían en el log de auditoría

**El Registry garantiza que:**
1. Toda operación lleva `username` en el contexto de logging
2. Toda conexión viene del pool centralizado con límite
3. Las operaciones críticas se registran en `audit.log`
4. No es posible ejecutar una operación sin pasar por un closure controlado

---

## 8. Integración con el Agente vCenter

```mermaid
sequenceDiagram
    participant FLASK as Flask (main_agent.py)
    participant AGENT as agent.py (module-level)
    participant REG as mcp_tool_registry
    participant WRAP as mcp_tool_wrappers
    participant AE as AgentExecutor

    Note over AGENT: Inicialización al importar el módulo (una vez)
    AGENT ->> REG: MCPToolRegistry(config, user_mapping, template_mapping)
    Note over REG: Instancia global única

    FLASK ->> AGENT: process_vcenter_query(username, message)
    AGENT ->> AGENT: get_user_context(username)

    alt Primera vez para este usuario
        AGENT ->> REG: create_tool_functions(username, session_abbr)
        REG -->> AGENT: Dict con 32 closures
        AGENT ->> WRAP: create_mcp_aware_tools(tool_functions, username)
        WRAP -->> AGENT: List[StructuredTool]
        AGENT ->> AE: AgentExecutor(agent, tools, memory)
        Note over AE: Cacheado en user_agents[username]
    else Usuario ya conocido
        AGENT ->> AE: Recuperar de user_agents[username]
    end

    AGENT ->> AE: invoke(input=message)
    AE -->> AGENT: output string
    AGENT -->> FLASK: respuesta
```

**Ciclo de vida de los objetos:**

| Objeto | Creación | Vida útil |
|--------|----------|-----------|
| `MCPToolRegistry` | Al importar `agent.py` | Toda la vida del proceso Flask |
| `VCenterConnectionPool` | Al importar | Toda la vida del proceso Flask |
| Closures (dict de funciones) | Primera petición de cada usuario | Hasta que se crea el `AgentExecutor` |
| `StructuredTool` | Primera petición de cada usuario | Toda la sesión del usuario |
| `AgentExecutor` | Primera petición de cada usuario | Hasta que `user_agents[username]` se limpia |
| `ConversationBufferMemory` | Primera petición de cada usuario | Hasta que la sesión expira (3600s) |

---

## 9. Añadir una nueva herramienta

Para añadir una herramienta al sistema MCP se siguen **3 pasos obligatorios**:

```mermaid
flowchart LR
    STEP1["Paso 1\nmcp_tool_registry.py\nañadir función closure\ndentro de create_tool_functions"]
    STEP2["Paso 2\nmcp_tool_registry.py\nagregar al dict return\ncon clave exacta"]
    STEP3["Paso 3 (opcional)\nmcp_vcenter_server.py\nañadir @mcp.tool()\npara uso externo"]

    STEP1 --> STEP2 --> STEP3
```

### Paso 1 — Función closure en el Registry

```python
# Dentro de MCPToolRegistry.create_tool_functions(username, session_abbr):

def mi_nueva_herramienta(vm_name: str, parametro: int = 10) -> str:
    """Descripción clara para el LLM — se usa como doc de la herramienta.
    USA esta herramienta cuando el usuario pida [casos de uso].
    Parámetros: vm_name (nombre de la VM), parametro (descripción)."""
    return mi_funcion_vcenter(si, vm_name, parametro, config, logger, username)
```

**Reglas:**
- `si`, `config`, `logger`, `username` siempre vienen del scope closure
- `__doc__` es la descripción que ve el LLM — hacerla explícita y con ejemplos de uso
- Type hints son obligatorios — `StructuredTool.from_function` los usa para el JSON Schema

### Paso 2 — Registrar en el dict de retorno

```python
return {
    # ... herramientas existentes ...
    'mi_nueva_herramienta': mi_nueva_herramienta,   # ← añadir aquí
}
```

### Paso 3 — Servidor FastMCP (opcional, para clientes externos)

```python
# En mcp_vcenter_server.py:
@mcp.tool()
def mi_nueva_herramienta(username: str, vm_name: str, parametro: int = 10) -> str:
    """Descripción para el servidor MCP externo."""
    si = get_user_si(username)
    return str(mi_funcion_vcenter(si, vm_name, parametro, config, logger, username))
```

### Resumen del patrón

```mermaid
flowchart TD
    VCENTER_TOOLS["src/utils/vcenter_tools.py\nImplementación pyvmomi"]
    REGISTRY["server/mcp_tool_registry.py\nClosure capturando si + context"]
    WRAPPERS["mcp_tool_wrappers.py\nStructuredTool.from_function (automático)"]
    AGENT["src/core/agent.py\ntools list para referencia"]
    MCP_SERVER["server/mcp_vcenter_server.py\n@mcp.tool() (opcional)"]

    VCENTER_TOOLS -->|llamada directa| REGISTRY
    REGISTRY -->|automático via __name__ + __doc__| WRAPPERS
    WRAPPERS -->|inyectado en AgentExecutor| AGENT
    REGISTRY -.->|duplicar manualmente| MCP_SERVER

    style REGISTRY fill:#2980b9,color:#fff
    style WRAPPERS fill:#27ae60,color:#fff
```

---

## 10. Referencia de archivos

| Archivo | Clase / Función | Propósito |
|---------|----------------|-----------|
| `server/mcp_tool_registry.py` | `MCPToolRegistry` | Registro central — 32 closures por usuario |
| `server/mcp_tool_wrappers.py` | `create_mcp_aware_tools()` | Convierte dict a `List[StructuredTool]` LangChain |
| `server/mcp_vcenter_server.py` | `FastMCP` + `@mcp.tool()` | Servidor MCP estándar para clientes externos |
| `server/mcp_client.py` | `MCPClient` | **INACTIVO** — cliente JSON-RPC para uso futuro |
| `src/core/agent.py` | `mcp_tool_registry` (global) | Punto de integración con el agente LangChain |
| `src/utils/vcenter_tools.py` | funciones pyvmomi | Implementación real de las operaciones vCenter |
| `src/utils/vcenter_tools.py` | `VCenterConnectionPool` | Pool de conexiones (max 5, timeout 30s) |
| `src/utils/vcenter_health_check.py` | `check_vcenter_health()` | TCP check para activar fallback vcsim |
| `src/utils/vcsim_manager.py` | `VcsimManager` | Gestión del simulador Docker vcsim |
| `config/config.json` | — | Credenciales vCenter, hosts ESXi |
| `config/user_mapping.json` | — | Mapeo `username → abreviatura` (ej. `jamb → JaMB`) |

---

*Documentación generada a partir del código fuente de `vcenter_agent_system/server/`.*
*Para detalles de la arquitectura general ver `CLAUDE.md` y `.github/copilot-instructions.md`.*
*Para seguridad de sesiones ver `DOCS_proyect/Login_Autentificación/`.*
