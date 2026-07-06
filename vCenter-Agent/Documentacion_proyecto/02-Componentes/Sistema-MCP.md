---
tipo: componente
versión: 3.0
tags: [mcp, tools, vcenter, security, pyvmomi]
última_actualización: 2026-04-20
---

# Sistema MCP — Model Context Protocol

Capa de seguridad y herramientas para el Agente vCenter. Centraliza las 36 herramientas MCP que actúan como intermediario obligatorio entre el LLM (razonamiento) y pyvmomi (ejecución VMware).

## Arquitectura del Sistema

```mermaid
graph TB
    subgraph AGENT["src/core/agent.py"]
        LLM[ChatOllama<br/>gpt-oss:20b]
        AE[AgentExecutor<br/>LangChain]
        LLM --> AE
    end

    subgraph REGISTRY["server/mcp_tool_registry.py"]
        REG[MCPToolRegistry<br/>Instancia global]
        CTF[create_tool_functions<br/>closures por usuario]
        POOL[VCenterConnectionPool<br/>max 5 · timeout 30s]
        REG --> CTF --> POOL
    end

    subgraph WRAPPERS["server/mcp_tool_wrappers.py"]
        WRAP[create_mcp_aware_tools<br/>StructuredTool]
    end

    subgraph TOOLS["36 Herramientas como Closures"]
        G0[Original: 15]
        G1[Snapshots: 4]
        G2[Reconfig VM: 3]
        G3[NICs: 3]
        G4[Datastore: 2]
        G5[ESXi: 3]
        G6[Eventos: 2]
        G7[Fechas: 1]
        G8[Config VM: 3]
    end

    subgraph VCENTER["VMware Infrastructure"]
        PVMOMI[pyvmomi<br/>vcenter_tools.py]
        VC[(vCenter)]
        PVMOMI --> VC
    end

    AE -->|invoke tool| WRAP
    WRAP -->|StructuredTool| CTF
    CTF --> G0 & G1 & G2 & G3 & G4 & G5 & G6 & G7 & G8
    G0 & G1 & G2 & G3 & G4 & G5 & G6 & G7 & G8 --> PVMOMI
    POOL --> PVMOMI
```

## Flujo de una Petición

```mermaid
sequenceDiagram
    participant U as Usuario
    participant AE as AgentExecutor
    participant LLM as ChatOllama
    participant ST as StructuredTool
    participant REG as MCPToolRegistry
    participant POOL as ConnectionPool
    participant VC as vCenter

    U ->> AE: "Muéstrame VMs del host esxi8-135"
    AE ->> LLM: Prompt + tools disponibles
    LLM -->> AE: tool_call: list_vms_by_host_tool("esxi8-135")

    AE ->> ST: invoke(args)
    Note over ST: StructuredTool.from_function(func)
    
    ST ->> REG: list_vms_by_host_tool("esxi8-135")
    Note over REG: Closure captura:<br/>si, config, logger,<br/>username, session_abbr

    REG ->> POOL: get_user_si(username)
    POOL -->> REG: ServiceInstance (cacheada o nueva)

    REG ->> VC: list_vms_by_host(si, host_name)
    VC -->> REG: Lista de VMs

    REG -->> ST: str resultado JSON
    ST -->> AE: observation
    AE ->> LLM: observation → respuesta final
    LLM -->> U: Respuesta formateada
```

## Catálogo Completo de 36 Herramientas

### Mapa Visual por Grupos

```mermaid
mindmap
  root((36 MCP Tools<br/>vCenter Agent))
    Grupo 0: Original - 15
      get_templates
      get_hosts_tool
      get_datastores_tool
      deploy_dev_env
      deploy_dev_env_2mcu
      list_vms_for_user
      delete_vms_tool ⚠️
      clone_mcu_template
      list_vms_by_host_tool
      list_all_vms_tool
      generate_resource_report_tool
      get_obsolete_vms_tool
      export_vm_performance_tool
      power_operations_tool
      get_vm_details_tool
    Grupo 1: Snapshots - 4
      create_snapshot_tool
      list_snapshots_tool
      revert_snapshot_tool ⚠️
      delete_snapshot_tool ⚠️
    Grupo 2: Reconfig VM - 3
      reconfigure_vm_tool ⚠️
      rename_vm_tool
      change_vm_network_tool ⚠️
    Grupo 3: Gestión NIC - 3
      add_vm_nic_tool
      remove_vm_nic_tool ⚠️
      list_available_networks_tool
    Grupo 4: ESXi Directo - 3
      get_esxi_status_tool
      get_esxi_resources_tool
      get_esxi_performance_tool
    Grupo 5: Datastore - 2
      browse_datastore_tool
      get_all_datastores_tool
    Grupo 6: Eventos y Alarmas - 2
      get_vm_events_tool
      get_active_alarms_tool
    Grupo 7: Fechas - 1
      get_vm_creation_dates_tool
    Grupo 8: Configuración VM - 3
      get_vm_network_details_tool
      get_vm_resource_limits_tool
      get_vm_storage_details_tool
```

### Tabla Detallada por Grupo

#### Grupo 0 — Original (15 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 1 | `get_templates` | — | Lista plantillas VM disponibles | No |
| 2 | `get_hosts_tool` | — | Hosts ESXi con CPU/memoria | No |
| 3 | `get_datastores_tool` | — | Datastores con capacidad | No |
| 4 | `deploy_dev_env` | `username_`, `mcu_template`, `eqsim_template` | Despliegue entorno desarrollo (1 MCU + 1 EqSim) | No |
| 5 | `deploy_dev_env_2mcu` | `username_`, `mcu_template`, `eqsim_template` | Despliegue entorno desarrollo (2 MCU + 1 EqSim) | No |
| 6 | `list_vms_for_user` | `username_=None` | VMs del usuario actual (filtro por session_abbr) | No |
| 7 | `delete_vms_tool` | `vm_names: list[str]` | **Elimina VMs permanentemente** | **Sí** |
| 8 | `clone_mcu_template` | `username_`, `template_name`, `count`, `host_name`, `datastore_name` | Clona plantilla N veces | No |
| 9 | `list_vms_by_host_tool` | `host_name: str` | VMs en un host específico | No |
| 10 | `list_all_vms_tool` | — | TODAS las VMs del vCenter (admin) | No |
| 11 | `generate_resource_report_tool` | `vm_name=None` | Genera ZIP con reporte recursos | No |
| 12 | `get_obsolete_vms_tool` | `days_threshold=30` | VMs inactivas por N días | No |
| 13 | `export_vm_performance_tool` | `vm_name: str` | Exporta CSV con métricas rendimiento | No |
| 14 | `power_operations_tool` | `vm_names`, `operation` | Operaciones energía (poweron/poweroff/suspend/reset) | No |
| 15 | `get_vm_details_tool` | `vm_names` | Detalles CPU/RAM/estado/red | No |

**Operaciones power_operations_tool:**
- `poweron` / `power_on`: Enciende VM
- `poweroff` / `power_off`: Apaga VM (forzado)
- `shutdown_guest`: Apagado graceful vía VMware Tools
- `suspend`: Suspende VM
- `reset`: Reinicia VM (forzado)

#### Grupo 1 — Snapshots (4 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 16 | `create_snapshot_tool` | `vm_name`, `snapshot_name`, `description` | Crea snapshot de VM | No |
| 17 | `list_snapshots_tool` | `vm_name` | Lista snapshots con fechas | No |
| 18 | `revert_snapshot_tool` | `vm_name`, `snapshot_name` | **Revierte VM a snapshot anterior** | **Sí** |
| 19 | `delete_snapshot_tool` | `vm_name`, `snapshot_name` | **Elimina snapshot permanentemente** | **Sí** |

**Ciclo de vida:**
```mermaid
flowchart LR
    VM[VM running] -->|create| S1[Snapshot A]
    S1 -->|create| S2[Snapshot B]
    S2 -->|revert| VM2[VM en estado A]
    S1 -->|delete| GONE[Snapshot eliminado]
    style GONE fill:#ff6b6b,color:#fff
```

#### Grupo 2 — Reconfiguración VM (3 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 20 | `reconfigure_vm_tool` | `vm_name`, `cpu_count`, `memory_mb`, `cores_per_socket` | **Modifica CPU/RAM (requiere VM apagada)** | **Sí** |
| 21 | `rename_vm_tool` | `vm_name`, `new_name` | Renombra VM | No |
| 22 | `change_vm_network_tool` | `vm_name`, `interface_index`, `network_name` | **Cambia NIC a otra red/VLAN** | **Sí** |

> **Nota:** `reconfigure_vm_tool` requiere VM **apagada** antes de modificar CPU o RAM.

#### Grupo 3 — Gestión NIC (3 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 23 | `add_vm_nic_tool` | `vm_name`, `network_name`, `adapter_type="vmxnet3"` | Añade NIC a VM | No |
| 24 | `remove_vm_nic_tool` | `vm_name`, `interface_index` | **Elimina NIC de VM** | **Sí** |
| 25 | `list_available_networks_tool` | — | VLANs y portgroups disponibles | No |

#### Grupo 4 — ESXi Directo (3 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 26 | `get_esxi_status_tool` | `host_id` | Estado general del host ESXi | No |
| 27 | `get_esxi_resources_tool` | `host_id` | CPU/memoria/datastores/VMs | No |
| 28 | `get_esxi_performance_tool` | `host_id` | Métricas en tiempo real | No |

**Métricas ESXi disponibles:**
```mermaid
mindmap
  root((ESXi Metrics))
    CPU
      usage_percent
      ready_percent
      cores_count
    Memoria
      usage_percent
      used_gb
      total_gb
      ballooning_mb
    Red
      received_kbps
      transmitted_kbps
    Storage
      read_kbps
      write_kbps
      latency_ms
```

#### Grupo 5 — Datastore (2 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 29 | `browse_datastore_tool` | `datastore_name`, `path="/"` | Lista archivos/carpetas en datastore | No |
| 30 | `get_all_datastores_tool` | — | Info completa todos los datastores | No |

#### Grupo 6 — Eventos y Alarmas (2 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 31 | `get_vm_events_tool` | `vm_name`, `max_events=20` | Historial eventos de VM | No |
| 32 | `get_active_alarms_tool` | — | Alarmas críticas y advertencias | No |

#### Grupo 7 — Fechas (1 tool)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 33 | `get_vm_creation_dates_tool` | `vm_names: str` (CSV) | Fechas de creación VMs | No |

#### Grupo 8 — Configuración detallada de VM (3 tools)

| # | Nombre | Parámetros | Descripción | Destructiva |
|---|--------|-----------|-------------|:-----------:|
| 34 | `get_vm_network_details_tool` | `vm_name` | VLAN/MAC/portgroup/vSwitch/DVS de la VM | No |
| 35 | `get_vm_resource_limits_tool` | `vm_name` | Reservas/límites/shares/hot-add/topología CPU | No |
| 36 | `get_vm_storage_details_tool` | `vm_name` | VMDKs, provisioning (thin/thick) y controladores | No |

### Herramientas de riesgo (selección)

```mermaid
quadrantChart
    title Herramientas por Riesgo e Irreversibilidad
    x-axis Bajo Riesgo --> Alto Riesgo
    y-axis Reversible --> Irreversible
    quadrant-1 CRÍTICAS
    quadrant-2 MONITOREAR
    quadrant-3 SEGURAS
    quadrant-4 CON CAUTELA

    delete_vms_tool: [0.95, 0.95]
    delete_snapshot_tool: [0.75, 0.85]
    revert_snapshot_tool: [0.65, 0.80]
    reconfigure_vm_tool: [0.70, 0.75]
    remove_vm_nic_tool: [0.60, 0.70]
    change_vm_network_tool: [0.55, 0.60]
    power_operations_tool: [0.40, 0.30]
    list_vms_for_user: [0.05, 0.05]
```

**Operaciones de riesgo (deben advertir al usuario y/o pedir confirmación según aplique):**
1. `delete_vms_tool` — Eliminación permanente de VMs
2. `delete_snapshot_tool` — Eliminación permanente de snapshots
3. `revert_snapshot_tool` — Pérdida de cambios posteriores
4. `reconfigure_vm_tool` — Cambios hardware permanentes
5. `remove_vm_nic_tool` — Pérdida de conectividad
6. `change_vm_network_tool` — Cambio de red/VLAN

## Connection Pool y Aislamiento por Usuario

```mermaid
flowchart TD
    subgraph REQUESTS["Peticiones Simultáneas"]
        U1[Usuario JaMB]
        U2[Usuario AnRG]
        U3[Usuario MaJB]
    end

    subgraph REGISTRY["MCPToolRegistry.user_connections"]
        C1["JaMB → SI_1"]
        C2["AnRG → SI_2"]
        C3["MaJB → SI_3"]
    end

    subgraph POOL["VCenterConnectionPool"]
        P1[Conexión 1]
        P2[Conexión 2]
        P3[Conexión 3]
        PX[... max 5]
    end

    subgraph FALLBACK["Fallback vcsim"]
        HEALTH[TCP health check]
        VCSIM[vcsim Docker<br/>Simulador vCenter]
        HEALTH --> VCSIM
    end

    U1 --> C1 --> P1
    U2 --> C2 --> P2
    U3 --> C3 --> P3
    P1 & P2 & P3 --> VC[(vCenter)]
    POOL -->|vCenter down| FALLBACK
```

**Reglas del pool:**
- **Max conexiones:** 5 simultáneas
- **Timeout:** 30 segundos
- **Caché por usuario:** Una `ServiceInstance` por usuario, reutilizada
- **Fallback:** vcsim Docker si vCenter inaccesible

### Por qué Closures

Cada llamada a `create_tool_functions(username, session_abbr)` genera 36 funciones donde `username` y `session_abbr` están **capturados en el closure**:

```python
def create_tool_functions(self, username: str, session_abbr: str):
    si = self.get_user_si(username)  # Conexión del usuario
    config = self.config

    def list_vms_for_user(username_: str = None) -> str:
        """Lista VMs del usuario actual."""
        # username y session_abbr capturados del scope externo
        return get_vms_for_user(si, session_abbr)  # si viene del closure
    
    return {'list_vms_for_user': list_vms_for_user, ...}
```

**Garantiza:**
- Cada usuario opera en su propio namespace de VMs
- Imposible inyectar otro `username` desde el LLM
- Conexiones aisladas por usuario

## Patrón de Seguridad MCP-Only

```mermaid
flowchart TD
    QUERY[Consulta usuario] --> AE[AgentExecutor]
    AE --> LLM[LLM decide tool]

    LLM --> CHECK{Tool en<br/>StructuredTools?}

    CHECK -->|No| REJECT[LLM no puede ejecutar<br/>operación no registrada]
    CHECK -->|Sí| ST[StructuredTool.invoke]

    ST --> CLOSURE[Closure Registry<br/>captura si + username]
    CLOSURE --> AUDIT[audit_logger.audit<br/>operación + usuario]
    AUDIT --> PYVMOMI[pyvmomi → vCenter]

    DIRECT["Llamada directa a<br/>vcenter_tools.py"] -.->|BLOQUEADO| PYVMOMI

    style DIRECT fill:#e74c3c,color:#fff
    style REJECT fill:#e67e22,color:#fff
    style AUDIT fill:#27ae60,color:#fff
```

### ¿Por qué no llamar directamente a vcenter_tools.py?

**Sin MCP Registry:**
- ❌ Sin trazabilidad de quién ejecutó qué
- ❌ Conexiones `si` reutilizadas entre usuarios sin control
- ❌ Sin normalización de usuario (`normalize_to_abbr`)
- ❌ Operaciones críticas sin auditoría

**Con MCP Registry:**
- ✅ Toda operación lleva `username` en logging
- ✅ Conexiones del pool centralizado con límite
- ✅ Operaciones críticas en `audit.log`
- ✅ Imposible ejecutar sin closure controlado

## Añadir Nueva Herramienta (Patrón 3 Pasos)

```mermaid
flowchart LR
    STEP1["Paso 1<br/>mcp_tool_registry.py<br/>función closure"]
    STEP2["Paso 2<br/>Agregar al dict return"]
    STEP3["Paso 3 (opcional)<br/>mcp_vcenter_server.py<br/>@mcp.tool()"]

    STEP1 --> STEP2 --> STEP3
```

### Paso 1 — Función closure en Registry

```python
# Dentro de MCPToolRegistry.create_tool_functions(username, session_abbr):

def mi_nueva_herramienta(vm_name: str, parametro: int = 10) -> str:
    """
    Descripción clara para el LLM — se usa como doc de la herramienta.
    
    USA esta herramienta cuando el usuario pida [casos de uso específicos].
    
    Parámetros:
    - vm_name: Nombre de la VM
    - parametro: Descripción del parámetro (default: 10)
    """
    try:
        with log_context(operation="mi_nueva_herramienta", user=username, vm=vm_name):
            # si, config, logger, username capturados del closure
            result = mi_funcion_vcenter(si, vm_name, parametro, config)
            logger.log_business_operation(
                "nueva_herramienta_ejecutada",
                {"vm": vm_name, "param": parametro}
            )
            return f"Operación exitosa: {result}"
    except Exception as e:
        logger.log_system_error("mi_nueva_herramienta", str(e))
        return f"Error: {str(e)}"
```

**Reglas:**
- `si`, `config`, `logger`, `username`, `session_abbr` vienen del closure
- `__doc__` debe ser explícita con casos de uso → LLM la usa para decidir
- **Type hints obligatorios** → StructuredTool infiere JSON Schema
- Usar `log_context` y `log_business_operation` para auditoría

### Paso 2 — Registrar en dict de retorno

```python
# Al final de create_tool_functions():
return {
    # ... herramientas existentes ...
    'mi_nueva_herramienta': mi_nueva_herramienta,  # ← añadir aquí
}
```

### Paso 3 — Servidor FastMCP (opcional)

Solo si necesitas exponer la herramienta a clientes MCP externos (Claude Desktop, etc.):

```python
# En server/mcp_vcenter_server.py:
@mcp.tool()
def mi_nueva_herramienta(username: str, vm_name: str, parametro: int = 10) -> str:
    """Descripción para servidor MCP externo."""
    si = get_user_si(username)
    return str(mi_funcion_vcenter(si, vm_name, parametro, config, logger, username))
```

### Flujo interno de una tool

```mermaid
sequenceDiagram
    participant LLM as LangChain LLM
    participant Tool as StructuredTool
    participant Registry as MCPToolRegistry
    participant Pool as ConnectionPool
    participant vC as vCenter API

    LLM->>Tool: llamar list_vms_for_user()
    Tool->>Registry: func() [closure captura username]
    Registry->>Registry: log_context(operation, user)
    Registry->>Pool: get_user_si(username)
    Pool->>Pool: Buscar conexión en _pool
    Pool-->>Registry: ServiceInstance (si)
    Registry->>vC: container_view(vim.VirtualMachine)
    vC-->>Registry: [ManagedObject, ...]
    Registry->>Registry: Filtrar por session_abbr
    Registry->>Registry: log_business_operation
    Registry-->>Tool: JSON string
    Tool-->>LLM: "VMs encontradas: [...]"
```

## Código Ejemplo: create_snapshot_tool

```python
# En MCPToolRegistry.create_tool_functions():

def create_snapshot_tool(vm_name: str, snapshot_name: str, description: str = "") -> str:
    """
    Crea un snapshot de una máquina virtual.
    
    USA esta herramienta cuando el usuario quiera:
    - Crear un punto de restauración antes de cambios
    - Guardar el estado actual de una VM
    - Hacer backup temporal antes de pruebas
    
    Parámetros:
    - vm_name: Nombre de la VM
    - snapshot_name: Nombre descriptivo del snapshot
    - description: Descripción opcional del snapshot
    """
    try:
        with log_context(operation="create_snapshot", user=username, vm=vm_name):
            si = self.get_user_si(username)  # Del pool
            
            # Buscar VM
            vm = find_vm_by_name(si, vm_name)
            if not vm:
                return f"Error: VM '{vm_name}' no encontrada"
            
            # Crear snapshot (pyvmomi)
            task = vm.CreateSnapshot_Task(
                name=snapshot_name,
                description=description,
                memory=False,
                quiesce=False
            )
            wait_for_task(task)
            
            logger.log_business_operation(
                "snapshot_created",
                {"vm": vm_name, "snapshot": snapshot_name}
            )
            return f"Snapshot '{snapshot_name}' creado exitosamente en VM '{vm_name}'"
            
    except Exception as e:
        logger.log_system_error("create_snapshot_tool", str(e), {"vm": vm_name})
        return f"Error creando snapshot: {str(e)}"
```

## Componentes del Sistema

| Archivo | Responsabilidad |
|---------|----------------|
| `server/mcp_tool_registry.py` | **Core:** Registro central, 36 closures por usuario |
| `server/mcp_tool_wrappers.py` | Convierte dict → `List[StructuredTool]` LangChain |
| `server/mcp_vcenter_server.py` | Servidor FastMCP para clientes externos (36 tools) |
| `server/mcp_client.py` | **INACTIVO:** Cliente JSON-RPC para uso futuro |
| `src/core/agent.py` | Integración: instancia global `mcp_tool_registry` |
| `src/utils/vcenter_tools.py` | Implementación pyvmomi de operaciones vCenter |
| `src/utils/vcenter_tools.py` | `VCenterConnectionPool` (max 5, timeout 30s) |
| `src/utils/vcenter_health_check.py` | TCP check para fallback vcsim |
| `src/utils/vcsim_manager.py` | Gestión simulador Docker vcsim |
| `config/config.json` | Credenciales vCenter, hosts ESXi |
| `config/user_mapping.json` | Mapeo `username → abreviatura` (ej. `jamb → JaMB`) |

## Integración con Agente vCenter

```mermaid
sequenceDiagram
    participant FLASK as Flask (main_agent.py)
    participant AGENT as agent.py (module)
    participant REG as mcp_tool_registry
    participant WRAP as mcp_tool_wrappers
    participant AE as AgentExecutor

    Note over AGENT: Inicialización al importar (una vez)
    AGENT ->> REG: MCPToolRegistry(config, user_mapping)
    Note over REG: Instancia global única

    FLASK ->> AGENT: process_vcenter_query(username, message)
    AGENT ->> AGENT: get_user_context(username)

    alt Primera vez para usuario
        AGENT ->> REG: create_tool_functions(username, session_abbr)
        REG -->> AGENT: Dict con 36 closures
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

### Ciclo de Vida de Objetos

| Objeto | Creación | Vida Útil |
|--------|----------|-----------|
| `MCPToolRegistry` | Al importar `agent.py` | Toda la vida del proceso Flask |
| `VCenterConnectionPool` | Al importar | Toda la vida del proceso Flask |
| Closures (dict funciones) | Primera petición usuario | Hasta crear `AgentExecutor` |
| `StructuredTool` | Primera petición usuario | Toda la sesión del usuario |
| `AgentExecutor` | Primera petición usuario | Hasta limpieza `user_agents[username]` |
| `ConversationBufferMemory` | Primera petición usuario | Hasta expiración sesión (3600s) |

## Relacionado

- [[Arquitectura-Agente-vCenter]] — Arquitectura completa del agente vCenter
- [[Agente-vCenter]] — Documentación funcional del agente
- [[Connection-Pool]] — Detalles del pool de conexiones VMware
- [[Seguridad]] — Controles de seguridad y auditoría
- [[Structured-Logging]] — Sistema de logs estructurados
- [[Glosario]] — Términos técnicos del proyecto
