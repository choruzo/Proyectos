# Herramientas MCP del Agente vCenter

Referencia técnica de las 33 herramientas MCP registradas en `server/mcp_tool_registry.py`.

---

## Diagrama de Grupos de Herramientas

```mermaid
mindmap
  root((33 MCP Tools<br/>vCenter Agent))
    Grupo Original - 15
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
    Snapshots - 4
      create_snapshot_tool
      list_snapshots_tool
      revert_snapshot_tool ⚠️
      delete_snapshot_tool ⚠️
    Reconfiguración VM - 3
      reconfigure_vm_tool ⚠️
      rename_vm_tool
      change_vm_network_tool ⚠️
    Gestión NIC - 3
      add_vm_nic_tool
      remove_vm_nic_tool ⚠️
      list_available_networks_tool
    ESXi Directo - 3
      get_esxi_status_tool
      get_esxi_resources_tool
      get_esxi_performance_tool
    Datastore - 2
      browse_datastore_tool
      get_all_datastores_tool
    Eventos y Alarmas - 2
      get_vm_events_tool
      get_active_alarms_tool
    Fechas - 1
      get_vm_creation_dates_tool
```

---

## Arquitectura del Registro MCP

```mermaid
flowchart TD
    subgraph Registry["MCPToolRegistry (mcp_tool_registry.py)"]
        INIT["__init__(config, user_mapping, template_mapping)"]
        CREATE["create_tool_functions(username, session_abbr)"]
        GET_SI["get_user_si(username)"]

        INIT --> CREATE
        CREATE --> GET_SI
    end

    subgraph Closures["33 Funciones Closure (por usuario)"]
        C1["list_vms_for_user()"]
        C2["deploy_dev_env(...)"]
        C3["delete_vms_tool(...)"]
        C33["... (30 más)"]
    end

    subgraph Wrappers["MCPToolWrappers (mcp_tool_wrappers.py)"]
        W["create_mcp_aware_tools(tool_functions)"]
        ST["StructuredTool.from_function(func)\n× 33"]
    end

    subgraph LangChain["LangChain AgentExecutor"]
        EXEC["AgentExecutor\n(agent + tools + memory)"]
    end

    CREATE --> Closures
    Closures --> W
    W --> ST
    ST --> EXEC
```

### Por qué se usan closures

Cada llamada a `create_tool_functions(username, session_abbr)` genera 33 funciones donde `username` y `session_abbr` están **capturados en el closure**. Esto garantiza:

- Cada usuario opera en su propio namespace de VMs
- Las funciones no necesitan recibir `username` como argumento visible al LLM
- Imposible que el LLM o un usuario inyecte otro `username`

---

## Referencia Completa de Herramientas

### Grupo 1 — Original (15 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 1 | `get_templates` | — | JSON lista de plantillas | No |
| 2 | `get_hosts_tool` | — | JSON hosts ESXi con CPU/mem | No |
| 3 | `get_datastores_tool` | — | JSON datastores con capacidad | No |
| 4 | `deploy_dev_env` | `username_`, `mcu_template`, `eqsim_template` | String resultado despliegue | No |
| 5 | `deploy_dev_env_2mcu` | `username_`, `mcu_template`, `eqsim_template` | String resultado despliegue | No |
| 6 | `list_vms_for_user` | `username_=None` | JSON VMs del usuario actual | No |
| 7 | `delete_vms_tool` | `vm_names: list[str]` | String confirmación | **Sí** |
| 8 | `clone_mcu_template` | `username_`, `template_name`, `count`, `host_name`, `datastore_name` | String resultado clonado | No |
| 9 | `list_vms_by_host_tool` | `host_name: str` | JSON VMs en el host | No |
| 10 | `list_all_vms_tool` | — | JSON TODAS las VMs del vCenter | No |
| 11 | `generate_resource_report_tool` | `vm_name=None` | Path al ZIP generado | No |
| 12 | `get_obsolete_vms_tool` | `days_threshold=30` | JSON VMs inactivas | No |
| 13 | `export_vm_performance_tool` | `vm_name: str` | Path al CSV generado | No |
| 14 | `power_operations_tool` | `vm_names`, `operation` | String resultado | No |
| 15 | `get_vm_details_tool` | `vm_names` | JSON detalles CPU/RAM/estado/red | No |

**Operaciones de `power_operations_tool`:**

```mermaid
stateDiagram-v2
    [*] --> Apagada
    Apagada --> Encendida: power_on
    Encendida --> Apagada: power_off / shutdown_guest
    Encendida --> Suspendida: suspend
    Suspendida --> Encendida: power_on
    Encendida --> Encendida: reset
```

---

### Grupo 2 — Snapshots (4 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 16 | `create_snapshot_tool` | `vm_name`, `snapshot_name`, `description` | String confirmación | No |
| 17 | `list_snapshots_tool` | `vm_name` | JSON lista con fechas | No |
| 18 | `revert_snapshot_tool` | `vm_name`, `snapshot_name` | String confirmación | **Sí** |
| 19 | `delete_snapshot_tool` | `vm_name`, `snapshot_name` | String confirmación | **Sí** |

**Ciclo de vida de snapshots:**

```mermaid
flowchart LR
    VM[VM en ejecución] -->|create_snapshot_tool| S1[Snapshot A]
    S1 -->|create_snapshot_tool| S2[Snapshot B]
    S2 -->|revert_snapshot_tool| VM2[VM en estado A]
    S1 -->|delete_snapshot_tool| S3[Snapshot A eliminado]

    style S3 fill:#ff6b6b,color:#fff
```

---

### Grupo 3 — Reconfiguración VM (3 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 20 | `reconfigure_vm_tool` | `vm_name`, `cpu_count`, `memory_mb`, `cores_per_socket` | String confirmación | **Sí** |
| 21 | `rename_vm_tool` | `vm_name`, `new_name` | String confirmación | No |
| 22 | `change_vm_network_tool` | `vm_name`, `interface_index`, `network_name` | String confirmación | **Sí** |

> **Nota:** `reconfigure_vm_tool` requiere que la VM esté **apagada** antes de modificar CPU o RAM.

---

### Grupo 4 — Gestión NIC (3 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 23 | `add_vm_nic_tool` | `vm_name`, `network_name`, `adapter_type="vmxnet3"` | String confirmación | No |
| 24 | `remove_vm_nic_tool` | `vm_name`, `interface_index` | String confirmación | **Sí** |
| 25 | `list_available_networks_tool` | — | JSON VLANs y portgroups | No |

---

### Grupo 5 — ESXi Directo (3 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 26 | `get_esxi_status_tool` | `host_id` | JSON estado general del host | No |
| 27 | `get_esxi_resources_tool` | `host_id` | JSON CPU/mem/datastores/VMs | No |
| 28 | `get_esxi_performance_tool` | `host_id` | JSON métricas en tiempo real | No |

**Métricas disponibles en `get_esxi_performance_tool`:**

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

---

### Grupo 6 — Datastore (2 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 29 | `browse_datastore_tool` | `datastore_name`, `path="/"` | JSON listado de archivos/carpetas | No |
| 30 | `get_all_datastores_tool` | — | JSON info completa todos los datastores | No |

---

### Grupo 7 — Eventos y Alarmas (2 tools)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 31 | `get_vm_events_tool` | `vm_name`, `max_events=20` | JSON historial de eventos | No |
| 32 | `get_active_alarms_tool` | — | JSON alarmas críticas y advertencias | No |

---

### Grupo 8 — Fechas (1 tool)

| # | Nombre | Parámetros | Retorna | Destructiva |
|---|--------|-----------|---------|:-----------:|
| 33 | `get_vm_creation_dates_tool` | `vm_names: str` | JSON fechas de creación | No |

---

## Herramientas Destructivas — Resumen de Riesgo

```mermaid
quadrantChart
    title Herramientas por Riesgo e Irreversibilidad
    x-axis Bajo Riesgo --> Alto Riesgo
    y-axis Reversible --> Irreversible
    quadrant-1 Críticas
    quadrant-2 Monitorear
    quadrant-3 Seguras
    quadrant-4 Con Cautela

    delete_vms_tool: [0.95, 0.95]
    reconfigure_vm_tool: [0.70, 0.75]
    revert_snapshot_tool: [0.65, 0.80]
    delete_snapshot_tool: [0.75, 0.85]
    remove_vm_nic_tool: [0.60, 0.70]
    change_vm_network_tool: [0.55, 0.60]
    power_operations_tool: [0.40, 0.30]
    deploy_dev_env: [0.30, 0.20]
    clone_mcu_template: [0.25, 0.15]
    list_vms_for_user: [0.05, 0.05]
    get_templates: [0.05, 0.05]
    get_esxi_performance_tool: [0.05, 0.05]
```

---

## Patrón de Implementación de una Tool

```python
# En server/mcp_tool_registry.py → create_tool_functions()

def list_vms_for_user(username_: str = None) -> str:
    """
    Lista las VMs pertenecientes al usuario actual en vCenter.
    Filtra por prefijo de abreviatura de usuario (session_abbr).
    Retorna JSON con nombre, estado, CPU, RAM y host de cada VM.
    """
    # username y session_abbr capturados del closure externo
    try:
        with log_context(operation="list_vms_for_user", user=username):
            si = self.get_user_si(username)          # Pool de conexiones
            vms = get_vms_for_user(si, session_abbr) # pyvmomi wrapper
            logger.log_business_operation(
                "vm_list", {"count": len(vms), "user": session_abbr}
            )
            return json.dumps(vms, ensure_ascii=False)
    except Exception as e:
        logger.log_system_error("list_vms_for_user", str(e))
        return f"Error listando VMs: {str(e)}"
```

### Flujo interno de una tool

```mermaid
sequenceDiagram
    participant LLM as LangChain LLM
    participant Tool as StructuredTool
    participant Registry as MCPToolRegistry closure
    participant Pool as ConnectionPool
    participant vC as vCenter API

    LLM->>Tool: llamar list_vms_for_user()
    Tool->>Registry: func() [closure captura username]
    Registry->>Registry: log_context(operation, user)
    Registry->>Pool: get_user_si(username)
    Pool->>Pool: Buscar conexión válida en _pool
    Pool-->>Registry: ServiceInstance (si)
    Registry->>vC: container_view(vim.VirtualMachine)
    vC-->>Registry: [ManagedObject, ...]
    Registry->>Registry: Filtrar por session_abbr
    Registry->>Registry: log_business_operation
    Registry-->>Tool: JSON string
    Tool-->>LLM: "VMs encontradas: [...]"
```
