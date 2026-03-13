# Arquitectura del Agente vCenter

## Visión General

El agente vCenter forma parte del sistema multi-agente. Recibe consultas en lenguaje natural del Orquestador y las ejecuta sobre infraestructura VMware vCenter a través de una capa MCP de 33 herramientas.

---

## Diagrama de Arquitectura General

```mermaid
graph TB
    subgraph Frontend["Frontend (Chat UI)"]
        UI[orchestrator_chat_auth.html]
        JS[orchestrator_chat_auth.js]
        UI --> JS
    end

    subgraph Orchestrator["Orquestador (main_agent.py)"]
        FLASK[Flask Routes<br/>/chat, /chat/stream]
        CLASSIFIER[QueryClassifier<br/>4 capas de clasificación]
        FLASK --> CLASSIFIER
    end

    subgraph vCenterAgent["Agente vCenter (agent.py)"]
        ENTRY[process_vcenter_query]
        CTX[get_user_context]
        EXECUTOR[AgentExecutor<br/>LangChain]
        LLM[ChatOllama<br/>gpt-oss:20b<br/>num_ctx=8192]
        MEMORY[ConversationBufferMemory<br/>por usuario]
        ENTRY --> CTX
        CTX --> EXECUTOR
        EXECUTOR --> LLM
        EXECUTOR --> MEMORY
    end

    subgraph MCPLayer["Capa MCP"]
        REGISTRY[MCPToolRegistry<br/>create_tool_functions]
        WRAPPERS[MCPToolWrappers<br/>StructuredTool]
        SERVER[FastMCP Server<br/>33 @mcp.tool]
        REGISTRY --> WRAPPERS
        WRAPPERS --> EXECUTOR
        REGISTRY -.->|protocolo MCP| SERVER
    end

    subgraph Infrastructure["Infraestructura"]
        POOL[VCenterConnectionPool<br/>max=5, timeout=30s]
        VCENTER[(VMware vCenter)]
        VCSIM[(vcsim<br/>simulador)]
        POOL -->|SmartConnect| VCENTER
        POOL -.->|fallback| VCSIM
    end

    subgraph Storage["Persistencia"]
        SQLITE[(SQLite<br/>data/users.db)]
        AUTH[(SQLite<br/>data/auth.db)]
    end

    JS -->|POST /chat| FLASK
    JS -->|SSE /chat/stream| FLASK
    CLASSIFIER -->|agent=vcenter| ENTRY
    REGISTRY --> POOL
    CTX --> SQLITE
```

---

## Flujo de Procesamiento de una Query

```mermaid
sequenceDiagram
    actor Usuario
    participant UI as Chat UI
    participant Flask as Flask /chat
    participant Orch as Orquestador
    participant Agent as agent.py
    participant Registry as MCPToolRegistry
    participant Pool as ConnectionPool
    participant vC as vCenter

    Usuario->>UI: "Muéstrame mis VMs"
    UI->>Flask: POST /chat {input: "..."}
    Flask->>Orch: classify_task(message)

    Note over Orch: 4 capas de clasificación:<br/>Layer 0: keywords exclusivos<br/>Layer 1: regex críticos<br/>Layer 2: intent detection<br/>Layer 3: scoring ponderado

    Orch-->>Flask: {agent: "vcenter"}
    Flask->>Agent: process_vcenter_query(username, msg)

    Agent->>Agent: get_user_context(username)

    alt Primera vez del usuario
        Agent->>Registry: create_tool_functions(username, abbr)
        Registry->>Pool: get_connection(host, user, pwd)
        Pool->>vC: SmartConnect(host, port=443)
        vC-->>Pool: ServiceInstance
        Pool-->>Registry: si (connection)
        Registry-->>Agent: {tool_name: func, ...} (33 tools)
        Agent->>Agent: create_mcp_aware_tools() → StructuredTools
        Agent->>Agent: AgentExecutor(llm, tools, memory)
    end

    Agent->>Agent: executor.invoke({input: "JaMB dice: Muéstrame mis VMs"})

    Note over Agent: LLM selecciona herramienta:<br/>list_vms_for_user()

    Agent->>Registry: list_vms_for_user(username)
    Registry->>vC: container_view(vim.VirtualMachine)
    vC-->>Registry: [VM1, VM2, ...]
    Registry-->>Agent: JSON con lista de VMs

    Agent->>Agent: LLM formatea respuesta + ConversationBufferMemory
    Agent-->>Flask: "Tienes 3 VMs activas: ..."
    Flask-->>UI: {response: "...", status: "ok"}
    UI->>Usuario: Respuesta renderizada con marked.js
```

---

## Inicialización del Agente por Usuario

```mermaid
flowchart TD
    A[get_user_context llamado] --> B{¿Contexto existe\nen user_contexts?}

    B -->|Sí| C[Retornar AgentExecutor cacheado]

    B -->|No| D[Crear nuevo contexto de usuario]
    D --> E[Obtener session_abbr\ndel user_mapping]
    E --> F[ConversationBufferMemory\npor usuario]
    F --> G[MCPToolRegistry.create_tool_functions\nusername, session_abbr]
    G --> H[VCenterConnectionPool.get_connection\nhost, user, pwd]
    H --> I{¿Conexión existente\ny válida?}

    I -->|Sí| J[Reusar ServiceInstance]
    I -->|No| K[SmartConnect a vCenter]
    K --> L{¿Conexión exitosa?}
    L -->|Sí| M[Nueva ServiceInstance en pool]
    L -->|No| N{¿Fallback habilitado?}
    N -->|Sí| O[Conectar a vcsim]
    N -->|No| P[Error: vCenter no disponible]

    J --> Q[Crear 33 closures de tools\ncon contexto del usuario]
    M --> Q
    O --> Q

    Q --> R[create_mcp_aware_tools\nStructuredTool.from_function]
    R --> S[create_tool_calling_agent\nllm + tools + prompt]
    S --> T[AgentExecutor\nagent + tools + memory]
    T --> U[Cachear en user_contexts]
    U --> C
```

---

## Selección de Herramienta por el LLM

```mermaid
flowchart LR
    A[user_input] --> B[AgentExecutor.invoke]
    B --> C[LLM analiza input\n+ system_prompt\n+ chat_history]

    C --> D{¿Qué herramienta\nseleccionar?}

    D -->|"mis vms / listar"| E[list_vms_for_user]
    D -->|"deploy / desplegar"| F[deploy_dev_env]
    D -->|"apagar / encender"| G[power_operations_tool]
    D -->|"snapshot"| H[create/list/revert/delete_snapshot_tool]
    D -->|"recursos / rendimiento"| I[get_esxi_performance_tool]
    D -->|"informe / reporte"| J[generate_resource_report_tool]
    D -->|"eliminar / borrar"| K[delete_vms_tool]
    D -->|"hosts / datastores"| L[get_hosts/datastores_tool]

    E & F & G & H & I & J & K & L --> M[Ejecutar función\nen MCPToolRegistry]
    M --> N[pyvmomi → vCenter]
    N --> O[Resultado JSON]
    O --> P[LLM post-procesa\ny formatea]
    P --> Q[Respuesta final]
```

---

## System Prompt del Agente

El system prompt define el comportamiento del agente. Se carga una vez al inicializar `agent.py` y se reutiliza para todos los usuarios.

```mermaid
mindmap
  root((System Prompt<br/>vCenter Agent))
    Identidad
      "Eres un agente especializado en vCenter"
      "Respondes en español"
    Selección de Herramientas
      "Usa siempre la herramienta más específica"
      "Nunca inventes datos"
      "Si no hay herramienta apropiada, informa al usuario"
    Contexto de Usuario
      "Identifica al usuario por su abreviatura"
      "session_abbr determina namespace de VMs"
    Seguridad
      "No expongas credenciales"
      "Confirma antes de acciones destructivas"
    Formato de Respuesta
      "Responde en Markdown"
      "Usa tablas para listas de VMs"
      "Incluye métricas cuando estén disponibles"
```

---

## Estructura de Archivos del Agente

```
vcenter_agent_system/
├── src/
│   └── core/
│       └── agent.py                 ← Entry point, AgentExecutor, sesiones Flask
│
├── server/
│   ├── mcp_tool_registry.py         ← 33 closures MCP por usuario (CRÍTICO)
│   ├── mcp_tool_wrappers.py         ← LangChain StructuredTool adapters
│   └── mcp_vcenter_server.py        ← FastMCP server (protocolo MCP externo)
│
└── src/utils/
    └── vcenter_tools.py             ← pyvmomi wrappers + VCenterConnectionPool
```

---

## Configuración Clave

| Parámetro | Valor | Archivo | Motivo |
|-----------|-------|---------|--------|
| `num_ctx` | 8192 tokens | `agent.py` | Contexto Ollama expandido (default: 4096) |
| `model` | `gpt-oss:20b` | `agent.py` | Modelo principal de razonamiento |
| `max_connections` | 5 | `vcenter_tools.py` | Pool máximo de conexiones vCenter |
| `connection_timeout` | 30s | `vcenter_tools.py` | Timeout para liberar conexiones inactivas |
| `memory_type` | ConversationBufferMemory | `agent.py` | Historial de chat por usuario |
| `verbose` | False | `agent.py` | No expone razonamiento interno al usuario |
