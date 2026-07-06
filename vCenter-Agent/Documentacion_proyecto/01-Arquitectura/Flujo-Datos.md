---
tipo: arquitectura
estado: actual
relacionado:
  - "[[Arquitectura-Sistema]]"
  - "[[Arquitectura-Chat]]"
  - "[[Arquitectura-Agente-vCenter]]"
  - "[[Orquestador]]"
tags: [arquitectura, flujo, diagramas]
version: 1.0
ultima_actualizacion: 2026-04-20
---

# 🔄 Flujo de Datos - vCenter Multi-Agent System

> Diagramas de secuencia y flujos de datos entre componentes del sistema multi-agente.

---

## 📋 Resumen

Este documento detalla los flujos de datos críticos del sistema:
1. **Query End-to-End**: Desde usuario hasta respuesta
2. **Clasificación 4-Capas**: Decisión de routing
3. **vCenter Operation**: Ejecución de herramienta MCP
4. **RAG v2.4 Pipeline**: Búsqueda de documentación
5. **Autenticación y Sesión**: Login y validación

---

## 1️⃣ Flujo End-to-End: Query de Usuario

```mermaid
sequenceDiagram
    actor Usuario
    participant Browser
    participant Flask
    participant Auth as Middleware Auth
    participant Orch as Orquestador
    participant Classifier as Clasificador 4-Capas
    participant VCAgent as vCenter Agent
    participant MCP as MCP Tools
    participant vCenter as vCenter API
    
    Usuario->>Browser: Escribe "Despliega VM test-01"
    Browser->>Flask: POST /chat {input, username}
    
    Note over Flask: T0 = Recepción request
    
    Flask->>Auth: @authenticated_action
    Auth->>Auth: Valida session['username']
    
    alt Sesión válida
        Auth-->>Flask: ✓ Authorized
    else Sesión inválida
        Auth-->>Browser: 401 Unauthorized
        Browser->>Usuario: Redirigir a /login
    end
    
    Flask->>Orch: chat_api(input, username)
    
    Note over Orch: T1 = Inicio clasificación
    
    Orch->>Classifier: classify_task(message)
    
    Note over Classifier: Layer 0: Keywords exclusivos<br/>Layer 1: Regex críticos<br/>Layer 2: Intent detection<br/>Layer 3: Scoring<br/>Layer 4: LLM fallback
    
    Classifier-->>Orch: agent = "vcenter"
    
    Note over Orch: T2 = Routing decidido<br/>Check sticky routing (last_agent)
    
    Orch->>Orch: ACTIVE_SESSIONS[user]['last_agent'] = 'vcenter'
    
    Note over Orch: T3 = Formatter opcional (si enabled)
    
    Orch->>VCAgent: process_vcenter_query(username, message)
    
    VCAgent->>VCAgent: get_user_context(username)
    
    Note over VCAgent: Si primera vez:<br/>Crear AgentExecutor + MCP Tools
    
    VCAgent->>MCP: AgentExecutor.invoke({input})
    
    Note over MCP: LLM analiza y selecciona tool:<br/>"deploy_dev_env"
    
    MCP->>vCenter: CloneVM_Task(...)
    Note over MCP: Despliegue de 1..N VMs según tool
    
    Note over vCenter: vCenter ejecuta clonación<br/>8-15 segundos típicos
    
    vCenter-->>MCP: Task result: SUCCESS
    MCP-->>VCAgent: {"status": "ok", "vm": "test-01"}
    
    VCAgent->>VCAgent: ConversationBufferMemory.save()
    VCAgent-->>Orch: "✅ VM test-01 desplegada..."
    
    Note over Orch: T4 = Respuesta lista
    
    Orch->>Orch: Log a business.log + audit.log
    Orch-->>Flask: {response, agent: "vcenter"}
    
    Flask-->>Browser: JSON 200 OK
    Browser->>Browser: renderMarkdown(response)
    Browser->>Usuario: Muestra respuesta formateada
    
    Note over Usuario: T5 = Usuario ve respuesta<br/>Total: 8-16 segundos
```

### Métricas Típicas

| Fase | Componente | Latencia P50 | Latencia P95 |
|------|-----------|--------------|--------------|
| T0→T1 | Flask middleware | 5-10ms | 15-25ms |
| T1→T2 | Clasificación 4-capas | 50-150ms | 200-400ms |
| T2→T3 | Sticky routing check | <5ms | <10ms |
| T3→T4 | Formatter (opcional) | 200-500ms | 800-1200ms |
| T4→T5 | vCenter operation | 300ms-15s | 1-30s |
| **Total Query** | **End-to-end** | **2-6s** | **5-30s** |

---

## 2️⃣ Clasificador 4-Capas: Decisión de Routing

```mermaid
flowchart TD
    Start([Mensaje del Usuario]) --> Layer0{Layer 0:<br/>Keywords<br/>Exclusivos}

    Layer0 -->|mcu, eqsim SOLO vcenter| ReturnVC1[✓ vcenter]
    Layer0 -->|doors, midat SOLO docs| ReturnDoc1[✓ documentation]
    Layer0 -->|No match| Layer1{Layer 1:<br/>Regex<br/>Críticos}

    Layer1 -->|despliega/crea/borra + vm/mcu| ReturnVC2[✓ vcenter<br/>HIGH confidence]
    Layer1 -->|cómo/qué es + funciona/configura| ReturnDoc2[✓ documentation<br/>HIGH confidence]
    Layer1 -->|No match| Layer2{Layer 2:<br/>Intent<br/>Detection}

    Layer2 -->|Imperativo:<br/>despliega, muestra| IncreaseVC[score_vcenter += 3]
    Layer2 -->|Learning:<br/>qué es, cómo| IncreaseDoc[score_doc += 3]
    Layer2 -->|Ambiguo| Layer3{Layer 3:<br/>Weighted<br/>Scoring}

    IncreaseVC --> Layer3
    IncreaseDoc --> Layer3

    Layer3 --> CountKeywords[Contar keywords<br/>de agents.yaml]
    CountKeywords --> CalculateScores["score_vcenter = suma pesos VC<br/>score_doc = suma pesos DOC"]

    CalculateScores --> CheckDelta{"diferencia de scores<br/>mayor que umbral 2.0?"}

    CheckDelta -->|delta mayor 2.0| CompareScores{score_vcenter<br/>mayor score_doc?}
    CompareScores -->|Sí| ReturnVC3[✓ vcenter]
    CompareScores -->|No| ReturnDoc3[✓ documentation]

    CheckDelta -->|empate, delta menor 2.0| Layer4{Layer 4:<br/>LLM<br/>Fallback}

    Layer4 --> LLMCall[LLM analiza mensaje<br/>gpt-oss:20b]
    LLMCall --> ParseResponse{Respuesta<br/>contiene<br/>vcenter?}

    ParseResponse -->|Sí| ReturnVC4[✓ vcenter]
    ParseResponse -->|No, contiene documentation| ReturnDoc4[✓ documentation]
    ParseResponse -->|No contiene ninguno| Heuristic[heuristic_fallback]

    Heuristic --> CheckLength{len msg<br/>mayor 100?}
    CheckLength -->|Sí| ReturnDoc5[✓ documentation<br/>Query larga = docs]
    CheckLength -->|No| ReturnVC5[✓ vcenter<br/>Default]

    style ReturnVC1 fill:#90EE90
    style ReturnVC2 fill:#90EE90
    style ReturnVC3 fill:#90EE90
    style ReturnVC4 fill:#90EE90
    style ReturnVC5 fill:#90EE90
    style ReturnDoc1 fill:#87CEEB
    style ReturnDoc2 fill:#87CEEB
    style ReturnDoc3 fill:#87CEEB
    style ReturnDoc4 fill:#87CEEB
    style ReturnDoc5 fill:#87CEEB
```

### Tabla de Decisión por Capa

| Layer | Método | Confianza | Casos de Uso | Exit Rate |
|-------|--------|-----------|--------------|-----------|
| **0** | Exclusive keywords | CRITICAL | Términos proyecto específicos ("mcu", "doors") | ~15% |
| **1** | Regex patterns | HIGH | Frases imperativas + objeto ("despliega una vm") | ~35% |
| **2** | Intent detection | MEDIUM | Detecta imperativo vs learning question | ~25% |
| **3** | Weighted scoring | MEDIUM | Cuenta keywords ponderadas de agents.yaml | ~20% |
| **4** | LLM + Heuristic | LOW-MEDIUM | Análisis semántico completo + fallback | ~5% |

---

## 3️⃣ Operación vCenter: Deploy (entorno / clonación)

```mermaid
sequenceDiagram
    participant Agent as vCenter Agent
    participant Executor as AgentExecutor
    participant LLM as ChatOllama
    participant Registry as MCPToolRegistry
    participant Pool as ConnectionPool
    participant vC as vCenter API
    participant ESXi as ESXi Host
    
    Agent->>Executor: invoke({"input": "Despliega VM..."})
    Executor->>LLM: Analizar input + system_prompt
    
    Note over LLM: Razonamiento interno:<br/>1. Usuario quiere desplegar entorno / VM(s)<br/>2. Necesito tool: deploy_dev_env (o clone_mcu_template)<br/>3. Parámetros: username_ + (opcional) versión plantilla
    
    LLM-->>Executor: Tool: deploy_dev_env<br/>Args: {username_: "JaMB", mcu_template_name: "p28", eqsim_template_name: "p28"}
    
    Executor->>Registry: deploy_dev_env(username_, mcu_template_name, eqsim_template_name)
    
    Note over Registry: Cierre con contexto usuario:<br/>username, session_abbr
    
    Registry->>Pool: get_connection(host, user, pwd)
    
    alt Conexión existe y es válida
        Pool-->>Registry: Reusa ServiceInstance
    else Primera conexión o expirada
        Pool->>vC: SmartConnect(host, 443, user, pwd)
        vC-->>Pool: Nueva ServiceInstance
        Pool-->>Registry: ServiceInstance
    end
    
    Registry->>Registry: get_dynamic_mapping()
    Note over Registry: Autodescubrimiento de plantillas + caché TTL
    Registry->>Registry: seleccionar plantilla (alias pNN → nombre real)
    
    Registry->>vC: CloneVM_Task(vm, folder, spec)
    
    Note over vC: Inicia tarea asíncrona
    
    vC->>ESXi: Ejecuta clonación en host
    
    Note over ESXi: 1. Reserva recursos<br/>2. Copia VMDK desde template<br/>3. Configura NICs<br/>4. Registra VM<br/>5. Power on (si requested)
    
    loop Poll cada 2s
        Registry->>vC: task.info.state
        vC-->>Registry: running | success | error
    end
    
    ESXi-->>vC: VM clonada exitosamente
    vC-->>Registry: Task SUCCESS
    
    Registry->>Registry: log_business_operation("vm_deploy")
    Registry-->>Executor: {"status": "ok", "vms": ["MCU-...", "Eqsim-..."], "detail": "..."}
    
    Executor->>LLM: Formatear respuesta con resultado
    LLM-->>Executor: "✅ VM test-01 desplegada en..."
    
    Executor->>Agent: ConversationBufferMemory.save_context()
    Executor-->>Agent: Response final
```

---

## 4️⃣ RAG v2.4 Pipeline: Query Documentación

```mermaid
flowchart TB
    Start(["¿Cómo configurar DNS en Ubuntu?"]) --> Phase1[1. Search Mode Detection]

    Phase1 --> CheckMode{Detecta<br/>keywords}
    CheckMode -->|SOLO vcenter| Strict[Strict Mode<br/>Solo carpeta vcenter]
    CheckMode -->|busca en esxi| Boost[Boosting Mode<br/>x2 boost esxi]
    CheckMode -->|Normal| Global[Global Mode<br/>Todas carpetas]

    Strict --> Phase2
    Boost --> Phase2
    Global --> Phase2[2. Query Normalization]

    Phase2 --> RemoveStop[Eliminar 23 stopwords<br/>españolas]
    RemoveStop --> RemoveFiller[Eliminar 12 filler<br/>phrases]
    RemoveFiller --> Phase3[3. Query Expansion]

    Phase3 --> Expand62["Expandir con 62 familias de términos<br/>dns → dns, bind, bind9, named, zona"]
    Expand62 --> Phase4[4. Hybrid Retrieval]

    Phase4 --> Vector[ChromaDB<br/>Vector Search<br/>nomic-embed-text]
    Phase4 --> Keyword[BM25<br/>Keyword Search<br/>k1=1.5, b=0.75]

    Vector --> Cache{Embedding<br/>en cache?}
    Cache -->|Hit 30%| UseCached[Usar cached<br/>embedding]
    Cache -->|Miss 70%| GenNew[Generar nuevo<br/>embedding Ollama]

    UseCached --> VectorResults[Top 20 docs<br/>score 0 a 1]
    GenNew --> VectorResults

    Keyword --> BM25Results[Top 20 docs<br/>score raw]

    VectorResults --> Normalize[Normalización<br/>scores 0 a 1]
    BM25Results --> Normalize

    Normalize --> AlphaCalc{Calcular<br/>alpha adaptativo}
    AlphaCalc -->|Query menos de 5 palabras| Alpha35[alpha = 0.35<br/>Favor keywords]
    AlphaCalc -->|Query más de 10 palabras| Alpha70[alpha = 0.70<br/>Favor semantic]
    AlphaCalc -->|Query media| Alpha50[alpha = 0.50<br/>Balance]

    Alpha35 --> Combine["Combinar scores<br/>alpha x vector + 1-alpha x bm25"]
    Alpha50 --> Combine
    Alpha70 --> Combine

    Combine --> InternalBoost[Internal Docs Boost<br/>+75% si archivo .md]
    InternalBoost --> Top40[Top 40 candidatos]

    Top40 --> Phase5[5. Folder Filtering]

    Phase5 --> ApplyFilter{Aplicar<br/>search mode}
    ApplyFilter -->|Strict| KeepOnly[Mantener solo<br/>carpeta target]
    ApplyFilter -->|Boost| MultiplyScores[x2 scores<br/>carpeta target]
    ApplyFilter -->|Global| NoFilter[Sin filtrado]

    KeepOnly --> Phase6
    MultiplyScores --> Phase6
    NoFilter --> Phase6[6. Reranking]

    Phase6 --> Rerank["Heurístico:<br/>25% original + 40% term_freq<br/>+ 15% length + 10% position"]

    Rerank --> Top8[Top 8 docs<br/>finales]

    Top8 --> Phase7[7. Metrics Logging]
    Phase7 --> LogMetrics[logs/retrieval_metrics.jsonl]

    LogMetrics --> Phase8[8. LLM Response]
    Phase8 --> LLMGen[gpt-oss:20b<br/>num_ctx=16384]

    LLMGen --> Response(["Respuesta con<br/>contexto de docs"])

    style Start fill:#FFE4B5
    style Response fill:#90EE90
    style Phase4 fill:#87CEEB
    style Top8 fill:#FFA07A
```

### Parámetros RAG v2.4

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **chunk_size** | 1400 chars | Balance contexto vs granularidad |
| **chunk_overlap** | 350 chars | Preservar continuidad semántica |
| **initial_k** | 40 | Candidatos pre-reranking |
| **rerank_top_k** | 8 | Contexto final para LLM |
| **base_alpha** | 0.5 | Balance vector/keyword (adaptativo) |
| **bm25_k1** | 1.5 | Saturación term frequency |
| **bm25_b** | 0.75 | Normalización document length |
| **internal_boost** | 0.75 | +75% para docs .md del proyecto |
| **embedding_cache** | 1000 | LRU cache queries frecuentes |
| **num_ctx** | 16384 | Contexto Ollama (vs 8192 vCenter) |

---

## 5️⃣ Autenticación y Sesión

```mermaid
sequenceDiagram
    actor Usuario
    participant Browser
    participant Flask
    participant Auth as auth_module
    participant PassMgr as password_manager
    participant SQLite as data/auth.db
    participant Sessions as ACTIVE_SESSIONS
    
    Usuario->>Browser: Navega a /
    Browser->>Flask: GET /
    Flask->>Flask: @authenticated_action
    Flask->>Sessions: session.get('username')
    
    alt Sin sesión
        Flask-->>Browser: redirect('/login')
        Browser->>Usuario: Muestra login form
        
        Usuario->>Browser: Ingresa user/pwd
        Browser->>Flask: POST /login {username, password}
        
        Flask->>Auth: authenticate_user(username, pwd)
        Auth->>SQLite: SELECT * FROM users WHERE username=?
        SQLite-->>Auth: {username, password_hash, role}
        
        Auth->>PassMgr: verify_password(pwd, hash)
        PassMgr->>PassMgr: bcrypt.checkpw(pwd, hash)
        
        alt Contraseña correcta
            PassMgr-->>Auth: ✓ Valid
            Auth-->>Flask: {valid: true, role: "user"}
            
            Flask->>Flask: session['username'] = username
            Flask->>Flask: session_id = generate_uuid()
            Flask->>Sessions: ACTIVE_SESSIONS[session_id] = {<br/>  username, role, login_time,<br/>  last_activity, last_agent<br/>}
            
            Flask->>SQLite: UPDATE users SET last_login=NOW()
            Flask-->>Browser: redirect('/chat')
            
        else Contraseña incorrecta
            PassMgr-->>Auth: ✗ Invalid
            Auth->>Auth: rate_limiter.record_failure(username)
            
            alt < 5 intentos
                Auth-->>Flask: {valid: false}
                Flask-->>Browser: "Credenciales inválidas"
            else ≥ 5 intentos
                Auth->>Auth: block_user(username, 5min)
                Auth-->>Flask: {valid: false, blocked: true}
                Flask-->>Browser: "Usuario bloqueado 5 min"
            end
        end
        
    else Con sesión válida
        Sessions->>Sessions: check timeout (3600s)
        
        alt Sesión vigente
            Sessions-->>Flask: ✓ Valid session
            Flask->>Sessions: update last_activity
            Flask-->>Browser: render('/chat')
            
        else Sesión expirada
            Sessions->>Sessions: delete ACTIVE_SESSIONS[session_id]
            Sessions-->>Flask: ✗ Expired
            Flask-->>Browser: redirect('/login?expired=true')
        end
    end
```

### Dual Session System

```mermaid
graph TB
    subgraph "In-Memory (ACTIVE_SESSIONS)"
        AS[ACTIVE_SESSIONS dict]
        AS --> K1["session_id contiene:<br/>username, role,<br/>login_time, last_activity,<br/>last_agent, last_agent_time"]
        K1 --> T1[Timeout: 3600s<br/>Limpieza automática]
    end

    subgraph "Persistent (SQLite)"
        SM[SessionManager<br/>data/users.db]
        SM --> T2[Timeout configurable<br/>Por defecto: 3600s]
        SM --> U1[user_contexts dict<br/>AgentExecutor por usuario]
    end

    AS -.->|username| SM
    SM -.->|session validation| AS

    style AS fill:#FFE4B5
    style SM fill:#87CEEB
```

**Diferencias clave:**
- **ACTIVE_SESSIONS**: Flask sessions, routing orquestador, in-memory
- **SessionManager**: Agent contexts, vCenter connections, persistent SQLite

---

## 📚 Documentos Relacionados

- [[Arquitectura-Sistema]] - Visión general completa
- [[Arquitectura-Chat]] - Sistema conversacional
- [[Arquitectura-Agente-vCenter]] - Agente VMware
- [[Orquestador]] - Clasificador 4-capas detallado
- [[Sistema-MCP]] - Herramientas vCenter
- [[Agente-Documentacion]] - RAG v2.4 completo
- [[Autenticacion]] - RBAC y sesiones

---

*Última actualización: 2026-03-24 | v1.0*
