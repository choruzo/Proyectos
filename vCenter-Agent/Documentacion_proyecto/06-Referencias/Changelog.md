---
tipo: referencia
estado: actual
relacionado:
  - "[[Propuestas-Funcionales]]"
  - "[[Arquitectura-Sistema]]"
tags: [changelog, mejoras, historial]
version: 3.7
ultima_actualizacion: 2026-05-11
---

# 📝 Changelog - vCenter Multi-Agent System

> Historial de mejoras, cambios y evoluciones del sistema multi-agente.

***
## Versión 3.7 - LangGraph Fase 5: Consolidación y Memoria de Sesión Unificada (Mayo 2026)

### 🗄️ Sesión de Agente Unificada en LangGraph Checkpointer

Elimina los sistemas de sesión en-memoria redundantes (`ACTIVE_SESSIONS` dict, `user_memories` dicts, `ConversationBufferMemory`) y unifica toda la persistencia de **estado conversacional** en el checkpointer SQLite de LangGraph (`data/lg_state.db`). La sesión web/autenticación sigue gestionada por `data/users.db` sin cambios.

#### Cambios implementados

- **`src/utils/langgraph_session_cleanup.py`** *(nuevo)*: Módulo de utilidades para la gestión unificada de sesiones LangGraph:
  - `build_orchestrator_thread_id(username)` → `{username}_orch`
  - `build_vcenter_thread_id(username)` → `{username}_vcenter`
  - `build_documentation_thread_id(username)` → `{username}_doc`
  - `clear_user_session_threads()` — limpia los 3 threads de un usuario
  - `expire_user_session_threads_if_needed()` — expiración lógica basada en `session_last_activity`
- **`src/api/main_agent.py`**: `ACTIVE_SESSIONS` dict eliminado. Función `_expire_conversation_threads_if_needed()` reemplaza la lógica de expiración manual; se invoca antes de cada mensaje. `/chat/clear` usa `clear_user_session_threads()`.
- **`src/core/agent.py`**: `user_memories` dict y `ConversationBufferMemory` eliminados. La memoria conversacional del agente vCenter persiste automáticamente vía checkpointer (`{username}_vcenter`).
- **`src/core/doc_consultant.py`**: `user_memories` dict eliminado. Memoria del agente de documentación en checkpointer (`{username}_doc`).
- **`src/api/orchestrator_graph.py`**: `OrchestratorState` ampliado con `session_last_activity: float` — actualizado en cada nodo para habilitar la expiración lógica.

#### Esquema de persistencia unificado

```
data/lg_state.db  ← estado conversacional (short-term)
├── checkpoints
│   ├── thread_id: {username}_orch      → estado del orquestador
│   ├── thread_id: {username}_vcenter   → estado del agente vCenter
│   └── thread_id: {username}_doc       → estado del agente de documentación
└── writes

data/users.db     ← sesiones web y autenticación (sin cambios)
data/auth.db      ← credenciales y roles (sin cambios)
```

#### Comparativa antes / después

| Aspecto | Antes | Ahora |
|---------|-------|-------|
| Estado conversacional | `ACTIVE_SESSIONS` dict (RAM) + `user_memories` dict (RAM) | Checkpointer SQLite (`data/lg_state.db`) |
| Memoria por agente | `ConversationBufferMemory` por usuario en RAM | Thread LangGraph por usuario en disco |
| Persistencia entre reinicios | No — se pierde al reiniciar Flask | Sí — SQLite persiste |
| Limpieza de sesión expirada | Comprobación inline al inicio de cada petición | `_expire_conversation_threads_if_needed()` |
| `/chat/clear` | Limpiaba dicts en RAM | Llama a `clear_user_session_threads()` |

#### Tests

- **`unitary_test/test_session_consolidation.py`** *(nuevo)*: 7 tests, todos pasan.

| Test | Qué verifica |
|------|-------------|
| `test_thread_id_builders_use_user_scope` | Thread IDs correctos por agente y usuario |
| `test_auth_session_manager_validates_web_session_independently` | Sesiones web siguen en `data/users.db` |
| `test_orchestrator_sticky_routing_persists_in_graph_state` | Sticky routing persiste entre invocaciones |
| `test_doc_graph_preserves_short_term_messages_with_stable_thread_id` | Memoria conversacional RAG persiste con `thread_id` estable |
| `test_doc_consultant_process_query_uses_stable_thread_id` | `process_query()` usa el mismo `thread_id` en llamadas sucesivas |
| `test_clear_user_session_threads_clears_all_graphs` | `clear_user_session_threads()` limpia los 3 grafos |
| `test_expire_user_session_threads_if_needed_uses_last_activity` | Expiración correcta basada en `session_last_activity` |

#### Archivos no modificados

El catálogo MCP (36 herramientas), el sistema RAG v2.4, la autenticación (`src/auth/`), los templates y el frontend no cambian.

***
## Versión 3.6 - LangGraph Fase 4: Pipeline RAG Adaptativo (Mayo 2026)

### 📄 Agente de Documentación Migrado a LangGraph StateGraph

Convierte el pipeline RAG lineal de `doc_consultant.py` en un `StateGraph` con lógica de reintento automático cuando el retrieval devuelve scores bajos. El comportamiento observable no cambia para el usuario; la diferencia es que ahora el sistema reintenta en modo global antes de abtenerse, en lugar de abtenerse directamente.

#### Cambios implementados

- **`src/core/doc_agent_graph.py`** *(nuevo)*: grafo RAG adaptativo con 6 nodos y routing condicional:
  - `expand_query`: normalización y expansión de la query (invoca `_prepare_rag_state`)
  - `retrieve`: retrieval híbrido (ChromaDB + BM25) con métricas JSONL
  - `retry_setup`: fuerza `search_mode="global"` e incrementa `retry_count`
  - `rerank`: pasa `retrieved_chunks` a `reranked_chunks` (hook para reranking futuro)
  - `generate`: llama al LLM con el prompt RAG construido
  - `abstain`: responde "no contiene información" con mensaje de abstención auditado
- **`src/core/doc_consultant.py`**: `process_query()` reescrita para usar `get_doc_graph().invoke()`. El bypass de exploración (`listar documentos`, `leer documento completo`) se mantiene antes de entrar al grafo. Cuatro métodos callback (`_doc_graph_expand_query`, `_doc_graph_retrieve`, `_doc_graph_generate`, `_doc_graph_abstain`) adaptan la lógica RAG existente a la interfaz del grafo.

#### Estado del grafo (`DocAgentState`)

```python
messages: Annotated[list, add_messages]  # historial conversacional
username: str
query: str                               # query original
cleaned_query: str | None               # tras normalización
expanded_query: str | None              # tras expansión
search_mode: str                        # "strict" | "boosting" | "global"
search_target_folder: str | None        # carpeta objetivo si strict/boosting
retrieved_chunks: list                  # resultado crudo del retrieval
reranked_chunks: list                   # resultado tras reranking
retrieval_score: float | None           # score promedio del top-k
retry_count: int                        # 0 o 1 (máximo 1 reintento)
final_response: str | None             # respuesta final al usuario
prompt: str | None                     # prompt enviado al LLM
sources: str | None                    # citas formateadas
abstention_message: str | None         # mensaje de abstención
is_procedural: bool                    # si la query es de pasos/procedimiento
search_mode_final: str | None          # modo de búsqueda efectivo final
long_term_context: str | None          # reservado para Hindsight
```

#### Flujo de reintento adaptativo

```
expand_query → retrieve
                  │
     score ≥ 0.35 │ score < 0.35 y modo ≠ global y retry_count < 1
                  │                    │
                  ↓                    ↓
               rerank           retry_setup (mode=global, retry_count+1)
                  │                    │
               generate          retrieve (global)
                  │                    │
                 END        score ≥ 0.35 → rerank → generate
                                       │
                             score < 0.35 → abstain → END
```

Score threshold: `0.35` (configurable en `build_doc_agent_graph`)

#### Ventajas sobre la implementación anterior

| Aspecto | Antes (lineal) | Ahora (grafo adaptativo) |
|---------|----------------|--------------------------|
| Score bajo en modo strict/boosting | Abstención directa | Reintento automático en modo global |
| Lógica de reintento | No existía | Rama explícita en el grafo |
| Auditoría de abstención | Log implícito | `_doc_graph_abstain` audita `reason`, `retry_count` y `search_mode_final` |
| Testabilidad | Acoplado a estado de instancia | Callbacks inyectables, grafo de test con `MemorySaver` |

#### Tests

- **`unitary_test/test_doc_agent_graph.py`** *(nuevo)*: 6 tests, todos pasan.

| Test | Qué verifica |
|------|-------------|
| `test_high_score_does_not_retry` | Score alto → 1 retrieve, genera respuesta, no abstiene |
| `test_low_score_retries_in_global_mode_once` | Score bajo en boosting → reintento global → genera |
| `test_second_low_score_abstains_after_one_retry` | Ambos intentos con score bajo → abstención final |
| `test_global_mode_low_score_abstains_without_retry` | Modo global con score bajo → abstención directa |
| `test_process_query_uses_graph_for_rag_mode` | `process_query()` delega al grafo para queries RAG |
| `test_process_query_keeps_exploration_bypass` | `"listar documentos"` no entra al grafo |

#### Archivos no modificados

El sistema RAG v2.4 completo (`hybrid_retriever.py`, `query_expander.py`, `reranker.py`, `search_modes.py`, etc.), el catálogo MCP, la autenticación, los templates y el frontend no cambian.

***
## Versión 3.5 - LangGraph Fase 3: Agente vCenter con Estado Tipado (Mayo 2026)

### 🤖 Agente vCenter Migrado a LangGraph StateGraph

Migra el agente vCenter de `AgentExecutor` + `ConversationBufferMemory` a un `StateGraph` completo con `ToolNode`, checkpointing SQLite y Progressive Disclosure preservado por turno. El comportamiento observable no cambia; la confirmación de acciones destructivas y la memoria conversacional son ahora persistentes entre reinicios de Flask.

#### Cambios implementados

- **`src/core/agent_graph.py`** *(versión 2.0 — Fase 3)*: actualizado de grafo mínimo (Fase 1) a grafo completo:
  - `execute_tools_node()`: usa `ToolNode` de `langgraph.prebuilt` con ejecución paralela de tool calls independientes. Reconstruye el subconjunto de tools por turno via `create_tool_functions_for_query()` (Progressive Disclosure preservado).
  - `_build_messages_for_llm()`: filtra `ToolMessage` y `AIMessage` con `tool_calls` del historial previo para evitar contaminación cross-turno con LLMs OpenAI-compatible estrictos.
  - `_select_tool_functions_from_state()`: usa `active_tool_names` del estado para consistencia entre nodo `agent` y nodo `tools`.
- **`src/core/agent.py`**: `AgentExecutor` eliminado. `process_vcenter_query()` invoca `vcenter_graph` (LangGraph). `ConversationBufferMemory` reemplazado por checkpointer SQLite. Nueva `clear_vcenter_conversation_state()`.
- **`src/api/main_agent.py`**: `chat_clear` llama a `clear_vcenter_conversation_state(username)`.

#### Estado del grafo (`VCenterAgentState`)

```python
messages: Annotated[list, add_messages]  # historial conversacional
username: str
session_abbr: str
pending_tool_call: dict | None           # tool destructiva pendiente
awaiting_confirmation: bool              # esperando confirmación
execution_confirmed: bool                # confirmado, ejecutar sin re-interceptar
last_user_query: str                     # última query del turno (para Progressive Disclosure)
active_tool_names: list[str]             # subconjunto de tools del turno
long_term_context: str | None            # reservado para Hindsight
```

#### Ventajas sobre la implementación anterior

| Aspecto | Antes (AgentExecutor) | Ahora (StateGraph) |
|---------|----------------------|-------------------|
| Memoria conversacional | In-memory (pérdida al reiniciar) | SQLite `data/lg_state.db` (persistente) |
| Paralelismo | Herramientas secuenciales | `ToolNode` ejecuta tool_calls en paralelo |
| Confirmaciones destructivas | Dict en memoria (Fase 1, ya SQLite) | Estado tipado en `VCenterAgentState` |
| Scratchpad cross-turno | Sin filtrado | `_build_messages_for_llm()` limpia historial |
| Progressive Disclosure | Por invocación completa | Por turno (nodo `agent` y nodo `tools` sincronizan subconjunto) |

#### Tests nuevos (añadidos a `test_vcenter_graph_interruption.py`)

| Clase | Qué verifica |
|-------|-------------|
| `TestToolNodeExecution.test_multiple_tool_calls_execute_in_parallel` | `Barrier` garantiza ejecución simultánea |
| `TestToolNodeExecution.test_tool_node_uses_same_subset_as_llm` | El `ToolNode` recibe el mismo subconjunto que `llm.bind_tools()` |
| `TestStatePersistence.test_previous_turn_tool_scratchpad_is_not_replayed_to_llm` | `ToolMessage` previos no contaminan el siguiente turno |
| `TestStatePersistence.test_invalid_historical_ai_turns_are_dropped_from_context` | AIMessages vacíos/de error se filtran del historial |
| `TestProgressCallback.*` | Eventos de progreso SSE emitidos correctamente |

#### Archivos no modificados

El catálogo MCP de 36 herramientas, el sistema RAG v2.4, la autenticación, los templates y el frontend no cambian.

***
## Versión 3.4 - LangGraph Fase 2: Orquestador como Grafo Supervisor (Mayo 2026)

### 🧠 Orquestador Migrado a LangGraph StateGraph

Convierte el routing imperativo de `main_agent.py` en un `StateGraph` explícito con aristas condicionales, checkpointing SQLite y soporte de streaming async SSE nativo. El comportamiento observable del sistema no cambia; el routing sigue siendo idéntico para el usuario.

#### Cambios implementados

- **`src/api/orchestrator_graph.py`** *(nuevo)*: grafo supervisor LangGraph con 4 nodos:
  - `classify`: encapsula `classify_task()` + sticky routing (180s). Registra `routing_layer` ("layer0"…"layer4_or_general", "sticky") en el estado.
  - `vcenter`: delega a `process_vcenter_query()`, aplica formateador si `ENABLE_QUERY_FORMATTING=true`.
  - `documentation`: delega a `process_documentation_query()` con mensaje original (sin formatear).
  - `general`: delega a `general_response()`.
- **`src/api/main_agent.py`**: instancia `orchestrator_graph` al arrancar (línea 524). Endpoint `/chat` usa `orchestrator_graph.invoke()`; endpoint `/chat/stream` usa `open_async_orchestrator_graph` + `astream_events()`.

#### Estado del grafo (`OrchestratorState`)

```python
messages: Annotated[list, add_messages]  # historial conversacional
username: str
active_agent: str | None                 # "vcenter" | "documentation" | "general"
last_agent: str | None                   # para sticky routing
last_agent_time: float                   # timestamp del último uso
routing_layer: str | None                # capa que tomó la decisión (debug)
followup_detected: bool                  # si el mensaje fue detectado como follow-up
used_sticky_routing: bool                # si se aplicó sticky routing en este turno
effective_message: str | None            # mensaje final enviado al subagente (puede ser formateado)
```

#### Mecanismos de progreso SSE (extensión al plan)

- **`progress_callback`** en `config["configurable"]`: los nodos emiten pasos intermedios ("Analizando routing...", "Ruta seleccionada.", "Delegando a agente vCenter...") sin acoplamiento al frontend.
- **`thinking_queue`** en `config["configurable"]`: canal para que los subagentes envíen eventos de herramientas en tiempo real durante el stream.

#### Streaming async

`open_async_orchestrator_graph` es un context manager async que instancia `AsyncSqliteSaver` para el endpoint SSE. Evita dependencias adicionales (`asgiref`) que el plan original mencionaba como alternativa.

#### Tests

- **`unitary_test/test_orchestrator_graph.py`** *(nuevo)*: 12 tests, todos pasan.
  - Routing a vcenter + formateador aplicado
  - Routing a documentation sin formateador
  - Routing a general
  - Sticky routing activo dentro de 180s
  - Sticky routing expira correctamente
  - Mensaje no follow-up ignora sticky
  - `routing_layer` reporta layer0/layer1/layer2 correctamente
  - Thread IDs por usuario son independientes
  - `progress_callback` y `thinking_queue` reciben pasos
  - Modo async con `AsyncSqliteSaver` y `astream_events`

#### Archivos no modificados

El catálogo MCP, el sistema RAG v2.4, la autenticación, los templates y el frontend no cambian. El clasificador de 4 capas (`query_classifier.py`) tampoco cambia — el grafo lo invoca sin modificar su firma.

***
## Versión 3.3 - LangGraph Fase 1: Confirmación de Acciones Destructivas (Mayo 2026)

### 🔒 Confirmación Persistente con LangGraph

Reemplaza el mecanismo ad-hoc de `_user_pending_confirmations` (dict en memoria) por un grafo LangGraph con checkpointing SQLite. El estado de confirmación ahora **sobrevive a reinicios del proceso Flask**.

#### Cambios implementados

- **`src/core/agent_graph.py`** *(nuevo)*: grafo LangGraph mínimo con 3 nodos (`agent`, `tools`, `confirm`) y routing condicional. Controla el flujo de 2 turnos para las 4 herramientas destructivas.
- **`server/mcp_tool_registry.py`**: eliminados `_user_pending_confirmations`, `_require_confirmation()` y sus 4 llamadas internas. Las herramientas destructivas ejecutan directamente; la interceptación es responsabilidad del grafo.
- **`src/core/agent.py`**: `process_vcenter_query()` reescrita para usar `vcenter_graph.invoke()`. El endpoint `/chat` standalone también migrado. `user_memories` (ConversationBufferMemory) reemplazado por checkpointer SQLite (`data/lg_state.db`).
- **`requirements_minimal.txt`**: añadidos `langgraph>=0.2.0,<0.3.0` y `langgraph-checkpoint-sqlite>=2.0.0,<3.0.0`.

#### Estado del grafo (`VCenterAgentState`)

```python
messages: Annotated[list, add_messages]  # historial conversacional
username: str
session_abbr: str
pending_tool_call: dict | None           # tool destructiva pendiente
awaiting_confirmation: bool              # esperando respuesta del usuario
execution_confirmed: bool                # usuario confirmó, ejecutar sin re-interceptar
long_term_context: str | None            # reservado para Hindsight
```

#### Herramientas destructivas interceptadas (4)

`delete_vms_tool`, `revert_snapshot_tool`, `delete_snapshot_tool`, `remove_vm_nic_tool`

#### Tests

- **`unitary_test/test_vcenter_graph_interruption.py`** *(nuevo)*: 12 tests, todos pasan.
  - Turn 1 destructiva → `awaiting_confirmation=True`
  - Turn 2 "sí" → tool ejecutada, `pending_tool_call=None`
  - Turn 2 "no" → `AIMessage("Operación cancelada")`
  - Estado persiste en checkpoint entre invocaciones
  - Aislamiento por usuario (thread IDs distintos)
  - Las 4 tools destructivas son interceptadas (parametrizado)

**Dependencia instalada**: `langgraph==0.2.76` + `langgraph-checkpoint==2.1.2` + `langgraph-checkpoint-sqlite==2.0.11`

***
## Versión 3.2 - Progressive Disclosure + ML Foundation (Marzo 2026)

### 🧠 Progressive Disclosure para Herramientas MCP (3 fases)

El agente vCenter exponía las 36 herramientas MCP del catálogo al LLM en cada llamada, consumiendo ~1.800–2.600 tokens extra por request. Progressive Disclosure filtra las herramientas relevantes según la intención detectada, reduciendo el contexto ~60-80% y mejorando la precisión de selección.

#### Fase 1 — Filtrado por Grupos según Intent
- **`create_tool_functions_for_query(username, session_abbr, query)`**: API en `mcp_tool_registry.py` que recibe la query del usuario y devuelve solo las herramientas del/los grupos relevantes
- **7 grupos semánticos**: `core`, `snapshots`, `reconfig`, `monitoring`, `info`, `vm_config`, `reports`
- **Detección por keywords**: matching en `GROUP_KEYWORDS` (sin coste de LLM)
- **Fallback**: si no se detecta grupo específico, se devuelven todas las herramientas

**Archivos**: `server/mcp_tool_registry.py`, `src/core/agent.py`

#### Fase 2 — Mejoras en Descripciones de Herramientas
- **Descripciones reescritas**: las 36 herramientas MCP con descripciones más precisas y diferenciadas para guiar mejor la selección del LLM
- **Docstrings estructurados**: formato "Cuándo usar / Cuándo NO usar" en herramientas con solapamiento semántico

**Archivo**: `server/mcp_tool_registry.py`

#### Fase 3 — Semantic Routing (nomic-embed-text)
- **`server/semantic_group_router.py`**: `SemanticGroupRouter` usa similitud coseno con embeddings `nomic-embed-text` como fallback cuando el matching por keywords no detecta un grupo específico
- **Group Persistence** (`_user_group_cache`, TTL=300s): conversaciones multi-turno heredan los grupos activos del turno anterior, alineado con el TTL de Sticky Routing
- **Detección híbrida en 3 capas**: keyword (0ms) → semántico (~10ms) → caché merge → fallback completo
- **Business logging**: `log_business_operation('progressive_disclosure')` con campos `detection_method`, `groups_selected`, `tool_count`, `reduction_pct`
- **Degradación graceful**: si Ollama no disponible al arrancar, opera en modo keyword-only
- **+17 tests** (55 total): Group Persistence, Semantic Router (mocked, sin Ollama), validación de logging

**Archivos**: `server/semantic_group_router.py`, `server/mcp_tool_registry.py`, `src/core/agent.py`

### 🤖 ML Predictivo — Data Foundation (Fase 0 completa)

Infraestructura de datos para futuros modelos de Machine Learning predictivo. Estrategia **datos primero, modelos después**.

#### Fase 0.1 — Extensión de Retención Histórica
- Retención aumentada de **7 días → 30 días** (`historical_retention_hours: 720` en `config/config.json`)
- El panel de monitorización sigue mostrando las últimas 24h sin cambios (`chart_window_hours: 24`)

#### Fase 0.2 — Almacén Parquet
- **`ml/data_store_builder.py`**: convierte JSON histórico en ficheros Parquet diarios por host (`data/ml_store/{host}/raw/YYYY-MM-DD.parquet`)
- **`ml/migrate_historical_to_parquet.py`**: migración one-shot de 7 días de datos sin pérdida — 2.937 filas migradas en 24 ficheros (3 hosts × 8 días)
- Los JSON originales del panel admin no se modifican; el almacén Parquet es independiente

#### Fase 0.3 — Pipeline de Sanitización
- **`ml/data_sanitizer.py`**: limpia Parquets raw → `processed/`:
  - Imputación de nulls y strings `"N/A"` por mediana
  - Label encoding de categóricos (`temperature_health`, `power_current_policy`, `host_connection_state`)
  - Detección y marcado de gaps temporales, outliers e inconsistencias (`data_quality_flag`)
- **`ml/migrate_raw_to_processed.py`**: script one-shot para sanitizar 24 Parquets raw existentes

#### Fase 0.4 — Etiquetado de Incidentes
- **Página admin dedicada**: `/admin/ml-labeling` para etiquetado de incidentes vCenter
- **Auto-descubrimiento de candidatos**: descubrimiento híbrido desde vCenter/ESXi con filtrado configurable
- **Modal de registro rápido**: formulario simplificado para registrar incidentes desde el panel admin
- **`ml/incident_store.py`**: almacenamiento SQLite de incidentes etiquetados

#### Fase 0.5 — Validación y Migraciones
- **`ml/validate_data_foundation.py`**: script end-to-end que verifica integridad de todo el pipeline Parquet
- Migraciones ejecutadas y validadas sobre los datos históricos existentes

**Directorio**: `vcenter_agent_system/ml/`

### 🔌 Multi-proveedor LLM (Ollama + llama.cpp server)

- **`src/utils/llm_factory.py`**: factory que devuelve `ChatOllama` o un cliente OpenAI-compatible según `llm_provider.provider` en `config.json`
- **Configuración en `config.json`** → sección `llm_provider`: cambiar de proveedor sin tocar código
  - `"provider": "ollama"` → usa `gpt-oss:20b` local
  - `"provider": "llama_cpp"` → usa `llama-server` con API OpenAI-compatible
- **`main_agent.py`, `agent.py`, `doc_consultant.py`**: actualizados para usar `get_llm()`
- **Instala**: `langchain-openai 0.3.16` (compatible con `langchain-core 0.3.x`)

**Archivo**: `src/utils/llm_factory.py`, `config/config.json`

### 📊 Live Log Viewer (SSE)

- **`GET /api/admin/logs/stream`** (superuser-only): endpoint SSE para streaming de logs en tiempo real
  - Polling cross-platform vía Python `seek/tell` (sin subprocess)
  - Soporta las 7 categorías: `api`, `audit`, `security`, `performance`, `business`, `system`, `retrieval_metrics`
  - History tail configurable (0-500 líneas) antes de iniciar el stream en vivo
  - Detección de rotación de fichero, heartbeat cada 5s, desconexión graceful
- **Tab "Logs en Vivo"** en `statistics.html`: chips de categoría/nivel, controles Start/Stop/Clear, indicador LED, auto-scroll, capped a 1000 entradas (anti-DOM-bloat)

**Archivos**: `src/api/main_agent.py`, `templates/admin/statistics.html`

### 🔧 Correcciones

- **ChromaDB lock en Windows**: fix en `src/utils/vector_store_manager.py` para evitar bloqueos de fichero en Windows al inicializar el cliente ChromaDB

***
## Versión 3.1 - Migración HTTPS + nginx (Marzo 2026)

### 🔒 Seguridad de Transporte

#### nginx como Proxy Inverso SSL
- **Arquitectura**: `Usuarios → https://host:5000 (nginx) → http://127.0.0.1:5001 (Flask)`
- **TLS**: TLSv1.2 + TLSv1.3, ciphers HIGH:!aNULL:!MD5
- **SSE preservado**: `proxy_buffering off` en `/chat/stream` para streaming sin cortes
- **Soporte dual**: Configs para Windows (`nginx_windows.conf`) y Ubuntu (`nginx_ubuntu.conf`)
- **Certificado**: Self-signed RSA 4096, instrucciones en `ssl/README.md`

**Archivos nuevos**: `vcenter_agent_system/nginx/nginx_windows.conf`, `nginx_ubuntu.conf`, `ssl/README.md`

#### Flask — Puerto Interno y Binding
- **Puerto**: `5000 → 5001` (interno, solo `127.0.0.1`)
- **Binding**: `0.0.0.0 → 127.0.0.1` (Flask ya no es accesible directamente desde la red)

**Archivo**: `vcenter_agent_system/run.py`

#### Session Cookie Flags
- `SESSION_COOKIE_SECURE=True` — cookie solo se envía por HTTPS
- `SESSION_COOKIE_HTTPONLY=True` — inaccesible desde JavaScript
- `SESSION_COOKIE_SAMESITE='Strict'` — protección CSRF
- `PREFERRED_URL_SCHEME='https'`

**Archivo**: `src/api/main_agent.py`

#### Security Headers HTTP (`@app.after_request`)
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (HSTS)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY` (anti-clickjacking)
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; ...`

**Archivo**: `src/api/main_agent.py`

#### Tests — URL Configurable
- Variable de entorno `TEST_BASE_URL` en 8 archivos de tests
- Default: `http://localhost:5001` (directo a Flask en local)
- Para testear contra nginx: `TEST_BASE_URL=https://localhost:5000`

***
## Versión 3.0 - RAG v2.4 + MCP Expansion (Febrero 2026)

### 🚀 Nuevas Características

#### RAG v2.4 - Sistema Híbrido de Retrieval
- **Búsqueda Híbrida**: ChromaDB (vector search) + BM25 (keyword search) con alpha adaptativo
- **Query Expansion**: 62 familias de términos (VMware + project tools) bidireccional
- **Embedding Cache**: LRU cache de 1000 queries (~30% speedup en repetidas)
- **Search Modes**: Strict, Boosting, Global para filtrado por carpeta
- **Reranking Heurístico**: Top 40 candidatos → top 8 con score optimizado
- **Internal Docs Boost**: +75% para archivos .md del proyecto
- **Metrics Logging**: JSONL con métricas de retrieval en `logs/retrieval_metrics.jsonl`
- **Pipeline 8-Fases**: Normalización, expansion, retrieval, filtering, reranking

**Archivos**: `src/utils/query_expander.py`, `embedding_cache.py`, `bm25_retriever.py`, `hybrid_retriever.py`, `reranker.py`, `search_modes.py`

#### MCP Tools v3.0 - Expansión de Herramientas
- **32 Tools Total** (vs 15 originales): 17 nuevas herramientas en 5 grupos
- **Grupo 1 - Snapshots** (4): list, delete, revert, info
- **Grupo 2 - Reconfiguración** (3): memory, CPU, disk
- **Grupo 2b - Datastores** (2): list, get_info  
- **Grupo 2c - NICs** (3): list, add, remove
- **Grupo 3 - ESXi Directo** (3): CPU usage, memory usage, list VMs
- **Grupo 5 - Eventos/Alarmas** (2): list events, list alarms
- **Grupo 6 - Fechas** (1): creation dates

**Archivos**: `server/mcp_tool_registry.py`, `mcp_tool_wrappers.py`

#### Chat Streaming SSE (Server-Sent Events)
- **Endpoint `/chat/stream`**: Respuestas progresivas en tiempo real
- **Eventos**: `routing`, `heartbeat`, `token`, `done`, `error`
- **Routing Inmediato**: Usuario ve el agente target en ~0ms
- **Heartbeats**: Indicador de progreso cada 2s durante procesamiento
- **Fallback Automático**: `/chat` original como backup si SSE falla

**Archivos**: `src/api/main_agent.py`, `static/js/orchestrator_chat_auth.js`

#### Renderizado Markdown en Chat
- **marked.js v15**: Parser Markdown en frontend
- **Syntax Highlighting**: Bloques de código con resaltado
- **GFM Support**: Tablas, listas, fenced code, negrita, cursiva
- **Preservación de Attachments**: Mantiene HTML de enlaces de descarga

**Archivos**: `static/js/marked.min.js`, `orchestrator_chat_auth.js`

### ⚙️ Mejoras de Arquitectura

#### Sticky Routing Conversacional
- **Memoria de Agente**: Sistema recuerda último agente usado por 180s
- **Auto-routing**: Follow-ups ("30 días", "datastore_35") van al mismo agente
- **Tracking en Sessions**: `last_agent`, `last_agent_time` en `ACTIVE_SESSIONS`

**Archivo**: `src/api/main_agent.py`

#### Clasificador 4-Capas Mejorado
- **Layer 0**: Exclusive keywords (agents.yaml)
- **Layer 1**: Critical regex patterns (alta confianza)
- **Layer 2**: Intent detection (imperativo vs learning)
- **Layer 3**: Weighted scoring con threshold
- **Layer 4**: LLM fallback + heuristic safety net

**Archivo**: `src/utils/query_classifier.py`

#### Connection Pool Optimizado
- **Max Connections**: 5 simultáneas (vs ilimitadas antes)
- **Timeout**: 30s inactividad para liberar conexiones
- **Reuso**: ~70% queries reusan conexión existente
- **Prevención de Leaks**: Automatic cleanup de conexiones antiguas

**Archivo**: `src/utils/vcenter_tools.py`

### 🔒 Seguridad

#### Rate Limiting
- **5 intentos fallidos** → bloqueo 5 minutos
- **Logging en `security.log`**

#### Decoradores de Seguridad
- `@authenticated_action` - Validación de sesión
- `@admin_required` - Solo admin/superuser
- `@superuser_required` - Solo superuser
- `@security_sensitive` - Log a security.log

**Archivos**: `src/utils/context_middleware.py`, `auth_module.py`

### 📊 Logging Estructurado

#### 6 Categorías de Logs
- `logs/api/` - HTTP requests/responses
- `logs/business/` - Operaciones vCenter
- `logs/security/` - Autenticación y accesos
- `logs/audit/` - Acciones de usuarios
- `logs/performance/` - Timing y bottlenecks
- `logs/system/` - Errores y excepciones

**Archivo**: `src/utils/structured_logger.py`

***
## Versión 2.5 - Background Agents (Enero 2026)

### 🤖 Agentes Background (6 agentes)

#### Report Scheduler
- **Frecuencia**: Diario a las 07:00
- **Función**: Dispara generación de informes PDF

#### Performance Report Agent
- **On-demand**: Generación de PDFs con estado vCenter
- **Contenido**: VMs, hosts, datastores, métricas

#### Colectores SNMP
- **TrueNAS Collector**: SNMP v3, métricas NAS cada 10 min
- **Cisco Catalyst Collector**: SNMP v2c, switch cada 10 min
- **Historical Data Collector**: Series temporales cada 10 min
- **Advanced ESXi Collector**: Métricas hosts cada 10 min

**Archivos**: `server/background_agents/`

***
## Versión 2.0 - Sistema Multi-Agente (Diciembre 2025)

### 🧠 Orquestador de Agentes

#### Agentes Especializados
- **vCenter Agent**: Operaciones VMware (VMs, hosts, datastores)
- **Documentation Consultant**: RAG v1.0 con Whoosh
- **General Agent**: Consultas generales

#### Sistema de Routing
- **Clasificación por Keywords**: Detección en `agents.yaml`
- **LLM Fallback**: Llama 3.1 8B para casos ambiguos

#### Memoria Conversacional
- **ConversationBufferMemory**: Por usuario, in-memory
- **Aislamiento**: Cada usuario tiene contexto independiente

**Archivos**: `src/api/main_agent.py`, `src/core/agent.py`, `doc_consultant.py`

***
## Versión 1.5 - Autenticación (Noviembre 2025)

### 🔐 Sistema de Autenticación

#### Bcrypt + SQLite
- **Hashing**: Bcrypt con cost factor 12
- **Base de datos**: `data/auth.db` con tabla users
- **Roles**: user, admin, superuser

#### Gestión de Sesiones
- **Dual System**: 
  - ACTIVE_SESSIONS (in-memory, 3600s)
  - SQLite SessionManager (persistent)
- **Flask Sessions**: Cookie-based

**Archivos**: `src/utils/password_manager.py`, `auth_module.py`

***
## Versión 1.0 - MVP (Octubre 2025)

### 🏗️ Sistema Base

#### Framework
- **Flask 2.0**: Web framework
- **LangChain**: Agent framework
- **pyvmomi**: VMware vSphere SDK

#### LLM
- **Ollama**: Runtime local
- **Llama 3.1 8B**: Modelo inicial

#### MCP v1.0
- **15 Tools**: Operaciones básicas vCenter
- **Connection**: SmartConnect directo (sin pool)

#### Frontend
- **HTML/CSS/JS**: Interfaz básica de chat
- **Temas**: Light/Dark mode

***
## 🗺️ Roadmap Futuro

Ver [[Propuestas-Funcionales]] para mejoras planificadas:

### Prioridad Alta 🔴
- Historial de conversación persistente (SQLite)
- Botón de copiar en respuestas
- Feedback de progreso en operaciones largas

### Prioridad Media 🟡
- Dashboard Grafana con métricas en tiempo real
- Multi-tenant isolation
- WebSocket bidireccional (vs SSE unidireccional)

### Prioridad Baja 🟢
- Export de conversaciones (JSON/PDF)
- Voice input/output
- Mobile responsive design mejorado

***
## 📚 Documentos Relacionados

- [[Arquitectura-Sistema]] - Visión general v3.0
- [[Propuestas-Funcionales]] - Mejoras futuras
- [[Propuestas-Informes]] - Background reporting extensions
- [[Sistema-MCP]] - Catálogo completo de 36 tools
- [[Agente-Documentacion]] - RAG v2.4 pipeline

***
## 🔧 Compatibilidad de Versiones

| Versión | Python | Flask | LangChain | Ollama | pyvmomi |
|---------|--------|-------|-----------|--------|---------|
| v3.0 | 3.9+ | 2.0+ | 0.3.x | 0.1.x | 8.0+ |
| v2.5 | 3.9+ | 2.0+ | 0.2.x | 0.1.x | 8.0+ |
| v2.0 | 3.8+ | 2.0+ | 0.1.x | 0.1.x | 7.0+ |
| v1.5 | 3.8+ | 2.0+ | 0.1.x | 0.1.x | 7.0+ |
| v1.0 | 3.8+ | 2.0+ | 0.0.x | 0.1.x | 7.0+ |

***
*Última actualización: 2026-05-11*
