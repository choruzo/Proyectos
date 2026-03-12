# Documentación Técnica — Sistema RAG v2.4

> **Agente de Documentación** | Sistema Híbrido ChromaDB + BM25
> Versión: 2.4.3 | Última actualización: 2026-03-12

---

## Índice

1. [Visión General](#1-visión-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Pipeline de Procesamiento — 8 Etapas](#3-pipeline-de-procesamiento--8-etapas)
4. [Componentes Técnicos](#4-componentes-técnicos)
   - 4.1 [Carga de Documentos](#41-carga-de-documentos-document_loaderpy)
   - 4.2 [Chunking Adaptativo](#42-chunking-adaptativo-chunkerpy)
   - 4.3 [Vector Store Manager](#43-vector-store-manager-vector_store_managerpy)
   - 4.4 [Embedding Cache](#44-embedding-cache-embedding_cachepy)
   - 4.5 [BM25 Retriever](#45-bm25-retriever-bm25_retrieverpy)
   - 4.6 [Hybrid Retriever](#46-hybrid-retriever-hybrid_retrieverpy)
   - 4.7 [Query Expander](#47-query-expander-query_expanderpy)
   - 4.8 [Search Modes](#48-search-modes-search_modespy)
   - 4.9 [Fast Reranker](#49-fast-reranker-rerankerpy)
   - 4.10 [Retrieval Metrics](#410-retrieval-metrics-retrieval_metricspy)
5. [Alpha Adaptativo](#5-alpha-adaptativo)
6. [Modos de Búsqueda por Carpeta](#6-modos-de-búsqueda-por-carpeta)
7. [Sistema de Abstención](#7-sistema-de-abstención)
8. [Flujo Conversacional y Memoria](#8-flujo-conversacional-y-memoria)
9. [Parámetros de Configuración](#9-parámetros-de-configuración)
10. [Métricas y Observabilidad](#10-métricas-y-observabilidad)
11. [Decisiones de Diseño](#11-decisiones-de-diseño)
12. [Referencia de Archivos](#12-referencia-de-archivos)

---

## 1. Visión General

El sistema RAG v2.4 es el motor de recuperación de información del **Agente de Documentación** (`doc_consultant.py`). Combina búsqueda vectorial semántica (ChromaDB + `nomic-embed-text`) con búsqueda léxica basada en frecuencia de términos (BM25) para responder consultas en lenguaje natural exclusivamente desde la documentación indexada.

### Principios de Diseño

| Principio | Implementación |
|-----------|----------------|
| **RAG Estricto** | El retrieval es **obligatorio** antes de cualquier respuesta LLM |
| **Sin alucinaciones** | Si no hay contexto recuperado, el sistema se abstiene |
| **Eficiencia** | Caché LRU para embeddings (~30% menos latencia en repeticiones) |
| **Explicabilidad** | Toda respuesta incluye citas de fuente obligatorias |
| **Adaptabilidad** | Alpha dinámico según longitud de query |

---

## 2. Arquitectura del Sistema

```mermaid
graph TB
    subgraph "Entrada"
        U[Usuario] -->|Consulta en lenguaje natural| ORC[Orchestrator]
        ORC -->|Clasificación: Documentation| DOC[DocConsultantAgent]
    end

    subgraph "Pipeline RAG — doc_tools.py"
        DOC --> SM[Search Mode Detection]
        SM --> QN[Query Normalization]
        QN --> QE[Query Expansion<br/>62 familias de términos]
        QE --> HR[Hybrid Retriever<br/>K=40 candidatos]

        subgraph "Retrieval Híbrido"
            HR --> VS[Vector Search<br/>ChromaDB + nomic-embed-text]
            HR --> BM[BM25 Keyword Search<br/>k1=1.5, b=0.75]
            VS --> NS[Score Normalization 0-1]
            BM --> NS
            NS --> AC[Alpha Adaptativo<br/>short: α=0.35 / long: α=0.70]
            AC --> IB[Internal Doc Boost<br/>+75% para .md]
        end

        IB --> FF[Folder Filter<br/>strict / boosting / global]
        FF --> RR[FastReranker<br/>top 40 → top 12]
        RR --> ML[Retrieval Metrics JSONL]
    end

    subgraph "Generación"
        ML --> CTX[Context Builder]
        CTX -->|Prompt estructurado| LLM[ChatOllama<br/>gpt-oss:20b<br/>num_ctx=16384<br/>temp=0.1]
        LLM --> VR[Validación Fuentes]
        VR --> RES[Respuesta + Fuentes]
    end

    subgraph "Almacenamiento"
        CHROMA[(ChromaDB<br/>data/chroma_db/)]
        BM25_IDX[(BM25 Index<br/>en memoria)]
        CACHE[(LRU Cache<br/>1000 embeddings)]
        DOCS[(docs/<br/>PDF · MD · TXT)]
    end

    DOCS -->|Carga + Chunks| CHROMA
    DOCS -->|Tokenización| BM25_IDX
    VS <--> CHROMA
    VS <--> CACHE
    BM <--> BM25_IDX
```

---

## 3. Pipeline de Procesamiento — 8 Etapas

```mermaid
flowchart TD
    Q([Consulta del Usuario]) --> E1

    E1["`**Etapa 1**
    Search Mode Detection
    strict / boosting / global`"]

    E1 --> E2["`**Etapa 2**
    Query Normalization
    stopwords + frases relleno`"]

    E2 --> E3["`**Etapa 3**
    Query Expansion
    62 familias bidireccional`"]

    E3 --> E4["`**Etapa 4**
    Hybrid Retrieval K=40
    Vector + BM25`"]

    subgraph E4detail["Detalle Etapa 4"]
        direction LR
        V[ChromaDB<br/>Vector Search] --> SN[Normalize 0-1]
        B[BM25<br/>Keyword Search] --> SN
        SN --> ALPHA[Alpha Adaptativo]
        ALPHA --> BOOST[Internal Doc Boost +75%]
    end

    E4 --- E4detail

    E4 --> E5["`**Etapa 5**
    Folder Filtering
    Según search mode`"]

    E5 --> E6["`**Etapa 6**
    FastReranker
    top 40 → top 12`"]

    E6 --> E7["`**Etapa 7**
    Metrics Logging
    retrieval_metrics.jsonl`"]

    E7 --> E8["`**Etapa 8**
    Return Results
    Formato legacy compatible`"]

    E8 --> LLM([LLM Response])

    style E1 fill:#4a90d9,color:#fff
    style E2 fill:#5ba85a,color:#fff
    style E3 fill:#e07b39,color:#fff
    style E4 fill:#9b59b6,color:#fff
    style E5 fill:#e74c3c,color:#fff
    style E6 fill:#1abc9c,color:#fff
    style E7 fill:#f39c12,color:#fff
    style E8 fill:#2c3e50,color:#fff
```

### Descripción por Etapa

| # | Etapa | Archivo | Función Principal |
|---|-------|---------|-------------------|
| 1 | Search Mode Detection | `search_modes.py` | `detect_search_mode(query)` |
| 2 | Query Normalization | `rag_retriever.py` | `normalize_query(query)` |
| 3 | Query Expansion | `query_expander.py` | `SimpleQueryExpander.expand(query)` |
| 4 | Hybrid Retrieval | `hybrid_retriever.py` | `ImprovedHybridRetriever.retrieve(query, k=40)` |
| 5 | Folder Filtering | `search_modes.py` | `filter_results_by_folder(results, folder, mode)` |
| 6 | Reranking | `reranker.py` | `FastReranker.rerank(query, docs, top_k=12)` |
| 7 | Metrics Logging | `retrieval_metrics.py` | `log_retrieval_metrics(metrics, logs_dir)` |
| 8 | Format Conversion | `doc_tools.py` | `_convert_hybrid_to_legacy_format(hybrid_results)` |

---

## 4. Componentes Técnicos

### 4.1 Carga de Documentos (`document_loader.py`)

Carga documentos en múltiples formatos preservando metadatos estructurales.

```mermaid
flowchart LR
    DOCS_DIR[docs/] --> PDF[PDF]
    DOCS_DIR --> MD[Markdown .md]
    DOCS_DIR --> TXT[TXT]

    PDF --> LOADER[document_loader.py<br/>load_documents_with_metadata]
    MD --> LOADER
    TXT --> LOADER

    LOADER --> META[Metadata enriquecida]

    META --> M1["source: ruta absoluta"]
    META --> M2["file_name: nombre.md"]
    META --> M3["file_type: .md / .pdf / .txt"]
    META --> M4["folder: subcarpeta docs/"]
    META --> M5["Header 1/2/3: sección MD"]
```

**Metadatos clave preservados:**

| Campo | Descripción | Uso |
|-------|-------------|-----|
| `source` | Ruta absoluta del archivo | Identificación única en BM25 |
| `file_name` | Nombre del archivo | Boost por herramienta en BM25 |
| `file_type` | `.md`, `.pdf`, `.txt` | Identificación de docs internos |
| `folder` | Subcarpeta bajo `docs/` | Filtrado por folder mode |
| `Header 1/2/3` | Secciones MD | Contexto en BM25 + citación |

---

### 4.2 Chunking Adaptativo (`chunker.py`)

```mermaid
flowchart TD
    DOC[Documento] --> FT{file_type?}
    FT -->|.md| MDC["MarkdownHeaderTextSplitter
    Divide por Headers H1/H2/H3
    Preserva contexto de sección"]
    FT -->|.pdf / .txt| REC["RecursiveCharacterTextSplitter
    chunk_size=1400 chars
    chunk_overlap=350 chars"]

    MDC --> CHUNKS[Chunks con metadata de sección]
    REC --> CHUNKS

    CHUNKS --> STATS["Estadísticas típicas:
    • 1400 chars/chunk
    • 350 chars overlap
    • ~30% contexto compartido"]
```

**Parámetros:**

```python
AdaptiveSemanticChunker(chunk_size=1400, chunk_overlap=350)
```

El overlap del 25% (`350/1400`) asegura que conceptos que cruzan límites de chunk no se pierdan.

---

### 4.3 Vector Store Manager (`vector_store_manager.py`)

Gestiona el ciclo de vida de ChromaDB con detección automática de cambios mediante manifesto SHA1.

```mermaid
stateDiagram-v2
    [*] --> CheckManifest : Inicialización

    CheckManifest --> NeedsRebuild : SHA1 diferente
    CheckManifest --> LoadExisting : SHA1 igual

    NeedsRebuild --> LoadDocuments : Cargar docs
    LoadDocuments --> ChunkDocuments : Chunking 1400 chars
    ChunkDocuments --> CreateVectorStore : ChromaDB.from_documents()
    CreateVectorStore --> SaveManifest : Guardar SHA1 nuevo
    SaveManifest --> Ready

    LoadExisting --> Ready : vectorstore.load()

    Ready --> [*]
```

**Ubicación:** `data/chroma_db/`
**Modelo de embedding:** `nomic-embed-text` (via Ollama)
**Forzar rebuild:** `config.json → rag_v2.vector_store.force_rebuild: true`

---

### 4.4 Embedding Cache (`embedding_cache.py`)

Cache LRU en memoria para embeddings de queries repetidas.

```mermaid
flowchart LR
    Q[Query] --> HASH{¿En caché?}
    HASH -->|HIT ~30%| EMB_CACHED[Embedding cacheado]
    HASH -->|MISS ~70%| OLLAMA[Ollama nomic-embed-text]
    OLLAMA --> STORE[Guardar en LRU]
    STORE --> EMB_NEW[Embedding nuevo]
    EMB_CACHED --> VS[Vector Search]
    EMB_NEW --> VS

    subgraph "LRU Cache"
        LRU_INFO["max_size: 1000 queries
        Política: Least Recently Used
        Hit tracking: hit_count / miss_count"]
    end
```

**Estadísticas disponibles:**

```python
cache.get_stats()
# {hit_count, miss_count, hit_rate, current_size, max_size}
```

---

### 4.5 BM25 Retriever (`bm25_retriever.py`)

Implementación del algoritmo BM25 (Best Matching 25) con indexación enriquecida.

#### Fórmula BM25

```
score(q, d) = Σ IDF(tᵢ) × [tf(tᵢ,d) × (k1+1)] / [tf(tᵢ,d) + k1×(1 - b + b×|d|/avgdl)]
```

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `k1` | 1.5 | Saturación de frecuencia de términos |
| `b` | 0.75 | Normalización por longitud de documento |

#### Texto indexable por documento

```mermaid
flowchart LR
    DOC[Document] --> PC[page_content]
    DOC --> FS["file_stem normalizado
    'Cantata-Know-How' → 'cantata know how'"]
    DOC --> H1["Header 1 (si existe)"]
    DOC --> H2["Header 2 (si existe)"]
    DOC --> H3["Header 3 (si existe)"]

    PC --> IDX[Texto indexable]
    FS --> IDX
    H1 --> IDX
    H2 --> IDX
    H3 --> IDX

    IDX --> TOKEN[Tokenización
    regex \w+
    Filtro: len > 2]
    TOKEN --> INDEX[(Índice BM25<br/>TF + IDF)]
```

> **Nota de diseño**: Incluir el nombre del archivo en el índice permite que búsquedas por herramienta (`"cantata"`, `"sonarqube"`) encuentren chunks aunque la palabra no aparezca literalmente en el contenido.

---

### 4.6 Hybrid Retriever (`hybrid_retriever.py`)

Combina los scores de búsqueda vectorial y BM25 mediante alpha adaptativo.

```mermaid
flowchart TD
    Q[Query expandida] --> VA[Alpha Adaptativo]

    subgraph "Búsqueda Paralela"
        Q --> VS["Vector Search
        ChromaDB + nomic-embed-text
        K=40 resultados"]
        Q --> BS["BM25 Search
        Índice en memoria
        K=40 resultados"]
    end

    VS --> VN["Normalización
    score → [0, 1]"]
    BS --> BN["Normalización
    score → [0, 1]"]

    VA --> COMBINE

    VN --> COMBINE["Combinación:
    score_v × alpha × boost
    + score_b × (1-alpha) × boost"]
    BN --> COMBINE

    COMBINE --> SORT["Ordenar descendente
    score final combinado"]
    SORT --> TOP["Top K resultados
    (típico K=40)"]
```

**Cálculo del score final:**

```
score_final = (score_vector × α × boost) + (score_bm25 × (1-α) × boost)

boost = 1.75 si es .md (interno)
boost = 1.00 si es .pdf o .txt
```

---

### 4.7 Query Expander (`query_expander.py`)

Expansión heurística sin LLM con 62 familias de términos bidireccionales.

```mermaid
flowchart LR
    Q["Query original:
    'cómo instalo dns'"]

    Q --> FWD["Búsqueda directa
    clave → sinónimos
    dns → [bind9, bind, named, ...]"]

    Q --> REV["Búsqueda inversa v2.4
    sinónimo → clave
    'virtual machine' → vm"]

    FWD --> EXP["Query expandida:
    'cómo instalo dns bind9 bind
    named resolución de nombres
    servidor dns ...'"]
    REV --> EXP
```

**Categorías de las 62 familias:**

| Categoría | Familias | Ejemplos |
|-----------|----------|---------|
| VMware Básico | 12 | vm, apagar, snapshot, disco, red, cpu |
| Infraestructura | 4 | host, datastore, cluster, vcenter |
| HA/Performance | 4 | ha, drs, vmotion, rendimiento |
| Red/Seguridad | 4 | vlan, vswitch, firewall, ip |
| Almacenamiento | 3 | vmdk, nfs, iscsi |
| Operaciones | 4 | arrancar, detener, backup, monitoreo |
| Administración | 5 | usuario, permiso, licencia, configurar |
| Recursos | 2 | recurso, límite |
| DNS/Infraestructura | 7 | dns, bind, bind9, named, zona, ubuntu |
| Herramientas Proyecto | 6 | gtr, dvd, build_dvds, pruebas, entrega, cantata |
| Herramientas Doc | 5 | doors, midat, sonarqube, sbr, concalnet |
| Logs/Acceso | 4 | log, logs, sta_element_history, sec_element_history |
| Métricas Red | 2 | mbps, ancho de banda |

---

### 4.8 Search Modes (`search_modes.py`)

Detección automática del scope de búsqueda mediante keywords en la query.

```mermaid
flowchart TD
    Q[Query] --> PARSE[Análisis de keywords]

    PARSE --> STRICT{"`¿Contiene
    'solo', 'únicamente', 'only'
    + nombre de carpeta?`"}

    STRICT -->|Sí| SM_STRICT["Modo STRICT
    Solo docs de carpeta target
    Filtro: folder == target"]

    STRICT -->|No| BOOST{"`¿Contiene
    'busca en', 'prioriza', 'enfoca en'
    + nombre de carpeta?`"}

    BOOST -->|Sí| SM_BOOST["Modo BOOSTING
    Score × 2.0 para carpeta target
    Re-ordenar después del boost"]

    BOOST -->|No| SM_GLOBAL["Modo GLOBAL
    Búsqueda sin filtrado
    (por defecto)"]

    SM_STRICT --> CLEAN["Query limpia
    sin keywords de modo"]
    SM_BOOST --> CLEAN
    SM_GLOBAL --> CLEAN

    CLEAN --> PIPELINE[Pipeline RAG]
```

**Carpetas conocidas:** `git, vmware, esxi, vcenter, sonarqube, cantata, doors, header, sbr, ttcf, documentacion, know-how`

**Ejemplos:**

```
"SOLO vcenter configuración de red"   → strict, folder=vcenter
"busca en esxi problemas de memoria"  → boosting, folder=esxi
"cómo configuro una VM"               → global
```

---

### 4.9 Fast Reranker (`reranker.py`)

Reordenamiento heurístico de candidatos sin llamadas adicionales al LLM.

```mermaid
flowchart TD
    INPUT["Top 40 candidatos
    (doc, score_híbrido)"]

    INPUT --> QTERMS["Extraer términos clave
    query - stopwords españolas"]

    QTERMS --> SCORE["Calcular score por doc"]

    subgraph "Función de Score"
        direction LR
        OS["Score original × 0.15"]
        TF["Term Frequency × 0.50
        (términos query en contenido)"]
        LN["Length Score × 0.15
        (min(len/1500, 1.0))"]
        PP["Position Penalty × 0.10
        (1 - pos/total × 0.3)"]
        IB["Internal Boost
        +0.75 si es .md"]

        OS --> FINAL
        TF --> FINAL
        LN --> FINAL
        PP --> FINAL
        IB --> FINAL["Score Final"]
    end

    SCORE --> SORT["Ordenar por Score Final"]
    SORT --> TOP["Top 12 resultados"]
```

**Pesos del scoring:**

| Factor | Peso | Razón |
|--------|------|-------|
| Term Frequency | 50% | Principal señal de relevancia |
| Score original | 15% | Confirmación híbrida |
| Length | 15% | Preferir chunks completos |
| Position | 10% | Penalizar posiciones bajas |
| Internal Boost | +75% | Docs `.md` = Know-How del proyecto |

**Stop words filtradas en term_freq:**
`como, cómo, qué, que, cuál, hay, un, una, el, la, los, las, es, en, de, del, al, y, o, a, para, con, se, por, lo, más, this, that, the, and, or, is, are, of, to, hago, quiero, puedo, necesito`

---

### 4.10 Retrieval Metrics (`retrieval_metrics.py`)

Logging estructurado en JSONL de cada operación de retrieval.

```mermaid
flowchart LR
    RETRIEVAL[Retrieval completado] --> METRICS

    subgraph METRICS["RetrievalMetrics"]
        Q_ORIG[query original]
        NC[num_chunks_retrieved]
        AVG[avg_similarity_score]
        SRCS[sources_used]
        CTX[context_length chars]
        METHOD[retrieval_method]
        FLAGS["reranked / query_expanded / cache_hit"]
        FOLDER[folder_mode]
    end

    METRICS --> LOG["logs/retrieval_metrics.jsonl"]
    LOG --> DIAG["Diagnóstico:
    context_length > 16000 → riesgo truncación
    avg_score < 0.3 → resultados pobres
    cache_hit: true → ~30% latencia"]
```

**Formato JSONL:**
```json
{
  "timestamp": "2026-03-12T10:30:00",
  "query": "cómo instalo bind9",
  "num_chunks_retrieved": 8,
  "avg_similarity_score": 0.74,
  "sources_used": ["Documentacion_Entorno_VMware.md"],
  "context_length": 9840,
  "retrieval_method": "semantic+rerank",
  "reranked": true,
  "query_expanded": true,
  "cache_hit": false,
  "folder_mode": "global"
}
```

---

## 5. Alpha Adaptativo

El parámetro alpha controla la proporción entre búsqueda vectorial (semántica) y BM25 (léxica) según la longitud de la query.

```mermaid
xychart-beta
    title "Alpha según longitud de query"
    x-axis ["1-4 palabras", "5-10 palabras", "11+ palabras"]
    y-axis "Alpha (peso vector)" 0.0 --> 1.0
    bar [0.35, 0.50, 0.70]
```

| Longitud | Alpha | Interpretación |
|----------|-------|----------------|
| < 5 palabras | **0.35** | Más peso a BM25 — queries cortas son términos exactos |
| 5–10 palabras | **0.50** | Balance equitativo (valor base) |
| > 10 palabras | **0.70** | Más peso a vector — queries largas tienen intención semántica |

**Fundamento:** Queries cortas como `"cantata licencia"` se benefician de matching exacto. Queries largas como `"cómo configurar el servidor DNS en Ubuntu para el entorno VMware"` se benefician de comprensión semántica.

---

## 6. Modos de Búsqueda por Carpeta

```mermaid
flowchart TD
    subgraph STRICT["Modo STRICT"]
        S1[Results: 40 candidatos] --> S2["Filtrar: solo folder == target"]
        S2 --> S3[Results: N ≤ 40]
        S3 --> S4[Rerank → top 12]
    end

    subgraph BOOST["Modo BOOSTING"]
        B1[Results: 40 candidatos] --> B2["Multiplicar score × 2.0
        para docs de carpeta target"]
        B2 --> B3[Re-ordenar por score]
        B3 --> B4[Rerank → top 12]
    end

    subgraph GLOBAL["Modo GLOBAL (default)"]
        G1[Results: 40 candidatos] --> G2[Sin filtrado]
        G2 --> G3[Rerank → top 12]
    end
```

---

## 7. Sistema de Abstención

El sistema implementa una regla de abstención estricta para evitar alucinaciones.

```mermaid
flowchart TD
    RETRIEVAL[Retrieval completado] --> CHECK{¿results vacíos?}

    CHECK -->|Sí| ABS["ABSTENCIÓN:
    No llamar al LLM
    Devolver mensaje pre-definido:
    'La documentación disponible no contiene...'
    + Sugerencias de búsqueda"]

    CHECK -->|No| BUILD["Construir prompt con contexto"]
    BUILD --> LLM[Invocar LLM]
    LLM --> VALIDATE{"`¿Respuesta contiene
    'Fuentes:'?`"}

    VALIDATE -->|No| APPEND["Añadir fuentes automáticamente
    regex: \\bfuentes\\b (IGNORECASE)"]
    VALIDATE -->|Sí| RETURN[Devolver respuesta]
    APPEND --> RETURN
```

**Mensajes de abstención:**
- Sin resultados: `"La documentación disponible no contiene información sobre: {query normalizada}"`
- Con sugerencia: `"Prueba con términos más generales o usa 'listar documentos'"`
- Error técnico: `"Error al buscar en la documentación. Por favor, intenta de nuevo."`

---

## 8. Flujo Conversacional y Memoria

```mermaid
sequenceDiagram
    participant U as Usuario
    participant ORC as Orchestrator
    participant DOC as DocConsultantAgent
    participant MEM as ConversationBufferMemory
    participant RAG as Pipeline RAG
    participant LLM as ChatOllama

    U->>ORC: Mensaje 1: "¿Cómo instalo DNS en Ubuntu?"
    ORC->>DOC: process_documentation_query(user, msg)
    DOC->>MEM: get_user_context(username)
    Note over MEM: Crear memoria si no existe<br/>per-user isolation
    DOC->>RAG: _enhance_query_with_context()
    RAG-->>DOC: {context, prompt, sources}
    DOC->>LLM: invoke({input: prompt_con_contexto})
    LLM-->>DOC: Respuesta con fuentes
    DOC->>MEM: Guardar turno en chat_history
    DOC-->>U: Respuesta

    U->>ORC: Mensaje 2: "¿Y para CentOS?"
    ORC->>DOC: process_documentation_query(user, msg2)
    DOC->>MEM: get_user_context(username)
    Note over MEM: Recuperar historial anterior
    DOC->>RAG: _enhance_query_with_context()
    Note over RAG: Query con contexto de turno anterior
    RAG-->>DOC: {context, prompt, sources}
    DOC->>LLM: invoke({input: prompt, chat_history: [turno1]})
    LLM-->>U: Respuesta contextualizada
```

**Aislamiento por usuario:**
- Cada usuario tiene su propia instancia de `ConversationBufferMemory`
- Cada usuario tiene su propio `AgentExecutor`
- Almacenado en `user_memories: Dict[str, ConversationBufferMemory]`

---

## 9. Parámetros de Configuración

### `config/config.json` → sección `rag_v2`

```json
{
  "rag_v2": {
    "enabled": true,
    "features": {
      "query_expansion_v2": true,
      "embedding_cache": true,
      "reranking": true,
      "folder_filtering": true,
      "hybrid_search": true
    },
    "vector_store": {
      "db_path": "data/chroma_db",
      "embedding_model": "nomic-embed-text",
      "chunk_size": 1200,
      "chunk_overlap": 250,
      "force_rebuild": false
    },
    "hybrid_retrieval": {
      "base_alpha": 0.5,
      "initial_k": 40,
      "bm25_k1": 1.5,
      "bm25_b": 0.75,
      "internal_docs_boost": 0.75
    }
  }
}
```

### Parámetros del LLM (hardcoded en `doc_consultant.py`)

```python
self.llm = ChatOllama(
    model="gpt-oss:20b",
    num_ctx=16384,   # Ollama default=4096 → truncación silenciosa con RAG
    temperature=0.1  # Baja creatividad = menos alucinaciones en RAG
)
```

> **CRÍTICO**: `num_ctx=16384` es esencial. Con `rerank_top_k=12` chunks de 1400 chars + system prompt el contexto supera 5500 tokens, superando el default de Ollama (4096) y causando truncación silenciosa.

### Tabla completa de parámetros

| Parámetro | Valor | Ubicación |
|-----------|-------|-----------|
| `chunk_size` | 1400 chars | `doc_tools.py:AdaptiveSemanticChunker` |
| `chunk_overlap` | 350 chars | `doc_tools.py:AdaptiveSemanticChunker` |
| `embedding_model` | `nomic-embed-text` | `doc_tools.py:OllamaEmbeddings` |
| `base_alpha` | 0.5 | `hybrid_retriever.py` |
| `alpha_short` (< 5w) | 0.35 | `hybrid_retriever.py` |
| `alpha_long` (> 10w) | 0.70 | `hybrid_retriever.py` |
| `internal_docs_boost` | 0.75 (+75%) | `hybrid_retriever.py`, `reranker.py` |
| `bm25_k1` | 1.5 | `bm25_retriever.py:search()` |
| `bm25_b` | 0.75 | `bm25_retriever.py:search()` |
| `initial_k` (retrieval) | 40 | `doc_tools.py:search_documents()` |
| `rerank_top_k` | 12 | `config.json → rag_v2.parameters` |
| `embedding_cache_size` | 1000 | `doc_tools.py:EmbeddingCache` |
| `num_ctx` (LLM) | 16384 | `doc_consultant.py` |
| `temperature` | 0.1 | `doc_consultant.py` |
| `folder_boost_factor` | 2.0 | `search_modes.py:BOOST_FACTOR` |

---

## 10. Métricas y Observabilidad

### Archivos de log

| Log | Ruta | Contenido |
|-----|------|-----------|
| Retrieval metrics | `logs/retrieval_metrics.jsonl` | Métricas por búsqueda (JSONL) |
| Sistema | `logs/system.log` | Errores críticos, inicialización |
| Negocio | `logs/business/` | Operaciones de usuario |
| Auditoría | `logs/audit/` | Consultas por usuario |
| Performance | `logs/performance/` | Latencias |

### Diagnóstico rápido

```powershell
# Ver últimas métricas de retrieval
Get-Content logs/retrieval_metrics.jsonl -Tail 5

# Monitorear errores del sistema RAG
Get-Content logs/system.log -Wait -Tail 20 | Select-String "chroma|bm25|rerank|embed"

# Ver estadísticas de cache
Get-Content logs/system.log -Tail 100 | Select-String "Cache stats"
```

### Indicadores de salud

```mermaid
flowchart LR
    subgraph "Señales de alerta"
        A["context_length > 16000
        → Riesgo truncación LLM"]
        B["avg_score < 0.3
        → Resultados pobres"]
        C["num_chunks = 0
        → ChromaDB o BM25 sin datos"]
        D["cache_hit_rate < 10%
        → Sin queries repetidas (normal)"]
    end

    subgraph "Señales saludables"
        E["context_length: 8000-14000 ✓"]
        F["avg_score > 0.5 ✓"]
        G["num_chunks: 6-12 ✓"]
        H["cache_hit_rate > 25% ✓"]
    end
```

---

## 11. Decisiones de Diseño

### ¿Por qué Híbrido (Vector + BM25)?

```
Vector search solo → Alta precisión semántica, baja precisión léxica
                     Falla en nombres propios: "cantata", "sonarqube", "sbr"

BM25 solo          → Alta precisión léxica, sin comprensión semántica
                     Falla en sinónimos: "apagar" ≠ "shutdown"

Híbrido            → Mejor de ambos mundos
                     Alpha adaptativo ajusta el balance por query
```

### ¿Por qué BM25 indexa metadatos (nombre de archivo + headers)?

Sin esto, una búsqueda por `"cantata"` no encontraría chunks del documento `Cantata-Know-How.md` si la palabra "cantata" no aparece en el texto del chunk. El nombre normalizado del archivo se añade al texto indexable.

### ¿Por qué `num_ctx=16384` y no el default?

Ollama usa 4096 tokens de contexto por defecto aunque el modelo soporte más. Con el pipeline RAG típico:
- System prompt: ~1800 tokens
- 12 chunks × 1400 chars ≈ 4200 tokens (sin contar overhead)
- Query + historial: ~500 tokens
- **Total: ~6500 tokens → truncación silenciosa con default 4096**

Con `num_ctx=16384` el margen es seguro hasta context lengths de ~14000 chars.

### ¿Por qué `temperature=0.1`?

El Agente de Documentación debe ser determinista y fiel al contexto recuperado. `temperature=0.1` reduce la variabilidad y la tendencia a "completar" información no presente en el contexto. El default de Ollama (`temperature=1.0`) es apropiado para tareas creativas, no para RAG estricto.

### ¿Por qué +75% boost para docs `.md`?

Los archivos `.md` en `docs/` son el Know-How del proyecto (documentación interna). Los archivos `.pdf` suelen ser documentación de terceros. El boost asegura que la documentación propia tenga prioridad cuando es igualmente relevante.

---

## 12. Referencia de Archivos

```mermaid
graph LR
    subgraph "Pipeline Core"
        DT[doc_tools.py<br/>MarkdownDocumentManager]
        DC[doc_consultant.py<br/>DocConsultantAgent]
        DT --> DC
    end

    subgraph "Retrieval"
        HR[hybrid_retriever.py<br/>ImprovedHybridRetriever]
        BM[bm25_retriever.py<br/>BM25Retriever]
        QE[query_expander.py<br/>SimpleQueryExpander]
        SM[search_modes.py<br/>detect_search_mode]
        HR --> BM
    end

    subgraph "Storage"
        VM[vector_store_manager.py<br/>VectorStoreManager]
        EC[embedding_cache.py<br/>EmbeddingCache]
        VM --> EC
    end

    subgraph "Procesamiento"
        CK[chunker.py<br/>AdaptiveSemanticChunker]
        DL[document_loader.py<br/>load_documents_with_metadata]
        RR[reranker.py<br/>FastReranker]
        RM[retrieval_metrics.py<br/>log_retrieval_metrics]
        RAG[rag_retriever.py<br/>Legacy bridge]
    end

    DT --> HR
    DT --> VM
    DT --> EC
    DT --> CK
    DT --> DL
    DT --> RR
    DT --> SM
    DT --> RM
    DT --> RAG
    DT --> QE
```

### Tabla completa de archivos

| Archivo | Clase / Función Principal | Propósito |
|---------|---------------------------|-----------|
| `src/core/doc_consultant.py` | `DocConsultantAgent` | Agente principal, RAG estricto, memoria por usuario |
| `src/utils/doc_tools.py` | `MarkdownDocumentManager` | Orquestador del pipeline, formato legacy |
| `src/utils/hybrid_retriever.py` | `ImprovedHybridRetriever` | Combinación Vector+BM25 con alpha adaptativo |
| `src/utils/bm25_retriever.py` | `BM25Retriever` | Algoritmo BM25 con indexación de metadatos |
| `src/utils/query_expander.py` | `SimpleQueryExpander` | 62 familias de expansión bidireccional |
| `src/utils/search_modes.py` | `detect_search_mode()` | Detección strict/boosting/global |
| `src/utils/reranker.py` | `FastReranker` | Reranking heurístico sin LLM |
| `src/utils/embedding_cache.py` | `EmbeddingCache` | LRU cache para embeddings |
| `src/utils/vector_store_manager.py` | `VectorStoreManager` | ChromaDB + manifesto SHA1 |
| `src/utils/chunker.py` | `AdaptiveSemanticChunker` | Chunking MD-aware |
| `src/utils/document_loader.py` | `load_documents_with_metadata()` | Carga PDF/MD/TXT con metadatos |
| `src/utils/retrieval_metrics.py` | `log_retrieval_metrics()` | Logging JSONL de métricas |
| `src/utils/rag_retriever.py` | `get_rag_retriever()` | Bridge legacy + normalize_query |
| `config/config.json` | sección `rag_v2` | Parámetros del sistema RAG |
| `data/chroma_db/` | — | Vector store persistente |
| `logs/retrieval_metrics.jsonl` | — | Métricas de retrieval por consulta |

---

*Documentación generada a partir del código fuente de `vcenter_agent_system/`.*
*Para detalles de migración desde v1.0 ver `RAG_V2_MIGRATION_README.md`.*
*Para detalles del sistema completo ver `HYBRID_SYSTEM_README.md`.*
