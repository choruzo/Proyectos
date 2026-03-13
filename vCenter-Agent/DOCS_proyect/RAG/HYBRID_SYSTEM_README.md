# Sistema RAG v2.4 — ChromaDB + BM25

## Estado: IMPLEMENTADO Y OPERATIVO

**Versión:** RAG v2.4 Hybrid
**Implementado:** 2026-02-01 | **Última mejora:** 2026-02-23

---

## Componentes

| Archivo | Función |
|---------|---------|
| `src/utils/query_expander.py` | 62 familias de términos (VMware + proyecto), expansión bidireccional |
| `src/utils/embedding_cache.py` | Caché LRU 1000 queries, tracking hit/miss |
| `src/utils/bm25_retriever.py` | BM25 (k1=1.5, b=0.75) — indexa page_content + nombre de archivo + headers |
| `src/utils/chunker.py` | Chunking adaptativo (MD vs otros formatos) |
| `src/utils/document_loader.py` | Carga PDF, MD, TXT con metadata de carpeta |
| `src/utils/vector_store_manager.py` | Gestión ChromaDB, manifest SHA1, auto-rebuild |
| `src/utils/hybrid_retriever.py` | Combina Vector + BM25 con alpha adaptativo |
| `src/utils/reranker.py` | Reranking heurístico, boost .md, filtro de stop words ES |
| `src/utils/search_modes.py` | Detección y filtrado strict/boosting/global |
| `src/utils/retrieval_metrics.py` | Log JSONL de métricas por query |

---

## Pipeline de Retrieval (8 etapas)

```
Query
  ↓ [1] detect_search_mode()  → strict / boosting / global
  ↓ [2] normalize_query()     → elimina stop words (23 ES) + frases de relleno (12)
  ↓ [3] expand_semantic_concepts()  → 62 familias de términos
  ↓ [4] hybrid_retriever.retrieve(k=40)
      ├─ Vector Search (ChromaDB + nomic-embed-text + LRU cache 1000)
      ├─ BM25 (k1=1.5, b=0.75 · indexa contenido + archivo + headers)
      ├─ Normalización de scores [0,1]
      ├─ Alpha adaptativo: cortas<5w α=0.35 · medias α=0.50 · largas>10w α=0.70
      └─ Boost +75% a docs .md
  ↓ [5] filter_by_folder()
      ├─ STRICT:   solo carpeta target
      ├─ BOOSTING: x2 score para carpeta target
      └─ GLOBAL:   sin filtrado
  ↓ [6] reranker.rerank(top_k=12)
      · Score original 25% · Term freq 40% · Length 15% · Position 10%
      · Boost interno +40% · Filtra stop words ES en term_freq
  ↓ [7] log_retrieval_metrics()  → logs/retrieval_metrics.jsonl
  ↓ [8] return results → doc_consultant.py → LLM (num_ctx=16384, temp=0.1)
```

---

## Parámetros Clave

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| chunk_size | 1400 chars | Hardcoded en doc_tools.py (MD-aware) |
| chunk_overlap | 350 chars | Preserva contexto entre chunks |
| embedding_cache | 1000 queries | LRU, ~30% más rápido en repetidas |
| rerank_candidates | 40 | Candidatos iniciales de retrieval |
| rerank_top_k | 12 | Resultados finales al LLM |
| base_alpha | 0.5 | Mix Vector/BM25 (adaptativo) |
| bm25_k1 / bm25_b | 1.5 / 0.75 | Parámetros estándar BM25 |
| internal_docs_boost | 0.75 | +75% boost para archivos .md |
| num_ctx (doc agent) | 16384 | Contexto Ollama expandido |
| temperature | 0.1 | Reduce alucinaciones en RAG |

---

## Configuración (`config/config.json → rag_v2`)

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

> **Nota:** `chunk_size` y `chunk_overlap` en config.json son valores de referencia. Los valores operativos (1400/350) están hardcoded en `doc_tools.py` para garantizar chunking semántico adecuado con documentos Markdown.

---

## Modos de Búsqueda

| Modo | Sintaxis | Comportamiento |
|------|----------|----------------|
| Strict | `"SOLO vcenter configuración"` | Solo documentos de la carpeta especificada |
| Boosting | `"busca en esxi rendimiento"` | x2 score para carpeta target, resto incluido |
| Global | Query normal | Todos los documentos sin filtro |

---

## Troubleshooting

**ChromaDB no inicializa:**
```powershell
ollama list                      # Verificar que nomic-embed-text está disponible
ollama pull nomic-embed-text     # Descargar si falta
# En config.json: "force_rebuild": true  → forzar rebuild
```

**Documentos nuevos no aparecen:**
El sistema detecta cambios automáticamente por SHA1. Si falla:
```powershell
Remove-Item -Recurse -Force data\chroma_db
python run.py   # Reconstruye el índice
```

**Ver métricas de retrieval:**
```powershell
Get-Content logs\retrieval_metrics.jsonl -Tail 10 | ConvertFrom-Json | Format-Table
Get-Content logs\business\business.log | Select-String "Cache stats"
```

**Validar instalación:**
```powershell
python validate_hybrid_system.py
```
