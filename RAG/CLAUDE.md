# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **improved RAG (Retrieval-Augmented Generation) system** for VMware ESXi documentation. It combines semantic vector search with keyword-based BM25 search to retrieve relevant documentation chunks, which are then passed to a local LLM (Ollama) for generating accurate, cited responses.

The system is designed to work entirely offline using local models and custom hybrid retrieval strategies.

## Architecture Overview

The RAG system follows a 7-stage pipeline:

1. **Query Expansion**: Short queries (<5 words) are expanded with synonyms and related technical terms using heuristic rules to improve search effectiveness
2. **Adaptive Alpha**: The balance between vector search (60-40%) and BM25 keyword search (40-60%) dynamically adjusts based on query length
3. **Hybrid Retrieval**: Combined vector similarity search + BM25 keyword search with score normalization and fusion
4. **Reranking**: Top 15 results are re-evaluated by the LLM for semantic relevance; combined with original scores (30% original, 70% LLM score)
5. **Context Building**: Top 5 reranked documents are assembled into a coherent context with source tracking
6. **Strict Relevance Checking**: Two-level validation—keyword overlap threshold (>5%) followed by LLM semantic validation (score ≥4/10)
7. **LLM Response Generation**: Enhanced prompt with critical instructions to use only context, avoid hallucinations, and cite sources

See `ARQUITECTURA_v2.md` for detailed flow diagrams and `MEJORAS_RELEVANCIA_EXPLICACION.md` for the technical improvements that solve low relevance problems.

## Development Setup

### Prerequisites
- **Python 3.9+**
- **Ollama** installed and running with models:
  - `llama3.1:8b` (main LLM)
  - `nomic-embed-text` (embeddings)
- **RAM**: 8GB minimum (16GB recommended for larger document sets)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Project Structure

```
RAG/
├── start_rag.py                  # Entry point with pre-flight checks
├── RAG_improved_v2_2_BOOSTING.py # Main system (latest version with all improvements)
├── check_dependencies.py         # Dependency verification
├── requirements.txt              # Python dependencies
│
├── docs/                         # User documents (PDFs, Markdown, TXT)
│   └── (auto-organized by user)
│
├── db_esxi/                      # Vector database (ChromaDB)
│   ├── chroma.sqlite3
│   └── index_manifest.json       # Tracks indexed files
│
├── logs/                         # Logs and metrics
│   ├── rag_YYYYMMDD.log         # Daily log
│   └── retrieval_metrics.jsonl  # Per-query metrics
│
└── Documentation files
    ├── README.md                 # User guide
    ├── ARQUITECTURA_v2.md        # Detailed architecture & flow
    └── MEJORAS_RELEVANCIA_EXPLICACION.md  # Improvement rationale
```

## Running the System

### Quick Start

```bash
# Run with pre-flight checks and clear progress indicators
python start_rag.py
```

This will:
1. Verify Ollama is running
2. Check dependencies
3. Validate documents exist
4. Check if vector DB needs rebuilding
5. Launch `RAG_improved_v2_2_BOOSTING.py` main system

### Direct Run (Skip Checks)

```bash
python RAG_improved_v2_2_BOOSTING.py
```

### Verify Installation

```bash
python check_dependencies.py
```

## Key Implementation Details

### Vector Database Location
- Default: `db_esxi/` directory
- Uses ChromaDB (local SQLite-based vector store)
- Automatically created on first run
- Persists across sessions—remove to force rebuild

### Document Loading
- **Supported formats**: PDF, Markdown, TXT
- **Location**: `docs/` directory (any subdirectory structure is fine)
- **Processing**:
  - PDFs: Loaded with `PyPDFLoader`, split by page
  - Markdown: `MarkdownHeaderTextSplitter` preserves heading hierarchy as metadata
  - TXT: Custom loader with UTF-8 encoding (Windows compatible)

### Chunking Strategy
- Recursive semantic splitting with fallback separators: `["\n\n\n", "\n\n", "\n", ". ", " "]`
- Default chunk size: 1000 characters with 200-character overlap
- For Markdown: Header hierarchy preserved to enable filtering/filtering by section

### Configuration
Core parameters in `RAG_improved_v2_2_BOOSTING.py` (top section):
```python
DB_DIR = "db_esxi"
MODEL_NAME = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
DOCS_DIR = "docs"
LOGS_DIR = "logs"
```

## Important Design Patterns

### Query Expansion (SimpleQueryExpander)
Uses heuristic term mapping rather than LLM calls for robustness:
- Maps common terms to VMware-specific synonyms (e.g., 'vm' → ['virtual machine', 'máquina virtual', 'guest'])
- Caches results to avoid redundant expansion
- Activates for queries <5 words

### Adaptive Alpha in Hybrid Retrieval
Dynamic weighting based on query characteristics:
- **Short queries** (< 8 words): Higher BM25 weight (60%) because keyword precision matters more
- **Long queries** (> 15 words): Higher vector weight (60%) to capture semantic meaning
- **Medium queries**: Balanced 50/50

### Reranking Pipeline
Cross-encoder style approach without external models:
- Top 15 results from hybrid search
- LLM re-evaluates each for relevance (0-10 score)
- Combined score: `original_score * 0.3 + llm_score * 0.7`
- Return top 5 to generator

### Two-Level Relevance Checking
1. **Fast reject**: Keyword overlap < 5% (avoids LLM calls for obvious mismatches)
2. **Semantic validate**: LLM scores overall context relevance ≥ 4/10

## Common Development Tasks

### Adding Support for New Document Format
1. Create loader in `RAG_improved_v2_2_BOOSTING.py` that returns `List[Document]`
2. Add file extension check in `load_documents()`
3. Test with sample document

### Adjusting Retrieval Quality
Edit these parameters in `RAG_improved_v2_2_BOOSTING.py`:
- `TOP_K_INITIAL`: Number of chunks from hybrid search (default 15)
- `TOP_K_RERANKED`: Final chunks passed to LLM (default 5)
- Relevance thresholds in `StrictRelevanceChecker`

### Changing LLM Model
Modify `MODEL_NAME` and `EMBEDDING_MODEL` in `RAG_improved_v2_2_BOOSTING.py`, then ensure models are pulled in Ollama:
```bash
ollama pull <model-name>
```

### Debugging Low Relevance Issues
Check `logs/retrieval_metrics.jsonl` for metrics on each query:
- `avg_similarity_score`: How well chunks matched semantically
- `sources_used`: Which documents were retrieved
- `retrieval_method`: Which path was taken (vector, BM25, or hybrid)
- `reranked`: Whether reranking was applied

## Important Notes

### Windows Compatibility
- UTF-8 encoding explicitly set for file handles and logging
- Path handling uses pathlib for cross-platform compatibility
- `start_rag.py` includes Windows-specific stdout/stderr reconfiguration

### Offline Operation
- All operations use local Ollama instance—no external API calls
- Vector embeddings stored locally in `db_esxi/`
- Logs stored locally in `logs/`

### Database Persistence
- `db_esxi/index_manifest.json` tracks which files have been indexed
- Changing documents in `docs/` automatically triggers re-indexing on next run
- Deleting `db_esxi/` forces complete rebuild (useful if document set changed significantly)

### Performance Considerations
- **First run**: Slow (embeddings generated for all documents)
- **Subsequent runs**: Fast (queries only, embeddings cached)
- **Large document sets**: Consider reducing `TOP_K_INITIAL` or `TOP_K_RERANKED` to speed up reranking phase
- **Memory usage**: ~50MB baseline + proportional to document set size

## Logging and Metrics

### Log Levels
Configured as INFO by default. Change in `RAG_improved_v2_2_BOOSTING.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

### Metric Files
- **Daily log**: `logs/rag_YYYYMMDD.log`
- **Query metrics**: `logs/retrieval_metrics.jsonl` (one JSON object per line)

Each query metric contains: `query`, `num_chunks_retrieved`, `avg_similarity_score`, `sources_used`, `context_length`, `retrieval_method`, `reranked`, `query_expanded`

## When to Make Changes

### Safe to change without breaking functionality:
- Document content in `docs/`
- Configuration values (model names, chunk sizes, weights)
- Prompt templates in `build_enhanced_prompt()`
- Log level and output formatting

### Requires careful testing:
- Chunking strategy or parameters
- Hybrid retrieval weighting algorithm
- Relevance checking thresholds
- Reranking score combination formula

### Risk of regression if changed:
- Query expansion term mappings (test with existing queries)
- LLM prompts for relevance checking/reranking (affects quality)
- Vector store initialization or metadata handling
