# 🤖 vCenter Agent System - Multi-Agent AI Platform

[![Python 3.12 recommended](https://img.shields.io/badge/python-3.12%20recommended-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LangGraph Runtime](https://img.shields.io/badge/langgraph-runtime-blueviolet.svg)](https://www.langchain.com/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AI platform for vCenter operations and technical documentation queries, with LangGraph orchestration, hybrid RAG, native MCP tool execution, predictive ML, and automated operational reporting.

**Key Highlights**:
- 🧠 **LangGraph-first runtime**: orchestrator, vCenter agent, and documentation agent run as persistent graphs with SQLite checkpointing
- 🔧 **Native tool execution**: progressive tool selection and provider-native tool calling in `server/tool_runtime.py`
- 🔍 **Advanced RAG**: ChromaDB + BM25 hybrid retrieval with query expansion, reranking, and retrieval metrics
- 🌐 **Web interface**: SSE streaming chat, admin dashboard, advanced monitoring, and operational panels
- 📋 **Operational reporting**: daily PDF reports, historical trend charts, AI commentary, and admin-triggered generation
- 🤖 **Predictive ML already in use**: data foundation + saturation prediction + workload classification + incident labeling
- 🔌 **Multi-provider LLM**: Ollama and `llama_cpp` via internal adapters in `llm_factory.py`
- 🔐 **Enterprise controls**: structured logging, audit trail, RBAC, CSRF protection, and session management

## 🌟 Features

### 🔧 vCenter Agent - VMware Operations
- **LangGraph-backed runtime**: the vCenter agent now runs on `src/core/agent_graph.py` with persistent per-user threads and checkpointing in `data/lg_state.db`
- **Native tool runtime**: tools are selected and executed through `server/tool_runtime.py`, with progressive disclosure, JSON-schema generation, and explicit confirmation for destructive actions
- **MCP catalog for VMware operations**: VM lifecycle, snapshots, reconfiguration, ESXi direct monitoring, datastore browsing, alarms, and events
- **User isolation**: per-user VM namespaces, session abbreviations, and conversation state
- **Connection pooling**: optimized pyVmomi reuse with automatic cleanup
- **Multi-provider LLM**: Ollama or `llama_cpp` through `src/utils/llm_factory.py` + `src/utils/model_adapters.py`
- **Advanced operations**:
  - VM deployment from templates
  - Power management and cleanup
  - Resource and datastore reporting
  - ESXi status and performance queries
  - Snapshot, network, and VM reconfiguration workflows

### 📚 Documentation Consultant - Advanced RAG v2.4
- **LangGraph doc agent**: retrieval/generation flow compiled in `src/core/doc_agent_graph.py`
- **Hybrid retrieval**: ChromaDB vector search + BM25 keyword search with adaptive alpha
- **Query expansion**: 62 term families (VMware + project tooling), bidirectional expansion
- **Smart reranking**: top results are re-ranked before answer generation
- **Folder-aware search modes**: strict / boosting / global
- **Low-hallucination setup**: `num_ctx=16384`, `temperature=0.1`, and abstention path when retrieval is weak
- **Source citation**: references in `file.md#Section` format
- **Metrics logging**: per-query telemetry in `logs/retrieval_metrics.jsonl`

### 💬 Chat Interface
- **Real-time streaming**: SSE-based token streaming with routing and heartbeat events
- **Markdown rendering**: marked.js v15 renders code blocks, tables, lists, and citations after stream completion
- **Fallback path**: legacy `/chat` endpoint remains available if SSE fails

### 🎯 Intelligent Orchestration
- **4-layer classifier**: keywords → critical regex → intent detection → weighted scoring → LLM fallback
- **Sticky routing**: follow-up messages reuse the last agent for a short window, enabling natural multi-turn conversations
- **Per-user persistence**: conversational state is stored by LangGraph checkpointer, not by classic LangChain memory buffers
- **Agent isolation**: vCenter and documentation flows keep separate tools, state, and knowledge sources

### 🧠 LangGraph Runtime - Migration Status
- **Migration completed in production**:
  - `src/api/orchestrator_graph.py` drives routing and sticky follow-ups
  - `src/core/agent_graph.py` handles the vCenter workflow, native tool execution, and destructive-action confirmation
  - `src/core/doc_agent_graph.py` manages the documentation retrieval/generation loop
- **Provider wrappers removed from the main runtime**: `src/utils/llm_factory.py` now returns internal adapters such as `OllamaChatAdapter` and `OpenAICompatibleChatAdapter`
- **ToolNode / bind_tools no longer drive the vCenter production flow**: the active path uses provider-native schemas from `ToolCatalog` and execution helpers in `server/tool_runtime.py`
- **Compatibility remains where useful**: `server/mcp_tool_wrappers.py` is kept as an interoperability layer, but it is no longer the primary execution path
- **LangChain still appears as support infrastructure**: `langchain_core.messages` remains useful for message objects and compatibility, while orchestration moved to LangGraph

### 🔐 Enterprise-Grade Security
- **Authentication**: bcrypt password hashing and role-based access control
- **Authorization**: admin/user separation with protected admin routes
- **Audit logging**: complete user-action trail and security-specific logs
- **Rate limiting and CSRF**: configurable endpoint protection and state-changing request validation
- **Session security**: web session controls plus LangGraph thread cleanup

### 📊 Observability & Monitoring
- **Structured logging**: categorized JSON logs for API, audit, security, performance, business, and system events
- **Performance telemetry**: request timing, agent progress, retrieval metrics, and ML logs
- **Admin dashboards**: advanced ESXi monitoring, reports UI, statistics, and ML incident management
- **Background automation**: historical collectors, report scheduler, and ML retraining jobs

### 📋 Automated Daily Reports
- **Scheduled generation**: `background_agents/report_scheduler.py` creates the operational PDF every day at **07:00**
- **Admin web panel**: `/admin/reports` lists generated PDFs, supports download, and exposes **Generate Now**
- **On-demand API**: `POST /api/admin/reports/generate` runs the report immediately and waits up to **900 seconds**
- **Multi-source collection**: the PDF aggregates ESXi metrics, vCenter inventory, alarms, events, datastore state, licenses, log analysis, user activity, and TrueNAS telemetry
- **7-day and historical trends**:
  - `collect_weekly_trend()` builds the rolling 7-day view
  - completed weeks are consolidated into `{host}_weekly_history.json`
  - if `matplotlib` is available, the PDF adds historical trend charts without writing temporary image files
- **ML-aware reports**: before rendering, the report agent refreshes cached ML outputs and can append:
  - **ML Predictions — Resource Saturation (Beta)**
  - **Workload Pattern Classification (ML)**
- **AI commentary**: sections can include `AI Analysis` blocks generated by the configured model
- **Fault tolerant**: source-specific failures are isolated so one subsystem does not abort the whole PDF
- **Output**: `reports/informe_rendimiento_YYYY-MM-DD.pdf`
- **Manual Trigger (Python)**:
  ```python
  from background_agents.performance_report_agent import PerformanceReportAgent
  pdf = PerformanceReportAgent().run()
  print(pdf)
  ```
- **Manual Trigger (API)**:
  ```bash
  curl -X POST http://localhost:5000/api/admin/reports/generate \
       -H "Content-Type: application/json" --cookie "session=<admin-session>"
  ```

### 🤖 Predictive ML - Phases 0, 1, and 2 Operational

The ML subsystem is no longer just in “data foundation”. Currently there is a **data pipeline**, **supervised prediction**, **unsupervised classification**, **incident labeling**, and **training automation**.

#### Phase 0 — Data Foundation (operational)
- **Extended retention**: `config/config.json` maintains a **30-day** history (`historical_retention_hours: 720`) while the dashboard still shows 24 h
- **Parquet store**: `ml/data_store_builder.py` generates `data/ml_store/{host}/raw/*.parquet`
- **Sanitization**: `ml/data_sanitizer.py` produces `processed/` with imputation, encoding, and `data_quality_flag`
- **Incident labeling**:
  - automatic queue in `data/ml_labels/pending_review.jsonl`
  - confirmed dataset in `data/ml_labels/incidents.jsonl`
  - admin UI at `/admin/ml-incidents`

#### Phase 1 — Saturation Prediction (operational)
- **Current model**: `LinearRegression` for CPU and memory
- **Training**: `ml/model_trainer.py`
- **Inference**: `ml/predictor.py`
- **Output**: `data/ml_predictions/latest.json` + `logs/ml/predictions.jsonl`
- **Main fields**:
  - `predicted_48h`
  - `predicted_7d`
  - `hours_to_warning`
  - `hours_to_critical`
  - `confidence`
- **API**: `GET /api/ml/predict/saturation`
- **Automation**: `background_agents/ml_trainer_scheduler.py` runs daily Parquet rebuild and weekly retraining

#### Phase 2 — Workload Pattern Classification (operational)
- **Current model**: `K-Means` over aggregated hourly windows
- **Implementation**: `ml/workload_classifier.py`
- **Output**: `data/ml_predictions/latest_workload.json` + `logs/ml/workload_classifications.jsonl`
- **Patterns**: `stable`, `office_hours`, `batch_nocturnal`, `erratic`, `saturated`, `underutilized`
- **API**: `GET /api/ml/classify/workload`
- **Integration**: classification also appears in the PDF when the feature is enabled

#### Validation and Usage

```bash
# Validate the ML database
python tests/validate_ml_data_foundation.py

# The ML scheduler starts alongside the application
python run.py
```

#### Pending Roadmap (Phases 3-5)

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **3** | Anomaly detection (Isolation Forest or similar) | Pending |
| **4** | Advanced forecasting with seasonality (e.g. Prophet) | Pending |
| **5** | Interactive predictions and trends dashboard | Pending |

## 📁 Project Structure

```
vcenter_agent_system/
├── src/                         # Source code
│   ├── core/                    # Core agent functionality
│   │   ├── agent.py            # vCenter agent entrypoint over LangGraph
│   │   ├── agent_graph.py      # LangGraph runtime for vCenter + native tools
│   │   ├── doc_consultant.py   # Doc agent entrypoint
│   │   └── doc_agent_graph.py  # LangGraph retrieval/generation workflow
│   ├── api/                     # Web API and routes
│   │   ├── main_agent.py       # Flask app, auth, admin, API routes
│   │   └── orchestrator_graph.py # LangGraph supervisor for routing
│   ├── auth/                    # Authentication system
│   │   ├── auth_manager.py     # User management, bcrypt hashing
│   │   ├── rate_limiter.py     # API rate limiting
│   │   └── session_store.py    # SQLite session persistence
│   └── utils/                   # Utility modules
│       ├── vcenter_tools.py    # pyVmomi operations, connection pool
│       ├── query_classifier.py # 4-layer query classifier (routing logic)
│       ├── doc_tools.py        # Hybrid RAG v2.4 initialization (ChromaDB + BM25)
│       ├── rag_retriever.py    # RAG core + v2.4 query expander bridge
│       ├── hybrid_retriever.py # Vector + BM25 combination with adaptive alpha
│       ├── query_expander.py   # 62 term families (VMware + project), bidirectional expansion
│       ├── embedding_cache.py  # LRU cache (1000 queries)
│       ├── reranker.py         # Heuristic multi-factor reranking
│       ├── search_modes.py     # Strict/boosting/global folder filtering
│       ├── bm25_retriever.py   # BM25 keyword-based retrieval
│       ├── chunker.py          # Adaptive semantic chunking (MD-aware)
│       ├── document_loader.py  # Multi-format loader (PDF, MD, TXT)
│       ├── vector_store_manager.py # ChromaDB + SHA1 manifest management
│       ├── retrieval_metrics.py # JSONL retrieval metrics logging
│       ├── structured_logger.py # Structured logging framework
│       ├── logging_config.py   # Logging configuration
│       ├── llm_factory.py      # Provider adapter factory (Ollama / llama_cpp)
│       ├── model_adapters.py   # Internal chat/embedding adapters
│       └── context_middleware.py # Request context tracking
├── server/                      # MCP Server integration
│   ├── mcp_vcenter_server.py   # Standalone MCP server (stdio transport)
│   ├── mcp_tool_registry.py    # Tool definitions and factory
│   ├── mcp_tool_wrappers.py    # Compatibility wrappers for integrations/tests
│   └── tool_runtime.py         # Native tool selection/execution runtime
├── background_agents/           # Autonomous background agents
│   ├── performance_report_agent.py # Daily PDF report generator + ML/historical sections
│   ├── report_scheduler.py     # APScheduler wrapper — triggers report at 07:00
│   ├── ml_trainer_scheduler.py # Weekly retraining + daily Parquet rebuild
│   ├── truenas_snmp_collector.py   # SNMP v3 collector for TrueNAS (ZFS, net, temp)
│   └── __init__.py
├── ml/                          # Predictive ML — phases 0, 1, and 2
│   ├── __init__.py
│   ├── data_store_builder.py   # Converts historical JSON → daily Parquet per host
│   ├── migrate_historical_to_parquet.py # One-shot migration (7d → raw Parquets)
│   ├── data_sanitizer.py       # Pipeline: raw/ → processed/ (imputation, encoding, flags)
│   ├── migrate_raw_to_processed.py # Batch sanitization of existing Parquets
│   ├── feature_builder.py      # Feature engineering for training
│   ├── model_trainer.py        # Supervised saturation training
│   ├── predictor.py            # Inference + prediction cache
│   ├── workload_classifier.py  # K-Means workload pattern classification
│   └── incident_labeler.py     # Atomic JSONL labeling + automatic candidate discovery
├── config/                      # Configuration files
│   ├── config.json             # vCenter credentials, ESXi hosts
│   ├── agents.yaml             # Agent routing keywords
│   └── logging_config.json     # Logging system config
├── templates/                   # Jinja2 HTML templates
│   ├── admin/                  # Admin dashboard (monitoring, users)
│   ├── chat/                   # Chat interface (streaming, history)
│   ├── auth/                   # Login/logout pages
│   └── index.html              # Landing page
├── static/                      # Static web assets
│   ├── css/                    # Stylesheets (responsive design)
│   └── js/                     # JavaScript (fetch API, Chart.js, marked.js v15)
├── docs/                        # Documentation repository
│   ├── *.md                    # Markdown documentation files
│   └── media/                  # Images and diagrams
├── tests/                       # Integration tests
│   ├── test_doc_agent.py       # Doc consultant tests
│   ├── test_orchestrator.py    # Multi-agent routing tests
│   ├── test_vcenter_tools.py   # vCenter operation tests
│   └── validate_ml_data_foundation.py # 4 checks Data Foundation (retención, Parquet, calidad, etiquetas)
├── unitary_test/                # Unit tests (188+ tests)
│   ├── test_routing.py         # Orchestrator routing tests (39 tests)
│   ├── test_query_classifier.py # 4-layer classifier tests (57 tests)
│   ├── test_mcp_bypass_prevention.py # MCP security tests (8 tests)
│   ├── test_rag_retriever.py   # RAG algorithm tests (28 tests)
│   ├── test_rag_integration.py # RAG with real docs (9 tests)
│   ├── test_doc_agent_e2e.py   # End-to-end tests (15 tests)
│   ├── test_doc_tools.py       # Document utilities (25 tests)
│   ├── conftest.py             # Shared fixtures and mock registry
│   ├── requirements_test.txt   # Test dependencies
│   └── run_tests.py            # Test runner with coverage
├── logs/                        # Structured log files (JSON format)
│   ├── api/                    # API request/response logs
│   ├── audit/                  # User actions, system changes
│   ├── security/               # Auth events, security alerts
│   └── performance/            # Timing metrics, resource usage
├── scripts/                     # Utility scripts
│   ├── log_maintenance.py      # Log rotation and cleanup
│   └── setup_log_maintenance.ps1 # Windows log maintenance setup
├── reports/                     # Generated PDF reports (manual and scheduled)
├── data/                        # Runtime data
│   ├── chroma_db/              # ChromaDB vector store (persistent)
│   ├── ml_store/               # Parquet store for ML — {host}/raw/ + {host}/processed/
│   ├── ml_labels/              # Incident labels — incidents.jsonl + pending_review.jsonl
│   ├── ml_models/              # Trained models (.pkl + metadata)
│   └── ml_predictions/         # latest.json + latest_workload.json
├── run.py                       # Main entry point (Flask server)
├── setup.py                     # Package setup (vcenter-agent command)
├── requirements_oficial.txt     # Production dependencies
├── pytest.ini                   # Pytest configuration
├── .coveragerc                  # Coverage settings
│
├── DOCS_proyect/                # Technical project documentation
│   ├── API/                    # REST API reference
│   ├── Background_agent/       # Background agents (ESXi, TrueNAS SNMP)
│   ├── Chat/                   # Chat UI architecture and guides
│   ├── Login_Autentificación/  # Authentication system docs
│   ├── MCP/                    # MCP architecture and security
│   ├── Mejoras/                # Proposed improvements
│   ├── RAG/                    # RAG v2.4 hybrid system docs
│   └── vCenter_Agent/          # vCenter agent docs (architecture, pool, tools, security)
├── Other_docs/                  # Supplementary documentation
│   ├── Implementation/         # Implementation summaries and change tracking
│   └── Investigaciones/        # Research and analysis documents
│
└── GitHub Copilot Agents/       # Custom agents for GitHub Copilot CLI
    ├── COPILOT_AGENTS.md        # Agent configuration and usage
    ├── COPILOT_AGENTS_QUICKSTART.md # Quick start guide
    ├── AGENT_RAG_ENGINEER.md    # RAG optimization specialist
    ├── AGENT_VCENTER_SPECIALIST.md # vCenter operations expert
    └── AGENT_TEST_EXPERT.md     # Testing automation specialist
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (**3.12 recommended**, aligned with production)
- **Ollama** installed and running ([https://ollama.ai](https://ollama.ai))
- **VMware vCenter** access (for vCenter agent)
- **Git** for cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vcenter_agent_system
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Production dependencies
   pip install -r requirements_oficial.txt

   # Optional: Development dependencies (includes testing)
   pip install -r unitary_test/requirements_test.txt
   ```

4. **Install Ollama models** (or configure `llama_cpp`)
   ```bash
    # Option A: Ollama (local inference)
    ollama pull gpt-oss:20b      # Main executor model

    # Option B: llama.cpp server (OpenAI-compatible API)
    # Edit config/config.json → llm_provider.provider: "llama_cpp"
    # and set base_url + model_name for your llama-server instance
   ```

5. **Configure vCenter connection**
   
   Edit `config/config.json`:
   ```json
   {
     "vcenter_host": "vcenter.example.com",
     "vcenter_user": "administrator@vsphere.local",
     "vcenter_pass": "your-secure-password",
     "vcenter_port": 443,
     "esxi_hosts": [
       "esxi-host-1.example.com",
       "esxi-host-2.example.com"
     ]
   }
   ```

6. **Add documentation files** (optional for Doc agent)
   ```bash
   # Place Markdown files in docs/ directory
   cp /path/to/your/docs/*.md vcenter_agent_system/docs/
   ```

7. **Run the application**
   ```bash
   # Development mode (Flask debug)
   python run.py

   # Production mode (with environment variables)
   export ORCH_EXECUTOR_MODEL="gpt-oss:20b"
   export ORCH_FORMATTER_MODEL="gpt-oss:20b"
   export FLASK_ENV="production"
   python run.py
   ```

8. **Access the web interface**
   - Open browser: `http://localhost:5000`
   - Login with default credentials:
     - Admin: `admin` / `admin123`
     - User: `user` / `user123`

### Running with start_app.bat (Windows)

```bash
# Simple one-click start (Windows only)
start_app.bat
```

This script:
- Activates the virtual environment
- Starts the Flask server
- Opens logs automatically

### Using the Multi-Agent System

The system automatically routes queries to the appropriate agent using hybrid routing (keyword matching + LLM classification).

#### 🔧 vCenter Queries (routed to vCenter Agent)

**VM Management**:
```
"List my VMs"
"Show details of VM mcu-test"
"Power on the MCU machine"
"Delete VM old-test"
"Clone template Ubuntu-20.04 to new-server"
```

**Deployment**:
```
"Deploy a new dev environment"
"Deploy 2-MCU configuration"
"Get available templates"
"Deploy from template Ubuntu-Server"
```

**Infrastructure**:
```
"Show ESXi host status"
"List all datastores"
"Check datastore capacity"
"Generate resource report"
"Find obsolete VMs older than 30 days"
```

**Performance**:
```
"Export VM performance metrics"
"Show CPU and memory usage for host esxi-01"
```

#### 📚 Documentation Queries (routed to Documentation Agent)

**General Search**:
```
"What does the documentation say about vCenter installation?"
"Search for VLAN configuration procedures"
"Find information about ESXi backup"
"How do I configure networking?"
```

**Specific Topics**:
```
"Explain the MCP Server implementation"
"What are the requirements for VM cloning?"
"Show me troubleshooting steps for connectivity issues"
"List all documents about configuration"
```

**Implicit Queries** (uses semantic search):
```
"What problems can I have with virtual machines?"
"What should I consider when deploying VMs?"
"Are there any common errors during installation?"
```

### Command Line Tools

```bash
# Start Flask development server
python run.py

# Run with specific model configuration
ORCH_EXECUTOR_MODEL=gpt-oss:20b ORCH_FORMATTER_MODEL=gpt-oss:20b python run.py

# Run MCP server standalone (for external clients)
python -m vcenter_agent_system.server.mcp_vcenter_server

# Run tests
python -m pytest tests/ -v
python -m pytest unitary_test/ -v

# Run with coverage
python -m pytest unitary_test/ --cov=vcenter_agent_system --cov-report=html

# Test RAG improvements
python demo_rag_improvement.py

# Validate document indexing
python -c "from vcenter_agent_system.src.utils.doc_tools import get_markdown_manager; m = get_markdown_manager(); print(f'Indexed {len(m.get_document_list())} documents')"
```

### Administrative Features

#### Structured Logging
The system provides comprehensive logging across multiple categories:

- **API Logs** (`logs/api/`): Request/response tracking
- **Audit Logs** (`logs/audit/`): User actions and system changes
- **Security Logs** (`logs/security/`): Authentication and security events
- **Performance Logs** (`logs/performance/`): System performance metrics

#### Log Monitoring
```bash
# View real-time logs
tail -f logs/api/api.log
tail -f logs/audit/audit.log
tail -f logs/security/security.log
```

### API Endpoints

#### Core Application Routes
- `GET /` - Main index page
- `POST /login` - User authentication
- `GET /chat` - Multi-agent chat interface
- `GET /admin_dashboard` - Admin dashboard
- `POST /logout` - User logout

#### vCenter API Routes
- `GET /api/hosts_status` - Host information
- `GET /api/datastores` - Datastore information
- `GET /api/vm_info` - Virtual machine details
- `POST /api/browse_datastore` - Browse datastore folders
- `POST /api/chat` - Chat endpoint with agent routing

#### Documentation API Routes
- `POST /api/search_documents` - Search documentation
- `GET /api/list_documents` - List available documents
- `POST /api/reindex_documents` - Reindex document collection

#### Admin API Routes
- `GET /api/system_status` - System health and metrics
- `GET /api/logs` - Access structured logs
- `POST /api/maintenance` - System maintenance operations

#### Report API Routes (admin only)
- `GET /admin/reports` - Web panel: list all generated PDF reports with download links and **Generate Now** button
- `POST /api/admin/reports/generate` - Trigger immediate report generation (runs in a worker thread, timeout 900 s)
- `GET /api/admin/reports/download/<filename>` - Download a specific PDF report by filename

#### ML API Routes
- `GET /admin/ml-incidents` - Admin UI for incident labeling and candidate review
- `GET /api/ml/predict/saturation` - Return cached or on-demand saturation forecasts (single host or all enabled hosts)
- `GET /api/ml/classify/workload` - Return cached or on-demand workload classification (single host or all enabled hosts)
- `GET /api/ml/incidents/candidates` - List pending ML incident candidates
- `POST /api/ml/incidents/collect` - Query vCenter and append new incident candidates for review

## Configuration

### vCenter Configuration (`config/config.json`)

```json
{
  "vcenter_host": "vcenter.example.com",
  "vcenter_user": "administrator@vsphere.local",
  "vcenter_pass": "password123",
  "vcenter_port": 443
}
```

### Agent Configuration (`config/agents.yaml`)

Configure AI agent behavior, tools, and routing:

```yaml
vcenter:
  type: "tool-calling"
  route_keywords: ["vm", "host", "datastore", "power", "esxi", "vcenter"]
  description: "vCenter operations agent"

documentation:
  type: "tool-calling"  
  route_keywords: ["doc", "documentation", "help", "guide", "manual"]
  description: "Documentation consultant agent"
```

### Logging Configuration (`config/logging_config.json`)

Configure structured logging behavior:

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "structured": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
  },
  "handlers": {
    "api": {
      "class": "logging.FileHandler",
      "filename": "logs/api/api.log",
      "formatter": "structured"
    },
    "audit": {
      "class": "logging.FileHandler", 
      "filename": "logs/audit/audit.log",
      "formatter": "structured"
    }
  }
}
```

## Development

### Adding New Features

#### Adding New vCenter Tools
1. **New Tools**: Add to `src/core/agent.py`
2. **New API Routes**: Add to `src/api/main_agent.py`
3. **New Templates**: Add to `templates/`
4. **New Utilities**: Add to `src/utils/vcenter_tools.py`

#### Adding New Documentation Features
1. **Document Tools**: Add to `src/utils/doc_tools.py`
2. **Agent Logic**: Modify `src/core/doc_consultant.py`
3. **Indexing**: Enhance document indexing capabilities

#### Adding Logging and Monitoring
1. **Structured Logging**: Use `src/utils/structured_logger.py`
2. **Context Management**: Leverage `src/utils/context_middleware.py`
3. **Performance Monitoring**: Add metrics to operations

## 🧪 Testing

### Test Structure

```
tests/                          # Integration tests
├── test_doc_agent.py          # Documentation agent tests
├── test_orchestrator.py       # Multi-agent routing tests
└── test_vcenter_tools.py      # vCenter operation tests (with mocks)

unitary_test/                   # Unit tests (77 total, 67 passing)
├── test_rag_retriever.py      # RAG algorithm tests (28 tests)
│   ├── Query normalization (5 tests)
│   ├── Semantic expansion (6 tests)
│   ├── Relevance calculation (5 tests)
│   ├── Implicit questions (4 tests)
│   └── Edge cases (4 tests)
├── test_rag_integration.py    # RAG with real docs (9 tests)
├── test_doc_agent_e2e.py      # End-to-end tests (15 tests)
├── test_doc_tools.py          # Document utilities (25 tests)
└── run_tests.py               # Test runner with coverage
```

### Running Tests

```bash
# All tests
python -m pytest

# With coverage report
python -m pytest --cov=vcenter_agent_system --cov-report=html
open htmlcov/index.html  # View coverage report

# Specific test suite
python -m pytest tests/test_doc_agent.py -v
python -m pytest unitary_test/test_rag_retriever.py -v

# Specific test
python -m pytest unitary_test/test_rag_retriever.py::test_query_normalization -v

# With markers (if configured)
pytest -m rag          # Only RAG tests
pytest -m vcenter      # Only vCenter tests
pytest -m integration  # Only integration tests

# Custom test runner (with coverage)
python unitary_test/run_tests.py
```

### Test Coverage (as of 2026-01-24)

| Component | Tests | Passing | Coverage |
|-----------|-------|---------|----------|
| RAG Retriever | 28 | 28 | 95% |
| RAG Integration | 9 | 9 | 88% |
| Doc Agent E2E | 15 | 5* | 65% |
| Doc Tools | 25 | 25 | 92% |
| **Total** | **77** | **67** | **87%** |

*10 tests skipped due to missing langchain in test environment

### Key Test Cases

**RAG False Negative Validation**:
```python
def test_issue_query_finds_results():
    """Validate query from issue #X returns results"""
    manager = get_markdown_manager()
    query = "segun la documentacion, que problemas hay con las maquinas virtuales"
    results = manager.search_documents(query, limit=10)
    
    assert len(results) > 0
    assert any('Esxi_desarrollo.md' in r['document'] for r in results)
    assert any('Problemas' in r.get('section', '') for r in results)
```

**vCenter Tool Mocking**:
```python
@pytest.fixture
def mock_si():
    """Mock pyVmomi ServiceInstance"""
    si = MagicMock()
    mock_vm = MagicMock()
    mock_vm.name = "test-vm"
    mock_vm.runtime.powerState = "poweredOn"
    # ... mock setup
    return si

def test_list_vms(mock_si):
    vms = list_vms(mock_si)
    assert len(vms) > 0
    assert vms[0]['name'] == "test-vm"
```

### Demo Scripts

```bash
# RAG improvement demo
python demo_rag_improvement.py

# Shows:
# - Query normalization steps
# - Semantic expansion (3 keywords → 36 terms)
# - Search results with relevance scores
# - Before/after comparison

# ML Data Foundation validation (4 checks)
python tests/validate_ml_data_foundation.py
# Checks: retention ≥14 days | Parquet store | sanitization quality | labels file
```

## 📊 Performance & Benchmarks

### Response Times (Average)

| Operation | Time | Notes |
|-----------|------|-------|
| Simple vCenter query | 2-4s | "List my VMs" |
| Complex vCenter query | 4-8s | "Deploy dev environment" |
| Doc search (cache hit) | 1-3s | Query already normalized |
| Doc search (cache miss) | 3-8s | Full RAG pipeline |
| First request (cold start) | 8-15s | Model loading + indexing |

### Resource Usage

**Memory**:
- Base Flask app: ~150 MB
- gpt-oss:20b model: ~8 GB
- gpt-oss:20b model: ~1.7 GB
- Document index / vector store: ~5-50 MB (depends on corpus size and embeddings)
- **Total peak**: ~10 GB

**CPU**:
- Idle: <5%
- Query processing: 50-100% (single core)
- Concurrent requests: Scales with CPU cores

**Disk**:
- Application: ~50 MB
- Dependencies: ~500 MB
- Models: ~10 GB
- Logs: ~1-10 MB/day (depends on usage)

### Optimization Strategies

**Connection Pooling**:
```python
# vCenter connections reused per user
VCenterConnectionPool:
  - Max connections: 10
  - Timeout: 30 minutes
  - Automatic cleanup: Every 5 minutes
```

**Document Caching**:
```python
# Document index cached for 5 minutes
_scan_documents(force_refresh=False):
  if cache_age < 300s:
    return cached_documents
```

**Query Optimization**:
- Normalized queries cached (in-memory)
- Semantic expansions pre-computed
- Relevance scores cached per query pattern

### Scaling Recommendations

**Small deployments** (1-10 users):
- 4 CPU cores
- 12 GB RAM
- Single server deployment

**Medium deployments** (10-50 users):
- 8 CPU cores
- 24 GB RAM
- Load balancer + 2 app servers
- Shared vCenter connection pool

**Large deployments** (50+ users):
- 16+ CPU cores
- 32+ GB RAM
- Load balancer + N app servers
- Redis for shared session store
- Separate vCenter agent + doc agent services

### Development Workflow

1. **Setup Development Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r merged-requirements.txt
   ```

2. **Run Pre-commit Checks**:
   ```bash
   # Verify all tests pass
   python test_imports.py && python test_structure.py
   
   # Check code quality
   flake8 src/ --max-line-length=100
   ```

3. **Development Best Practices**:
   - Use structured logging for all operations
   - Maintain agent isolation principles
   - Test both individual agents and orchestration
   - Follow security best practices

### Building Distribution

```bash
# Create distribution packages
python setup.py sdist bdist_wheel
```

## 🏗️ Architecture

### Multi-Agent System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Flask / SSE Web Layer                     │
│               auth, admin, reports, chat routes              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                LangGraph Orchestrator Graph                  │
│        classify_task + sticky routing + per-user state       │
└───────────────────────┬───────────────────────┬──────────────┘
                        │                       │
             ┌──────────▼──────────┐  ┌────────▼─────────┐
             │   vCenter Graph     │  │   Doc Graph      │
             │ agent_graph.py      │  │ doc_agent_graph  │
             └──────────┬──────────┘  └────────┬─────────┘
                        │                      │
        ┌───────────────▼──────────────┐  ┌────▼─────────────────────┐
        │ ToolCatalog + ToolRuntime    │  │ Hybrid Retriever v2.4    │
        │ provider-native tool calling │  │ ChromaDB + BM25 + rerank │
        └───────────────┬──────────────┘  └────┬─────────────────────┘
                        │                      │
             ┌──────────▼─────────┐   ┌───────▼─────────────┐
             │ pyVmomi + MCP      │   │ docs/ + Chroma store│
             │ connection pool    │   │ + retrieval metrics │
             └────────────────────┘   └─────────────────────┘
```

### RAG System Architecture (Documentation Agent)

```
User Query: "What problems are there with the VMs?"
     ↓
┌──────────────────────────────────────────────────────┐
│  1. Search Mode Detection                            │
│     strict / boosting / global                      │
└────────────────┬─────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────┐
│  2. Query Normalization + Expansion                  │
│     stop-words, filler phrases, 62 term families    │
└────────────────┬─────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────┐
│  3. Hybrid Retrieval                                │
│     ChromaDB vector search + BM25 keyword scoring   │
└────────────────┬─────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────┐
│  4. Score Fusion + Folder Filtering                  │
│     adaptive alpha + strict/boosted folders         │
└────────────────┬─────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────┐
│  5. Reranking                                        │
│     term frequency + length + position heuristics   │
└────────────────┬─────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────┐
│  6. Generate / Abstain                               │
│     factual answer with sources or safe abstention  │
└──────────────────────────────────────────────────────┘
```

**RAG Metrics (as of 2026-01-24)**:
- False Negative Rate: **~20%** (down from 60%, target: <15%)
- Average Relevance Score: **12.5/20**
- Context Utilization: improved with `num_ctx=16384`
- Query Response Time: **3-8 seconds**
- Test Coverage: **77 tests** (28 RAG-specific, 67 passing)

### MCP Integration Architecture

```
┌─────────────────────────────────────────────┐
│ vCenter LangGraph node                      │
│ - builds ToolCatalog per user/session       │
│ - selects active groups by query/follow-up  │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ ToolRuntime                                 │
│ - provider-native JSON schemas              │
│ - active tool subset (progressive disclosure)│
│ - safety level + destructive confirmations  │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ MCP registry + pyVmomi wrappers             │
│ - create_tool_functions(username, session)  │
│ - connection pool reuse                     │
│ - per-user context isolation                │
└─────────────────────────────────────────────┘

Standalone MCP Server (optional):
┌─────────────────────────────────────────┐
│  mcp_vcenter_server.py                  │
│  - Stdio transport                      │
│  - Exposes the vCenter MCP catalog      │
│  - Can be used by external MCP clients  │
└─────────────────────────────────────────┘
```

**MCP Tool Groups**:

| Group | Tools |
|-------|-------|
| Core VM | Template listing, deployment, clone, delete, power operations, VM detail and cleanup/report tools |
| Snapshots | Create, list, revert, and delete snapshots |
| VM Reconfiguration | CPU/RAM/network changes, rename, NIC management |
| ESXi Direct | Host status, resources, and performance |
| Datastore | Browse and enumerate datastores |
| Events / Alarms | VM events and active alarms |

### Connection Management

```python
# VCenter Connection Pool
class VCenterConnectionPool:
    connections: Dict[str, ServiceInstance]
    last_used: Dict[str, float]
    timeout: 1800  # 30 minutes
    
    def get_connection(username):
        # Returns per-user connection
        # Automatic cleanup of expired connections
        # Thread-safe with locks
    
# Usage in tools
si = get_user_si(username)  # ALWAYS use this, never get_si()
```

### Session Management (Dual System)

```python
# System 1: Flask Sessions (HTTP requests)
session['username'] = 'jmartinb'
session['role'] = 'admin'

# System 2: LangGraph checkpointer (conversational state)
data/lg_state.db:
  - thread_id por usuario/agente
  - message history
  - sticky routing state
  - pending confirmations / retrieval state
```

## Dependencies

### Core Framework
- **Flask**: Web framework and API server
- **LangGraph**: graph orchestration and checkpointed conversational state
- **langchain-core**: message objects and compatibility primitives
- **Ollama**: Local LLM runtime

### vCenter Agent Dependencies
- **pyvmomi**: vCenter SDK for Python (v9.0.0.0+)
- **pyVim**: VMware vSphere automation
- **bcrypt**: Password hashing and security

### Documentation Agent Dependencies
- **ChromaDB**: Persistent vector store
- **pypdf / Markdown / TXT loaders**: Multi-format document ingestion
- **Custom BM25 retriever**: keyword scoring on top of the same corpus

### Predictive ML Dependencies
- **pyarrow** (≥14.0.0): Parquet engine for the ML data store (`data/ml_store/`)
- **pandas**: DataFrame manipulation and transformation (sanitization, imputation)
- **scikit-learn**: LinearRegression, K-Means, and model persistence

### Reporting Dependencies
- **reportlab**: PDF generation
- **APScheduler**: report scheduling and retraining
- **matplotlib**: optional historical charts in the PDF

### LLM Provider Dependencies
- **Internal adapters**: `OllamaChatAdapter` and `OpenAICompatibleChatAdapter`
- **llama_cpp / llama-server**: OpenAI-compatible endpoint for chat and embeddings

### System Dependencies
- **PyYAML**: Configuration management
- **Requests**: HTTP client for API interactions
- **Flask-Limiter**: API rate limiting
- **Werkzeug**: WSGI utilities and security

### Development Dependencies
- **pytest**: Testing framework
- **pytest-cov**: Test coverage reporting
- **flake8**: Code linting and style checking

### Logging and Monitoring
- **Custom Structured Logger**: Built-in logging framework
- **Context Middleware**: Request context management
- **Performance Monitoring**: Built-in metrics collection

## Logging

### Structured Logging System
The system implements comprehensive structured logging with categorization:

```
logs/
├── api/              # API request/response logs
│   └── api.log
├── audit/            # User actions and system changes
│   └── audit.log
├── security/         # Authentication and security events
│   └── security.log
├── performance/      # Performance metrics and monitoring
│   └── performance.log
└── system.log        # General system events
```

### Log Categories
- **Business Operations**: Core business logic and operations
- **API Operations**: Web API requests and responses
- **Security Events**: Authentication, authorization, and security
- **Audit Trail**: User actions and system modifications
- **System Events**: Application lifecycle and errors
- **Performance Metrics**: Response times and resource usage

### Accessing Logs
```bash
# View real-time logs
tail -f logs/api/api.log
tail -f logs/audit/audit.log

# Search logs for specific events
grep "user_login" logs/security/security.log
grep "vm_deployment" logs/audit/audit.log

# Monitor performance
grep "performance" logs/performance/performance.log
```

### Log Management
- **Automatic Rotation**: Configured log rotation and archival
- **Size Management**: Automatic cleanup of old log files
- **Structured Format**: JSON-formatted logs for easy parsing
- **Context Preservation**: Full request context in all logs

## Security

### Authentication & Authorization
- **Multi-user Support**: Admin and regular user roles
- **Secure Password Hashing**: bcrypt with salt
- **Session Management**: Secure token-based sessions with timeout
- **Role-based Access**: Different permissions for admin/user roles

### Security Monitoring
- **Audit Logging**: Complete audit trail of user actions
- **Security Event Logging**: Authentication attempts, authorization failures
- **Context Tracking**: Full request context for security analysis
- **Session Security**: Automatic session timeout and cleanup

### Input Validation & Protection
- **Input Sanitization**: All user inputs validated and sanitized
- **Path Traversal Protection**: File system access controls
- **XSS Prevention**: Template escaping and output sanitization
- **CSRF Protection**: Session-based request validation
- **Rate Limiting**: API endpoint protection against abuse

### vCenter Security
- **Credential Management**: Secure credential storage and handling
- **Connection Security**: HTTPS/SSL connections to vCenter
- **Permission Inheritance**: Respects vCenter user permissions
- **API Key Management**: Secure API token handling

### Data Protection
- **Configuration Security**: Secure configuration file handling
- **Log Security**: Protected log file access
- **Document Security**: Isolated document access for documentation agent
- **Memory Protection**: Secure handling of sensitive data in memory

## 🤖 GitHub Copilot CLI Agents

This project includes custom specialized agents for **GitHub Copilot CLI** to enhance development productivity with context-aware assistance.

### Available Agents

#### 🔍 RAG Engineer (`AGENT_RAG_ENGINEER.md`)
**Specialization**: RAG system optimization and false negative reduction

**Use for**:
- Reducing false negatives (currently ~20%, target <15%)
- Optimizing semantic search and query normalization
- Adjusting fragment sizes and context windows
- Implementing new semantic families
- Debugging retrieval issues

**Example**:
```bash
gh copilot suggest "Using AGENT_RAG_ENGINEER.md context: 
Implement fix to increase fragment size from 300 to 800 chars 
in doc_tools.py line 282 with backward compatibility"
```

**Key Knowledge**:
- Files: `doc_consultant.py`, `doc_tools.py`, `rag_retriever.py`
- Current issues: 60% false negative rate (improved to ~20%)
- Fragment limit: 300 chars (should be 800)
- Context usage: 14% (should be >25%)

---

#### ⚙️ vCenter Specialist (`AGENT_VCENTER_SPECIALIST.md`)
**Specialization**: VMware vCenter/ESXi integration via pyVmomi

**Use for**:
- Adding new MCP tools for vCenter operations
- Debugging pyVmomi connection issues
- Implementing VM lifecycle operations (snapshots, cloning)
- Optimizing connection pool
- User isolation and VLAN mapping

**Example**:
```bash
gh copilot suggest "Using AGENT_VCENTER_SPECIALIST.md context:
Add MCP tool for VM snapshot management with create, list, 
delete, and revert operations. Include user permission validation."
```

**Key Knowledge**:
- Files: `agent.py`, `vcenter_tools.py`, `mcp_tool_registry.py`
- 17 existing MCP tools
- Connection pooling (30 min timeout)
- User mapping and isolation patterns

---

#### 🧪 Test Expert (`AGENT_TEST_EXPERT.md`)
**Specialization**: Comprehensive testing (unit, integration, E2E)

**Use for**:
- Creating tests for new features
- Mocking vCenter (pyVmomi), Ollama, Flask
- Validating RAG improvements
- Increasing test coverage (currently 87%)
- Writing fixtures and parametrized tests

**Example**:
```bash
gh copilot suggest "Using AGENT_TEST_EXPERT.md context:
Create comprehensive tests for fragment size increase:
- Unit test for 800 char truncation
- Integration test with real docs
- Regression test for existing queries"
```

**Key Knowledge**:
- Files: `tests/`, `unitary_test/`
- 77 total tests (28 RAG, 25 doc_tools, 15 E2E)
- Mocking patterns for vCenter and LLM
- pytest configuration and fixtures

---

### Quick Start with Agents

**Option 1: Direct Context** (simplest)
```bash
gh copilot suggest "Context: AGENT_RAG_ENGINEER.md

Task: Analyze why query 'clonación VMs' returns 'no contiene información' 
despite citing Vcenter.md#Clonación section"
```

**Option 2: Environment Variables** (recommended)
```powershell
# PowerShell - add to $PROFILE
$env:RAG_AGENT = Get-Content ".\AGENT_RAG_ENGINEER.md" -Raw
$env:VCENTER_AGENT = Get-Content ".\AGENT_VCENTER_SPECIALIST.md" -Raw
$env:TEST_AGENT = Get-Content ".\AGENT_TEST_EXPERT.md" -Raw

# Usage
gh copilot suggest "$env:RAG_AGENT

TASK: Implement fragment size increase"
```

```bash
# Bash - add to .bashrc
export RAG_AGENT=$(cat ./AGENT_RAG_ENGINEER.md)
export VCENTER_AGENT=$(cat ./AGENT_VCENTER_SPECIALIST.md)
export TEST_AGENT=$(cat ./AGENT_TEST_EXPERT.md)

# Usage
gh copilot suggest "$RAG_AGENT

TASK: Fix false negatives"
```

### Agent Selection Matrix

| Your Need | Agent | File |
|-----------|-------|------|
| RAG not finding existing info | RAG Engineer | `AGENT_RAG_ENGINEER.md` |
| False negatives in search | RAG Engineer | `AGENT_RAG_ENGINEER.md` |
| Add vCenter operation | vCenter Specialist | `AGENT_VCENTER_SPECIALIST.md` |
| pyVmomi connection issues | vCenter Specialist | `AGENT_VCENTER_SPECIALIST.md` |
| Create tests for new feature | Test Expert | `AGENT_TEST_EXPERT.md` |
| Mock vCenter/Ollama | Test Expert | `AGENT_TEST_EXPERT.md` |
| Increase test coverage | Test Expert | `AGENT_TEST_EXPERT.md` |

### Documentation

- **`COPILOT_AGENTS_QUICKSTART.md`**: Quick start guide with copy-paste examples
- **`COPILOT_AGENTS.md`**: Complete configuration and usage reference
- **Individual agent files**: Detailed context and expertise for each agent

### Workflow Example: Fixing RAG False Negatives

```bash
# Step 1: Analysis
gh copilot suggest "CONTEXT: AGENT_RAG_ENGINEER.md
Diagnose why 60% of queries return 'no contiene información' 
Focus on fragment size, context window, and scoring"

# Step 2: Implementation  
gh copilot suggest "CONTEXT: AGENT_RAG_ENGINEER.md
Implement fragment size increase from 300 to 800 chars
File: doc_tools.py line 282"

# Step 3: Testing
gh copilot suggest "CONTEXT: AGENT_TEST_EXPERT.md
Create tests validating false negative reduction:
- test_fragment_size_800()
- test_false_negative_clonacion()
- test_regression_suite()"

# Step 4: Validation
python -m pytest unitary_test/test_rag_*.py -v
python demo_rag_improvement.py
```

**See `COPILOT_AGENTS_QUICKSTART.md` for more examples and workflows.**

### Development Guidelines
1. **Fork the repository** and create a feature branch
2. **Follow coding standards**: PEP 8 for Python, proper documentation
3. **Maintain agent isolation**: Keep vCenter and documentation agents separate
4. **Use structured logging**: Implement proper logging in all new features
5. **Add comprehensive tests**: Include both unit and integration tests
6. **Update documentation**: Keep README and AGENTS.md current

### Pull Request Process
1. Ensure all tests pass: `python test_imports.py && python test_structure.py`
2. Run the full test suite: `python -m pytest tests/ -v`
3. Check code quality: `flake8 src/ --max-line-length=100`
4. Update documentation as needed
5. Submit pull request with clear description

### Development Setup
```bash
# Clone and setup development environment
git clone <repository-url>
cd vcenter_agent_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r merged-requirements.txt

# Run development checks
python test_imports.py
python test_structure.py
python tests/test_doc_agent.py
```

### Code Review Checklist
- [ ] Code follows project patterns and standards
- [ ] Proper error handling and logging implemented
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Security considerations addressed
- [ ] Performance impact considered
- [ ] Agent isolation maintained

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| **README.md** (this file) | Complete project overview and quick start |
| **CLAUDE.md** | Development guide for Claude Code IDE |
| **.github/copilot-instructions.md** | Complete development guide for GitHub Copilot |

### DOCS_proyect/ (Technical Documentation)

| Document | Description |
|----------|-------------|
| **DOCS_proyect/API/API_REFERENCE.md** | Complete REST API endpoint reference |
| **DOCS_proyect/Background_agent/BACKGROUND_AGENTS_TECHNICAL_DOCUMENTATION.md** | Background agents (ESXi historical, TrueNAS SNMP) |
| **DOCS_proyect/Chat/ARQUITECTURA_CHAT.md** | SSE streaming architecture and Markdown rendering |
| **DOCS_proyect/Chat/GUIA_FUNCIONAMIENTO.md** | Chat usage guide |
| **DOCS_proyect/Chat/GUIA_IMPLEMENTACION_TECNICA.md** | Technical implementation guide |
| **DOCS_proyect/Login_Autentificación/AUTENTICACION.md** | Authentication system and session management |
| **DOCS_proyect/MCP/MCP_TECHNICAL_DOCUMENTATION.md** | MCP architecture, tool registry, security patterns |
| **DOCS_proyect/Mejoras/MEJORAS_PROPUESTAS.md** | Proposed technical improvements |
| **DOCS_proyect/Mejoras/MEJORAS_PROPUESTAS_funcionalidad.md** | Proposed functional improvements |
| **DOCS_proyect/RAG/HYBRID_SYSTEM_README.md** | RAG v2.4 hybrid system (ChromaDB + BM25) |
| **DOCS_proyect/RAG/RAG_TECHNICAL_DOCUMENTATION.md** | RAG pipeline deep-dive |
| **DOCS_proyect/vCenter_Agent/ARQUITECTURA.md** | vCenter agent architecture |
| **DOCS_proyect/vCenter_Agent/CONNECTION_POOL.md** | Connection pool design |
| **DOCS_proyect/vCenter_Agent/MCP_TOOLS.md** | MCP tool catalog reference |
| **DOCS_proyect/vCenter_Agent/SEGURIDAD.md** | Security model |

### Other_docs/ (Supplementary Documentation)

| Document | Description |
|----------|-------------|
| **Other_docs/Implementation/LOGGING_IMPLEMENTATION_SUMMARY.md** | Structured logging system |
| **Other_docs/Implementation/STICKY_ROUTING_IMPLEMENTATION.md** | Conversational memory and sticky routing |
| **Other_docs/Implementation/IMPLEMENTACION_AUTH_MODULE.md** | Auth module implementation |
| **Other_docs/Implementation/SQLITE_MIGRATION.md** | SQLite migration guide |
| **Other_docs/Implementation/SUPERUSER_IMPLEMENTATION_SUMMARY.md** | Superuser role implementation |
| **Other_docs/Implementation/TESTING_GUIDE.md** | Testing guide and best practices |
| **Other_docs/Investigaciones/OLLAMA_CONCURRENCY_ASYNC_ANALYSIS.md** | Ollama concurrency and async analysis |

### Know-How Documentation (docs/)

Technical documentation indexed by the RAG agent:
- `Vcenter.md` - vCenter installation and configuration
- `Esxi_desarrollo.md` - ESXi development environment
- `Configuracion_templates.md` - Template configuration
- `Monitorización.md` - Monitoring and performance
- Additional guides for DNS, Git, TrueNAS, Cantata, Doors, GTR, SBR, SonarQube, etc.

## 🤝 Contributing

### Development Guidelines

1. **Follow project patterns**: Use existing code structure and naming conventions
2. **Maintain agent isolation**: Keep vCenter and Documentation agents completely separated
3. **Use structured logging**: Implement proper logging in all new features
4. **Add comprehensive tests**: Include unit, integration, and E2E tests
5. **Update documentation**: Keep README and relevant docs current
6. **Use GitHub Copilot Agents**: Leverage custom agents for context-aware development

### Pull Request Checklist

- [ ] All tests pass (`pytest tests/ unitary_test/ -v`)
- [ ] Code follows PEP 8 (`flake8 src/ --max-line-length=100`)
- [ ] New features have tests (target: >85% coverage)
- [ ] Documentation updated (README, AGENTS.md, etc.)
- [ ] No secrets in code or config files
- [ ] Structured logging implemented for new operations
- [ ] Agent isolation maintained (no cross-agent dependencies)

### Development Workflow

```bash
# 1. Setup environment
git clone <repository-url>
cd vcenter_agent_system
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements_oficial.txt
pip install -r unitary_test/requirements_test.txt

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Develop with GitHub Copilot agents
gh copilot suggest "Using AGENT_RAG_ENGINEER.md context: ..."

# 4. Run tests
python -m pytest tests/ unitary_test/ -v
python -m pytest --cov=vcenter_agent_system --cov-report=html

# 5. Check code quality
flake8 src/ --max-line-length=100
black src/ --check  # Optional: auto-formatting

# 6. Commit and push
git add .
git commit -m "feat: descriptive commit message"
git push origin feature/my-feature

# 7. Create Pull Request
# Include description of changes, related issues, test results
```

### Code Style

**Python**:
- Follow PEP 8 (max line length: 100)
- Use type hints where appropriate
- Docstrings for all public functions
- Meaningful variable names

**Logging**:
```python
from src.utils.structured_logger import get_structured_logger, LogCategory

logger = get_structured_logger('my_module')
logger.info("Operation completed", LogCategory.BUSINESS, 
           operation="deploy_vm", vm_name="test-vm", duration=3.5)
```

**Error Handling**:
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error("Operation failed", LogCategory.SYSTEM, 
                error=str(e), operation="risky_operation")
    return {"success": False, "error": str(e)}
```

## 📞 Support

### Getting Help

1. **Check documentation** in this README and related files
2. **Review GitHub Copilot agents** for context-aware assistance
3. **Search existing issues** in the repository
4. **Check logs** in `logs/` directory for error details
5. **Run diagnostic tests**: `pytest tests/ -v`

### Troubleshooting

#### Common Issues

**"Import Error: No module named 'langgraph' / 'langchain_core'"**
```bash
# Install missing dependencies
pip install -r requirements_oficial.txt
```

**"Connection refused to Ollama"**
```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve

# Verify model is available
ollama pull gpt-oss:20b
```

**"vCenter connection failed"**
```bash
# Verify credentials in config/config.json
# Test connection manually
python -c "from vcenter_agent_system.src.utils.vcenter_tools import get_si; si = get_si(); print(si.content.about.fullName)"
```

**"Documentation agent returns 'no contiene información'"**
```bash
# Rebuild/validate the hybrid retriever inputs
python tests/validate_hybrid_system.py
```

**"ML predictions are empty or unavailable"**
```bash
# Check ML configuration and cached outputs
python tests/validate_ml_data_foundation.py

# Confirm ml scheduler is enabled in config/ml_config.json
# Then restart the app so run.py launches the scheduler
python run.py
```

**"Tests failing with LangGraph / ML imports"**
```bash
# Install test dependencies
pip install -r unitary_test/requirements_test.txt
```

#### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL="DEBUG"
python run.py

# Watch specific log category
tail -f logs/api/api.log
tail -f logs/security/security.log
tail -f logs/performance/performance.log
```

#### Performance Issues

```bash
# Check resource usage
tail -f logs/performance/performance.log | grep "duration"

# Profile specific operation
python -m cProfile -o profile.stats run.py
python -m pstats profile.stats

# Check vCenter connection pool
grep "connection_pool" logs/system.log
```

### Reporting Issues

When reporting issues, please include:

- **Error message**: Full stack trace from logs
- **Steps to reproduce**: Exact query or operation that failed
- **Environment**: 
  - Python version (`python --version`)
  - Ollama version (`ollama --version`)
  - OS and version
- **Configuration**: Relevant config (without passwords)
- **Logs**: Relevant log excerpts from `logs/`
- **Test results**: Output of `pytest tests/ -v`

If the issue is specific to reports, also include:

- Whether the failure happened in the **07:00 scheduler** or via **Generate Now**
- Whether `matplotlib` is installed (historical charts are optional)
- Whether ML sections were expected (`config/ml_config.json` with predictors enabled)
- A sample PDF filename from `reports/` if one was produced partially

## 🔗 Related Projects

### LLM & AI Frameworks
- **LangGraph**: [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)
- **LangChain Core**: [https://python.langchain.com/](https://python.langchain.com/)
- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **ChromaDB**: [https://www.trychroma.com/](https://www.trychroma.com/)

### VMware Integration
- **pyVmomi**: [https://github.com/vmware/pyvmomi](https://github.com/vmware/pyvmomi)
- **vSphere Automation SDK**: [https://github.com/vmware/vsphere-automation-sdk-python](https://github.com/vmware/vsphere-automation-sdk-python)

### MCP (Model Context Protocol)
- **Anthropic MCP**: [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol)
- **MCP Python SDK**: [https://github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)

## 📜 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Anthropic** for Claude and MCP specification
- **LangGraph / LangChain** ecosystem for the graph and message abstractions
- **Ollama** for local LLM runtime
- **VMware** for pyVmomi and vSphere SDK
- **Flask** team for the robust web framework

---

**Project Status**: ✅ Active Development
**Latest Update**: 2026-06-02
**Version**: 2.1
**Maintainer**: jmartinb

---

<div align="center">

**[⬆ Back to Top](#-vcenter-agent-system---multi-agent-ai-platform)**

Made with ❤️ and 🤖 AI

</div>
