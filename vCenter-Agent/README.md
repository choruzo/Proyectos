# 🤖 vCenter Agent System - Multi-Agent AI Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/langchain-0.3.27+-orange.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive AI-powered multi-agent system for intelligent vCenter management and documentation consultation, featuring advanced RAG (Retrieval-Augmented Generation), MCP (Model Context Protocol) integration, and enterprise-grade security.

**Key Highlights**:
- 🧠 **Dual AI Agents**: vCenter Specialist + Documentation Consultant
- 🔍 **Advanced RAG**: Semantic search with 77 tests (87% passing)
- 🌐 **Web Interface**: Real-time chat, admin dashboard, monitoring
- 🔐 **Enterprise Security**: Audit trails, structured logging, role-based access
- ⚡ **Performance**: Connection pooling, caching, optimized queries
- 📊 **MCP Integration**: 17 standardized tools for vCenter operations

## 🌟 Features

### 🔧 vCenter Agent - VMware Operations
- **Natural Language Interface**: Talk to your vCenter infrastructure in plain language
- **17 MCP Tools**: Standardized operations (VMs, hosts, datastores, snapshots, cloning)
- **User Isolation**: Per-user VM namespaces and VLAN mapping
- **Connection Pooling**: Optimized pyVmomi connections with automatic cleanup
- **Dual Model Architecture**: Query formatter (qwen3:1.7b) + executor (llama3.1:8b)
- **Real-time Monitoring**: Live metrics from ESXi hosts and VMs
- **Advanced Operations**:
  - VM deployment from templates (single/multi-MCU environments)
  - Power management (on/off/reset/suspend)
  - Resource reporting and cleanup (obsolete VM detection)
  - Performance metrics export
  - Datastore browsing and management

### 📚 Documentation Consultant - Advanced RAG System
- **Semantic Search**: Query normalization + concept expansion (7 semantic families)
- **Smart Retrieval**: Chunk-based search with relevance scoring (keyword: 3.0, semantic: 1.5, section: 5.0)
- **Source Citation**: Automatic references in format `file.md#Section`
- **Markdown Native**: Optimized for .md files with section extraction
- **Low False Negatives**: Improved from 60% to <20% (target: <15%)
- **Auto-Indexing**: Whoosh-based indexing with cache management
- **Context Optimization**: 800-char fragments (up from 300) for better LLM context
- **Testing**: 77 comprehensive tests (RAG, integration, E2E)

### 🎯 Intelligent Orchestration
- **Hybrid Routing**: Keyword matching (agents.yaml) + LLM classification
- **Dual Session System**: Flask sessions + SQLite persistence
- **Context Preservation**: Per-user conversation memory with LangChain
- **Agent Isolation**: Complete separation of tools and knowledge sources
- **Fallback Handling**: Graceful degradation on routing failures
- **Performance**: <3s average response time for simple queries

### 🔐 Enterprise-Grade Security
- **Authentication**: bcrypt password hashing with argon2 support
- **Authorization**: Role-based access (admin/user) with permission inheritance
- **Audit Logging**: Complete trail of user actions and system changes
- **Rate Limiting**: Configurable API endpoint protection
- **Session Security**: 30-minute timeout, secure token management
- **Input Validation**: Comprehensive sanitization and XSS prevention
- **Security Events**: Dedicated logging category for security analysis

### 📊 Observability & Monitoring
- **Structured Logging**: JSON logs categorized by type (API, audit, security, performance)
- **Performance Metrics**: Request timing, resource usage, agent execution time
- **Real-time Dashboards**: Admin interface with system health monitoring
- **Log Rotation**: Automatic cleanup and archival of old logs
- **Context Middleware**: Request tracking with correlation IDs
- **Error Tracking**: Comprehensive error logging with stack traces

## 📁 Project Structure

```
vcenter_agent_system/
├── src/                         # Source code
│   ├── core/                    # Core agent functionality
│   │   ├── agent.py            # vCenter agent (17 MCP tools, LangChain integration)
│   │   └── doc_consultant.py   # Doc agent (RAG system, semantic search)
│   ├── api/                     # Web API and routes
│   │   └── main_agent.py       # Flask orchestrator (routing, sessions, auth)
│   ├── auth/                    # Authentication system
│   │   ├── auth_manager.py     # User management, bcrypt hashing
│   │   ├── rate_limiter.py     # API rate limiting
│   │   └── session_store.py    # SQLite session persistence
│   └── utils/                   # Utility modules
│       ├── vcenter_tools.py    # pyVmomi operations, connection pool
│       ├── doc_tools.py        # Markdown processing, search (800 char fragments)
│       ├── rag_retriever.py    # RAG core (semantic families, scoring)
│       ├── structured_logger.py # Structured logging framework
│       ├── logging_config.py   # Logging configuration
│       └── context_middleware.py # Request context tracking
├── server/                      # MCP Server integration
│   ├── mcp_vcenter_server.py   # Standalone MCP server (stdio transport)
│   ├── mcp_tool_registry.py    # Tool definitions and factory
│   └── mcp_tool_wrappers.py    # LangChain tool wrappers
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
│   └── js/                     # JavaScript (fetch API, Chart.js)
├── docs/                        # Documentation repository
│   ├── *.md                    # Markdown documentation files
│   └── media/                  # Images and diagrams
├── tests/                       # Integration tests
│   ├── test_doc_agent.py       # Doc consultant tests
│   ├── test_orchestrator.py    # Multi-agent routing tests
│   └── test_vcenter_tools.py   # vCenter operation tests
├── unitary_test/                # Unit tests (77 tests, 67 passing)
│   ├── test_rag_retriever.py   # RAG algorithm tests (28 tests)
│   ├── test_rag_integration.py # RAG with real docs (9 tests)
│   ├── test_doc_agent_e2e.py   # End-to-end tests (15 tests)
│   ├── test_doc_tools.py       # Document utilities (25 tests)
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
├── reports/                     # Generated reports (ZIP exports)
├── run.py                       # Main entry point (Flask server)
├── setup.py                     # Package setup (vcenter-agent command)
├── requirements_oficial.txt     # Production dependencies
├── pytest.ini                   # Pytest configuration
├── .coveragerc                  # Coverage settings
│
├── Documentation/               # Project documentation
│   ├── CLAUDE.md               # Development guide for Claude Code
│   ├── AGENTS.md               # Multi-agent architecture guide
│   ├── IMPLEMENTATION_SUMMARY.md # System implementation details
│   ├── LOGGING_IMPLEMENTATION_SUMMARY.md # Logging system docs
│   ├── MCP_SERVER_IMPLEMENTATION.md # MCP integration guide
│   ├── RAG_IMPROVEMENT_SUMMARY.md # RAG optimization history
│   └── RAG_INVESTIGATION_REPORT.md # Latest RAG analysis (2026-01-24)
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

- **Python 3.8+** (tested on 3.9, 3.10, 3.11)
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

4. **Install Ollama models**
   ```bash
   # Recommended models
   ollama pull llama3.1:8b      # Main executor model (8GB RAM)
   ollama pull qwen3:1.7b       # Query formatter (1.7GB RAM)

   # Alternative models
   ollama pull llama3.2:3b      # Lighter alternative
   ollama pull mistral:7b       # Alternative executor
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
   export ORCH_EXECUTOR_MODEL="llama3.1:8b"
   export ORCH_FORMATTER_MODEL="qwen3:1.7b"
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
ORCH_EXECUTOR_MODEL=llama3.1:8b ORCH_FORMATTER_MODEL=qwen3:1.7b python run.py

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
- llama3.1:8b model: ~8 GB
- qwen3:1.7b model: ~1.7 GB
- Document index (Whoosh): ~5-50 MB (depends on doc count)
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
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Flask)                   │
│                    http://localhost:5000                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Orchestrator (main_agent.py)                    │
│   ┌────────────────────────────────────────────────────┐    │
│   │  Routing Logic (Hybrid)                            │    │
│   │  1. Keyword Match (agents.yaml)                    │    │
│   │  2. LLM Classification (fallback)                  │    │
│   └────────────┬───────────────────────────┬───────────┘    │
└────────────────┼───────────────────────────┼────────────────┘
                 │                           │
        ┌────────▼────────┐         ┌───────▼────────┐
        │  vCenter Agent  │         │   Doc Agent    │
        │   (agent.py)    │         │(doc_consultant)│
        └────────┬────────┘         └───────┬────────┘
                 │                           │
    ┌────────────▼────────────┐   ┌─────────▼──────────┐
    │   MCP Tool Registry     │   │   RAG Retriever    │
    │  (17 vCenter tools)     │   │ (semantic search)  │
    └────────────┬────────────┘   └─────────┬──────────┘
                 │                           │
    ┌────────────▼────────────┐   ┌─────────▼──────────┐
    │   pyVmomi (vCenter)     │   │  Whoosh Index      │
    │   Connection Pool       │   │  (Markdown docs)   │
    └─────────────────────────┘   └────────────────────┘
```

### RAG System Architecture (Documentation Agent)

```
User Query: "¿Qué problemas hay con las VMs?"
     ↓
┌────────────────────────────────────────────────┐
│  1. Query Normalization                        │
│     - Remove: "según", "la", "en"              │
│     - Remove: "según la documentación"         │
│     - Extract: ["problemas", "vms"]            │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│  2. Semantic Expansion (7 families)            │
│     problemas → {error, fallo, limitación,     │
│                  consideración, requisito...}  │
│     vms → {vm, máquina virtual, virtual        │
│            machine, máquinas virtuales...}     │
│     Result: 3 keywords → 36 expanded terms     │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│  3. Chunk-based Search (Whoosh)                │
│     - Search at paragraph level                │
│     - Include section metadata                 │
│     - Max 800 chars per fragment               │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│  4. Relevance Scoring                          │
│     - Direct keyword match: weight 3.0         │
│     - Semantic match: weight 1.5               │
│     - Important section: weight 5.0            │
│     - Sort by total relevance                  │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│  5. Context Assembly                           │
│     - Top 5 documents                          │
│     - 5 paragraphs per doc                     │
│     - 800 chars per paragraph                  │
│     - Total: ~20KB context                     │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│  6. LLM Response Generation                    │
│     Model: llama3.1:8b (8K context window)     │
│     Prompt: Strict RAG (no hallucination)      │
│     Output: Answer + Sources (file.md#Section) │
└────────────────────────────────────────────────┘
```

**RAG Metrics (as of 2026-01-24)**:
- False Negative Rate: **~20%** (down from 60%, target: <15%)
- Average Relevance Score: **12.5/20**
- Context Utilization: **25%** (20KB of 8K tokens ~= 80KB capacity)
- Query Response Time: **3-8 seconds**
- Test Coverage: **77 tests** (28 RAG-specific, 67 passing)

### MCP Integration Architecture

```
┌─────────────────────────────────────────┐
│     vCenter Agent (agent.py)            │
│  ┌───────────────────────────────────┐  │
│  │   LangChain Tool Interface        │  │
│  │  (AgentExecutor with tools)       │  │
│  └─────────────┬─────────────────────┘  │
│                ↓                         │
│  ┌───────────────────────────────────┐  │
│  │   MCP Tool Registry               │  │
│  │  - create_tool_functions()        │  │
│  │  - Per-user context               │  │
│  │  - User mapping                   │  │
│  └─────────────┬─────────────────────┘  │
│                ↓                         │
│  ┌───────────────────────────────────┐  │
│  │   MCP Tool Wrappers               │  │
│  │  @tool decorators for LangChain   │  │
│  └─────────────┬─────────────────────┘  │
│                ↓                         │
│  ┌───────────────────────────────────┐  │
│  │   vCenter Tools (vcenter_tools.py)│  │
│  │  - Connection pooling             │  │
│  │  - pyVmomi operations             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Standalone MCP Server (optional):
┌─────────────────────────────────────────┐
│  mcp_vcenter_server.py                  │
│  - Stdio transport                      │
│  - Exposes all 17 tools via MCP         │
│  - Can be used by external MCP clients  │
└─────────────────────────────────────────┘
```

**MCP Tools Available** (17 total):
1. `get_templates` - List VM templates
2. `get_hosts` - List ESXi hosts
3. `get_datastores` - List datastores
4. `deploy_from_template` - Deploy single VM
5. `deploy_dev_env` - Deploy 1-MCU environment
6. `deploy_dev_env_2mcu` - Deploy 2-MCU environment
7. `list_vms_for_user` - User's VMs
8. `list_vms_by_host` - VMs on specific host
9. `list_all_vms` - All VMs
10. `delete_vms` - Delete VMs (with validation)
11. `clone_mcu_template` - Clone MCU template
12. `clone_multiple_mcu_template` - Clone multiple MCUs
13. `generate_resource_report` - Resource usage
14. `get_obsolete_vms` - Find old VMs
15. `export_vm_performance` - Performance metrics
16. `power_operations` - Power on/off/reset/suspend
17. `get_vm_details` - Detailed VM info

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

# System 2: SQLite Persistence (cross-restart)
sessions_db:
  - session_id (primary key)
  - username
  - created_at
  - last_activity
  - data (JSON)
```

## Dependencies

### Core Framework
- **Flask**: Web framework and API server
- **LangChain**: AI agent framework (v0.3.27+)
- **langchain-ollama**: Ollama integration (v0.3.7+)
- **Ollama**: Local LLM runtime

### vCenter Agent Dependencies
- **pyvmomi**: vCenter SDK for Python (v9.0.0.0+)
- **pyVim**: VMware vSphere automation
- **bcrypt**: Password hashing and security

### Documentation Agent Dependencies
- **python-docx**: Document processing and extraction
- **Whoosh**: Full-text search indexing engine
- **Markdown**: Document formatting and processing

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
| **AGENTS.md** | Multi-agent architecture and development guide |
| **IMPLEMENTATION_SUMMARY.md** | System implementation details and patterns |
| **LOGGING_IMPLEMENTATION_SUMMARY.md** | Structured logging system documentation |
| **MCP_SERVER_IMPLEMENTATION.md** | MCP integration architecture and tools |

### RAG Documentation

| Document | Description |
|----------|-------------|
| **RAG_IMPROVEMENT_SUMMARY.md** | Historical RAG improvements and optimizations |
| **RAG_INVESTIGATION_REPORT.md** | Latest RAG analysis (2026-01-24) |
| **Resumen_ivestigacion_RAG.md** | RAG investigation summary (Spanish) |

### GitHub Copilot CLI Agents

| Document | Description |
|----------|-------------|
| **COPILOT_AGENTS_QUICKSTART.md** | Quick start guide with examples ⭐ START HERE |
| **COPILOT_AGENTS.md** | Complete agent configuration reference |
| **AGENT_RAG_ENGINEER.md** | RAG optimization specialist (8.7 KB) |
| **AGENT_VCENTER_SPECIALIST.md** | vCenter operations expert (13.4 KB) |
| **AGENT_TEST_EXPERT.md** | Testing automation specialist (15.8 KB) |

### Project Documentation (docs/)

Technical documentation in Markdown format:
- `Vcenter.md` - vCenter installation and configuration
- `Esxi_desarrollo.md` - ESXi development environment
- `MCP_SERVER_IMPLEMENTATION.md` - MCP server architecture
- `Configuracion_templates.md` - Template configuration
- `Monitorización.md` - Monitoring and performance
- Additional guides for DNS, Git, TrueNAS, etc.

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

**"Import Error: No module named 'langchain'"**
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
ollama pull llama3.1:8b
```

**"vCenter connection failed"**
```bash
# Verify credentials in config/config.json
# Test connection manually
python -c "from vcenter_agent_system.src.utils.vcenter_tools import get_si; si = get_si(); print(si.content.about.fullName)"
```

**"Documentation agent returns 'no contiene información'"**
```bash
# This is a known issue (false negatives ~20%)
# Check RAG_INVESTIGATION_REPORT.md for solutions
# Quick fix: Increase fragment size in doc_tools.py line 282
```

**"Tests failing with 'langchain' import error"**
```bash
# Install test dependencies
pip install -r unitary_test/requirements_test.txt

# Or skip LLM-dependent tests
pytest -m "not llm" unitary_test/ -v
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

## 🔗 Related Projects

### LLM & AI Frameworks
- **LangChain**: [https://python.langchain.com/](https://python.langchain.com/)
- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **Whoosh**: [https://whoosh.readthedocs.io/](https://whoosh.readthedocs.io/)

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
- **LangChain** team for the excellent AI framework
- **Ollama** for local LLM runtime
- **VMware** for pyVmomi and vSphere SDK
- **Flask** team for the robust web framework

---

**Project Status**: ✅ Active Development  
**Latest Update**: 2026-01-24  
**Version**: 2.0  
**Maintainer**: jmartinb

---

<div align="center">

**[⬆ Back to Top](#-vcenter-agent-system---multi-agent-ai-platform)**

Made with ❤️ and 🤖 AI

</div>

![Login interface](./assets/Login.gif)
![Admin interface](./assets/Admin.gif)
![Chat interface](./assets/Chat.gif)

