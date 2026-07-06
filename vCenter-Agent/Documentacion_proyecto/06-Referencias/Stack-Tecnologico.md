---
tipo: referencia
estado: actual
relacionado:
  - "[[Arquitectura-Sistema]]"
  - "[[Changelog]]"
  - "[[Guia-Implementacion]]"
tags: [stack, tecnologias, dependencias, versiones]
version: 3.0
ultima_actualizacion: 2026-03-24
---

# 🔧 Stack Tecnológico

> Dependencias, versiones y tecnologías utilizadas en vCenter Multi-Agent System v3.0.

***
## 📋 Resumen por Capa

| Capa | Tecnologías Principales |
|------|-------------------------|
| **Backend** | Flask 2.0+, Python 3.9+ |
| **LLM/AI** | Ollama, gpt-oss:20b, nomic-embed-text |
| **Agent Framework** | LangChain 0.3.x, LangChain-Ollama |
| **VMware SDK** | pyvmomi 8.0+ |
| **Storage** | ChromaDB, SQLite, BM25 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Monitoring** | APScheduler, PySNMP |
| **Reporting** | ReportLab, (matplotlib planeado) |

***
## 🐍 Python Backend

### Core Dependencies

```python
# requirements_oficial.txt

# Web Framework
Flask==2.0.3
Flask-Cors==4.0.0

# LangChain - Agent Framework
langchain==0.3.2
langchain-community==0.3.1
langchain-ollama==0.2.1
langchain-text-splitters==0.3.0

# VMware SDK
pyvmomi==8.0.3.0.1

# Database & Storage
chromadb==0.5.9
sqlite3  # Python stdlib

# Authentication & Security
bcrypt==4.2.0

# Scheduling & Background
APScheduler==3.10.4

# SNMP Monitoring
pysnmp==4.4.12

# PDF Generation
reportlab==3.6.13

# Utilities
python-dotenv==1.0.0
requests==2.31.0
```

### Python Version

- **Mínima:** Python 3.9
- **Recomendada:** Python 3.12
- **Testeado:** Python 3.12

**Justificación:** LangChain 0.3.x requiere Python 3.9+. Type hints modernos.

***
## 🧠 LLM Stack

### Ollama Runtime

```bash
# Instalación Windows
https://ollama.com/download

# Instalación Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verificar instalación
ollama --version  # v0.1.x
```

**Endpoint por defecto:** `http://localhost:11434`

### Modelos LLM

| Modelo | Uso | Parámetros | Context Window |
|--------|-----|------------|----------------|
| **gpt-oss:20b** | Executor principal | 20B | 8192 tokens (vCenter/Orch) |
| **gpt-oss:20b** | Formatter opcional | 20B | 4096 tokens |
| **gpt-oss:20b** | Documentation agent | 20B | 16384 tokens (RAG) |

**Instalación:**
```bash
ollama pull gpt-oss:20b
```

**Configuración Ollama:**
```bash
# Variables de entorno
OLLAMA_HOST=0.0.0.0:11434  # Escuchar todas las interfaces
OLLAMA_MODELS=/path/to/models  # Ubicación de modelos
OLLAMA_NUM_PARALLEL=2  # Requests paralelas
```

### Modelo de Embeddings

| Modelo | Dimensiones | Uso |
|--------|-------------|-----|
| **nomic-embed-text** | 768 | Vector search ChromaDB |

**Instalación:**
```bash
ollama pull nomic-embed-text
```

**Performance:**
- Latencia: ~50-100ms por embedding (CPU)
- Throughput: ~20 embeddings/seg (CPU)
- Cache hit rate: ~30% con LRU 1000

***
## 📦 LangChain Ecosystem

### Versiones Compatibles

```
langchain==0.3.2
langchain-community==0.3.1  # Tools community
langchain-ollama==0.2.1     # Ollama integration
langchain-text-splitters==0.3.0  # Chunking
```

### Componentes Utilizados

| Componente | Uso |
|------------|-----|
| `ChatOllama` | LLM interface |
| `AgentExecutor` | Agent execution loop |
| `ConversationBufferMemory` | Memoria por usuario |
| `StructuredTool` | MCP tool wrappers |
| `RecursiveCharacterTextSplitter` | Chunking documentos |

***
## 🗄️ Storage & Retrieval

### ChromaDB

```python
chromadb==0.5.9
```

**Configuración:**
```python
import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.create_collection(
    name="documentation",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)
```

**Características:**
- Persistent storage en disco
- Vector similarity search (cosine)
- Metadata filtering
- Batch operations

**Uso de disco:** ~50-100 MB para 500 documentos (1400 chars/chunk)

### SQLite

**Versión:** 3.x (Python stdlib)

**Bases de datos:**
- `data/auth.db` - Usuarios y autenticación
- `data/users.db` - Sesiones agentes

**Schema management:** Manual (sin ORM)

### BM25 (Keyword Search)

**Implementación:** Custom (`src/utils/bm25_retriever.py`)

**Parámetros:**
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (document length normalization)

**Índice:** In-memory dict, reconstruido en cada inicio

***
## 🖥️ VMware Stack

### pyvmomi (VMware SDK)

```python
pyvmomi==8.0.3.0.1
```

**Compatibilidad vCenter:**
- vCenter 7.0+
- vCenter 8.0+ (recomendado)
- ESXi 7.0+ standalone

**Objetos principales:**
- `vim.VirtualMachine`
- `vim.HostSystem`
- `vim.Datastore`
- `vim.Task`
- `vim.ServiceInstance`

### Connection Pool Custom

**Implementación:** `src/utils/vcenter_tools.py::VCenterConnectionPool`

**Parámetros:**
- `max_connections`: 5
- `timeout`: 30s
- `ssl_verification`: False (dev/test)

***
## 🌐 Frontend Stack

### HTML5 + CSS3 + Vanilla JS

**No frameworks** - Zero dependencies frontend

**Archivos:**
```
templates/
  └── chat/
      └── orchestrator_chat_auth.html

static/
  ├── css/
  │   └── orchestrator_chat_auth.css
  └── js/
      ├── orchestrator_chat_auth.js
      └── marked.min.js  # Markdown parser (40KB)
```

### Marked.js (Markdown Rendering)

```html
<script src="{{ url_for('static', filename='js/marked.min.js') }}"></script>
```

**Versión:** v15.x  
**Tamaño:** ~40 KB minificado  
**Propósito:** Renderizar Markdown en respuestas del chat

**Configuración:**
```javascript
marked.use({
  breaks: true,          // \n → <br>
  gfm: true,             // GitHub Flavored Markdown
  headerIds: false,
  mangle: false
});
```

### Server-Sent Events (SSE)

**API:** Native `fetch()` + `ReadableStream`

**No usa `EventSource`** (solo GET, no POST)

**Protocolo custom:**
```
event: routing
data: {"agent": "vcenter"}

event: token
data: {"t": "palabra", "i": 0}

event: done
data: {"agent": "vcenter", "total_tokens": 47}
```

***
## 📊 Monitoring & Reporting

### APScheduler

```python
APScheduler==3.10.4
```

**Configuración:**
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    generate_daily_report,
    trigger='cron',
    hour=7, minute=0,
    id='daily_report'
)
scheduler.start()
```

**Jobs activos:**
- Daily report: 07:00
- Historical collector: cada 10 min
- TrueNAS SNMP: cada 10 min
- Cisco SNMP: cada 10 min

### PySNMP

```python
pysnmp==4.4.12
```

**Protocolos soportados:**
- SNMP v2c (Cisco Catalyst)
- SNMP v3 (TrueNAS, auth + priv)

**OIDs monitoreados:**
- `1.3.6.1.2.1.2.2.1.10` - ifInOctets (tráfico entrante)
- `1.3.6.1.2.1.2.2.1.16` - ifOutOctets (tráfico saliente)
- `1.3.6.1.4.1.2021.11.9` - ssCpuIdle
- `1.3.6.1.4.1.2021.4.6` - memTotalFree

### ReportLab (PDF Generation)

```python
reportlab==3.6.13
```

**Uso:**
```python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph

doc = SimpleDocTemplate("report.pdf", pagesize=A4)
story = [...]
doc.build(story)
```

**Características:**
- Generación programática de PDFs
- Tablas, párrafos, imágenes
- Estilos personalizados

### Matplotlib (Planeado - Mejora 2)

```python
# matplotlib>=3.5.0  # Actualmente comentado
```

**Uso futuro:** Gráficas de tendencias en PDFs

***
## 🔐 Security Stack

### Bcrypt

```python
bcrypt==4.2.0
```

**Configuración:**
```python
import bcrypt

# Hash password
salt = bcrypt.gensalt(rounds=12)  # Cost factor 12
hashed = bcrypt.hashpw(password.encode(), salt)

# Verify
bcrypt.checkpw(password.encode(), hashed)  # ~400ms
```

**Cost factor:** 12 (balance seguridad/UX)

***
## 🚀 Deployment Stack

### Development

```bash
# Flask development server
python run.py

# Ollama local
ollama serve
```

### Production (Recomendado)

```bash
# Gunicorn (WSGI server)
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Docker (Opcional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_oficial.txt .
RUN pip install -r requirements_oficial.txt

COPY . .
EXPOSE 5000

CMD ["python", "run.py"]
```

**Nota:** Ollama debe ejecutarse en host o container separado.

***
## 📏 Versiones Mínimas

| Dependencia | Mínima | Recomendada | Razón |
|-------------|--------|-------------|-------|
| Python | 3.9 | 3.12 | LangChain 0.3.x requiere 3.9+ |
| Flask | 2.0 | 2.3+ | Async support |
| Ollama | 0.1.0 | 0.1.x | API stability |
| pyvmomi | 8.0 | 8.0.3+ | vCenter 8.0 compatibility |
| ChromaDB | 0.5.0 | 0.5.9+ | Performance improvements |

***
## 🔄 Matriz de Compatibilidad

### vCenter Versions

| vCenter | pyvmomi 8.0.3 | MCP Tools | Notas |
|---------|---------------|-----------|-------|
| 7.0 | ✅ | ✅ | Totalmente compatible |
| 8.0 | ✅ | ✅ | Recomendado |
| 6.7 | ⚠️ | ⚠️ | Algunos features limitados |

### Operating Systems

| OS | Python 3.9+ | Ollama | SQLite | Notas |
|----|-------------|--------|--------|-------|
| Windows 10/11 | ✅ | ✅ | ✅ | Desarrollo + Producción |
| Ubuntu 20.04+ | ✅ | ✅ | ✅ | Producción recomendada |
| macOS 12+ | ✅ | ✅ | ✅ | Desarrollo |
| RHEL 8+ | ✅ | ✅ | ✅ | Producción enterprise |

***
## 📦 Instalación Completa

### 1. Python Dependencies

```bash
pip install -r requirements_oficial.txt
```

### 2. Ollama + Models

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### 3. Verificar Instalación

```python
python tests/check_system_status.py
```

**Output esperado:**
```
✅ Flask: 2.0.3
✅ LangChain: 0.3.2
✅ pyvmomi: 8.0.3.0.1
✅ Ollama: Running on localhost:11434
✅ ChromaDB: 0.5.9
✅ SQLite: 3.x
✅ All systems operational
```

***
## 🔧 Variables de Entorno

### Requeridas

```bash
# vCenter (si no en config.json)
VCENTER_HOST=vcenter.example.com
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=SecurePass123

# Flask
SECRET_KEY=your-secret-key-here-min-32-chars
```

### Opcionales

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Modelos LLM
ORCH_EXECUTOR_MODEL=gpt-oss:20b
ORCH_FORMATTER_MODEL=gpt-oss:20b  # Deshabilitado por defecto

# Logging
LOG_LEVEL=INFO  # DEBUG | INFO | WARNING | ERROR
```

***
## 📚 Documentos Relacionados

- [[Arquitectura-Sistema]] - Visión general del sistema
- [[Changelog]] - Evolución de versiones
- [[Guia-Implementacion]] - Deploy paso a paso
- [[Glosario]] - Términos técnicos

***
## 🆘 Troubleshooting Dependencias

### Problema: ChromaDB no instala

```bash
# Error: Building wheel for hnswlib failed
# Solución: Instalar build tools
pip install --upgrade pip setuptools wheel
pip install chromadb==0.5.9
```

### Problema: pyvmomi SSL errors

```python
# Solución: Deshabilitar verificación SSL (dev/test)
import ssl
context = ssl._create_unverified_context()
SmartConnect(..., sslContext=context)
```

### Problema: Ollama no responde

```bash
# Verificar servicio
curl http://localhost:11434/api/version

# Reiniciar Ollama
systemctl restart ollama  # Linux
# O relanzar desde terminal en Windows
```

***
*Última actualización: 2026-03-24 | v3.0*
