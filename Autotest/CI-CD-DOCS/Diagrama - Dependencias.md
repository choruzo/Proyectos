# 🔗 Diagrama - Dependencias

## Visión General

Mapa completo de módulos, scripts, archivos de configuración y sus relaciones en el sistema CI/CD.

**Relacionado con**:
- [[Arquitectura del Pipeline]] - Sistema de ejecución
- [[Arquitectura Web UI]] - Sistema web
- [[Diagrama - Flujo Completo]] - Flujo del pipeline

---

## Módulos del Sistema

```mermaid
graph TB
    subgraph "Daemon Systemd"
        CICD[cicd.service]
        WEB[cicd-web.service]
    end
    
    subgraph "Orquestador"
        MAIN[ci_cd.sh]
    end
    
    subgraph "Scripts Bash"
        GM[git_monitor.sh]
        COMP[compile.sh]
        DEP[deploy.sh]
        COMMON[common.sh]
    end
    
    subgraph "Scripts Python"
        SQ[sonar_check.py]
        VC[vcenter_api.py]
    end
    
    subgraph "Web UI"
        APP[app.py<br/>Flask]
        TPL[templates/*.html]
        STATIC[static/css,js]
    end
    
    subgraph "Configuración"
        YAML[ci_cd_config.yaml]
        ENV[.env]
        SONAR_PROP[sonar-project.properties]
    end
    
    subgraph "Persistencia"
        DB[(pipeline.db<br/>SQLite)]
        LOGS[logs/*.log]
    end
    
    subgraph "Herramientas Externas"
        GIT[Git Repo]
        SONARQUBE[SonarQube API]
        VCENTER[vCenter API]
        TARGET[Target VM SSH]
    end
    
    CICD --> MAIN
    WEB --> APP
    
    MAIN --> GM
    MAIN --> COMP
    MAIN --> SQ
    MAIN --> VC
    MAIN --> DEP
    
    GM --> COMMON
    COMP --> COMMON
    DEP --> COMMON
    
    MAIN --> YAML
    MAIN --> ENV
    SQ --> SONAR_PROP
    
    MAIN --> DB
    MAIN --> LOGS
    APP --> DB
    APP --> LOGS
    
    APP --> TPL
    APP --> STATIC
    
    GM --> GIT
    SQ --> SONARQUBE
    VC --> VCENTER
    DEP --> TARGET
    
    COMMON --> YAML
    COMMON --> DB
    
    style CICD fill:#FF6B6B,color:#fff
    style WEB fill:#4ECDC4,color:#fff
    style MAIN fill:#FF6B6B,color:#fff
    style APP fill:#4ECDC4,color:#fff
    style DB fill:#95E1D3,color:#000
```

---

## Dependencias por Módulo

### `ci_cd.sh` (Orquestador)

**Depende de**:
- `scripts/common.sh` (functions)
- `config/ci_cd_config.yaml` (config)
- `config/.env` (credentials)
- `db/pipeline.db` (state)
- `logs/*.log` (logging)

**Invoca**:
- `scripts/git_monitor.sh`
- `scripts/compile.sh`
- `python/sonar_check.py`
- `python/vcenter_api.py`
- `scripts/deploy.sh`

**Ver**: [[Arquitectura del Pipeline#Orquestador Principal]]

---

### `scripts/git_monitor.sh`

**Depende de**:
- `scripts/common.sh`
- Git CLI tool
- `config/ci_cd_config.yaml` (git.url, git.tag_pattern)
- `db/pipeline.db` (processed_tags table)

**APIs externas**:
- `git ls-remote` → Git repository

**Ver**: [[Pipeline - Git Monitor]]

---

### `scripts/compile.sh`

**Depende de**:
- `scripts/common.sh`
- Make, gcc/g++ (build tools)
- `config/ci_cd_config.yaml` (compilation.compile_dir, timeout)
- `db/pipeline.db` (build_logs table)

**Invoca**:
- `build_DVDs.sh` (del código fuente)

**Ver**: [[Pipeline - Compilación]]

---

### `python/sonar_check.py`

**Depende de**:
- Python 3.6+
- Libraries: `requests`, `PyYAML`, `urllib3`
- `config/ci_cd_config.yaml` (sonarqube.*)
- `config/sonar-project.properties`
- `utils/sonar-scanner/`
- `utils/build-wrapper/`
- `db/pipeline.db` (sonar_results table)

**APIs externas**:
- SonarQube REST API (https://YOUR_SONARQUBE_SERVER)

**Ver**: [[Pipeline - SonarQube]]

---

### `python/vcenter_api.py`

**Depende de**:
- Python 3.6+
- Libraries: `requests`, `PyYAML`, `urllib3`
- `config/ci_cd_config.yaml` (vcenter.*)
- `config/.env` (VCENTER_USER, VCENTER_PASSWORD)

**APIs externas**:
- vCenter REST API

**Ver**: [[Pipeline - vCenter]]

---

### `scripts/deploy.sh`

**Depende de**:
- `scripts/common.sh`
- SSH client + key authentication
- `config/ci_cd_config.yaml` (target_vm.*)
- `~/.ssh/id_rsa` (SSH key)

**APIs externas**:
- SSH a YOUR_TARGET_VM_IP (target VM)

**Ver**: [[Pipeline - SSH Deploy]]

---

### `web/app.py` (Flask Web UI)

**Depende de**:
- Python 3.6+
- Libraries: `Flask`, `Gunicorn`, `PyYAML`
- `web/config.py` (Flask config)
- `db/pipeline.db` (read-only queries)
- `logs/*.log` (read-only)
- `web/templates/*.html`
- `web/static/css/*.css`
- `web/static/js/*.js`

**No invoca nada del pipeline** (solo lectura)

**Ver**: [[Arquitectura Web UI]]

---

## Dependencias de Configuración

```mermaid
graph LR
    ENV[.env<br/>Credentials] -->|Variables| YAML[ci_cd_config.yaml]
    YAML -->|config_get| SCRIPTS[Bash Scripts]
    YAML -->|load_config| PYTHON[Python Scripts]
    YAML -->|Flask Config| WEB[Web UI]
    
    SONAR_PROP[sonar-project.properties] --> SQ[sonar_check.py]
    
    style ENV fill:#FFD93D,color:#000
    style YAML fill:#6BCB77,color:#fff
```

**Flujo**:
1. `.env` define credenciales (GIT_PASSWORD, SONAR_TOKEN, etc.)
2. `ci_cd_config.yaml` referencia con `${VAR_NAME}`
3. Scripts usan `config_get` (bash) o `load_config()` (Python) para expandir

**Ver**: [[Referencia - Configuración]]

---

## Dependencias de Base de Datos

```mermaid
erDiagram
    CI_CD_SH ||--o{ DEPLOYMENTS : "writes"
    GIT_MONITOR ||--o{ PROCESSED_TAGS : "writes"
    COMPILE_SH ||--o{ BUILD_LOGS : "writes"
    SONAR_CHECK ||--o{ SONAR_RESULTS : "writes"
    CI_CD_SH ||--o{ EXECUTION_LOG : "writes"
    
    WEB_APP ||--o{ DEPLOYMENTS : "reads"
    WEB_APP ||--o{ BUILD_LOGS : "reads"
    WEB_APP ||--o{ SONAR_RESULTS : "reads"
    WEB_APP ||--o{ PROCESSED_TAGS : "reads"
```

**Escritura** (Pipeline):
- `ci_cd.sh` → `deployments`, `execution_log`
- `git_monitor.sh` → `processed_tags`
- `compile.sh` → `build_logs`
- `sonar_check.py` → `sonar_results`

**Lectura** (Web UI):
- `app.py` → Todas las tablas (read-only)

**Ver**: [[Modelo de Datos]]

---

## Dependencias de Logs

```mermaid
graph TD
    PIPELINE[Pipeline Scripts] -->|write| LOG_FILES[logs/*.log]
    WEB[Web UI] -->|write| WEB_LOGS[logs/web_*.log]
    
    LOG_FILES -->|read via API| WEB_API[Web UI API]
    LOG_FILES -->|tail -f| MONITORING[Monitoring Tools]
    
    WEB_LOGS -->|journalctl| SYSTEMD[systemd journal]
```

**Archivos de log**:
- `logs/pipeline_YYYYMMDD.log` - Pipeline general
- `logs/compile_YYYYMMDD_HHMMSS.log` - Compilación
- `logs/deploy_YYYYMMDD_HHMMSS.log` - Deployment
- `logs/web_access.log` - Web UI access
- `logs/web_error.log` - Web UI errors

**Ver**: [[Referencia - Logs]]

---

## Dependencias de Herramientas

### Pipeline

```mermaid
graph LR
    PIPELINE[Pipeline Scripts] --> GIT[git CLI]
    PIPELINE --> MAKE[make / gcc]
    PIPELINE --> SQLITE[sqlite3]
    PIPELINE --> SSH[ssh client]
    PIPELINE --> YQ[yq - YAML parser]
    
    SQ_SCRIPT[sonar_check.py] --> SCANNER[sonar-scanner]
    SQ_SCRIPT --> WRAPPER[build-wrapper]
```

**Herramientas bundled**:
- `utils/sonar-scanner/` - SonarQube scanner CLI
- `utils/build-wrapper/` - Build wrapper para C/C++

**Herramientas del sistema** (deben instalarse):
- `git`
- `make`, `gcc`, `g++`
- `sqlite3`
- `ssh`
- `yq` (YAML query tool)
- `python3.6+`

---

### Web UI

```mermaid
graph LR
    WEB[Web UI] --> FLASK[Flask 2.0.3]
    WEB --> GUNICORN[Gunicorn 20.1.0]
    WEB --> PYYAML[PyYAML]
    
    FRONTEND[Frontend] --> ALPINE[Alpine.js 3.x<br/>CDN]
    FRONTEND --> TAILWIND[Tailwind CSS<br/>CDN]
    FRONTEND --> CHART[Chart.js<br/>CDN]
```

**Python packages** (en `web/requirements.txt`):
- Flask==2.0.3
- Gunicorn==20.1.0
- PyYAML==6.0.1

**Frontend libraries** (CDN):
- Alpine.js 3.x
- Tailwind CSS
- Chart.js 4.x

---

## Dependencias de Servicios Externos

```mermaid
graph TB
    subgraph "CI/CD System"
        CICD[CI/CD Pipeline]
    end
    
    subgraph "External Services"
        GIT[Git Repository<br/>YOUR_GIT_SERVER]
        SQ[SonarQube<br/>YOUR_SONARQUBE_SERVER]
        VC[vCenter<br/>vcenter.example.com]
        VM[Target VM<br/>YOUR_TARGET_VM_IP]
    end
    
    CICD -->|HTTPS + Basic Auth| GIT
    CICD -->|HTTPS + Token Auth| SQ
    CICD -->|HTTPS + Session Auth| VC
    CICD -->|SSH + Key Auth| VM
    
    style GIT fill:#F1502F
    style SQ fill:#549DD0
    style VC fill:#0091DA
    style VM fill:#00C176
```

**Autenticación**:
- **Git**: Basic auth con usuario/password (HTTPS)
- **SonarQube**: Token auth (token como username, password vacío)
- **vCenter**: Session-based (POST /session, luego header `vmware-api-session-id`)
- **Target VM**: SSH key-based (sin password)

**Ver**: [[Referencia - APIs Externas]]

---

## Orden de Inicialización

```mermaid
flowchart TD
    START([System Boot]) --> INSTALL[1. Install Dependencies<br/>git, make, python3.6, etc.]
    INSTALL --> SETUP[2. Run setup_phase0.sh<br/>Create dirs, init DB]
    SETUP --> ENV[3. Configure .env<br/>Credentials]
    ENV --> SSHKEY[4. Generate SSH key<br/>ssh-keygen + ssh-copy-id]
    SSHKEY --> INIT[5. Initialize DB<br/>./ci_cd.sh init]
    INIT --> SERVICE1[6. Install cicd.service<br/>./install_service.sh]
    SERVICE1 --> SERVICE2[7. Install cicd-web.service<br/>./install_web.sh]
    SERVICE2 --> START_SVC[8. Start Services<br/>systemctl start cicd cicd-web]
    START_SVC --> READY([✅ System Ready])
```

**Ver guía completa**: [[Operación - Instalación]]

---

## Comunicación entre Módulos

### Pipeline → Base de Datos

```bash
# common.sh provee:
db_query "INSERT INTO deployments ..."

# Usado por:
- ci_cd.sh (deployments, execution_log)
- git_monitor.sh (processed_tags)
- compile.sh (build_logs)
- sonar_check.py (sonar_results)
```

### Pipeline → Configuración

```bash
# common.sh provee:
value=$(config_get "path.to.key")

# Usado por todos los scripts para leer config
```

### Web UI → Base de Datos

```python
# app.py usa sqlite3 directamente:
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
result = conn.execute("SELECT ...").fetchall()
```

### Pipeline → Logs

```bash
# common.sh provee:
log_info "message" >&2

# Todos los logs van a stderr, redirigidos a archivos por systemd
```

---

## Puntos de Extensión

### Añadir Nueva Fase al Pipeline

```mermaid
graph LR
    MAIN[ci_cd.sh] -->|1. Añadir invocación| NEWSCRIPT[scripts/new_phase.sh]
    NEWSCRIPT -->|2. Source| COMMON[common.sh]
    NEWSCRIPT -->|3. Read config| YAML[ci_cd_config.yaml]
    NEWSCRIPT -->|4. Write data| DB[(pipeline.db)]
    NEWSCRIPT -->|5. Opcional: API| EXTERNAL[External Service]
```

**Pasos**:
1. Crear `scripts/new_phase.sh`
2. Source `common.sh`
3. Integrar en `ci_cd.sh` (función `run_pipeline()`)
4. Añadir config en YAML si necesario
5. Crear tabla en DB si necesario

### Añadir Nuevo Endpoint Web UI

```mermaid
graph LR
    ROUTE[app.py<br/>@app.route] -->|Query| DB[(pipeline.db)]
    ROUTE -->|Render| TEMPLATE[templates/new_page.html]
    TEMPLATE -->|Use| ALPINE[Alpine.js components]
    TEMPLATE -->|Fetch| API[/api/new-endpoint]
```

**Ver**: [[Arquitectura Web UI#Extensión de la Web UI]]

---

## Enlaces Relacionados

- [[Arquitectura del Pipeline]] - Sistema de ejecución
- [[Arquitectura Web UI]] - Sistema web
- [[Diagrama - Flujo Completo]] - Flujo del pipeline
- [[Diagrama - Estados]] - Estados de deployments
- [[Modelo de Datos]] - Esquema de base de datos
- [[Referencia - Configuración]] - Configuración YAML y .env
