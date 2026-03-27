# 🏗️ Arquitectura del Pipeline - Sistema de Ejecución

## Visión General

El **Pipeline de Ejecución** es el núcleo del sistema CI/CD. Se ejecuta como daemon systemd que monitoriza el repositorio Git cada 5 minutos y procesa nuevos tags automáticamente.

**Relacionado con**:
- [[Arquitectura Web UI]] - Sistema de monitorización
- [[Modelo de Datos]] - Persistencia de datos
- [[Diagrama - Flujo Completo]] - Visualización del flujo

---

## Arquitectura General

```mermaid
graph TB
    subgraph "Daemon systemd"
        A[ci_cd.sh] --> B[git_monitor.sh]
        A --> C[compile.sh]
        A --> D[sonar_check.py]
        A --> E[vcenter_api.py]
        A --> F[deploy.sh]
        A --> G[notify.sh]
    end
    
    subgraph "Librería Compartida"
        H[common.sh]
    end
    
    subgraph "Configuración"
        I[ci_cd_config.yaml]
        J[.env]
    end
    
    subgraph "Persistencia"
        K[(pipeline.db)]
        L[Logs]
    end
    
    B --> H
    C --> H
    F --> H
    
    A --> I
    A --> J
    
    A --> K
    A --> L
    
    style A fill:#FF6B6B,color:#fff
    style H fill:#4ECDC4,color:#fff
    style K fill:#95E1D3,color:#000
```

---

## Componentes del Pipeline

### 🎯 Orquestador Principal - `ci_cd.sh`

**Responsabilidad**: Coordinar todas las fases del pipeline secuencialmente.

**Funciones clave**:
- `run_pipeline()` - Ejecuta todas las fases en orden
- `daemon_mode()` - Loop infinito con polling cada 5 minutos
- `verify_environment()` - Verifica prerequisites del sistema
- `show_status()` - Muestra estadísticas de deployments

**Estados gestionados**:
```
pending → compiling → analyzing → deploying → success/failed
```

**Ver detalles**: [[Diagrama - Estados]]

**Código**: `/home/YOUR_USER/cicd/ci_cd.sh`

---

## Fases del Pipeline

### 📡 Fase 1: Git Monitor

**Script**: [[Pipeline - Git Monitor]] (`scripts/git_monitor.sh`)

**Funciones**:
- `detect_new_tags()` - Consulta `git ls-remote` y filtra por regex
- `checkout_tag()` - Clona/actualiza repo y hace checkout del tag
- `mark_tag_processed()` - Marca en tabla `processed_tags`

**Flujo**:
```mermaid
sequenceDiagram
    participant GM as git_monitor.sh
    participant Git as Git Repo
    participant DB as SQLite
    
    GM->>Git: git ls-remote --tags
    Git-->>GM: Lista de tags
    GM->>GM: Filtrar por regex
    GM->>DB: SELECT FROM processed_tags
    DB-->>GM: Tags ya procesados
    GM->>GM: Comparar y detectar nuevos
    GM->>Git: git clone/checkout TAG
    GM->>DB: INSERT INTO processed_tags
```

**Configuración relevante**:
```yaml
git:
  url: https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
  branch: YOUR_GIT_BRANCH
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"
```

---

### ⚙️ Fase 2: Compilación

**Script**: [[Pipeline - Compilación]] (`scripts/compile.sh`)

**Funciones**:
- `prepare_compile_directory()` - Copia fuentes a `/home/YOUR_USER/compile`
- `execute_build_script()` - Ejecuta `build_DVDs.sh` con timeout de 3600s
- `validate_output()` - Verifica existencia de `InstallationDVD.iso`
- `record_build_metrics()` - Guarda logs en tabla `build_logs`

**Flujo**:
```mermaid
sequenceDiagram
    participant CS as compile.sh
    participant FS as File System
    participant BUILD as build_DVDs.sh
    participant DB as SQLite
    
    CS->>FS: mkdir -p /home/YOUR_USER/compile
    CS->>FS: cp -r sources/* compile/
    CS->>FS: find -exec chmod +x
    CS->>BUILD: ./build_DVDs.sh
    BUILD-->>CS: Exit code + logs
    CS->>FS: Verificar InstallationDVD.iso
    CS->>DB: INSERT INTO build_logs
```

**Output esperado**:
- `/home/YOUR_USER/compile/InstallationDVD.iso` (≈ 3-4 GB)

**Timeout**: 3600 segundos (1 hora)

**Métricas registradas**:
- Tiempo de compilación
- Tamaño del ISO resultante
- Logs de stderr/stdout

---

### 🔍 Fase 3: Análisis SonarQube

**Script**: [[Pipeline - SonarQube]] (`python/sonar_check.py`)

**Funciones**:
- `run_analysis()` - Ejecuta build-wrapper + sonar-scanner
- `check_quality_gate()` - Consulta API de SonarQube
- `validate_thresholds()` - Compara métricas con umbrales configurados
- `store_results()` - Guarda en tabla `sonar_results`

**Flujo**:
```mermaid
sequenceDiagram
    participant SC as sonar_check.py
    participant BW as build-wrapper
    participant SS as sonar-scanner
    participant SQ as SonarQube API
    participant DB as SQLite
    
    SC->>BW: build-wrapper --out-dir build-wrapper-output make
    SC->>SS: sonar-scanner (usa sonar-project.properties)
    SS->>SQ: Sube análisis
    SQ-->>SS: Task ID
    SC->>SQ: GET /api/qualitygates/project_status
    SQ-->>SC: Quality gate status + métricas
    SC->>SC: Validar umbrales
    SC->>DB: INSERT INTO sonar_results
```

**Quality Gates (umbrales)**:
```yaml
sonarqube:
  thresholds:
    coverage: 80         # ≥ 80%
    bugs: 0              # = 0
    vulnerabilities: 0   # = 0
    security_hotspots: 0 # = 0
    code_smells: 10      # ≤ 10
  allow_override: false  # Bloquea deployment si falla
```

**Métricas registradas**:
- Coverage (cobertura de tests)
- Bugs, vulnerabilities, security_hotspots
- Code smells, duplications
- Lines of code

**API calls**:
- `POST /api/ce/task` - Obtener tarea de análisis
- `GET /api/qualitygates/project_status` - Estado quality gate
- `GET /api/measures/component` - Métricas detalladas

---

### ☁️ Fase 4: vCenter Integration

**Script**: [[Pipeline - vCenter]] (`python/vcenter_api.py`)

**Funciones**:
- `authenticate()` - Login a vCenter REST API (session-based)
- `upload_iso()` - Sube ISO al datastore YOUR_DATASTORE
- `configure_cdrom()` - Configura CD-ROM en VM `Releases`
- `power_on_vm()` - Enciende la VM

**Flujo**:
```mermaid
sequenceDiagram
    participant VA as vcenter_api.py
    participant VC as vCenter API
    participant DS as Datastore
    participant VM as VM Releases
    
    VA->>VC: POST /rest/com/vmware/cis/session
    VC-->>VA: Session token
    VA->>DS: PUT [YOUR_DATASTORE] P27/Versiones/TAG.iso
    VA->>VC: GET /rest/vcenter/vm?names=Releases
    VC-->>VA: VM ID
    VA->>VC: PATCH /rest/vcenter/vm/{id}/hardware/cdrom
    VA->>VC: POST /rest/vcenter/vm/{id}/power/start
    VC-->>VA: VM powered on
```

**Configuración relevante**:
```yaml
vcenter:
  api_url: https://vcenter.example.com/rest
  datacenter: YOUR_DATACENTER
  vm_name: Releases
  datastore: YOUR_DATASTORE
  datastore_path: P27/Versiones/
```

**Notas importantes**:
- **No usa pyvmomi** (solo REST API nativo)
- Session expira a los 30 minutos (reconexión automática en 401)
- ISO path format: `[YOUR_DATASTORE] P27/Versiones/{TAG}.iso`

---

### 🚀 Fase 5: SSH Deploy

**Script**: [[Pipeline - SSH Deploy]] (`scripts/deploy.sh`)

**Funciones**:
- `ssh_execute()` - Ejecuta comandos remotos en VM destino
- `mount_iso()` - Monta ISO en `/mnt/cdrom`
- `copy_and_install()` - Copia contenido y ejecuta `install.sh`
- `cleanup()` - Desmonta y limpia archivos temporales

**Flujo**:
```mermaid
sequenceDiagram
    participant DS as deploy.sh
    participant SSH as SSH Client
    participant VM as Target VM (YOUR_TARGET_VM_IP)
    participant ISO as Mounted ISO
    
    DS->>SSH: ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP
    SSH->>VM: mount /dev/cdrom /mnt/cdrom
    SSH->>VM: cp -r /mnt/cdrom/* /root/install/
    SSH->>VM: cd /root/install && ./install.sh "ope 1 - YES yes"
    VM-->>SSH: Exit code
    SSH->>VM: umount /mnt/cdrom
    SSH->>VM: rm -rf /root/install
    SSH-->>DS: Deployment result
```

**Comandos ejecutados en VM**:
```bash
# 1. Montar ISO
mount /dev/cdrom /mnt/cdrom

# 2. Copiar contenido
mkdir -p /root/install
cp -r /mnt/cdrom/* /root/install/

# 3. Ejecutar instalador
cd /root/install
./install.sh "ope 1 - YES yes"

# 4. Limpieza
umount /mnt/cdrom
rm -rf /root/install
```

**Configuración relevante**:
```yaml
target_vm:
  host: YOUR_TARGET_VM_IP
  user: root
  ssh_key: /home/YOUR_USER/.ssh/id_rsa
  install_args: "ope 1 - YES yes"
```

**Prerequisito**: SSH key pública en `~/.ssh/authorized_keys` del VM destino

---

## Librería Compartida

### 📚 common.sh

**Ubicación**: `scripts/common.sh`

**Funciones disponibles**:

#### Logging
```bash
log_info "Mensaje informativo"      # [INFO]
log_warn "Advertencia"               # [WARN]
log_error "Error crítico"            # [ERROR]
log_ok "Operación exitosa"           # [OK]
log_debug "Debug info"               # [DEBUG]
```

**Importante**: Todos los logs van a **stderr** para no interferir con captura de valores.

#### Configuración
```bash
# Lee valores de ci_cd_config.yaml
value=$(config_get "git.url")
pattern=$(config_get "git.tag_pattern")
```

**Implementación**: Usa `yq` para parsear YAML con expansión de variables de entorno desde `.env`.

#### Base de Datos
```bash
# Ejecuta queries SQL
db_query "INSERT INTO deployments (tag_name, status) VALUES ('V24_02_15_01', 'pending')"

# Con captura de resultado
count=$(db_query "SELECT COUNT(*) FROM deployments WHERE status='success'")
```

**Implementación**: Wrapper de `sqlite3` con manejo de errores y lock timeout.

#### Utilidades
```bash
# Espera condicional con timeout
wait_for "command_to_check" 300 "Description"

# Validación de comandos
require_command git "Git is required"
require_command python3.6 "Python 3.6+ required"
```

**Ver detalles**: [[Pipeline - Common Functions]]

---

## Sistema de Configuración

### Jerarquía de Configuración

```mermaid
graph LR
    A[.env] -->|Variables entorno| B[ci_cd_config.yaml]
    B -->|config_get| C[Scripts]
    
    style A fill:#FFD93D,color:#000
    style B fill:#6BCB77,color:#fff
    style C fill:#4D96FF,color:#fff
```

### 1. Variables de Entorno (`.env`)

**Ubicación**: `config/.env` (NO commitear)

**Contenido típico**:
```bash
GIT_USER=automation
GIT_PASSWORD=secret_token_here
SONAR_TOKEN=sonarqube_token_here
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=vcenter_password_here
```

**Ver template**: `config/.env.example`

### 2. Configuración YAML (`ci_cd_config.yaml`)

**Ubicación**: `config/ci_cd_config.yaml`

**Expansión de variables**:
```yaml
git:
  url: "https://${GIT_USER}:${GIT_PASSWORD}@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO"

sonarqube:
  token: "${SONAR_TOKEN}"
```

Las variables `${VAR_NAME}` se reemplazan desde `.env` al cargar la configuración.

**Ver referencia completa**: [[Referencia - Configuración]]

---

## Persistencia de Datos

### Base de Datos SQLite

**Ubicación**: `db/pipeline.db`

**Tablas principales**:
- `deployments` - Tracking de pipeline runs
- `build_logs` - Logs de compilación
- `sonar_results` - Resultados SonarQube
- `processed_tags` - Tags ya procesados
- `execution_log` - Log de eventos del sistema

**Ver schema completo**: [[Modelo de Datos]]

**Ejemplo de registro de deployment**:
```sql
-- 1. Pipeline detecta nuevo tag
INSERT INTO processed_tags (tag_name, detected_at) 
VALUES ('MAC_1_V24_02_15_01', CURRENT_TIMESTAMP);

-- 2. Inicia deployment
INSERT INTO deployments (tag_name, status, started_at) 
VALUES ('MAC_1_V24_02_15_01', 'pending', CURRENT_TIMESTAMP);

-- 3. Actualiza status por fase
UPDATE deployments SET status='compiling' WHERE tag_name='MAC_1_V24_02_15_01';
UPDATE deployments SET status='analyzing' WHERE tag_name='MAC_1_V24_02_15_01';
UPDATE deployments SET status='deploying' WHERE tag_name='MAC_1_V24_02_15_01';
UPDATE deployments SET status='success', finished_at=CURRENT_TIMESTAMP WHERE tag_name='MAC_1_V24_02_15_01';
```

### Sistema de Logs

**Ubicación**: `logs/`

**Tipos de logs**:
```
logs/
├── pipeline_YYYYMMDD.log          # Log general del orquestador
├── compile_YYYYMMDD_HHMMSS.log    # Logs de compilación
├── deploy_YYYYMMDD_HHMMSS.log     # Logs de deployment
└── web_access.log                 # Logs de Web UI
```

**Formato**:
```
[2026-03-20 10:05:33] [INFO] Git monitor: Checking for new tags...
[2026-03-20 10:05:34] [OK] Found 1 new tag: MAC_1_V24_02_15_01
[2026-03-20 10:05:35] [INFO] Starting pipeline for tag MAC_1_V24_02_15_01
```

**Ver referencia**: [[Referencia - Logs]]

---

## Daemon systemd

### Servicio: `cicd.service`

**Ubicación**: `/etc/systemd/system/cicd.service`

**Configuración**:
```ini
[Unit]
Description=GALTTCMC CI/CD Pipeline
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/home/YOUR_USER/cicd
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/YOUR_USER/cicd/config/.env
ExecStart=/home/YOUR_USER/cicd/ci_cd.sh daemon
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
```

**Características**:
- **Auto-restart**: Si el proceso muere, systemd lo reinicia automáticamente
- **EnvironmentFile**: Carga `.env` automáticamente
- **Logs**: Via journald (`journalctl -u cicd`)

**Ver instalación**: [[Operación - Instalación#Servicio systemd]]

---

## Flujo Completo de un Deployment

### Diagrama de Secuencia

```mermaid
sequenceDiagram
    participant D as Daemon
    participant GM as Git Monitor
    participant C as Compile
    participant SQ as SonarQube
    participant VC as vCenter
    participant SSH as SSH Deploy
    participant DB as SQLite
    
    loop Every 5 minutes
        D->>GM: detect_new_tags()
        GM->>DB: Check processed_tags
        GM-->>D: New tag: MAC_1_V24_02_15_01
        
        D->>DB: INSERT deployment (status=pending)
        D->>GM: checkout_tag()
        GM->>DB: INSERT processed_tags
        
        D->>DB: UPDATE status=compiling
        D->>C: execute_build()
        C-->>D: InstallationDVD.iso
        
        D->>DB: UPDATE status=analyzing
        D->>SQ: run_analysis()
        SQ-->>D: Quality gate PASSED
        
        D->>DB: UPDATE status=deploying
        D->>VC: upload_iso() + configure_cdrom() + power_on()
        D->>SSH: mount_iso() + install()
        SSH-->>D: Installation OK
        
        D->>DB: UPDATE status=success
    end
```

**Ver diagrama ampliado**: [[Diagrama - Flujo Completo]]

---

## Gestión de Errores

### Estrategia de Rollback

**Si falla Fase 2 (Compilación)**:
- Status → `failed`
- Deployment **no continúa**
- Logs guardados en `build_logs`

**Si falla Fase 3 (SonarQube)**:
- Quality gate KO → status `failed`
- **Bloquea deployment** (si `allow_override: false`)
- Resultados guardados en `sonar_results`

**Si falla Fase 4 (vCenter)**:
- Status → `failed`
- ISO no se sube
- VM no se enciende

**Si falla Fase 5 (SSH Deploy)**:
- Status → `failed`
- **Rollback manual** requerido en VM destino
- Logs en `deploy_*.log`

### Manejo de Excepciones

**En Bash scripts**:
```bash
set -euo pipefail  # Abort on error, undefined var, pipe failure

# Captura de errores específicos
if ! git clone "$GIT_URL" "$COMPILE_DIR"; then
    log_error "Failed to clone repository"
    db_query "UPDATE deployments SET status='failed', error_message='Git clone failed' WHERE tag_name='$TAG'"
    exit 1
fi
```

**En Python scripts**:
```python
try:
    response = requests.get(api_url, verify=False)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    logging.error("API call failed: {}".format(e))
    update_deployment_status(tag, 'failed', str(e))
    sys.exit(1)
```

**Ver troubleshooting**: [[Operación - Troubleshooting]]

---

## Extensión del Pipeline

### Añadir Nueva Fase

**1. Crear script en `scripts/`**:
```bash
cd /home/YOUR_USER/cicd/scripts
nano new_phase.sh
```

**2. Usar template**:
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TAG_NAME="${1:-}"

log_info "Starting new phase for tag: $TAG_NAME"

# Tu lógica aquí

log_ok "New phase completed successfully"
```

**3. Integrar en `ci_cd.sh`**:
```bash
# En función run_pipeline()
if ! "$SCRIPT_DIR/scripts/new_phase.sh" "$TAG_NAME"; then
    log_error "New phase failed"
    mark_deployment_failed "$TAG_NAME" "New phase execution failed"
    return 1
fi
```

### Modificar Quality Gates

**Editar `config/ci_cd_config.yaml`**:
```yaml
sonarqube:
  thresholds:
    coverage: 85          # Aumentar a 85%
    bugs: 0
    vulnerabilities: 0
    security_hotspots: 0
    code_smells: 5        # Reducir a 5
    duplications: 3.0     # Añadir nuevo umbral
```

**Actualizar `python/sonar_check.py`**:
```python
def validate_thresholds(metrics, thresholds):
    # ... código existente ...
    
    # Añadir validación de duplications
    if 'duplications' in thresholds:
        dup_threshold = float(thresholds['duplications'])
        dup_actual = float(metrics.get('duplicated_lines_density', 0))
        if dup_actual > dup_threshold:
            errors.append("Duplications: {:.1f}% > {:.1f}%".format(dup_actual, dup_threshold))
```

---

## Métricas y Monitorización

### Métricas Clave

**Performance**:
- Tiempo promedio de compilación: ~45 minutos
- Tiempo total pipeline: ~60 minutos
- Frecuencia polling: 5 minutos

**Calidad**:
- Success rate objetivo: >80%
- Coverage mínimo: 80%
- Bugs permitidos: 0

**Capacidad**:
- Tags procesados por día: ~10-20
- Tamaño promedio ISO: 3.5 GB
- Espacio disco requerido: ~50 GB

### Dashboard Web UI

**Ver en tiempo real**:
- http://YOUR_PIPELINE_HOST_IP:8080
- Métricas últimas 24h
- Gráficos de tendencias
- Logs en vivo

**Ver detalles**: [[Arquitectura Web UI]]

---

## Enlaces Relacionados

### Documentación de Fases
- [[Pipeline - Git Monitor]]
- [[Pipeline - Compilación]]
- [[Pipeline - SonarQube]]
- [[Pipeline - vCenter]]
- [[Pipeline - SSH Deploy]]
- [[Pipeline - Common Functions]]

### Diagramas
- [[Diagrama - Flujo Completo]]
- [[Diagrama - Estados]]
- [[Diagrama - Dependencias]]

### Operación
- [[Operación - Instalación]]
- [[Operación - Monitorización]]
- [[Operación - Troubleshooting]]
- [[Operación - Mantenimiento]]

### Referencia
- [[Referencia - Configuración]]
- [[Modelo de Datos]]
- [[Referencia - APIs Externas]]
- [[Referencia - Logs]]
