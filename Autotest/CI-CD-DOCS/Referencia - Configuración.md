# ⚙️ Referencia - Configuración

## Visión General

Documentación completa del archivo `ci_cd_config.yaml` y variables de entorno en `.env`.

**Relacionado con**:
- [[Arquitectura del Pipeline#Sistema de Configuración]]
- [[Pipeline - Common Functions#config_get()]]

---

## Archivos de Configuración

### 1. `config/ci_cd_config.yaml`

**Propósito**: Configuración principal del pipeline (commiteable)

**Ubicación**: `/home/YOUR_USER/cicd/config/ci_cd_config.yaml`

**Formato**: YAML con expansión de variables `${VAR_NAME}`

### 2. `config/.env`

**Propósito**: Credenciales y secrets (NO commitear)

**Ubicación**: `/home/YOUR_USER/cicd/config/.env`

**Template**: `config/.env.example`

---

## Estructura Completa del YAML

```yaml
# Git Configuration
git:
  url: "https://${GIT_USER}:${GIT_PASSWORD}@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO"
  branch: "YOUR_GIT_BRANCH"
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"

# Compilation Configuration
compilation:
  compile_dir: "/home/YOUR_USER/compile"
  source_dir: "/home/YOUR_USER/compile"
  timeout: 3600  # seconds (1 hour)
  min_iso_size: 3221225472  # 3 GB in bytes
  build_script_path: "Development_TTCF/ttcf/utils/dvds/build_DVDs.sh"

# SonarQube Configuration
sonarqube:
  url: "https://YOUR_SONARQUBE_SERVER"
  token: "${SONAR_TOKEN}"
  project_key: "GALTTCMC"
  thresholds:
    coverage: 80
    bugs: 0
    vulnerabilities: 0
    security_hotspots: 0
    code_smells: 10
  allow_override: false

# vCenter Configuration
vcenter:
  api_url: "https://vcenter.example.com/rest"
  user: "${VCENTER_USER}"
  password: "${VCENTER_PASSWORD}"
  datacenter: "YOUR_DATACENTER"
  vm_name: "Releases"
  datastore: "YOUR_DATASTORE"
  datastore_path: "P27/Versiones/"
  verify_ssl: false

# Target VM Configuration
target_vm:
  host: "YOUR_TARGET_VM_IP"
  user: "root"
  ssh_key: "/home/YOUR_USER/.ssh/id_rsa"
  install_args: "ope 1 - YES yes"
  connection_timeout: 30

# Database Configuration
database:
  path: "db/pipeline.db"
  journal_mode: "WAL"
  busy_timeout: 5000

# Logging Configuration
logging:
  log_dir: "logs"
  retention_days: 30
  level: "INFO"  # DEBUG, INFO, WARN, ERROR

# Daemon Configuration
daemon:
  polling_interval: 300  # 5 minutes
  max_concurrent_deployments: 1
```

---

## Secciones Detalladas

### Git

```yaml
git:
  url: "https://${GIT_USER}:${GIT_PASSWORD}@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO"
  branch: "YOUR_GIT_BRANCH"
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"
```

**Campos**:
- `url`: URL del repositorio con credenciales expandidas desde `.env`
- `branch`: Branch principal a monitorizar
- `tag_pattern`: Regex para validar formato de tags

**Variables requeridas en `.env`**:
```bash
GIT_USER=automation
GIT_PASSWORD=ghp_xxxxxxxxxxxx
```

**Usado por**: [[Pipeline - Git Monitor]]

---

### Compilation

```yaml
compilation:
  compile_dir: "/home/YOUR_USER/compile"
  source_dir: "/home/YOUR_USER/compile"
  timeout: 3600
  min_iso_size: 3221225472
  build_script_path: "Development_TTCF/ttcf/utils/dvds/build_DVDs.sh"
```

**Campos**:
- `compile_dir`: Directorio temporal de compilación
- `source_dir`: Directorio de fuentes (generalmente igual a compile_dir)
- `timeout`: Timeout en segundos (3600 = 1 hora)
- `min_iso_size`: Tamaño mínimo del ISO en bytes (3 GB)
- `build_script_path`: Path relativo al script de build

**Usado por**: [[Pipeline - Compilación]]

---

### SonarQube

```yaml
sonarqube:
  url: "https://YOUR_SONARQUBE_SERVER"
  token: "${SONAR_TOKEN}"
  project_key: "GALTTCMC"
  thresholds:
    coverage: 80
    bugs: 0
    vulnerabilities: 0
    security_hotspots: 0
    code_smells: 10
  allow_override: false
```

**Campos**:
- `url`: URL base de SonarQube
- `token`: Token de autenticación (expandido desde `.env`)
- `project_key`: Key del proyecto en SonarQube
- `thresholds`: Umbrales de quality gates
- `allow_override`: Si `false`, bloquea deployment cuando quality gate falla

**Variables requeridas en `.env`**:
```bash
SONAR_TOKEN=squ_yyyyyyyyyyyy
```

**Umbrales**:
- `coverage`: Cobertura de tests (%)
- `bugs`: Número de bugs (0 = ninguno permitido)
- `vulnerabilities`: Vulnerabilidades de seguridad
- `security_hotspots`: Hotspots de seguridad a revisar
- `code_smells`: Problemas de mantenibilidad

**Usado por**: [[Pipeline - SonarQube]]

---

### vCenter

```yaml
vcenter:
  api_url: "https://vcenter.example.com/rest"
  user: "${VCENTER_USER}"
  password: "${VCENTER_PASSWORD}"
  datacenter: "YOUR_DATACENTER"
  vm_name: "Releases"
  datastore: "YOUR_DATASTORE"
  datastore_path: "P27/Versiones/"
  verify_ssl: false
```

**Campos**:
- `api_url`: Endpoint REST API de vCenter
- `user`, `password`: Credenciales (expandidas desde `.env`)
- `datacenter`: Nombre del datacenter
- `vm_name`: Nombre de la VM objetivo
- `datastore`: Nombre del datastore para ISOs
- `datastore_path`: Path dentro del datastore
- `verify_ssl`: Verificar certificados SSL (false en dev)

**Variables requeridas en `.env`**:
```bash
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=vcenter_password
```

**Usado por**: [[Pipeline - vCenter]]

---

### Target VM

```yaml
target_vm:
  host: "YOUR_TARGET_VM_IP"
  user: "root"
  ssh_key: "/home/YOUR_USER/.ssh/id_rsa"
  install_args: "ope 1 - YES yes"
  connection_timeout: 30
```

**Campos**:
- `host`: IP/hostname de la VM destino
- `user`: Usuario SSH
- `ssh_key`: Path a private key SSH
- `install_args`: Argumentos para `install.sh`
- `connection_timeout`: Timeout de conexión SSH (segundos)

**Usado por**: [[Pipeline - SSH Deploy]]

---

### Database

```yaml
database:
  path: "db/pipeline.db"
  journal_mode: "WAL"
  busy_timeout: 5000
```

**Campos**:
- `path`: Path relativo a la base de datos SQLite
- `journal_mode`: Modo de journaling (`WAL` recomendado)
- `busy_timeout`: Timeout para locks (milisegundos)

**Ver**: [[Modelo de Datos]]

---

### Logging

```yaml
logging:
  log_dir: "logs"
  retention_days: 30
  level: "INFO"
```

**Campos**:
- `log_dir`: Directorio de logs
- `retention_days`: Días de retención de logs
- `level`: Nivel de logging (DEBUG, INFO, WARN, ERROR)

**Ver**: [[Referencia - Logs]]

---

### Daemon

```yaml
daemon:
  polling_interval: 300
  max_concurrent_deployments: 1
```

**Campos**:
- `polling_interval`: Intervalo de polling Git en segundos (300 = 5 min)
- `max_concurrent_deployments`: Deployments simultáneos (1 = secuencial)

---

## Variables de Entorno (`.env`)

### Template (`.env.example`)

```bash
# Git Credentials
GIT_USER=your_git_username
GIT_PASSWORD=your_git_token_or_password

# SonarQube
SONAR_TOKEN=your_sonarqube_token

# vCenter
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=your_vcenter_password
```

### Crear `.env` Real

```bash
# Copiar template
cp config/.env.example config/.env

# Editar con valores reales
nano config/.env
```

**⚠️ NUNCA commitear `.env`**

Verificar que `.gitignore` incluye:
```
config/.env
*.env
```

---

## Acceso a Configuración

### Desde Bash Scripts

```bash
# Source common.sh primero
source "$SCRIPT_DIR/common.sh"

# Leer valores
GIT_URL=$(config_get "git.url")
TIMEOUT=$(config_get "compilation.timeout")
COVERAGE=$(config_get "sonarqube.thresholds.coverage")
```

### Desde Python Scripts

```python
import yaml
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Expandir variables de entorno
    for key, value in os.environ.items():
        content = content.replace('${{{}}}'.format(key), value)
    
    return yaml.safe_load(content)

# Uso
config = load_config('config/ci_cd_config.yaml')
git_url = config['git']['url']
timeout = config['compilation']['timeout']
```

---

## Validación de Configuración

### Verificar Sintaxis YAML

```bash
# Instalar yamllint
pip install --user yamllint

# Validar
yamllint config/ci_cd_config.yaml
```

### Verificar Variables de Entorno

```bash
# Cargar .env y verificar expansión
source config/.env
yq eval '.git.url' config/ci_cd_config.yaml | envsubst
# Debe mostrar URL con credenciales expandidas
```

### Comando de Verificación

```bash
./ci_cd.sh verify

# Verifica:
# - Archivos existen
# - Variables requeridas definidas
# - Permisos correctos
# - Conectividad externa
```

---

## Ejemplos de Personalización

### Aumentar Timeout de Compilación

```yaml
compilation:
  timeout: 7200  # 2 horas
```

### Relajar Quality Gates (Desarrollo)

```yaml
sonarqube:
  thresholds:
    coverage: 70  # Reducir de 80% a 70%
    code_smells: 20  # Permitir más code smells
  allow_override: true  # No bloquear deployment
```

### Múltiples VMs Destino (No Implementado)

```yaml
target_vms:
  - name: "VM_TEST"
    host: "10.0.0.100"
    user: "root"
  - name: "VM_PROD"
    host: "10.0.0.200"
    user: "deploy"
```

---

## Mejores Prácticas

### Seguridad

1. **Nunca commitear credenciales**
2. **Usar tokens en lugar de passwords**
3. **Rotar credenciales periódicamente**
4. **Permisos restrictivos en `.env`**: `chmod 600 config/.env`

### Mantenimiento

1. **Documentar cambios** en config con comments YAML
2. **Versionar .env.example** con placeholders
3. **Backup de config** antes de cambios grandes

### Testing

1. **Test en dev** antes de aplicar en producción
2. **Verificar con `./ci_cd.sh verify`** después de cambios
3. **Deployment manual** de test: `./ci_cd.sh --tag TEST_TAG`

---

## Enlaces Relacionados

- [[Arquitectura del Pipeline#Sistema de Configuración]]
- [[Pipeline - Common Functions#config_get()]]
- [[Operación - Instalación#Configurar Credenciales]]
- [[01 - Quick Start#Verificar Estado del Sistema]]
