# 📚 Pipeline - Common Functions

## Visión General

**common.sh** es la librería compartida que proporciona funciones de logging, configuración, base de datos y utilidades usadas por todos los scripts del pipeline.

**Relacionado con**:
- [[Arquitectura del Pipeline#Librería Compartida]]
- Usada por: [[Pipeline - Git Monitor]], [[Pipeline - Compilación]], [[Pipeline - SSH Deploy]]

---

## Ubicación

**Script**: `scripts/common.sh`

**Uso**:
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Ahora puedes usar: log_info, config_get, db_query, etc.
```

---

## Funciones de Logging

### Niveles

```bash
log_info "mensaje"     # [INFO] mensaje
log_warn "mensaje"     # [WARN] mensaje (amarillo)
log_error "mensaje"    # [ERROR] mensaje (rojo)
log_ok "mensaje"       # [OK] mensaje (verde)
log_debug "mensaje"    # [DEBUG] mensaje (solo si DEBUG=1)
```

### Implementación

```bash
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" >&2
}

log_warn() {
    echo -e "\033[33m[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $*\033[0m" >&2
}

log_error() {
    echo -e "\033[31m[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*\033[0m" >&2
}

log_ok() {
    echo -e "\033[32m[$(date +'%Y-%m-%d %H:%M:%S')] [OK] $*\033[0m" >&2
}

log_debug() {
    if [ "${DEBUG:-0}" = "1" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [DEBUG] $*" >&2
    fi
}
```

**⚠️ Importante**: Todos los logs van a **stderr** (`>&2`), no stdout, para no interferir con captura de valores.

---

## Funciones de Configuración

### `config_get()`

**Propósito**: Leer valores de `ci_cd_config.yaml` con expansión de variables de entorno.

**Sintaxis**:
```bash
value=$(config_get "yaml.path.to.key")
```

**Implementación**:
```bash
config_get() {
    local KEY="$1"
    local CONFIG_FILE="${CONFIG_FILE:-config/ci_cd_config.yaml}"
    
    # Usar yq para parsear YAML
    local value=$(yq eval ".${KEY}" "$CONFIG_FILE")
    
    # Expandir variables de entorno ${VAR}
    value=$(echo "$value" | envsubst)
    
    echo "$value"
}
```

**Ejemplos**:
```bash
GIT_URL=$(config_get "git.url")
# Returns: https://user:pass@YOUR_GIT_SERVER/... (con variables expandidas)

TIMEOUT=$(config_get "compilation.timeout")
# Returns: 3600

COVERAGE=$(config_get "sonarqube.thresholds.coverage")
# Returns: 80
```

**Ver config**: [[Referencia - Configuración]]

---

## Funciones de Base de Datos

### `db_query()`

**Propósito**: Ejecutar queries SQL en SQLite con manejo de errores.

**Sintaxis**:
```bash
db_query "SQL query here"

# Capturar resultado
result=$(db_query "SELECT COUNT(*) FROM deployments")
```

**Implementación**:
```bash
db_query() {
    local QUERY="$1"
    local DB_PATH="${DB_PATH:-db/pipeline.db}"
    
    if ! sqlite3 "$DB_PATH" "$QUERY" 2>&1; then
        log_error "Database query failed: $QUERY"
        return 1
    fi
}
```

**Ejemplos**:
```bash
# Insert
db_query "INSERT INTO processed_tags (tag_name) VALUES ('V24_02_15_01')"

# Select con resultado
count=$(db_query "SELECT COUNT(*) FROM deployments WHERE status='success'")
echo "Success count: $count"

# Update
db_query "UPDATE deployments SET status='compiling' WHERE tag_name='V24_02_15_01'"
```

**Ver schema**: [[Modelo de Datos]]

---

## Funciones de Utilidad

### `wait_for()`

**Propósito**: Esperar condicionalmente con timeout.

**Sintaxis**:
```bash
wait_for "command_to_check" timeout_seconds "description"
```

**Implementación**:
```bash
wait_for() {
    local CHECK_CMD="$1"
    local TIMEOUT="$2"
    local DESC="${3:-condition}"
    
    log_info "Waiting for $DESC (timeout: ${TIMEOUT}s)..."
    
    local elapsed=0
    while [ $elapsed -lt $TIMEOUT ]; do
        if eval "$CHECK_CMD" >/dev/null 2>&1; then
            log_ok "$DESC ready after ${elapsed}s"
            return 0
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    log_error "Timeout waiting for $DESC after ${TIMEOUT}s"
    return 1
}
```

**Ejemplo**:
```bash
# Esperar a que VM esté encendida
wait_for "ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP 'echo OK'" 300 "VM SSH access"

# Esperar a que archivo exista
wait_for "[ -f /home/YOUR_USER/compile/InstallationDVD.iso ]" 3600 "ISO compilation"
```

### `require_command()`

**Propósito**: Verificar que comando existe en PATH.

**Implementación**:
```bash
require_command() {
    local CMD="$1"
    local MSG="${2:-$CMD is required}"
    
    if ! command -v "$CMD" >/dev/null 2>&1; then
        log_error "$MSG"
        exit 1
    fi
}
```

**Ejemplo**:
```bash
require_command git "Git is required for repository operations"
require_command python3.6 "Python 3.6+ is required"
require_command sqlite3 "SQLite3 is required for database"
```

### `format_duration()`

**Propósito**: Formatear segundos a formato legible.

**Implementación**:
```bash
format_duration() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${secs}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${secs}s"
    else
        echo "${secs}s"
    fi
}
```

**Ejemplo**:
```bash
format_duration 3665  # Returns: "1h 1m 5s"
format_duration 125   # Returns: "2m 5s"
format_duration 45    # Returns: "45s"
```

---

## Variables de Entorno Globales

```bash
# Configuración
CONFIG_FILE="${CONFIG_FILE:-config/ci_cd_config.yaml}"
DB_PATH="${DB_PATH:-db/pipeline.db}"
LOG_DIR="${LOG_DIR:-logs}"

# Directorios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Debug mode
DEBUG="${DEBUG:-0}"  # Set DEBUG=1 para verbose logging
```

---

## Convenciones

### 1. Set Strict Mode

```bash
set -euo pipefail
```

- `set -e` - Exit on error
- `set -u` - Exit on undefined variable
- `set -o pipefail` - Pipe failure detection

### 2. Logging a stderr

```bash
# ✅ CORRECTO
log_info "Processing tag: $TAG_NAME" >&2

# ❌ INCORRECTO (interfiere con captura)
echo "Processing tag: $TAG_NAME"
```

### 3. Funciones que retornan valores

```bash
# Usar echo a stdout (sin >&2)
get_latest_tag() {
    local latest=$(db_query "SELECT tag_name FROM deployments ORDER BY started_at DESC LIMIT 1")
    echo "$latest"  # stdout, sin >&2
}

# Captura
TAG=$(get_latest_tag)
```

### 4. Manejo de errores

```bash
# Siempre verificar exit codes
if ! some_command; then
    log_error "Command failed"
    return 1
fi

# O usar && / ||
some_command && log_ok "Success" || log_error "Failed"
```

---

## Ejemplo de Uso Completo

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

main() {
    local TAG_NAME="$1"
    
    # Logging
    log_info "Starting deployment for tag: $TAG_NAME"
    
    # Configuración
    local GIT_URL=$(config_get "git.url")
    local COMPILE_DIR=$(config_get "compilation.compile_dir")
    
    # Verificar prerequisites
    require_command git
    require_command make
    
    # Base de datos
    db_query "INSERT INTO deployments (tag_name, status) VALUES ('$TAG_NAME', 'pending')"
    
    # Operación con timer
    local start_time=$(date +%s)
    
    if ! compile_project "$COMPILE_DIR"; then
        log_error "Compilation failed"
        db_query "UPDATE deployments SET status='failed' WHERE tag_name='$TAG_NAME'"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_ok "Compilation completed in $(format_duration $duration)"
    
    db_query "UPDATE deployments SET status='success', duration_seconds=$duration WHERE tag_name='$TAG_NAME'"
}

main "$@"
```

---

## Testing

### Test de Logging

```bash
# Test todos los niveles
source scripts/common.sh

log_info "This is info"
log_warn "This is warning"
log_error "This is error"
log_ok "This is success"

DEBUG=1 log_debug "This is debug (only shown if DEBUG=1)"
```

### Test de config_get

```bash
source scripts/common.sh

# Test lectura de config
echo "Git URL: $(config_get 'git.url')"
echo "Timeout: $(config_get 'compilation.timeout')"

# Test expansión de variables
export GIT_PASSWORD="test123"
echo "Expanded: $(config_get 'git.url')"  # Debe mostrar password expandido
```

### Test de db_query

```bash
source scripts/common.sh

# Test query simple
count=$(db_query "SELECT COUNT(*) FROM deployments")
echo "Total deployments: $count"

# Test insert
db_query "INSERT INTO processed_tags (tag_name) VALUES ('TEST_TAG')"

# Cleanup
db_query "DELETE FROM processed_tags WHERE tag_name='TEST_TAG'"
```

---

## Performance Tips

### Evitar Subshells Innecesarios

```bash
# ❌ LENTO (crea subshell)
value=$(config_get "key")
echo "Value: $value"

# ✅ RÁPIDO (lectura directa, si es posible)
yq eval ".key" config.yaml
```

### Cache de Config Values

```bash
# Si usas mismo valor múltiples veces
GIT_URL=$(config_get "git.url")
COMPILE_DIR=$(config_get "compilation.compile_dir")

# Reusar variables en lugar de llamar config_get repetidamente
```

### Logging Condicional

```bash
# Solo log debug si DEBUG=1
[ "$DEBUG" = "1" ] && log_debug "Detailed info here"

# En lugar de siempre llamar log_debug (que internamente hace el check)
```

---

## Enlaces Relacionados

- [[Arquitectura del Pipeline#Librería Compartida]]
- [[Referencia - Configuración]] - Estructura del YAML
- [[Modelo de Datos]] - Schema SQLite
- [[Referencia - Logs]] - Sistema de logging

**Usado por**:
- [[Pipeline - Git Monitor]]
- [[Pipeline - Compilación]]
- [[Pipeline - SSH Deploy]]
