#!/bin/bash
#===============================================================================
# common.sh - Funciones compartidas para el pipeline CI/CD
#===============================================================================
# Incluir en otros scripts con: source "$(dirname "$0")/common.sh"

# Directorio base del pipeline
CICD_HOME="${CICD_HOME:-/home/agent/cicd}"
CONFIG_FILE="${CONFIG_FILE:-$CICD_HOME/config/ci_cd_config.yaml}"
DB_PATH="${DB_PATH:-$CICD_HOME/db/pipeline.db}"
LOG_DIR="${LOG_DIR:-$CICD_HOME/logs}"

# Cargar variables de entorno si existe el fichero .env
if [[ -f "$CICD_HOME/config/.env" ]]; then
    set -a
    source "$CICD_HOME/config/.env"
    set +a
fi

#===============================================================================
# Logging
#===============================================================================

# Asegurar que existe el directorio de logs
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Fichero de log del día
LOG_FILE="${LOG_FILE:-$LOG_DIR/pipeline_$(date +%Y%m%d).log}"

# Colores (solo si stdout es terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Función base de logging
log() {
    local level=$1
    shift
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="[$timestamp] [$level] $*"
    
    # Escribir a fichero con flush inmediato
    {
        echo "$message"
        # Forzar flush del buffer del archivo
    } >> "$LOG_FILE"
    
    # Escribir a stderr con colores (NO a stdout para no interferir con captura de valores)
    case $level in
        DEBUG) echo -e "${CYAN}$message${NC}" >&2 ;;
        INFO)  echo -e "${BLUE}$message${NC}" >&2 ;;
        WARN)  echo -e "${YELLOW}$message${NC}" >&2 ;;
        ERROR) echo -e "${RED}$message${NC}" >&2 ;;
        OK)    echo -e "${GREEN}$message${NC}" >&2 ;;
        *)     echo "$message" >&2 ;;
    esac
}

log_debug() { log "DEBUG" "$@"; }
log_info()  { log "INFO" "$@"; }
log_warn()  { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_ok()    { log "OK" "$@"; }

#===============================================================================
# Gestión de configuración YAML
#===============================================================================

# Obtener valor de configuración YAML
# Uso: config_get "git.repo_url"
config_get() {
    local key=$1
    local default=${2:-}
    
    if command -v yq &>/dev/null; then
        local value
        value=$(yq ".$key" "$CONFIG_FILE" 2>/dev/null)
        if [[ "$value" != "null" && -n "$value" ]]; then
            # Expandir variables de entorno en el valor
            eval "echo \"$value\""
        else
            echo "$default"
        fi
    else
        log_warn "yq no disponible, usando valor por defecto para $key"
        echo "$default"
    fi
}

# Cargar variables de configuración principales
load_config() {
    # Git
    GIT_REPO_URL=$(config_get "git.repo_url")
    export GIT_REPO_URL
    GIT_BRANCH=$(config_get "git.branch")
    export GIT_BRANCH
    REPO_LOCAL_PATH=$(config_get "git.repo_local_path" "/home/agent/GALTTCMC")
    export REPO_LOCAL_PATH
    COMPILE_PATH=$(config_get "git.compile_path" "/home/agent/compile")
    export COMPILE_PATH
    TAG_PATTERN=$(config_get "git.tag_pattern")
    export TAG_PATTERN
    
    # Compilación
    BUILD_SCRIPT=$(config_get "compilation.build_script")
    export BUILD_SCRIPT
    OUTPUT_ISO=$(config_get "compilation.output_iso" "InstallationDVD.iso")
    export OUTPUT_ISO
    COMPILE_TIMEOUT=$(config_get "compilation.timeout_seconds" "3600")
    export COMPILE_TIMEOUT
    
    # SonarQube
    SONAR_URL=$(config_get "sonarqube.url")
    export SONAR_URL
    SONAR_PROJECT_KEY=$(config_get "sonarqube.project_key")
    export SONAR_PROJECT_KEY
    
    # VM destino
    TARGET_VM_IP=$(config_get "target_vm.ip")
    export TARGET_VM_IP
    TARGET_VM_USER=$(config_get "target_vm.ssh_user" "root")
    export TARGET_VM_USER
    TARGET_VM_KEY=$(config_get "target_vm.ssh_key_path")
    export TARGET_VM_KEY
    
    # General
    POLLING_INTERVAL=$(config_get "general.polling_interval_seconds" "300")
    export POLLING_INTERVAL
}

#===============================================================================
# Base de datos SQLite
#===============================================================================

# Ejecutar query SQL
# Uso: db_query "SELECT * FROM deployments"
db_query() {
    local query=$1
    sqlite3 "$DB_PATH" "$query"
}

# Ejecutar query y devolver resultado con headers
db_query_headers() {
    local query=$1
    sqlite3 -header -column "$DB_PATH" "$query"
}

# Insertar log de ejecución
# Uso: db_log_execution $deployment_id "compile" "Iniciando compilación" "INFO"
db_log_execution() {
    local deployment_id=$1
    local phase=$2
    local message=$3
    local level=${4:-INFO}
    
    db_query "INSERT INTO execution_log (deployment_id, phase, message, level) 
              VALUES ($deployment_id, '$phase', '$message', '$level')"
}

# Obtener ID del último deployment
db_last_deployment_id() {
    db_query "SELECT id FROM deployments ORDER BY id DESC LIMIT 1"
}

# Comprobar si un tag ya fue procesado
# Uso: if db_tag_processed "V01_02_03_04"; then ...
db_tag_processed() {
    local tag=$1
    local count
    count=$(db_query "SELECT COUNT(*) FROM deployments WHERE tag_name='$tag' AND status='success'")
    [[ "$count" -gt 0 ]]
}

#===============================================================================
# Utilidades SSH
#===============================================================================

# Ejecutar comando en VM destino
# Uso: ssh_exec "whoami"
ssh_exec() {
    local cmd=$1
    local ssh_key="${TARGET_VM_KEY:-/home/agent/.ssh/id_rsa}"
    local ssh_opts="-o StrictHostKeyChecking=no -o BatchMode=yes -i $ssh_key"
    
    ssh $ssh_opts "${TARGET_VM_USER}@${TARGET_VM_IP}" "$cmd"
}

# Copiar fichero a VM destino
# Uso: ssh_copy "/local/path" "/remote/path"
ssh_copy() {
    local local_path=$1
    local remote_path=$2
    local ssh_key="${TARGET_VM_KEY:-/home/agent/.ssh/id_rsa}"
    local ssh_opts="-o StrictHostKeyChecking=no -o BatchMode=yes -i $ssh_key"
    
    scp $ssh_opts "$local_path" "${TARGET_VM_USER}@${TARGET_VM_IP}:${remote_path}"
}

# Verificar conectividad SSH
ssh_check() {
    ssh_exec "echo ok" &>/dev/null
}

#===============================================================================
# Utilidades generales
#===============================================================================

# Verificar que comando existe
require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" &>/dev/null; then
        log_error "Comando requerido no encontrado: $cmd"
        return 1
    fi
}

# Esperar con timeout
# Uso: wait_for "ssh_check" 30 5  # Esperar hasta 30 intentos, 5 segundos entre cada uno
wait_for() {
    local check_cmd=$1
    local max_attempts=${2:-30}
    local interval=${3:-10}
    local attempt=1
    
    while ! eval "$check_cmd"; do
        if [[ $attempt -ge $max_attempts ]]; then
            return 1
        fi
        sleep $interval
        ((attempt++))
    done
    return 0
}

# Obtener duración formateada
# Uso: format_duration 125  # Devuelve "2m 5s"
format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local remaining=$((seconds % 60))
    
    if [[ $minutes -gt 0 ]]; then
        echo "${minutes}m ${remaining}s"
    else
        echo "${seconds}s"
    fi
}

# Inicializar (crear directorio de logs si no existe)
init_common() {
    mkdir -p "$LOG_DIR"
    load_config
}

# Inicializar automáticamente al cargar
init_common
