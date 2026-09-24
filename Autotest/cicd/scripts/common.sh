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

# Fichero de log del día. Si no se fija LOG_FILE desde fuera, log() lo
# recalcula en cada llamada: un proceso que sigue vivo al cambiar de día (el
# daemon) pasa a escribir en el fichero del día nuevo.
if [[ -n "${LOG_FILE:-}" ]]; then
    LOG_FILE_FIXED=1
else
    LOG_FILE_FIXED=0
    printf -v LOG_FILE '%s/pipeline_%(%Y%m%d)T.log' "$LOG_DIR" -1
fi

# Nivel mínimo de log (general.log_level, se aplica en load_config):
# DEBUG=0, INFO/OK=1, WARN=2, ERROR=3. Hasta leer la configuración, INFO.
LOG_LEVEL_NUM=1

# Convertir un nombre de nivel en su número (INFO si no es válido)
log_level_num() {
    case ${1^^} in
        DEBUG) echo 0 ;;
        WARN|WARNING) echo 2 ;;
        ERROR) echo 3 ;;
        *) echo 1 ;;
    esac
}

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

    # Filtrar por nivel. ERROR y los niveles desconocidos se escriben siempre.
    # Siempre devuelve 0: con 'set -e' un log filtrado no puede abortar el script.
    local num=3
    case $level in
        DEBUG) num=0 ;;
        INFO|OK) num=1 ;;
        WARN) num=2 ;;
    esac
    if [[ $num -lt ${LOG_LEVEL_NUM:-1} ]]; then
        return 0
    fi

    local timestamp
    printf -v timestamp '%(%Y-%m-%d %H:%M:%S)T' -1
    local message="[$timestamp] [$level] $*"

    if [[ "${LOG_FILE_FIXED:-0}" != "1" ]]; then
        printf -v LOG_FILE '%s/pipeline_%(%Y%m%d)T.log' "$LOG_DIR" -1
    fi
    echo "$message" >> "$LOG_FILE"
    
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

# yq es obligatorio: sin él config_get devolvería valores vacíos (con
# TAG_PATTERN vacío cualquier tag coincide) y el pipeline fallaría en silencio.
if ! command -v yq &>/dev/null; then
    log_error "yq no está instalado y es obligatorio para leer $CONFIG_FILE (https://github.com/mikefarah/yq)"
    exit 1
fi

if [[ ! -r "$CONFIG_FILE" ]]; then
    log_error "Archivo de configuración no encontrado o sin permisos de lectura: $CONFIG_FILE"
    exit 1
fi

# Obtener valor de configuración YAML
# Uso: config_get "git.repo_url"
# Devuelve 1 si yq no puede leer el fichero (YAML mal formado, etc.)
config_get() {
    local key=$1
    local default=${2:-}
    local value

    if ! value=$(yq ".$key" "$CONFIG_FILE" 2>&1); then
        log_error "Error leyendo '$key' de $CONFIG_FILE: $value"
        return 1
    fi

    if [[ "$value" != "null" && -n "$value" ]]; then
        expand_env_vars "$value"
    else
        echo "$default"
    fi
}

# Sustituir ${VAR} y $VAR por su valor del entorno (vacío si no existe).
# No ejecuta nada: $(...), comillas invertidas, comillas y '$' sueltos
# (p. ej. el '$' final de una regex o una contraseña) se dejan tal cual.
# Uso: expand_env_vars 'https://${GIT_USER}@host'
expand_env_vars() {
    local rest=$1 out="" name
    local re='\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
    while [[ $rest =~ $re ]]; do
        name=${BASH_REMATCH[1]:-${BASH_REMATCH[2]}}
        out+=${rest%%"${BASH_REMATCH[0]}"*}${!name:-}
        rest=${rest#*"${BASH_REMATCH[0]}"}
    done
    printf '%s\n' "$out$rest"
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
    # Nivel de log: general.log_level, o CICD_LOG_LEVEL para una ejecución puntual
    local log_level
    log_level=$(config_get "general.log_level" "INFO")
    LOG_LEVEL_NUM=$(log_level_num "${CICD_LOG_LEVEL:-$log_level}")

    POLLING_INTERVAL=$(config_get "general.polling_interval_seconds" "300")
    export POLLING_INTERVAL
    DB_BUSY_TIMEOUT_MS=$(config_get "general.db_busy_timeout_ms" "10000")
    [[ "$DB_BUSY_TIMEOUT_MS" =~ ^[0-9]+$ ]] || DB_BUSY_TIMEOUT_MS=10000
    export DB_BUSY_TIMEOUT_MS

    # Claves sin las que el pipeline no puede funcionar
    local var missing=()
    for var in GIT_REPO_URL TAG_PATTERN REPO_LOCAL_PATH COMPILE_PATH BUILD_SCRIPT TARGET_VM_IP; do
        [[ -n "${!var:-}" ]] || missing+=("$var")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Configuración incompleta en $CONFIG_FILE, faltan: ${missing[*]}"
        return 1
    fi

    # grep devuelve 2 si la regex no es válida
    local rc=0
    grep -Eq -- "$TAG_PATTERN" <<< "" || rc=$?
    if [[ $rc -eq 2 ]]; then
        log_error "git.tag_pattern no es una regex extendida válida: $TAG_PATTERN"
        return 1
    fi
}

# Comprobar que un nombre de tag cumple git.tag_pattern
# Uso: if tag_is_valid "$tag"; then ...
tag_is_valid() {
    local tag=${1:-}
    [[ -n "$tag" && -n "${TAG_PATTERN:-}" && "$tag" =~ $TAG_PATTERN ]]
}

#===============================================================================
# Base de datos SQLite
#===============================================================================

# Invocar sqlite3 con busy timeout: la Web y el pipeline acceden a la vez y,
# sin él, cualquier escritura concurrente falla con "database is locked".
# Se usa '.timeout' (y no PRAGMA busy_timeout) porque no escribe nada en stdout.
db_sqlite() {
    sqlite3 -cmd ".timeout ${DB_BUSY_TIMEOUT_MS:-10000}" "$@"
}

# Ejecutar query SQL
# Uso: db_query "SELECT * FROM deployments"
# Los valores de texto deben ir escapados con sql_escape.
db_query() {
    local query=$1
    db_sqlite "$DB_PATH" "$query"
}

# Ejecutar query y devolver resultado con headers
db_query_headers() {
    local query=$1
    db_sqlite -header -column "$DB_PATH" "$query"
}

# Escapar un valor para usarlo dentro de un literal SQL entre comillas simples
# Uso: db_query "... WHERE tag_name='$(sql_escape "$tag")'"
sql_escape() {
    local value=${1:-}
    local q="'"
    printf '%s' "${value//$q/$q$q}"
}

# Insertar log de ejecución
# Uso: db_log_execution $deployment_id "compile" "Iniciando compilación" "INFO"
db_log_execution() {
    local deployment_id=$1
    local phase=$2
    local message=$3
    local level=${4:-INFO}

    if [[ ! "$deployment_id" =~ ^[0-9]+$ ]]; then
        log_warn "db_log_execution: deployment_id no válido: '$deployment_id'"
        return 1
    fi

    db_query "INSERT INTO execution_log (deployment_id, phase, message, level)
              VALUES ($deployment_id, '$(sql_escape "$phase")', '$(sql_escape "$message")', '$(sql_escape "$level")')"
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
    count=$(db_query "SELECT COUNT(*) FROM deployments WHERE tag_name='$(sql_escape "$tag")' AND status='success'")
    [[ "$count" -gt 0 ]]
}

# Migraciones ligeras del schema (idempotentes)
# - journal_mode=WAL: lectores (Web) y escritor (pipeline) no se bloquean entre sí.
#   Es persistente en el fichero de BD, basta con aplicarlo una vez.
# - processed_tags.attempts / last_error: control de reintentos de tags fallidos
# - deployments.current_phase / failed_phase: seguimiento de fases en la Web
db_ensure_schema() {
    [[ -f "$DB_PATH" ]] || return 0

    local mode
    mode=$(db_query "PRAGMA journal_mode;") || return 1
    if [[ "$mode" != "wal" ]]; then
        db_query "PRAGMA journal_mode=WAL;" >/dev/null || return 1
        log_info "Migración BD: journal_mode cambiado a WAL"
    fi

    local cols
    cols=$(db_query "PRAGMA table_info(processed_tags);" | cut -d'|' -f2) || return 1

    if ! grep -qx "attempts" <<< "$cols"; then
        db_query "ALTER TABLE processed_tags ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0" || return 1
        log_info "Migración BD: añadida columna processed_tags.attempts"
    fi
    if ! grep -qx "last_error" <<< "$cols"; then
        db_query "ALTER TABLE processed_tags ADD COLUMN last_error TEXT" || return 1
        log_info "Migración BD: añadida columna processed_tags.last_error"
    fi

    # deployments.current_phase / failed_phase: fase en curso y fase fallida (Web UI)
    cols=$(db_query "PRAGMA table_info(deployments);" | cut -d'|' -f2) || return 1
    local col
    for col in current_phase failed_phase; do
        if ! grep -qx "$col" <<< "$cols"; then
            db_query "ALTER TABLE deployments ADD COLUMN $col TEXT" || return 1
            log_info "Migración BD: añadida columna deployments.$col"
        fi
    done

    # deployments.tag_name ya no es UNIQUE: cada reproceso de un tag es una
    # fila nueva (deployments.attempt) y se conserva el historial.
    if db_query "PRAGMA index_list(deployments);" | cut -d'|' -f4 | grep -qx "u"; then
        db_rebuild_deployments || return 1
        log_info "Migración BD: deployments.tag_name sin UNIQUE y columna deployments.attempt"
    elif ! grep -qx "attempt" <<< "$cols"; then
        db_query "ALTER TABLE deployments ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1" || return 1
        log_info "Migración BD: añadida columna deployments.attempt"
    fi
}

# Recrear la tabla deployments sin el UNIQUE de tag_name (SQLite no permite
# quitarlo con ALTER TABLE). Todo en una transacción: si algo falla, sqlite3
# termina sin COMMIT y la BD queda como estaba. Las vistas se borran y se
# vuelven a crear porque el RENAME falla si alguna referencia a la tabla.
db_rebuild_deployments() {
    local views indexes drop_views
    views=$(db_query "SELECT group_concat(sql || ';', char(10)) FROM sqlite_master WHERE type='view'") || return 1
    drop_views=$(db_query "SELECT group_concat('DROP VIEW IF EXISTS \"' || name || '\";', char(10))
                           FROM sqlite_master WHERE type='view'") || return 1
    indexes=$(db_query "SELECT group_concat(sql || ';', char(10)) FROM sqlite_master
                        WHERE type='index' AND tbl_name='deployments' AND sql IS NOT NULL") || return 1

    local cols="id, tag_name, status, started_at, completed_at, duration_seconds, triggered_by,
                error_message, current_phase, failed_phase, created_at"
    db_query "BEGIN IMMEDIATE;
$drop_views
CREATE TABLE deployments_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL,
    status TEXT CHECK(status IN ('pending', 'compiling', 'analyzing', 'deploying', 'success', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds INTEGER,
    triggered_by TEXT DEFAULT 'daemon',
    error_message TEXT,
    current_phase TEXT,
    failed_phase TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    attempt INTEGER NOT NULL DEFAULT 1
);
INSERT INTO deployments_new ($cols, attempt) SELECT $cols, 1 FROM deployments;
DROP TABLE deployments;
ALTER TABLE deployments_new RENAME TO deployments;
$indexes
$views
COMMIT;"
}

# Número máximo de intentos automáticos por tag (general.max_tag_attempts)
tag_max_attempts() {
    local max
    max=$(config_get "general.max_tag_attempts" "3")
    [[ "$max" =~ ^[1-9][0-9]*$ ]] || max=3
    echo "$max"
}

# Registrar un intento fallido de un tag en processed_tags.
# Deja el tag en 'pending' (el daemon lo reintentará tras general.retry_delay_seconds)
# o en 'skipped' si ha agotado general.max_tag_attempts.
# Uso: db_register_tag_failure "V08_00_00_02" "[deploy_ssh] Error en despliegue SSH"
db_register_tag_failure() {
    local tag_raw=$1
    local tag err max
    tag=$(sql_escape "$tag_raw")
    err=$(sql_escape "${2:-}")
    max=$(tag_max_attempts)

    db_ensure_schema || log_warn "No se pudo verificar el schema de processed_tags"

    db_query "INSERT OR IGNORE INTO processed_tags (tag_name, status) VALUES ('$tag', 'pending');
              UPDATE processed_tags
                 SET attempts = attempts + 1,
                     last_error = '$err',
                     processed_at = datetime('now'),
                     status = CASE WHEN attempts + 1 >= $max THEN 'skipped' ELSE 'pending' END
               WHERE tag_name = '$tag';" || return 1

    local state
    state=$(db_query "SELECT status || '|' || attempts FROM processed_tags WHERE tag_name='$tag'") || return 0
    if [[ "${state%%|*}" == "skipped" ]]; then
        log_warn "Tag '$tag_raw' descartado tras ${state##*|} intento(s) fallido(s): no se reintentará automáticamente"
    else
        log_info "Tag '$tag_raw': intento ${state##*|}/$max fallido, el daemon lo reintentará más tarde"
    fi
}

# Marcar un tag como descartado ('skipped'): la detección lo salta y no bloquea
# a los tags siguientes. Se puede reprocesar a mano con: ci_cd.sh --tag <tag>
# Uso: db_mark_tag_skipped "V08_00_00_00_Foo" "no cumple git.tag_pattern"
db_mark_tag_skipped() {
    local tag err
    tag=$(sql_escape "$1")
    err=$(sql_escape "${2:-skipped}")

    db_query "INSERT OR IGNORE INTO processed_tags (tag_name, status) VALUES ('$tag', 'skipped');
              UPDATE processed_tags
                 SET status = 'skipped', last_error = '$err', processed_at = datetime('now')
               WHERE tag_name = '$tag';"
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

# Ejecutar una acción de vcenter_api.py con un tiempo máximo total.
# Cada petición HTTP ya tiene su propio timeout; este límite cubre la
# llamada completa (reintentos, esperas de estado, subida lenta del ISO).
# Uso: vcenter_call <acción> [args...]   (la salida es la de vcenter_api.py)
# Devuelve el código de vcenter_api.py, o 124/137 si se supera el límite.
vcenter_call() {
    local action=$1
    local key="vcenter.action_timeout_seconds"
    local default=900
    if [[ "$action" == "upload_iso" ]]; then
        key="vcenter.upload_action_timeout_seconds"
        default=10800
    fi

    local max_seconds
    max_seconds=$(config_get "$key" "$default")
    [[ "$max_seconds" =~ ^[1-9][0-9]*$ ]] || max_seconds=$default

    local vcenter_script
    vcenter_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/python/vcenter_api.py"

    local rc=0
    timeout --kill-after=30 "$max_seconds" python3 "$vcenter_script" "$CONFIG_FILE" "$@" || rc=$?
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        log_error "vcenter_api.py $action superó el tiempo máximo ($(format_duration "$max_seconds"))"
    fi
    return $rc
}

# Borrar los logs con fecha en el nombre (pipeline_*, compile_*, deploy_*)
# que no se han modificado en general.log_retention_days días (3 por defecto).
# Los logs de nombre fijo (service*.log, web_*.log) los rota logrotate
# (cicd.logrotate): borrarlos aquí dejaría a systemd/gunicorn escribiendo en
# un fichero ya eliminado.
purge_old_logs() {
    local days
    days=$(config_get "general.log_retention_days" "3")
    [[ "$days" =~ ^[1-9][0-9]*$ ]] || days=3

    local f count=0
    while IFS= read -r -d '' f; do
        if rm -f -- "$f"; then
            count=$((count + 1))
        fi
    done < <(find "$LOG_DIR" -maxdepth 1 -type f \
                 \( -name 'pipeline_*.log' -o -name 'compile_*.log' -o -name 'deploy_*.log' \) \
                 -mmin +$((days * 1440)) -print0 2>/dev/null)

    if [[ $count -gt 0 ]]; then
        log_info "Rotación de logs: eliminados $count fichero(s) con más de $days día(s) en $LOG_DIR"
    fi
    return 0
}

# Inicializar (crear directorio de logs si no existe)
init_common() {
    mkdir -p "$LOG_DIR"
    load_config
}

# Inicializar automáticamente al cargar (sin configuración válida no se continúa)
init_common || exit 1
