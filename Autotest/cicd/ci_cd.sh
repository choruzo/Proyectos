#!/bin/bash
#===============================================================================
# ci_cd.sh - Orquestador Principal del Pipeline CI/CD
#===============================================================================
# Script principal que coordina todas las fases del pipeline:
#   1. Monitorización de tags Git
#   2. Compilación
#   3. Análisis SonarQube
#   4. Generación de checksums (sha256sum + ZIP) y documentación Doxygen
#   5. Despliegue en vCenter + VM
#   6. Finalización y notificaciones
#
# Uso:
#   ./ci_cd.sh daemon           # Modo daemon (polling continuo)
#   ./ci_cd.sh --tag TAG_NAME   # Procesar tag específico manualmente
#   ./ci_cd.sh status           # Ver estado del último despliegue
#   ./ci_cd.sh init             # Inicializar base de datos
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/ci_cd_config.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
DB_PATH="$SCRIPT_DIR/db/pipeline.db"

# Cargar configuración común
source "$SCRIPT_DIR/scripts/common.sh"

# Cargar variables de configuración desde YAML
load_config

# PID file del daemon (informativo, lo usa 'status')
PID_FILE="$SCRIPT_DIR/.cicd.pid"

# Locks (flock). Se liberan solos al morir el proceso: no hay locks huérfanos.
#  - DAEMON_LOCK_FILE: una sola instancia del daemon.
#  - PIPELINE_LOCK_FILE: un solo pipeline a la vez en cualquier modo (daemon,
#    --tag). Los pipelines comparten compile_path, .last_iso_path, el snapshot
#    y la VM destino. El descriptor lo heredan los procesos hijos, así que el
#    lock se mantiene mientras siga vivo cualquier proceso del pipeline.
DAEMON_LOCK_FILE="$SCRIPT_DIR/.cicd.daemon.lock"
PIPELINE_LOCK_FILE="$SCRIPT_DIR/.cicd.lock"
DAEMON_LOCK_FD=""
PIPELINE_LOCK_FD=""

#===============================================================================
# Verificación de instancia única y lock del pipeline
#===============================================================================

check_already_running() {
    exec {DAEMON_LOCK_FD}>>"$DAEMON_LOCK_FILE"
    if ! flock -n "$DAEMON_LOCK_FD"; then
        log_error "Ya hay un daemon ejecutándose (PID: $(cat "$PID_FILE" 2>/dev/null || echo '?'))"
        return 1
    fi
    return 0
}

# Tomar el lock del pipeline sin esperar. Devuelve 1 si otro proceso lo tiene.
acquire_pipeline_lock() {
    [[ -n "$PIPELINE_LOCK_FD" ]] && return 0

    local fd
    exec {fd}>>"$PIPELINE_LOCK_FILE"
    if ! flock -n "$fd"; then
        exec {fd}>&-
        return 1
    fi
    PIPELINE_LOCK_FD=$fd
    return 0
}

# Cerrar el descriptor del lock. No se hace 'flock -u': si quedara vivo algún
# proceso hijo del pipeline, el lock debe seguir tomado hasta que termine.
release_pipeline_lock() {
    [[ -n "$PIPELINE_LOCK_FD" ]] || return 0
    exec {PIPELINE_LOCK_FD}>&-
    PIPELINE_LOCK_FD=""
}

create_pid_file() {
    echo $$ > "$PID_FILE"
}

remove_pid_file() {
    rm -f "$PID_FILE"
}

#===============================================================================
# Inicialización
#===============================================================================

init_database() {
    log_info "Inicializando base de datos..."
    
    local db_dir
    db_dir=$(dirname "$DB_PATH")
    mkdir -p "$db_dir"
    
    local init_sql="$SCRIPT_DIR/db/init_db.sql"
    
    if [[ ! -f "$init_sql" ]]; then
        log_error "Archivo de inicialización no encontrado: $init_sql"
        return 1
    fi
    
    if db_sqlite "$DB_PATH" < "$init_sql" 2>&1 && db_ensure_schema; then
        log_ok "Base de datos inicializada: $DB_PATH"
        return 0
    else
        log_error "Error inicializando base de datos"
        return 1
    fi
}

reset_database() {
    log_warn "¡ATENCIÓN! Esta operación eliminará todos los datos de la base de datos."

    # Pedir confirmación si hay terminal interactiva
    if [[ -t 0 ]]; then
        read -r -p "¿Seguro que deseas resetear la base de datos? [s/N]: " confirm
        if [[ ! "$confirm" =~ ^[sS]$ ]]; then
            log_info "Operación cancelada."
            return 0
        fi
    fi

    # Hacer backup si la BD ya existe
    if [[ -f "$DB_PATH" ]]; then
        local backup_path
        backup_path="${DB_PATH%.db}_backup_$(date +%Y%m%d_%H%M%S).db"
        cp "$DB_PATH" "$backup_path"
        log_info "Backup guardado en: $backup_path"

        rm -f "$DB_PATH"
        log_ok "Base de datos eliminada: $DB_PATH"
    else
        log_info "La base de datos no existía aún."
    fi

    init_database
}

verify_environment() {
    log_info "Verificando entorno..."
    
    local errors=0
    
    # Verificar directorios
    for dir in "$LOG_DIR" "$SCRIPT_DIR/db"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Directorio creado: $dir"
        fi
    done
    
    # Verificar configuración
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Archivo de configuración no encontrado: $CONFIG_FILE"
        ((errors++))
    fi
    
    # Verificar base de datos
    if [[ ! -f "$DB_PATH" ]]; then
        log_warn "Base de datos no existe, inicializando..."
        init_database || ((errors++))
    fi
    db_ensure_schema || log_warn "No se pudo aplicar la migración del schema de la BD"
    
    # Verificar scripts requeridos
    local scripts=(
        "scripts/git_monitor.sh"
        "scripts/compile.sh"
        "scripts/deploy.sh"
        "scripts/notify.sh"
        "python/sonar_check.py"
        "python/vcenter_api.py"
    )
    
    for script in "${scripts[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            log_debug "Script OK: $script"
        else
            log_error "Script no encontrado: $script"
            ((errors++))
        fi
    done
    
    # Verificar comandos requeridos
    local commands=(git sqlite3 python3 yq flock timeout)
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            log_debug "Comando OK: $cmd"
        else
            log_error "Comando requerido no encontrado: $cmd"
            ((errors++))
        fi
    done
    
    if [[ $errors -gt 0 ]]; then
        log_error "Se encontraron $errors errores de configuración"
        return 1
    fi
    
    log_ok "Entorno verificado correctamente"
    return 0
}

#===============================================================================
# Gestión de fallos del pipeline
#===============================================================================

# Estado de la ejecución en curso. Cada pipeline corre en su propio subshell
# (ver run_pipeline_isolated), así que no se arrastra entre ejecuciones.
PIPELINE_TAG=""
PIPELINE_DEPLOYMENT_ID=""
PIPELINE_START_TIME=""
PIPELINE_FAILURE_HANDLED=0
PIPELINE_PID=""

# Registrar el fallo del pipeline (BD, processed_tags, notificación).
# Idempotente: solo actúa la primera vez, para que un fallo explícito seguido
# del trap ERR no sobrescriba la causa real ni duplique notificaciones.
# Devuelve 0 para no disparar a su vez el trap ERR.
cleanup_on_error() {
    local error_msg=${1:-"Error desconocido"}
    local phase=${2:-"unknown"}
    local full_msg="[$phase] $error_msg"

    if [[ "$PIPELINE_FAILURE_HANDLED" == "1" ]]; then
        log_debug "Fallo ya registrado, se ignora: $full_msg"
        return 0
    fi
    PIPELINE_FAILURE_HANDLED=1

    log_error "════════════════════════════════════════════════════════"
    log_error "PIPELINE FALLIDO"
    log_error "════════════════════════════════════════════════════════"
    log_error "Tag: $PIPELINE_TAG"
    log_error "Fase: $phase"
    log_error "Error: $error_msg"

    if [[ -n "$PIPELINE_DEPLOYMENT_ID" ]]; then
        local duration="NULL"
        if [[ -n "$PIPELINE_START_TIME" ]]; then
            duration=$(( $(date +%s) - PIPELINE_START_TIME ))
        fi
        db_query "UPDATE deployments SET status='failed', error_message='$(sql_escape "$full_msg")',
                  completed_at=datetime('now'), duration_seconds=$duration
                  WHERE id=$PIPELINE_DEPLOYMENT_ID" \
            || log_warn "No se pudo registrar el fallo en deployments (id=$PIPELINE_DEPLOYMENT_ID)"
    fi

    if [[ -n "$PIPELINE_TAG" ]]; then
        db_register_tag_failure "$PIPELINE_TAG" "$full_msg" \
            || log_warn "No se pudo registrar el intento fallido en processed_tags"
    fi

    "$SCRIPT_DIR/scripts/notify.sh" both failure "$PIPELINE_TAG" "$full_msg" 2>/dev/null || true

    return 0
}

# Handler del trap ERR: fallo no controlado dentro del pipeline
on_pipeline_unexpected_error() {
    local exit_code=$1
    local line=$2
    local cmd=${3:0:200}

    # Con errtrace el trap también salta dentro de $(...) y subshells. Ahí no
    # se decide el fallo: si es relevante, su código de salida llega al
    # proceso principal del pipeline y el trap salta allí.
    if [[ "$BASHPID" != "$PIPELINE_PID" ]]; then
        log_warn "Comando fallido en subproceso (código $exit_code, línea $line): $cmd"
        return 0
    fi

    cleanup_on_error "Error inesperado (código $exit_code) en línea $line: $cmd" "unexpected"
}

# Cerrar la ejecución de un tag que quedó a medias porque el proceso del
# pipeline murió sin pasar por cleanup_on_error (timeout global, señal, crash).
# Si el pipeline ya registró su fallo, no hay nada abierto y no hace nada.
close_unfinished_run() {
    local tag=$1
    local phase=$2
    local msg="[$2] $3"
    local tag_sql ids id
    tag_sql=$(sql_escape "$tag")

    ids=$(db_query "SELECT id FROM deployments WHERE tag_name='$tag_sql'
                    AND status IN ('pending', 'compiling', 'analyzing', 'deploying')") || ids=""
    [[ -n "$ids" ]] || return 0

    log_error "PIPELINE ABORTADO ($phase) - Tag: $tag - $3"
    while IFS= read -r id; do
        [[ -n "$id" ]] || continue
        db_query "UPDATE deployments SET status='failed', completed_at=datetime('now'),
                  duration_seconds=CAST(strftime('%s','now') - strftime('%s', started_at) AS INTEGER),
                  error_message='$(sql_escape "$msg")'
                  WHERE id=$id" || log_warn "No se pudo registrar el fallo en deployments (id=$id)"
    done <<< "$ids"

    db_register_tag_failure "$tag" "$msg" || log_warn "No se pudo registrar el intento fallido en processed_tags"
    "$SCRIPT_DIR/scripts/notify.sh" both failure "$tag" "$msg" 2>/dev/null || true
}

# Ejecutar run_pipeline en un proceso aparte (ci_cd.sh __run_pipeline) bajo
# 'timeout' (general.pipeline_timeout_seconds). El resultado se deja en
# PIPELINE_RESULT y la función devuelve 0; si llega INT/TERM, cierra la
# ejecución y termina el proceso.
# - Hay que tener tomado el lock del pipeline (lo hereda el proceso hijo).
# - 'timeout' crea su propio grupo de procesos y, al vencer, mata al grupo
#   entero (compilación, ssh, subida del ISO...), no solo al script.
# - Por eso Ctrl-C no le llega directamente: INT/TERM se reenvían a mano.
# - El proceso aparte aísla cd, traps y variables entre ejecuciones.
PIPELINE_RESULT=0
run_pipeline_isolated() {
    local tag=$1
    local triggered_by=$2

    if [[ -z "$PIPELINE_LOCK_FD" ]]; then
        log_error "run_pipeline_isolated llamado sin el lock del pipeline"
        PIPELINE_RESULT=1
        return 0
    fi

    local max_seconds
    max_seconds=$(config_get "general.pipeline_timeout_seconds" "14400")
    [[ "$max_seconds" =~ ^[1-9][0-9]*$ ]] || max_seconds=14400
    log_info "Timeout global del pipeline: $(format_duration "$max_seconds")"

    local child rc=0 signal_rc=0
    CICD_PIPELINE_CHILD=1 timeout --kill-after=120 "$max_seconds" \
        "$SCRIPT_DIR/ci_cd.sh" __run_pipeline "$tag" "$triggered_by" &
    child=$!

    trap 'signal_rc=130; kill -TERM "$child" 2>/dev/null || true' INT
    trap 'signal_rc=143; kill -TERM "$child" 2>/dev/null || true' TERM
    # 'wait' vuelve antes de tiempo si llega una señal con trap: repetir
    # hasta que el hijo haya terminado de verdad.
    while :; do
        rc=0
        wait "$child" || rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    trap - INT TERM

    PIPELINE_RESULT=$rc
    if [[ $signal_rc -ne 0 ]]; then
        close_unfinished_run "$tag" "interrupted" "Pipeline interrumpido por señal (código $rc)"
        release_pipeline_lock
        exit "$signal_rc"
    elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
        close_unfinished_run "$tag" "timeout" "Superado el timeout global del pipeline ($(format_duration "$max_seconds"))"
    elif [[ $rc -ne 0 ]]; then
        close_unfinished_run "$tag" "unexpected" "El proceso del pipeline terminó con código $rc sin registrar el fallo"
    fi

    return 0
}

# Recuperar ejecuciones interrumpidas (proceso muerto a mitad de pipeline:
# reinicio del servicio, kill, reboot...). Se llama con el lock del pipeline
# tomado, así que ninguna de las ejecuciones abiertas puede seguir viva.
recover_interrupted_runs() {
    if [[ -z "$PIPELINE_LOCK_FD" ]]; then
        log_warn "recover_interrupted_runs llamado sin el lock del pipeline, se omite"
        return 0
    fi

    local stale id tag
    stale=$(db_query "SELECT id || '|' || tag_name FROM deployments
                      WHERE status IN ('pending', 'compiling', 'analyzing', 'deploying')") || return 0
    while IFS='|' read -r id tag; do
        [[ -z "$id" ]] && continue
        log_warn "Ejecución interrumpida: $tag (deployment_id=$id), se marca como fallida"
        db_query "UPDATE deployments SET status='failed', completed_at=datetime('now'),
                  error_message='[interrupted] Ejecución interrumpida: el proceso terminó sin cerrar el pipeline'
                  WHERE id=$id" || true
        db_register_tag_failure "$tag" "[interrupted] Ejecución interrumpida" || true
    done <<< "$stale"

    # Tags en 'processing' sin ejecución activa
    local orphans
    orphans=$(db_query "SELECT tag_name FROM processed_tags WHERE status='processing'") || return 0
    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        log_warn "Tag '$tag' en 'processing' sin ejecución activa, se registra como intento fallido"
        db_register_tag_failure "$tag" "[interrupted] Estado 'processing' sin ejecución activa" || true
    done <<< "$orphans"
}

#===============================================================================
# Pipeline Principal
#===============================================================================

# Ejecutar SIEMPRE a través de run_pipeline_isolated
run_pipeline() {
    local tag=$1
    local triggered_by=${2:-"daemon"}
    local deployment_id=""

    if ! tag_is_valid "$tag"; then
        log_error "Tag no válido según git.tag_pattern ($TAG_PATTERN): $tag"
        return 1
    fi
    [[ "$triggered_by" =~ ^(daemon|manual)$ ]] || triggered_by="manual"

    # Literal SQL del tag (escapado)
    local tag_sql
    tag_sql=$(sql_escape "$tag")

    local start_time
    start_time=$(date +%s)

    PIPELINE_TAG="$tag"
    PIPELINE_START_TIME="$start_time"
    PIPELINE_FAILURE_HANDLED=0
    PIPELINE_PID="$BASHPID"

    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "INICIANDO PIPELINE CI/CD"
    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "Tag: $tag"
    log_info "Trigger: $triggered_by"
    log_info "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "═══════════════════════════════════════════════════════════════════"
    
    # Verificar si el tag ya existe en deployments
    local existing_deployment
    existing_deployment=$(db_query "SELECT id FROM deployments WHERE tag_name='$tag_sql'" | head -n1 || echo "")
    
    if [[ -n "$existing_deployment" ]]; then
        log_warn "El tag '$tag' ya fue procesado anteriormente (deployment_id: $existing_deployment)"
        log_warn "Eliminando registro anterior para reprocesar..."
        
        # Eliminar registros anteriores.
        # processed_tags NO se borra: conserva el contador de intentos, si no
        # un tag que falla siempre se reintentaría indefinidamente.
        db_query "DELETE FROM deployments WHERE tag_name='$tag_sql'" || log_warn "No se pudo borrar el deployment anterior"
        db_query "DELETE FROM build_logs WHERE tag='$tag_sql'" || log_warn "No se pudieron borrar los build_logs anteriores"
        db_query "DELETE FROM sonar_results WHERE tag='$tag_sql'" || log_warn "No se pudieron borrar los sonar_results anteriores"

        log_ok "Registros anteriores eliminados, continuando con reprocesamiento..."
    fi
    
    # Registrar inicio en BD
    deployment_id=$(db_query \
        "INSERT INTO deployments (tag_name, status, started_at, triggered_by) 
         VALUES ('$tag_sql', 'pending', datetime('now'), '$triggered_by');
         SELECT last_insert_rowid();")
    
    if [[ -z "$deployment_id" || "$deployment_id" == "0" ]]; then
        log_error "Error crítico: No se pudo crear registro en deployments"
        return 1
    fi
    
    log_debug "Deployment ID: $deployment_id"
    PIPELINE_DEPLOYMENT_ID="$deployment_id"

    # Trap para errores inesperados (con errtrace también cubre funciones anidadas)
    trap 'on_pipeline_unexpected_error $? $LINENO "$BASH_COMMAND"' ERR
    
    #---------------------------------------------------------------------------
    # FASE 1: Checkout del tag
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[1/6] CHECKOUT DEL TAG"
    log_info "───────────────────────────────────────────────────────────"
    
    db_query "UPDATE deployments SET status='compiling' WHERE id=$deployment_id"
    
    if ! "$SCRIPT_DIR/scripts/git_monitor.sh" checkout "$tag"; then
        cleanup_on_error "Fallo en checkout del tag $tag" "checkout"
        return 1
    fi
    
    log_ok "Checkout completado"
    
    #---------------------------------------------------------------------------
    # FASE 2: Compilación
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[2/6] COMPILACIÓN"
    log_info "───────────────────────────────────────────────────────────"
    
    # Notificar inicio de compilación
    "$SCRIPT_DIR/scripts/notify.sh" wall compiling "$tag" 2>/dev/null || true
    
    if ! "$SCRIPT_DIR/scripts/compile.sh"; then
        cleanup_on_error "Fallo en compilación" "compile"
        return 1
    fi
    
    log_ok "Compilación completada"
    
    #---------------------------------------------------------------------------
    # FASE 3: Análisis SonarQube
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[3/6] ANÁLISIS SONARQUBE"
    log_info "───────────────────────────────────────────────────────────"
    
    db_query "UPDATE deployments SET status='analyzing' WHERE id=$deployment_id"
    
    # Obtener rutas de configuración
    local compile_path
    compile_path=$(config_get "git.compile_path" "/home/agent/compile")
    
    # Copiar herramientas al directorio de compilación
    log_info "Preparando herramientas de análisis..."
    
    if [[ ! -d "$compile_path/utils" ]]; then
        log_info "Copiando carpeta utils al directorio de compilación..."
        cp -r "$SCRIPT_DIR/utils" "$compile_path/" 2>&1 | tee -a "$LOG_FILE"
        log_ok "Carpeta utils copiada"
    else
        log_debug "Carpeta utils ya existe en compile"
    fi
    
    # Copiar sonar-project.properties al directorio de compilación
    if [[ -f "$SCRIPT_DIR/config/sonar-project.properties" ]]; then
        log_info "Copiando sonar-project.properties..."
        cp "$SCRIPT_DIR/config/sonar-project.properties" "$compile_path/" 2>&1 | tee -a "$LOG_FILE"
        log_ok "sonar-project.properties copiado"
    fi
    
    # Dar permisos de ejecución a los binarios
    log_info "Configurando permisos de ejecución..."
    chmod +x "$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/bin/sonar-scanner" 2>/dev/null || true
    chmod +x "$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/jre/bin/java" 2>/dev/null || true
    export JAVA_HOME=/usr/lib64/jvm/java-25-openjdk-25
    log_ok "Permisos configurados"

    # Rutas locales en el directorio de compilación
    local sonar_scanner="$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/bin/sonar-scanner"
    local bw_output_dir="$compile_path/bw-output"

    # PASO 1: Salida de build-wrapper para el análisis C/C++. La genera
    # compile.sh al ejecutar build_DVDs.sh dentro de build-wrapper, así que
    # aquí no se vuelve a compilar (build_DVDs.sh también genera mmi.jar).
    if [[ ! -f "$bw_output_dir/build-wrapper-dump.json" && ! -f "$bw_output_dir/compile_commands.json" ]]; then
        log_error "No existe la salida de build-wrapper en: $bw_output_dir"
        cleanup_on_error "Salida de build-wrapper no encontrada (la genera la fase de compilación)" "sonarqube_prepare"
        return 1
    fi
    log_ok "Salida de build-wrapper disponible: $bw_output_dir"

    cd "$compile_path" || {
        log_error "No se puede acceder a: $compile_path"
        cleanup_on_error "Directorio de compilación no accesible" "sonarqube_prepare"
        return 1
    }
    
    # PASO 2: Preparar análisis Java - Extraer mmi.jar en target/
    log_info "Preparando análisis Java: extrayendo mmi.jar..."
    
    local mmi_jar="$compile_path/mmi.jar"
    local target_dir="$compile_path/target"
    
    if [[ -f "$mmi_jar" ]]; then
        mkdir -p "$target_dir"
        cd "$target_dir" || {
            log_error "No se puede acceder a: $target_dir"
            cleanup_on_error "Error creando directorio target" "sonarqube_prepare"
            return 1
        }
        
        log_info "Extrayendo $mmi_jar en $target_dir..."
        if jar xf "$mmi_jar" 2>&1 | tee -a "$LOG_FILE"; then
            log_ok "mmi.jar extraído correctamente"
        else
            log_warn "No se pudo extraer mmi.jar con 'jar', intentando con 'unzip'..."
            if unzip -q "$mmi_jar" 2>&1 | tee -a "$LOG_FILE"; then
                log_ok "mmi.jar extraído con unzip"
            else
                log_error "No se pudo extraer mmi.jar"
                cleanup_on_error "Error extrayendo mmi.jar" "sonarqube_prepare"
                return 1
            fi
        fi
        
        cd "$compile_path" || true
    else
        log_warn "mmi.jar no encontrado en: $mmi_jar"
        log_warn "El análisis Java puede ser incompleto"
    fi
    
    # PASO 3: Ejecutar sonar-scanner con configuración correcta
    log_info "Ejecutando análisis SonarQube..."
    
    if [[ ! -f "$sonar_scanner" ]]; then
        log_error "sonar-scanner no encontrado en: $sonar_scanner"
        cleanup_on_error "sonar-scanner no disponible" "sonarqube"
        return 1
    fi
    
    cd "$compile_path" || {
        log_error "No se puede acceder a: $compile_path"
        cleanup_on_error "Directorio de compilación no accesible" "sonarqube"
        return 1
    }
    
    # Rama fija (sonarqube.branch): cada análisis sobrescribe al anterior.
    # sonar_check.py consulta la misma rama.
    local sonar_branch
    sonar_branch=$(config_get "sonarqube.branch" "V08_00_00_00")

    log_info "Ejecutando: $sonar_scanner con proyecto GALTTCMC_interno (rama: $sonar_branch)"
    log_info "Directorio de trabajo: $compile_path"
    log_info "Configuración: sonar-project.properties"

    if ! "$JAVA_HOME/bin/java" -jar /home/agent/cicd/utils/sonar-scanner-7.2.0.5079-linux-x64/lib/sonar-scanner-cli-7.2.0.5079.jar  -Dproject.settings=sonar-project.properties -Dsonar.projectKey=GALTTCMC_interno -Dsonar.projectName=GALTTCMC_interno -Dsonar.branch.name="$sonar_branch" -Dsonar.projectVersion="$sonar_branch" \
        2>&1 | tee -a "$LOG_FILE"; then
        log_error "sonar-scanner falló"
        cleanup_on_error "Error en análisis SonarQube" "sonarqube"
        return 1
    fi
    
    log_ok "sonar-scanner completado"
    
    # PASO 4: Verificar resultados via API
    log_info "Verificando resultados en SonarQube..."
    
    # Pasar el report-task.txt para que sonar_check.py espere a que
    # SonarQube termine de procesar el análisis antes de consultar el
    # quality gate (SonarQube procesa de forma asíncrona).
    local report_task_file="$compile_path/.scannerwork/report-task.txt"
    
    local sonar_result=0
    python3 "$SCRIPT_DIR/python/sonar_check.py" "$CONFIG_FILE" "$tag" "$report_task_file" || sonar_result=$?
    
    if [[ $sonar_result -ne 0 ]]; then
        log_warn "Quality Gate no superado"
        "$SCRIPT_DIR/scripts/notify.sh" wall sonar_failed "$tag" 2>/dev/null || true
        
        # Verificar si se permite override
        local allow_override
        allow_override=$(config_get "sonarqube.allow_override" "false")
        
        if [[ "$allow_override" != "true" ]]; then
            cleanup_on_error "Quality Gate no superado y override no permitido" "sonarqube"
            return 1
        fi
        
        log_warn "Override habilitado, continuando con el despliegue..."
    else
        log_ok "Análisis SonarQube: APROBADO"
    fi
    
    #---------------------------------------------------------------------------
    # FASE 4: Generación de checksums y documentación Doxygen
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[4/6] GENERACIÓN DE CHECKSUMS Y DOCUMENTACIÓN"
    log_info "───────────────────────────────────────────────────────────"

    log_info "Generando documentación Doxygen..."

    (
        doxygen_dir="$compile_path/Development_TTCF/ttcf/utils/doxygen"

        if [[ ! -d "$doxygen_dir" ]]; then
            log_warn "Directorio Doxygen no encontrado: $doxygen_dir, omitiendo generación de documentación"
            exit 0
        fi

        if ! command -v doxygen >/dev/null 2>&1; then
            log_warn "Comando 'doxygen' no disponible, omitiendo generación de documentación"
            exit 0
        fi

        cd "$doxygen_dir" || {
            log_warn "No se puede acceder a: $doxygen_dir, omitiendo generación de documentación"
            exit 0
        }

        for doxyfile in Doxyfile_C_Figures Doxyfile_C_NoFigures Doxyfile_Java_Figures Doxyfile_Java_NoFigures; do
            if [[ -f "$doxyfile" ]]; then
                log_info "Ejecutando: doxygen $doxyfile"
                if ! doxygen "$doxyfile" 2>&1 | tee -a "$LOG_FILE"; then
                    log_warn "doxygen falló para $doxyfile, continuando..."
                fi
            else
                log_warn "Doxyfile no encontrado: $doxygen_dir/$doxyfile, omitiendo"
            fi
        done

        doxygen_out_dir="$doxygen_dir"
        dirs_to_zip=()
        for d in C_Figures C_NoFigures Java_Figures Java_NoFigures; do
            if [[ -d "$d" ]]; then
                dirs_to_zip+=("$d")
            fi
        done

        if [[ ${#dirs_to_zip[@]} -eq 0 ]] && [[ -d "$compile_path" ]]; then
            for d in C_Figures C_NoFigures Java_Figures Java_NoFigures; do
                if [[ -d "$compile_path/$d" ]]; then
                    dirs_to_zip+=("$d")
                fi
            done
            if [[ ${#dirs_to_zip[@]} -gt 0 ]]; then
                log_info "Carpetas de documentación encontradas en $compile_path (OUTPUT_DIRECTORY del Doxyfile), no en $doxygen_dir"
                doxygen_out_dir="$compile_path"
                cd "$doxygen_out_dir" || {
                    log_warn "No se puede acceder a: $doxygen_out_dir, omitiendo compresión"
                    exit 0
                }
            fi
        fi

        if [[ ${#dirs_to_zip[@]} -eq 0 ]]; then
            log_warn "No se generó ninguna carpeta de documentación Doxygen, omitiendo compresión"
            exit 0
        fi

        doxygen_zip="doxygen_docs_$(date +%Y%m%d_%H%M%S).zip"

        log_info "Comprimiendo documentación Doxygen en: ${doxygen_zip}"

        if command -v zip >/dev/null 2>&1; then
            zip -q -r -9 "${doxygen_zip}" "${dirs_to_zip[@]}"
        else
            log_warn "Comando 'zip' no disponible; usando python3 para crear el ZIP"
            python3 - "${doxygen_zip}" "${dirs_to_zip[@]}" <<'PY'
import os
import sys
import zipfile

zip_name = sys.argv[1]
dirs = sys.argv[2:]

zf = zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED)
try:
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                full_path = os.path.join(root, f)
                zf.write(full_path, arcname=full_path)
finally:
    zf.close()

print('ZIP creado: {0}'.format(zip_name))
PY
        fi

        if [[ "$doxygen_out_dir" != "$compile_path" ]]; then
            cp "${doxygen_zip}" "$compile_path/${doxygen_zip}" 2>/dev/null || log_warn "No se pudo copiar ${doxygen_zip} a $compile_path"
        fi

        log_ok "Documentación Doxygen generada: $compile_path/${doxygen_zip}"
    ) || log_warn "Fallo generando documentación Doxygen (no crítico), continuando con el pipeline..."

    log_info "Generando checksums en: $compile_path"

    (
        cd "$compile_path" || {
            log_error "No se puede acceder a: $compile_path"
            exit 1
        }

        # Nota: truncamos ficheros antes de usar '>>' para evitar duplicados entre ejecuciones
        : > contents_schemas.txt
        : > contents_scripts.txt
        : > contents_scripts2.txt

        log_info "Calculando checksums completos (contents_RPM_Completos.txt)..."
        find . -type f -exec sha256sum {} \; > contents_RPM_Completos.txt
        log_ok "contents_RPM_Completos.txt generado"

        log_info "Calculando checksums de esquemas XSD (contents_schemas.txt)..."
        find . -type f -name "*.xsd" -exec sha256sum {} \; >> contents_schemas.txt
        log_ok "contents_schemas.txt generado"

        log_info "Calculando checksums de scripts .sh (contents_scripts.txt)..."
        find . -type f -name "*.sh" -exec sha256sum {} \; >> contents_scripts.txt
        log_ok "contents_scripts.txt generado"

        log_info "Calculando checksums de scripts .pl (contents_scripts2.txt)..."
        find . -type f -name "*.pl" -exec sha256sum {} \; >> contents_scripts2.txt
        log_ok "contents_scripts2.txt generado"

        checksum_zip="sha256sum_files_$(date +%Y%m%d_%H%M%S).zip"

        log_info "Comprimiendo ficheros de checksums en: ${checksum_zip}"

        if command -v zip >/dev/null 2>&1; then
            zip -q -9 "${checksum_zip}" \
                contents_RPM_Completos.txt \
                contents_schemas.txt \
                contents_scripts.txt \
                contents_scripts2.txt
        else
            log_warn "Comando 'zip' no disponible; usando python3 para crear el ZIP"
            python3 - "${checksum_zip}" <<'PY'
from __future__ import print_function

import os
import sys
import zipfile

zip_name = sys.argv[1]
files = [
    'contents_RPM_Completos.txt',
    'contents_schemas.txt',
    'contents_scripts.txt',
    'contents_scripts2.txt',
]

zf = zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED)
try:
    for f in files:
        # Guardar solo el nombre del fichero (sin rutas)
        zf.write(f, arcname=os.path.basename(f))
finally:
    zf.close()

print('ZIP creado: {0}'.format(zip_name))
PY
        fi

        echo "${checksum_zip}" > .last_checksums_zip
        log_ok "ZIP generado: $compile_path/${checksum_zip}"
    ) || {
        cleanup_on_error "Error generando checksums/ZIP en $compile_path" "checksums"
        return 1
    }

    log_ok "Checksums y ZIP generados correctamente en $compile_path"

    #---------------------------------------------------------------------------
    # FASE 5: Despliegue en vCenter + VM
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[5/6] DESPLIEGUE"
    log_info "───────────────────────────────────────────────────────────"
    
    db_query "UPDATE deployments SET status='deploying' WHERE id=$deployment_id"
    
    # Notificar inicio de despliegue
    "$SCRIPT_DIR/scripts/notify.sh" wall deploying "$tag" 2>/dev/null || true
    
    # Obtener ruta del ISO generado
    local iso_path
    iso_path=$(cat "$compile_path/.last_iso_path" 2>/dev/null || find "$compile_path" -name "*.iso" -type f | head -1)
    
    if [[ -z "$iso_path" || ! -f "$iso_path" ]]; then
        cleanup_on_error "ISO no encontrado después de compilación" "deploy"
        return 1
    fi
    
    log_info "ISO a desplegar: $iso_path"
    
    # 4.1 Subir ISO al datastore
    log_info "Subiendo ISO al datastore..."
    local upload_output
    local upload_status=0
    upload_output=$(vcenter_call upload_iso "$iso_path" 2>&1) || upload_status=$?
    echo "$upload_output" | tee -a "$LOG_FILE"
    
    if [[ $upload_status -ne 0 ]]; then
        cleanup_on_error "Error subiendo ISO al datastore" "deploy_upload"
        return 1
    fi
    
    # Extraer el path remoto del ISO desde la salida
    local remote_iso_path
    remote_iso_path=$(echo "$upload_output" | grep -oP '\[REMOTE_ISO_PATH\] \K.*' || echo "")
    
    if [[ -z "$remote_iso_path" ]]; then
        # Fallback: construir path manualmente si no se pudo extraer
        local datastore iso_folder iso_filename
        datastore=$(config_get "vcenter.datastore" "NAS_LIBRERIA")
        iso_folder=$(config_get "vcenter.iso_path" "/ISO")
        # Eliminar barra inicial del iso_folder para coincidir con Python
        iso_folder="${iso_folder#/}"
        iso_filename=$(basename "$iso_path")
        remote_iso_path="[${datastore}] ${iso_folder}/${iso_filename}"
        log_warn "No se pudo extraer path remoto, usando fallback: $remote_iso_path"
    else
        log_debug "Path remoto del ISO: $remote_iso_path"
    fi
    
    # 4.2 Revertir snapshot (garantizar estado limpio antes de configurar hardware)
    log_info "Revirtiendo snapshot de la VM..."
    if ! vcenter_call revert_snapshot; then
        cleanup_on_error "Error al revertir snapshot" "deploy_snapshot"
        return 1
    fi

    # 4.3 Esperar POWERED_OFF (el revert apaga la VM)
    log_info "Esperando a que la VM esté apagada tras el revert..."
    if ! vcenter_call wait_powered_off; then
        cleanup_on_error "Timeout esperando POWERED_OFF tras revert" "deploy_snapshot_wait"
        return 1
    fi

    # 4.4 Configurar CD-ROM de la VM
    log_info "Configurando CD-ROM de la VM..."
    if ! vcenter_call configure_cdrom "$remote_iso_path"; then
        cleanup_on_error "Error configurando CD-ROM" "deploy_cdrom"
        return 1
    fi

    # 4.5 Encender VM
    log_info "Encendiendo VM..."
    if ! vcenter_call power_on; then
        cleanup_on_error "Error encendiendo VM" "deploy_power"
        return 1
    fi

    # 4.6 Esperar POWERED_ON antes del despliegue SSH
    log_info "Esperando a que la VM esté encendida..."
    if ! vcenter_call wait_powered_on; then
        cleanup_on_error "Timeout esperando POWERED_ON antes de SSH deploy" "deploy_power_wait"
        return 1
    fi

    # 4.7 Despliegue vía SSH
    log_info "Ejecutando despliegue en VM destino..."
    if ! "$SCRIPT_DIR/scripts/deploy.sh"; then
        cleanup_on_error "Error en despliegue SSH" "deploy_ssh"
        return 1
    fi
    
    log_ok "Despliegue completado"

    #---------------------------------------------------------------------------
    # FASE 6: Finalización y notificaciones
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[6/6] FINALIZACIÓN"
    log_info "───────────────────────────────────────────────────────────"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Actualizar BD
    db_query "UPDATE deployments SET status='success', completed_at=datetime('now'), 
              duration_seconds=$duration WHERE id=$deployment_id"
    
    # Marcar tag como completado en processed_tags (INSERT if not exists, UPDATE if exists)
    db_query "INSERT OR IGNORE INTO processed_tags (tag_name, status) VALUES ('$tag_sql', 'completed')" || true
    db_query "UPDATE processed_tags SET status='completed', processed_at=datetime('now') 
              WHERE tag_name='$tag_sql'" || {
        log_warn "No se pudo actualizar processed_tags para $tag"
    }
    
    # Notificaciones finales
    "$SCRIPT_DIR/scripts/notify.sh" both success "$tag" || log_warn "No se pudo enviar la notificación de éxito"
    
    # Desactivar trap
    trap - ERR
    
    log_info ""
    log_info "═══════════════════════════════════════════════════════════════════"
    log_ok "PIPELINE COMPLETADO EXITOSAMENTE"
    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "Tag: $tag"
    log_info "Duración: $(format_duration $duration)"
    log_info "Fin: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "═══════════════════════════════════════════════════════════════════"
    
    return 0
}

#===============================================================================
# Modo Daemon
#===============================================================================

run_daemon() {
    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "INICIANDO MODO DAEMON"
    log_info "═══════════════════════════════════════════════════════════════════"
    
    # Verificar instancia única
    if ! check_already_running; then
        exit 1
    fi
    
    # Crear PID file
    create_pid_file
    trap remove_pid_file EXIT
    
    # Trap para errores en el daemon (no matar el proceso, solo loguear)
    trap 'log_error "Error en daemon loop en línea $LINENO, continuando..."' ERR
    
    # Verificar entorno
    verify_environment || exit 1
    
    local polling_interval
    polling_interval=$(config_get "general.polling_interval_seconds" "300")
    
    log_info "Polling interval: ${polling_interval}s ($(format_duration $polling_interval))"
    log_info "PID: $$"
    log_info "Log: $LOG_FILE"
    log_info "═══════════════════════════════════════════════════════════════════"
    
    # Loop infinito con manejo robusto de errores
    while true; do
        log_info "───────────────────────────────────────────────────────────"
        log_info "Verificando nuevos tags... ($(date '+%H:%M:%S'))"

        # Todo el ciclo (recuperación, detección y pipeline) va bajo el lock:
        # si hay un pipeline manual en curso, se espera al siguiente ciclo.
        if acquire_pipeline_lock; then
            run_daemon_cycle || log_warn "El ciclo del daemon terminó con errores"
            release_pipeline_lock
        else
            log_warn "Hay un pipeline en curso (ejecución manual), se omite este ciclo"
        fi

        # Forzar flush de buffers antes de dormir
        sync 2>/dev/null || true

        log_debug "Próxima verificación en ${polling_interval}s..."
        sleep "$polling_interval"
    done
}

# Un ciclo del daemon. Se ejecuta con el lock del pipeline tomado.
run_daemon_cycle() {
    # Cerrar ejecuciones que quedaron a medias (reinicio, kill...) para que
    # sus tags entren en la política de reintentos en vez de bloquearse
    recover_interrupted_runs || log_warn "No se pudieron revisar las ejecuciones interrumpidas"

    # Detectar nuevo tag (logs van a stderr, solo el tag a stdout)
    local new_tag=""
    local detect_exit_code=0
    new_tag=$("$SCRIPT_DIR/scripts/git_monitor.sh" detect 2>&2) || detect_exit_code=$?

    # Limpiar espacios, saltos de línea y caracteres de control
    new_tag=$(echo "$new_tag" | tr -d '[:space:][:cntrl:]')

    if [[ $detect_exit_code -ne 0 ]]; then
        log_warn "git_monitor.sh detect falló con código $detect_exit_code, reintentando en siguiente ciclo..."
    elif [[ -z "$new_tag" ]]; then
        log_info "No hay tags nuevos"
    elif tag_is_valid "$new_tag"; then
        log_ok "Nuevo tag detectado: $new_tag"

        # Ejecutar pipeline aislado (resultado en PIPELINE_RESULT)
        run_pipeline_isolated "$new_tag" "daemon"
        local pipeline_result=$PIPELINE_RESULT

        if [[ $pipeline_result -eq 0 ]]; then
            log_ok "Pipeline completado para: $new_tag"
        else
            log_error "Pipeline fallido para: $new_tag (código: $pipeline_result)"
        fi
    elif [[ ${#new_tag} -le 200 ]]; then
        # Se marca como descartado para que no bloquee la detección de los
        # tags siguientes en cada ciclo.
        log_warn "git_monitor.sh devolvió un tag que no cumple git.tag_pattern ($TAG_PATTERN): $new_tag"
        log_warn "Se marca como 'skipped' (reprocesar a mano con: ci_cd.sh --tag <tag>, tras ajustar git.tag_pattern)"
        db_mark_tag_skipped "$new_tag" "No cumple git.tag_pattern" \
            || log_warn "No se pudo marcar el tag como 'skipped': $new_tag"
    else
        log_warn "Salida inesperada de git_monitor.sh detect (${#new_tag} caracteres), posible problema de captura"
    fi

    return 0
}

#===============================================================================
# Procesar tag manual
#===============================================================================

process_manual_tag() {
    local tag=$1
    
    if ! tag_is_valid "$tag"; then
        log_error "Tag no válido: '$tag' no cumple git.tag_pattern ($TAG_PATTERN)"
        return 1
    fi

    log_info "Procesando tag manualmente: $tag"

    # Verificar entorno
    verify_environment || exit 1

    if ! acquire_pipeline_lock; then
        log_error "Hay otro pipeline en curso (daemon o manual). Inténtalo cuando termine."
        return 1
    fi

    # Ejecutar pipeline aislado (resultado en PIPELINE_RESULT)
    run_pipeline_isolated "$tag" "manual"
    release_pipeline_lock
    if [[ $PIPELINE_RESULT -eq 0 ]]; then
        log_ok "Pipeline completado para: $tag"
        return 0
    else
        log_error "Pipeline fallido para: $tag"
        return 1
    fi
}

#===============================================================================
# Estado y utilidades
#===============================================================================

show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "                    ESTADO DEL PIPELINE CI/CD"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    
    # Verificar si hay daemon corriendo
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Daemon: EJECUTÁNDOSE (PID: $pid)"
        else
            echo "Daemon: DETENIDO (PID file obsoleto)"
        fi
    else
        echo "Daemon: NO INICIADO"
    fi
    
    echo ""
    echo "─── Últimos 5 despliegues ─────────────────────────────────────────"
    db_query_headers "SELECT tag_name, status, started_at, duration_seconds 
                      FROM deployments 
                      ORDER BY id DESC 
                      LIMIT 5" 2>/dev/null || echo "(sin datos)"
    
    echo ""
    echo "─── Estadísticas ──────────────────────────────────────────────────"
    db_query_headers "SELECT * FROM v_deployment_stats" 2>/dev/null || echo "(sin datos)"
    
    echo ""
    echo "─── Últimos resultados SonarQube ──────────────────────────────────"
    db_query_headers "SELECT tag, coverage, bugs, vulnerabilities, 
                             CASE WHEN passed=1 THEN 'PASS' ELSE 'FAIL' END as result
                      FROM sonar_results 
                      ORDER BY id DESC 
                      LIMIT 5" 2>/dev/null || echo "(sin datos)"
    
    echo ""
}

show_logs() {
    local lines=${1:-50}
    
    if [[ -f "$LOG_FILE" ]]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "No hay logs disponibles"
    fi
}

#===============================================================================
# Main
#===============================================================================

usage() {
    cat <<EOF
═══════════════════════════════════════════════════════════════════
                     CI/CD Pipeline - GALTTCMC
═══════════════════════════════════════════════════════════════════

Uso: $(basename "$0") <comando> [argumentos]

Comandos:
  daemon              Iniciar modo daemon (polling continuo)
  --tag <TAG>         Procesar tag específico manualmente
  status              Ver estado del pipeline
  logs [N]            Ver últimas N líneas de log (default: 50)
  init                Inicializar base de datos
  reset               Eliminar y recrear la base de datos (hace backup previo)
  verify              Verificar entorno y configuración
  help                Mostrar esta ayuda

Ejemplos:
  $(basename "$0") daemon                    # Iniciar daemon
  $(basename "$0") --tag V01_02_03_04        # Procesar tag manual
  $(basename "$0") status                    # Ver estado

Configuración:
  Config:    $CONFIG_FILE
  Base datos: $DB_PATH
  Logs:       $LOG_DIR

Para ejecutar como servicio systemd, ver: cicd.service

EOF
}

main() {
    # Crear directorio de logs
    mkdir -p "$LOG_DIR"
    
    local cmd="${1:-}"
    
    case "$cmd" in
        daemon)
            run_daemon
            ;;
        --tag|-t)
            if [[ -z "${2:-}" ]]; then
                log_error "Debe especificar un tag"
                echo "Uso: $0 --tag TAG_NAME"
                exit 1
            fi
            process_manual_tag "$2"
            ;;
        __run_pipeline)
            # Uso interno: lo lanza run_pipeline_isolated con el lock tomado
            if [[ "${CICD_PIPELINE_CHILD:-}" != "1" || -z "${2:-}" ]]; then
                log_error "__run_pipeline es de uso interno; usa: $0 --tag <TAG>"
                exit 1
            fi
            set -E
            run_pipeline "$2" "${3:-daemon}"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        init)
            init_database
            ;;
        reset)
            reset_database
            ;;
        verify)
            verify_environment
            ;;
        help|--help|-h|"")
            usage
            ;;
        *)
            log_error "Comando no reconocido: $cmd"
            usage
            exit 1
            ;;
    esac
}

main "$@"
