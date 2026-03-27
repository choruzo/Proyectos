#!/bin/bash
#===============================================================================
# ci_cd.sh - Orquestador Principal del Pipeline CI/CD
#===============================================================================
# Script principal que coordina todas las fases del pipeline:
#   1. Monitorización de tags Git
#   2. Compilación
#   3. Análisis SonarQube
#   4. Despliegue en vCenter + VM
#   5. Notificaciones
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

# PID file para evitar múltiples instancias
PID_FILE="$SCRIPT_DIR/.cicd.pid"

#===============================================================================
# Verificación de instancia única
#===============================================================================

check_already_running() {
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log_error "Ya hay una instancia ejecutándose (PID: $old_pid)"
            log_error "Si crees que es un error, elimina: $PID_FILE"
            return 1
        else
            # Proceso antiguo ya no existe, limpiar PID file
            rm -f "$PID_FILE"
        fi
    fi
    return 0
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
    
    if sqlite3 "$DB_PATH" < "$init_sql" 2>&1; then
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
    local commands=(git sqlite3 python3)
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            log_debug "Comando OK: $cmd"
        else
            log_warn "Comando no encontrado: $cmd"
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
# Pipeline Principal
#===============================================================================

run_pipeline() {
    local tag=$1
    local triggered_by=${2:-"daemon"}
    local deployment_id=""
    
    local start_time
    start_time=$(date +%s)
    
    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "INICIANDO PIPELINE CI/CD"
    log_info "═══════════════════════════════════════════════════════════════════"
    log_info "Tag: $tag"
    log_info "Trigger: $triggered_by"
    log_info "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "═══════════════════════════════════════════════════════════════════"
    
    # Verificar si el tag ya existe en deployments
    local existing_deployment
    existing_deployment=$(db_query "SELECT id FROM deployments WHERE tag_name='$tag'" 2>/dev/null | head -n1 || echo "")
    
    if [[ -n "$existing_deployment" ]]; then
        log_warn "El tag '$tag' ya fue procesado anteriormente (deployment_id: $existing_deployment)"
        log_warn "Eliminando registro anterior para reprocesar..."
        
        # Eliminar registros anteriores
        db_query "DELETE FROM deployments WHERE tag_name='$tag'" 2>/dev/null || true
        db_query "DELETE FROM build_logs WHERE tag='$tag'" 2>/dev/null || true
        db_query "DELETE FROM sonar_results WHERE tag='$tag'" 2>/dev/null || true
        db_query "DELETE FROM processed_tags WHERE tag_name='$tag'" 2>/dev/null || true
        
        log_ok "Registros anteriores eliminados, continuando con reprocesamiento..."
    fi
    
    # Registrar inicio en BD
    deployment_id=$(db_query \
        "INSERT INTO deployments (tag_name, status, started_at, triggered_by) 
         VALUES ('$tag', 'pending', datetime('now'), '$triggered_by');
         SELECT last_insert_rowid();")
    
    if [[ -z "$deployment_id" || "$deployment_id" == "0" ]]; then
        log_error "Error crítico: No se pudo crear registro en deployments"
        return 1
    fi
    
    log_debug "Deployment ID: $deployment_id"
    
    # Función de cleanup en caso de error
    cleanup_on_error() {
        local error_msg=${1:-"Error desconocido"}
        local phase=${2:-"unknown"}
        
        log_error "════════════════════════════════════════════════════════"
        log_error "PIPELINE FALLIDO"
        log_error "════════════════════════════════════════════════════════"
        log_error "Tag: $tag"
        log_error "Fase: $phase"
        log_error "Error: $error_msg"
        
        # Actualizar BD
        db_query "UPDATE deployments SET status='failed', error_message='$error_msg', 
                  completed_at=datetime('now') WHERE id=$deployment_id" 2>/dev/null || true
        
        # Notificar
        "$SCRIPT_DIR/scripts/notify.sh" both failure "$tag" "$error_msg" 2>/dev/null || true
        
        return 1
    }
    
    # Trap para errores inesperados
    trap 'cleanup_on_error "Error inesperado en línea $LINENO" "unknown"' ERR
    
    #---------------------------------------------------------------------------
    # FASE 1: Checkout del tag
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[1/5] CHECKOUT DEL TAG"
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
    log_info "[2/5] COMPILACIÓN"
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
    log_info "[3/5] ANÁLISIS SONARQUBE"
    log_info "───────────────────────────────────────────────────────────"
    
    db_query "UPDATE deployments SET status='analyzing' WHERE id=$deployment_id"
    
    # Obtener rutas de configuración
    local compile_path
    compile_path=$(config_get "git.compile_path" "/home/YOUR_USER/compile")
    
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
    chmod +x "$compile_path/utils/build-wrapper-linux-x86/build-wrapper-linux-x86-64" 2>/dev/null || true
    chmod +x "$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/bin/sonar-scanner" 2>/dev/null || true
    chmod +x "$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/jre/bin/java" 2>/dev/null || true
    export JAVA_HOME=/usr/lib64/jvm/java-21-openjdk-21
    log_ok "Permisos configurados"
    
    # Rutas locales en el directorio de compilación
    local build_wrapper="$compile_path/utils/build-wrapper-linux-x86/build-wrapper-linux-x86-64"
    local sonar_scanner="$compile_path/utils/sonar-scanner-7.2.0.5079-linux-x64/bin/sonar-scanner"
    local bw_output_dir="$compile_path/bw-output"
    local compile_all_script="Development_TTCF/ttcf/utils/makefile/compile_all.sh"
    
    # PASO 1: Ejecutar compilación con build-wrapper para análisis C/C++
    log_info "Ejecutando build-wrapper para capturar compilación C/C++..."
    
    if [[ ! -f "$build_wrapper" ]]; then
        log_error "build-wrapper no encontrado en: $build_wrapper"
        cleanup_on_error "build-wrapper no disponible" "sonarqube_prepare"
        return 1
    fi
    
    mkdir -p "$bw_output_dir"
    
    cd "$compile_path" || {
        log_error "No se puede acceder a: $compile_path"
        cleanup_on_error "Directorio de compilación no accesible" "sonarqube_prepare"
        return 1
    }
    
    # Verificar que existe el script de compilación
    if [[ ! -f "$compile_path/$compile_all_script" ]]; then
        log_error "Script de compilación no encontrado: $compile_path/$compile_all_script"
        cleanup_on_error "compile_all.sh no encontrado" "sonarqube_prepare"
        return 1
    fi
    
    chmod +x "$compile_path/$compile_all_script"
    
    log_info "Ejecutando: $build_wrapper --out-dir $bw_output_dir $compile_all_script"
    
    if ! "$build_wrapper" --out-dir "$bw_output_dir" "$compile_path/$compile_all_script" 2>&1 | tee -a "$LOG_FILE"; then
        log_error "build-wrapper falló durante la compilación"
        cleanup_on_error "Error en build-wrapper" "sonarqube_prepare"
        return 1
    fi
    
    log_ok "build-wrapper completado"
    
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
    
    log_info "Ejecutando: $sonar_scanner con proyecto GALTTCMC_interno"
    log_info "Directorio de trabajo: $compile_path"
    log_info "Configuración: sonar-project.properties"
    
    if ! $JAVA_HOME/bin/java -jar /home/YOUR_USER/cicd/utils/sonar-scanner-7.2.0.5079-linux-x64/lib/sonar-scanner-cli-7.2.0.5079.jar  -Dproject.settings=sonar-project.properties -Dsonar.projectKey=GALTTCMC_interno -Dsonar.projectName=GALTTCMC_interno -Dsonar.branch.name=V08_00_00_00 -Dsonar.projectVersion=V08_00_00_00 \
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
    # FASE 4: Despliegue en vCenter + VM
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[4/6] DESPLIEGUE"
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
    upload_output=$(python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" upload_iso "$iso_path" 2>&1)
    local upload_status=$?
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
        datastore=$(config_get "vcenter.datastore" "YOUR_DATASTORE")
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
    if ! python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" revert_snapshot; then
        cleanup_on_error "Error al revertir snapshot" "deploy_snapshot"
        return 1
    fi

    # 4.3 Esperar POWERED_OFF (el revert apaga la VM)
    log_info "Esperando a que la VM esté apagada tras el revert..."
    if ! python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" wait_powered_off; then
        cleanup_on_error "Timeout esperando POWERED_OFF tras revert" "deploy_snapshot_wait"
        return 1
    fi

    # 4.4 Configurar CD-ROM de la VM
    log_info "Configurando CD-ROM de la VM..."
    if ! python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" configure_cdrom "$remote_iso_path"; then
        cleanup_on_error "Error configurando CD-ROM" "deploy_cdrom"
        return 1
    fi

    # 4.5 Encender VM
    log_info "Encendiendo VM..."
    if ! python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" power_on; then
        cleanup_on_error "Error encendiendo VM" "deploy_power"
        return 1
    fi

    # 4.6 Esperar POWERED_ON antes del despliegue SSH
    log_info "Esperando a que la VM esté encendida..."
    if ! python3 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" wait_powered_on; then
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
    # FASE 5: Generación de checksums
    #---------------------------------------------------------------------------
    log_info ""
    log_info "[5/6] GENERACIÓN DE CHECKSUMS"
    log_info "───────────────────────────────────────────────────────────"

    log_info "Generando checksums en: $compile_path"

    (
        cd "$compile_path" || {
            log_error "No se puede acceder a: $compile_path"
            exit 1
        }

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
    ) || {
        cleanup_on_error "Error generando checksums en $compile_path" "checksums"
        return 1
    }

    log_ok "Checksums generados correctamente en $compile_path"

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
    db_query "INSERT OR IGNORE INTO processed_tags (tag_name, status) VALUES ('$tag', 'completed')" 2>/dev/null || true
    db_query "UPDATE processed_tags SET status='completed', processed_at=datetime('now') 
              WHERE tag_name='$tag'" 2>/dev/null || {
        log_warn "No se pudo actualizar processed_tags para $tag"
    }
    
    # Notificaciones finales
    "$SCRIPT_DIR/scripts/notify.sh" both success "$tag"
    
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
        
        # Detectar nuevo tag (logs van a stderr, solo el tag a stdout)
        # Capturar también el código de salida para detectar errores
        local new_tag=""
        local detect_exit_code=0
        
        # Redirigir stderr del comando a stderr del script explícitamente
        # y capturar solo stdout
        new_tag=$("$SCRIPT_DIR/scripts/git_monitor.sh" detect 2>&2) || detect_exit_code=$?
        
        # Limpiar espacios, saltos de línea y caracteres de control
        new_tag=$(echo "$new_tag" | tr -d '[:space:][:cntrl:]')
        
        if [[ $detect_exit_code -ne 0 ]]; then
            log_warn "git_monitor.sh detect falló con código $detect_exit_code, reintentando en siguiente ciclo..."
        elif [[ -n "$new_tag" && "$new_tag" =~ ^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$ ]]; then
            log_ok "Nuevo tag detectado: $new_tag"
            
            # Ejecutar pipeline con manejo de errores
            set +e  # Temporalmente desactivar exit on error
            run_pipeline "$new_tag" "daemon"
            local pipeline_result=$?
            set -e  # Reactivar exit on error
            
            if [[ $pipeline_result -eq 0 ]]; then
                log_ok "Pipeline completado para: $new_tag"
            else
                log_error "Pipeline fallido para: $new_tag (código: $pipeline_result)"
            fi
        elif [[ -n "$new_tag" ]]; then
            # Solo debug si la salida parece tener contenido sospechoso
            if [[ ${#new_tag} -lt 100 ]]; then
                log_debug "Salida no es un tag válido: $new_tag"
            else
                log_debug "Salida inesperada (${#new_tag} caracteres), posible problema de captura"
            fi
        else
            log_info "No hay tags nuevos"
        fi
        
        # Forzar flush de buffers antes de dormir
        sync 2>/dev/null || true
        
        log_debug "Próxima verificación en ${polling_interval}s..."
        sleep "$polling_interval"
    done
}

#===============================================================================
# Procesar tag manual
#===============================================================================

process_manual_tag() {
    local tag=$1
    
    log_info "Procesando tag manualmente: $tag"
    
    # Verificar entorno
    verify_environment || exit 1
    
    # Ejecutar pipeline
    if run_pipeline "$tag" "manual"; then
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
