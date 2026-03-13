#!/bin/bash
#===============================================================================
# compile.sh - Gestión del proceso de compilación
#===============================================================================
# Fase 2 del pipeline CI/CD
# Prepara workspace, ejecuta build_DVDs.sh y valida resultado
#
# Uso:
#   ./compile.sh                    # Ejecutar compilación completa
#   ./compile.sh prepare            # Solo preparar workspace
#   ./compile.sh build              # Solo ejecutar build
#   ./compile.sh validate           # Solo validar resultado
#   ./compile.sh clean              # Limpiar directorio de compilación
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cargar funciones comunes
source "$SCRIPT_DIR/common.sh"

# Variables específicas de compilación
COMPILE_LOG_FILE="${LOG_DIR}/compile_$(date +%Y%m%d_%H%M%S).log"

#===============================================================================
# Preparación del workspace
#===============================================================================

# Limpiar directorio de compilación
clean_compile_dir() {
    log_info "Limpiando directorio de compilación: $COMPILE_PATH"
    
    if [[ -d "$COMPILE_PATH" ]]; then
        # Preservar algunos ficheros si es necesario
        rm -rf "${COMPILE_PATH:?}"/* 2>/dev/null || true
        rm -rf "${COMPILE_PATH:?}"/.[!.]* 2>/dev/null || true
        log_ok "Directorio limpiado"
    else
        log_info "Directorio no existe, creando..."
        mkdir -p "$COMPILE_PATH"
        log_ok "Directorio creado: $COMPILE_PATH"
    fi
}

# Preparar workspace: copiar código y asignar permisos
prepare_workspace() {
    local source_dir="$REPO_LOCAL_PATH"
    local target_dir="$COMPILE_PATH"
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "PREPARACIÓN DEL WORKSPACE"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Origen: $source_dir"
    log_info "Destino: $target_dir"
    
    # Verificar que existe el directorio origen
    if [[ ! -d "$source_dir" ]]; then
        log_error "Directorio origen no existe: $source_dir"
        log_error "Ejecute primero: git_monitor.sh checkout <tag>"
        return 1
    fi
    
    # Limpiar directorio destino
    clean_compile_dir
    
    # Copiar código fuente
    log_info "Copiando código fuente..."
    local start_time
    start_time=$(date +%s)
    
    if ! cp -r "$source_dir"/* "$target_dir"/ 2>&1 | tee -a "$COMPILE_LOG_FILE"; then
        log_error "Error copiando código fuente"
        return 1
    fi
    
    # Copiar ficheros ocultos también (excepto .git para ahorrar espacio)
    cp -r "$source_dir"/.[!.]* "$target_dir"/ 2>/dev/null || true
    rm -rf "$target_dir/.git" 2>/dev/null || true
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Calcular tamaño
    local size
    size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
    log_ok "Código copiado (${size}, ${duration}s)"
    
    # Asignar permisos de ejecución a scripts
    log_info "Asignando permisos de ejecución a scripts..."
    local script_count
    script_count=$(find "$target_dir" -name "*.sh" -type f 2>/dev/null | wc -l)
    
    find "$target_dir" -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
    
    log_ok "Permisos asignados a $script_count scripts"
    
    
    # Registrar fase en BD
    register_build_phase "prepare" 0 "$duration"
    
    log_ok "Workspace preparado correctamente"
    return 0
}

#===============================================================================
# Ejecución de compilación
#===============================================================================

# Ejecutar script de compilación
run_compilation() {
    local build_script="$BUILD_SCRIPT"
    local compile_dir="$COMPILE_PATH"
    local timeout_secs="${COMPILE_TIMEOUT:-3600}"
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "EJECUCIÓN DE COMPILACIÓN"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Directorio: $compile_dir"
    log_info "Script: $build_script"
    log_info "Timeout: ${timeout_secs}s ($(format_duration $timeout_secs))"
    
    # Verificar que existe el script de build
    local full_build_path="$compile_dir/$build_script"
    if [[ ! -f "$full_build_path" ]]; then
        log_error "Script de compilación no encontrado: $full_build_path"
        return 1
    fi
    
    # Asegurar permisos de ejecución
    chmod +x "$full_build_path"

    # Comentar líneas con 'read' en el script de build para evitar pausas interactivas
    log_info "Desactivando comandos 'read' en: $full_build_path"
    sed -i 's/^\([[:space:]]*read\b\)/#\1/' "$full_build_path"
    
    # Cambiar al directorio de compilación
    cd "$compile_dir" || {
        log_error "No se puede acceder a: $compile_dir"
        return 1
    }
    
    log_info "Iniciando compilación..."
    log_info "Log de compilación: $COMPILE_LOG_FILE"
    
    local start_time
    start_time=$(date +%s)
    local exit_code=0
    
    # Ejecutar con timeout y capturar salida
    # Usamos set +e para capturar el código de salida sin que el script falle
    set +e
    
    # Ejecutar el script de build
    # Redirigir stdout y stderr al log, y también mostrar en pantalla
    timeout "$timeout_secs" bash -c "
        cd '$compile_dir'
        './$build_script' 2>&1
    " 2>&1 | tee -a "$COMPILE_LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    set -e
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Registrar en BD
    register_build_phase "compile" "$exit_code" "$duration"
    
    # Interpretar código de salida
    case $exit_code in
        0)
            log_ok "Compilación completada exitosamente ($(format_duration $duration))"
            ;;
        124)
            log_error "Compilación TIMEOUT después de $(format_duration $timeout_secs)"
            return 1
            ;;
        *)
            log_error "Compilación FALLIDA con código de salida: $exit_code"
            log_error "Revise el log: $COMPILE_LOG_FILE"
            return 1
            ;;
    esac
    
    return 0
}

#===============================================================================
# Validación del resultado
#===============================================================================

# Validar que la compilación produjo el ISO esperado
validate_build() {
    local expected_iso="$OUTPUT_ISO"
    local compile_dir="$COMPILE_PATH"
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "VALIDACIÓN DE COMPILACIÓN"
    log_info "═══════════════════════════════════════════════════════════"
    
    # Buscar el ISO en el directorio de compilación
    local iso_path=""
    
    # Primero buscar en la ruta exacta
    if [[ -f "$compile_dir/$expected_iso" ]]; then
        iso_path="$compile_dir/$expected_iso"
    else
        # Buscar recursivamente
        log_info "Buscando $expected_iso en subdirectorios..."
        iso_path=$(find "$compile_dir" -name "$expected_iso" -type f 2>/dev/null | head -1)
    fi
    
    if [[ -z "$iso_path" || ! -f "$iso_path" ]]; then
        log_error "ISO no encontrado: $expected_iso"
        log_error "Directorios con ISOs encontrados:"
        find "$compile_dir" -name "*.iso" -type f 2>/dev/null | while read -r f; do
            log_error "  - $f"
        done || log_error "  (ninguno)"
        
        register_build_phase "package" 1 0
        return 1
    fi
    
    # Obtener información del ISO
    local iso_size
    local iso_md5
    iso_size=$(du -h "$iso_path" | cut -f1)
    iso_md5=$(md5sum "$iso_path" 2>/dev/null | cut -d' ' -f1 || echo "N/A")
    
    log_ok "ISO encontrado: $iso_path"
    log_info "  Tamaño: $iso_size"
    log_info "  MD5: $iso_md5"
    
    # Verificar que el ISO tiene contenido (no está vacío o corrupto)
    local iso_bytes
    iso_bytes=$(stat -c%s "$iso_path" 2>/dev/null || stat -f%z "$iso_path" 2>/dev/null || echo "0")
    
    if [[ "$iso_bytes" -lt 1048576 ]]; then  # Menor a 1MB probablemente es inválido
        log_warn "ISO sospechosamente pequeño: $iso_bytes bytes"
        log_warn "Verifique que la compilación fue correcta"
    fi
    
    # Guardar ruta del ISO para uso posterior
    export BUILT_ISO_PATH="$iso_path"
    echo "$iso_path" > "$compile_dir/.last_iso_path"
    
    # Registrar fase exitosa
    register_build_phase "package" 0 0
    
    log_ok "Validación completada: ISO listo para despliegue"
    return 0
}

# Obtener ruta del último ISO generado
get_last_iso_path() {
    local marker_file="$COMPILE_PATH/.last_iso_path"
    
    if [[ -f "$marker_file" ]]; then
        cat "$marker_file"
    else
        # Buscar el ISO más reciente
        find "$COMPILE_PATH" -name "*.iso" -type f -printf '%T@ %p\n' 2>/dev/null | \
            sort -n | tail -1 | cut -d' ' -f2-
    fi
}

#===============================================================================
# Registro en base de datos
#===============================================================================

# Registrar fase de build en SQLite
register_build_phase() {
    local phase=$1
    local exit_code=$2
    local duration=$3
    
    # Obtener tag actual (si está disponible)
    local current_tag
    current_tag=$(get_current_tag)
    
    if [[ -z "$current_tag" ]]; then
        current_tag="unknown"
    fi
    
    # Obtener deployment_id si existe
    local deployment_id
    deployment_id=$(db_query "SELECT id FROM deployments WHERE tag_name='$current_tag' ORDER BY id DESC LIMIT 1" 2>/dev/null || echo "")
    
    local start_time
    start_time=$(date +%s)
    start_time=$((start_time - duration))
    
    # Insertar registro
    if [[ -n "$deployment_id" ]]; then
        db_query "INSERT INTO build_logs (deployment_id, tag, phase, start_time, duration, exit_code, log_file)
                  VALUES ($deployment_id, '$current_tag', '$phase', $start_time, $duration, $exit_code, '$COMPILE_LOG_FILE')" 2>/dev/null || true
    else
        db_query "INSERT INTO build_logs (tag, phase, start_time, duration, exit_code, log_file)
                  VALUES ('$current_tag', '$phase', $start_time, $duration, $exit_code, '$COMPILE_LOG_FILE')" 2>/dev/null || true
    fi
    
    log_debug "Fase '$phase' registrada en BD (exit_code=$exit_code, duration=${duration}s)"
}

# Obtener tag actual del checkout
get_current_tag() {
    if [[ -d "$REPO_LOCAL_PATH" ]]; then
        cd "$REPO_LOCAL_PATH" 2>/dev/null || return
        git describe --tags --exact-match 2>/dev/null || echo ""
    fi
}

#===============================================================================
# Pipeline completo
#===============================================================================

# Ejecutar pipeline de compilación completo
run_full_build() {
    log_info "═══════════════════════════════════════════════════════════"
    log_info "PIPELINE DE COMPILACIÓN COMPLETO"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local total_start
    total_start=$(date +%s)
    
    # Fase 1: Preparar workspace
    log_info ""
    log_info "[1/3] Preparando workspace..."
    if ! prepare_workspace; then
        log_error "FALLO en preparación del workspace"
        return 1
    fi
    
    # Fase 2: Ejecutar compilación
    log_info ""
    log_info "[2/3] Ejecutando compilación..."
    if ! run_compilation; then
        log_error "FALLO en compilación"
        return 1
    fi
    
    # Fase 3: Validar resultado
    log_info ""
    log_info "[3/3] Validando resultado..."
    if ! validate_build; then
        log_error "FALLO en validación"
        return 1
    fi
    
    local total_end
    total_end=$(date +%s)
    local total_duration=$((total_end - total_start))
    
    log_info ""
    log_info "═══════════════════════════════════════════════════════════"
    log_ok "COMPILACIÓN COMPLETADA EXITOSAMENTE"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Duración total: $(format_duration $total_duration)"
    log_info "ISO generado: $(get_last_iso_path)"
    log_info "Log: $COMPILE_LOG_FILE"
    log_info "Fin: $(date '+%Y-%m-%d %H:%M:%S')"
    
    return 0
}

#===============================================================================
# Utilidades
#===============================================================================

# Mostrar estado de la última compilación
show_status() {
    log_info "Estado de compilaciones recientes:"
    echo ""
    
    db_query_headers "SELECT tag, phase, exit_code, duration, 
                             datetime(start_time, 'unixepoch', 'localtime') as started
                      FROM build_logs 
                      ORDER BY id DESC 
                      LIMIT 10" 2>/dev/null || echo "(sin datos)"
    
    echo ""
    echo "Último ISO generado:"
    local last_iso
    last_iso=$(get_last_iso_path)
    if [[ -n "$last_iso" && -f "$last_iso" ]]; then
        ls -lh "$last_iso"
    else
        echo "(ninguno)"
    fi
}

# Verificar prerequisitos
verify_prerequisites() {
    log_info "Verificando prerequisitos de compilación..."
    
    local errors=0
    
    # Verificar directorios
    if [[ -d "$REPO_LOCAL_PATH" ]]; then
        log_ok "Repositorio: $REPO_LOCAL_PATH"
    else
        log_error "Repositorio no existe: $REPO_LOCAL_PATH"
        ((errors++))
    fi
    
    # Verificar que hay un checkout válido
    if [[ -d "$REPO_LOCAL_PATH/.git" ]]; then
        local current_ref
        current_ref=$(cd "$REPO_LOCAL_PATH" && git describe --tags 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "")
        if [[ -n "$current_ref" ]]; then
            log_ok "Checkout actual: $current_ref"
        else
            log_warn "No se puede determinar ref actual"
        fi
    fi
    
    # Verificar script de build
    local build_path="$REPO_LOCAL_PATH/$BUILD_SCRIPT"
    if [[ -f "$build_path" ]]; then
        log_ok "Script de build: $BUILD_SCRIPT"
    else
        log_error "Script de build no encontrado: $build_path"
        ((errors++))
    fi
    
    # Verificar espacio en disco
    local available_space
    available_space=$(df -BG "$COMPILE_PATH" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")
    if [[ "$available_space" -gt 10 ]]; then
        log_ok "Espacio disponible: ${available_space}G"
    else
        log_warn "Espacio disponible bajo: ${available_space}G (recomendado >10G)"
    fi
    
    # Verificar comandos necesarios
    for cmd in timeout find chmod cp; do
        if command -v "$cmd" &>/dev/null; then
            log_debug "Comando disponible: $cmd"
        else
            log_error "Comando no encontrado: $cmd"
            ((errors++))
        fi
    done
    
    if [[ $errors -gt 0 ]]; then
        log_error "Se encontraron $errors errores"
        return 1
    fi
    
    log_ok "Prerequisitos verificados correctamente"
    return 0
}

#===============================================================================
# Main
#===============================================================================

usage() {
    cat <<EOF
Uso: $(basename "$0") [comando]

Comandos:
  (sin args)    Ejecutar pipeline completo (prepare + build + validate)
  prepare       Solo preparar workspace (copiar código, permisos)
  build         Solo ejecutar compilación
  validate      Solo validar resultado (buscar ISO)
  clean         Limpiar directorio de compilación
  status        Ver estado de compilaciones recientes
  verify        Verificar prerequisitos
  help          Mostrar esta ayuda

Ejemplos:
  $(basename "$0")              # Pipeline completo
  $(basename "$0") prepare      # Solo preparar
  $(basename "$0") build        # Solo compilar
  $(basename "$0") validate     # Solo validar

Variables de entorno usadas:
  REPO_LOCAL_PATH   Directorio del repositorio clonado
  COMPILE_PATH      Directorio de compilación
  BUILD_SCRIPT      Ruta relativa del script de build
  OUTPUT_ISO        Nombre del ISO esperado
  COMPILE_TIMEOUT   Timeout en segundos (default: 3600)

EOF
}

main() {
    local cmd="${1:-full}"
    
    # Crear directorio de logs si no existe
    mkdir -p "$LOG_DIR"
    
    case "$cmd" in
        full|"")
            run_full_build
            ;;
        prepare)
            prepare_workspace
            ;;
        build)
            run_compilation
            ;;
        validate)
            validate_build
            ;;
        clean)
            clean_compile_dir
            ;;
        status)
            show_status
            ;;
        verify)
            verify_prerequisites
            ;;
        help|--help|-h)
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
