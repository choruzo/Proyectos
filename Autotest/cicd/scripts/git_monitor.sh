#!/bin/bash
#===============================================================================
# git_monitor.sh - Monitorización de tags Git
#===============================================================================
# Fase 1 del pipeline CI/CD
# Detecta nuevos tags en el repositorio remoto y realiza checkout
#
# Uso:
#   ./git_monitor.sh detect          # Detectar y mostrar nuevo tag (si existe)
#   ./git_monitor.sh checkout <tag>  # Hacer checkout de un tag específico
#   ./git_monitor.sh list            # Listar todos los tags remotos
#   ./git_monitor.sh status          # Ver estado de tags procesados
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cargar funciones comunes
source "$SCRIPT_DIR/common.sh"

#===============================================================================
# Funciones de obtención de tags
#===============================================================================

# Obtener todos los tags remotos que coinciden con el patrón configurado
# Devuelve lista ordenada por fecha de creación real (más reciente primero)
# Si existe repo local usa 'git for-each-ref --sort=-creatordate' (más fiable);
# si no, usa 'git ls-remote' + 'sort -V' como fallback.
get_remote_tags() {
    local repo_url="$GIT_REPO_URL"
    local pattern="$TAG_PATTERN"
    
    log_debug "Consultando tags remotos de: $repo_url"
    log_debug "Patrón de filtrado: $pattern"
    
    # Si existe el repositorio local clonado, ordenar por fecha real de creación
    # Esto evita que tags con sufijos (_IntermediateVersion, etc.) ordenen
    # como "más nuevos" por versión cuando en realidad son anteriores.
    if [[ -d "${REPO_LOCAL_PATH:-}/.git" ]]; then
        # Traer los tags del remoto antes de listarlos: sin esto, los tags
        # creados después del último checkout no se verían nunca.
        # stdout de git va a stderr: la salida de 'detect' se captura como valor.
        log_debug "Actualizando tags del repo local (git fetch --tags)..."
        if GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$REPO_LOCAL_PATH" fetch --tags --force --quiet origin 1>&2; then
            log_debug "Repo local disponible, ordenando por fecha de creación..."
            local local_tags
            local_tags=$(git -C "$REPO_LOCAL_PATH" for-each-ref \
                --sort=-creatordate \
                --format='%(refname:short)' \
                'refs/tags/' 2>/dev/null | \
                grep -E "$pattern" || echo "")
            if [[ -n "$local_tags" ]]; then
                echo "$local_tags"
                return 0
            fi
            log_debug "No se encontraron tags en repo local, consultando remoto..."
        else
            log_warn "git fetch de tags falló en $REPO_LOCAL_PATH, usando git ls-remote (orden por versión)"
        fi
    fi
    
    # Fallback: obtener tags remotos via git ls-remote
    local all_tags
    if [[ -n "${GIT_PASSWORD:-}" ]]; then
        local auth_url
        auth_url=$(echo "$repo_url" | sed "s|https://|https://${GIT_USERNAME:-agent}:${GIT_PASSWORD}@|")
        all_tags=$(git ls-remote --tags "$auth_url" 2>/dev/null || echo "")
    else
        all_tags=$(git ls-remote --tags "$repo_url" 2>/dev/null || echo "")
    fi
    
    if [[ -z "$all_tags" ]]; then
        log_warn "No se pudieron obtener tags remotos o el repositorio está vacío"
        return 1
    fi
    
    # Filtrar por patrón y extraer nombres (orden por versión como fallback)
    echo "$all_tags" | \
        grep -v '\^{}' | \
        awk '{print $2}' | \
        sed 's|refs/tags/||' | \
        grep -E "$pattern" | \
        sort -V -r
}

# Obtener tags ya procesados exitosamente desde SQLite
get_processed_tags() {
    log_debug "Consultando tags procesados en BD..."
    
    # Tags con status 'success' o 'completed'
    db_query "SELECT tag_name FROM deployments WHERE status='success'
              UNION
              SELECT tag_name FROM processed_tags WHERE status='completed'" 2>/dev/null || echo ""
}

# Obtener tags pendientes o en proceso
get_pending_tags() {
    db_query "SELECT tag_name FROM deployments WHERE status IN ('pending', 'compiling', 'analyzing', 'deploying')
              UNION
              SELECT tag_name FROM processed_tags WHERE status='processing'" 2>/dev/null || echo ""
}

#===============================================================================
# Detección de nuevos tags
#===============================================================================

# Detectar el primer tag nuevo no procesado
# Devuelve el nombre del tag si hay uno nuevo, vacío si no hay
detect_new_tag() {
    log_info "Iniciando detección de nuevos tags..."

    db_ensure_schema || log_warn "No se pudo verificar el schema de processed_tags"

    # Obtener tags remotos
    local remote_tags
    remote_tags=$(get_remote_tags) || {
        log_error "Error obteniendo tags remotos"
        return 1
    }
    
    if [[ -z "$remote_tags" ]]; then
        log_info "No hay tags remotos que coincidan con el patrón"
        return 0
    fi
    
    # Obtener tags ya procesados
    local processed_tags
    processed_tags=$(get_processed_tags)
    
    # Obtener tags en proceso (para no lanzarlos dos veces)
    local pending_tags
    pending_tags=$(get_pending_tags)

    local max_attempts retry_delay
    max_attempts=$(tag_max_attempts)
    retry_delay=$(config_get "general.retry_delay_seconds" "1800")
    [[ "$retry_delay" =~ ^[0-9]+$ ]] || retry_delay=1800

    # La lista remote_tags está ordenada de más reciente a más antiguo:
    #  - Al llegar al primer tag ya procesado se para: todo lo que sigue es
    #    más antiguo y nunca debe desplegarse automáticamente.
    #  - Si el candidato está en curso o esperando reintento, se espera en
    #    lugar de saltar a un tag más antiguo.
    #  - Un tag descartado ('skipped', agotó los reintentos) se salta.
    local new_tag="" tag info status attempts elapsed
    while IFS= read -r tag; do
        if [[ -z "$tag" ]]; then
            continue
        fi

        if grep -Fxq -- "$tag" <<< "$processed_tags"; then
            log_debug "Alcanzado el último tag procesado ($tag), no hay tags más recientes pendientes"
            break
        fi

        if grep -Fxq -- "$tag" <<< "$pending_tags"; then
            log_info "Tag en proceso: $tag, se espera a que termine"
            break
        fi

        # Estado de reintentos del tag (vacío si nunca se ha procesado)
        info=$(db_query "SELECT status || '|' || attempts || '|' ||
                                CAST(strftime('%s','now') - strftime('%s', COALESCE(processed_at, first_seen_at)) AS INTEGER)
                         FROM processed_tags WHERE tag_name='$(sql_escape "$tag")'" 2>/dev/null || echo "")
        if [[ -n "$info" ]]; then
            IFS='|' read -r status attempts elapsed <<< "$info"
            attempts=${attempts:-0}
            elapsed=${elapsed:-0}

            if [[ "$status" == "skipped" ]]; then
                log_debug "Tag descartado tras $attempts intento(s): $tag (reprocesar con: ci_cd.sh --tag $tag)"
                continue
            fi

            if [[ "$attempts" -gt 0 && "$elapsed" -lt "$retry_delay" ]]; then
                log_info "Tag $tag falló (intento $attempts/$max_attempts), próximo reintento en $(format_duration $((retry_delay - elapsed)))"
                break
            fi

            if [[ "$attempts" -gt 0 ]]; then
                log_info "Reintentando tag $tag (intento $((attempts + 1))/$max_attempts)"
            fi
        fi

        new_tag="$tag"
        break
    done <<< "$remote_tags"
    
    if [[ -n "$new_tag" ]]; then
        log_ok "Nuevo tag detectado: $new_tag"
        echo "$new_tag"
        return 0
    else
        log_info "No hay tags nuevos pendientes de procesar"
        return 0
    fi
}

# Listar todos los tags remotos (para diagnóstico)
list_remote_tags() {
    log_info "Listando tags remotos..."
    
    local tags
    tags=$(get_remote_tags) || {
        log_error "Error obteniendo tags"
        return 1
    }
    
    if [[ -z "$tags" ]]; then
        echo "No se encontraron tags que coincidan con el patrón: $TAG_PATTERN"
        return 0
    fi
    
    echo "Tags remotos encontrados (patrón: $TAG_PATTERN):"
    echo "================================================"
    
    local processed
    processed=$(get_processed_tags)
    
    while IFS= read -r tag; do
        if [[ -z "$tag" ]]; then
            continue
        fi
        
        if echo "$processed" | grep -Fxq "$tag"; then
            echo "  [✓] $tag (procesado)"
        else
            echo "  [ ] $tag (pendiente)"
        fi
    done <<< "$tags"
}

#===============================================================================
# Checkout de tags
#===============================================================================

# Hacer checkout de un tag específico
# Prepara el código fuente para compilación
checkout_tag() {
    local tag=$1
    
    if [[ -z "$tag" ]]; then
        log_error "Debe especificar un tag"
        return 1
    fi
    
    log_info "Iniciando checkout del tag: $tag"
    
    local repo_path="$REPO_LOCAL_PATH"
    
    # Verificar si el directorio del repo existe
    if [[ ! -d "$repo_path" ]]; then
        log_info "Directorio del repositorio no existe, clonando..."
        clone_repository || return 1
    fi
    
    cd "$repo_path" || {
        log_error "No se puede acceder al directorio: $repo_path"
        return 1
    }
    
    # Limpiar cambios locales
    log_debug "Limpiando cambios locales..."
    git reset --hard HEAD 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    
    # Actualizar referencias remotas
    log_info "Actualizando referencias del repositorio..."
    if ! git fetch --all --tags --prune 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Error actualizando repositorio"
        return 1
    fi
    
    # Verificar que el tag existe
    if ! git tag -l "$tag" | grep -Fxq "$tag"; then
        # Intentar con refs/tags/
        if ! git rev-parse "refs/tags/$tag" &>/dev/null; then
            log_error "El tag no existe en el repositorio: $tag"
            return 1
        fi
    fi
    
    # Hacer checkout del tag
    log_info "Checkout del tag: $tag"
    if ! git checkout "tags/$tag" -f 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Error en checkout del tag: $tag"
        return 1
    fi
    
    # Registrar en BD como pendiente
    register_tag_pending "$tag"
    
    # Verificar estado
    local current_ref
    current_ref=$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)
    log_ok "Checkout completado. Ref actual: $current_ref"
    
    # Mostrar info del tag
    log_info "Información del tag:"
    git --no-pager log -1 --format="  Commit: %H%n  Autor: %an <%ae>%n  Fecha: %ai%n  Mensaje: %s"
    
    return 0
}

# Clonar repositorio si no existe
clone_repository() {
    local repo_url="$GIT_REPO_URL"
    local repo_path="$REPO_LOCAL_PATH"
    local branch="$GIT_BRANCH"
    
    log_info "Clonando repositorio: $repo_url"
    log_info "Directorio destino: $repo_path"
    
    # Crear directorio padre si no existe
    mkdir -p "$(dirname "$repo_path")"
    
    # Construir URL con credenciales si están disponibles
    local clone_url="$repo_url"
    if [[ -n "${GIT_PASSWORD:-}" ]]; then
        clone_url=$(echo "$repo_url" | sed "s|https://|https://${GIT_USERNAME:-agent}:${GIT_PASSWORD}@|")
    fi
    
    # Clonar repositorio
    if ! git clone --branch "$branch" "$clone_url" "$repo_path" 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Error clonando repositorio"
        return 1
    fi
    
    log_ok "Repositorio clonado correctamente"
    return 0
}

#===============================================================================
# Registro en base de datos
#===============================================================================

# Registrar tag como pendiente de procesar
register_tag_pending() {
    local tag=$1
    
    # Insertar en processed_tags si no existe
    db_query "INSERT OR IGNORE INTO processed_tags (tag_name, status) 
              VALUES ('$tag', 'processing')" 2>/dev/null || true
    
    # Actualizar estado si ya existía
    db_query "UPDATE processed_tags SET status='processing', processed_at=datetime('now') 
              WHERE tag_name='$tag'" 2>/dev/null || true
    
    log_debug "Tag registrado como pendiente: $tag"
}

# Marcar tag como completado
mark_tag_completed() {
    local tag=$1
    
    db_query "UPDATE processed_tags SET status='completed', processed_at=datetime('now') 
              WHERE tag_name='$tag'" 2>/dev/null || true
    
    log_debug "Tag marcado como completado: $tag"
}

# Marcar tag como fallido/saltado
mark_tag_skipped() {
    local tag=$1
    local reason=${2:-"skipped"}
    
    db_query "UPDATE processed_tags SET status='skipped' 
              WHERE tag_name='$tag'" 2>/dev/null || true
    
    log_debug "Tag marcado como saltado: $tag ($reason)"
}

#===============================================================================
# Estado y diagnóstico
#===============================================================================

# Mostrar estado de tags procesados
show_status() {
    log_info "Estado de tags procesados:"
    echo ""
    
    echo "=== Últimos 10 deployments ==="
    db_query_headers "SELECT tag_name, status, started_at, duration_seconds 
                      FROM deployments 
                      ORDER BY id DESC 
                      LIMIT 10" 2>/dev/null || echo "(sin datos)"
    
    echo ""
    echo "=== Tags en processed_tags ==="
    db_query_headers "SELECT tag_name, status, first_seen_at, processed_at 
                      FROM processed_tags 
                      ORDER BY id DESC 
                      LIMIT 10" 2>/dev/null || echo "(sin datos)"
}

# Verificar configuración
verify_config() {
    log_info "Verificando configuración de git_monitor..."
    
    local errors=0
    
    # Verificar variables requeridas
    if [[ -z "${GIT_REPO_URL:-}" ]]; then
        log_error "GIT_REPO_URL no configurado"
        ((errors++))
    else
        log_ok "GIT_REPO_URL: $GIT_REPO_URL"
    fi
    
    if [[ -z "${TAG_PATTERN:-}" ]]; then
        log_error "TAG_PATTERN no configurado"
        ((errors++))
    else
        log_ok "TAG_PATTERN: $TAG_PATTERN"
    fi
    
    if [[ -z "${REPO_LOCAL_PATH:-}" ]]; then
        log_error "REPO_LOCAL_PATH no configurado"
        ((errors++))
    else
        log_ok "REPO_LOCAL_PATH: $REPO_LOCAL_PATH"
    fi
    
    # Verificar comandos requeridos
    for cmd in git sqlite3; do
        if command -v "$cmd" &>/dev/null; then
            log_ok "Comando disponible: $cmd"
        else
            log_error "Comando no encontrado: $cmd"
            ((errors++))
        fi
    done
    
    # Verificar BD
    if [[ -f "$DB_PATH" ]]; then
        log_ok "Base de datos: $DB_PATH"
    else
        log_warn "Base de datos no existe: $DB_PATH (se creará automáticamente)"
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Se encontraron $errors errores de configuración"
        return 1
    fi
    
    log_ok "Configuración verificada correctamente"
    return 0
}

#===============================================================================
# Main
#===============================================================================

usage() {
    cat <<EOF
Uso: $(basename "$0") <comando> [argumentos]

Comandos:
  detect              Detectar nuevo tag pendiente (devuelve nombre si existe)
  checkout <tag>      Hacer checkout de un tag específico
  list                Listar todos los tags remotos
  status              Ver estado de tags procesados
  verify              Verificar configuración

Ejemplos:
  $(basename "$0") detect
  $(basename "$0") checkout V01_02_03_04
  $(basename "$0") list

EOF
}

main() {
    local cmd="${1:-}"
    
    case "$cmd" in
        detect)
            detect_new_tag
            ;;
        checkout)
            if [[ -z "${2:-}" ]]; then
                log_error "Debe especificar un tag"
                echo "Uso: $0 checkout <tag>"
                exit 1
            fi
            checkout_tag "$2"
            ;;
        list)
            list_remote_tags
            ;;
        status)
            show_status
            ;;
        verify)
            verify_config
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
