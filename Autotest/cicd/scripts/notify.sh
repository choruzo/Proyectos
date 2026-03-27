#!/bin/bash
#===============================================================================
# notify.sh - Sistema de notificaciones
#===============================================================================
# Fase 5 del pipeline CI/CD
# Notificaciones wall y actualización de /etc/profile.d/informacion.sh
#
# Uso:
#   ./notify.sh wall <tipo> <tag> [detalles]
#   ./notify.sh profile <tag> <estado>
#   ./notify.sh both <tipo> <tag> [detalles]
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cargar funciones comunes
source "$SCRIPT_DIR/common.sh"

# Configuración de notificaciones
PROFILE_SCRIPT="${PROFILE_SCRIPT:-/etc/profile.d/informacion.sh}"

#===============================================================================
# Notificación wall (broadcast a usuarios conectados)
#===============================================================================

notify_wall() {
    local message_type=$1
    local tag=$2
    local details=${3:-""}
    
    local wall_enabled
    wall_enabled=$(config_get "notifications.wall_enabled" "true")
    
    if [[ "$wall_enabled" != "true" ]]; then
        log_debug "Notificaciones wall deshabilitadas"
        return 0
    fi
    
    log_info "Enviando notificación wall: $message_type"
    
    local message=""
    
    case $message_type in
        success)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          NUEVA VERSION DESPLEGADA CORRECTAMENTE               #
#                                                               #
#   Tag: $tag
#   Estado: DESPLEGADO OK
#   Fecha: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#   La nueva version esta disponible en la VM Releases.         #
#   Puedes clonar la maquina o revisar el informe SonarQube.    #
#                                                               #
#################################################################

EOF
)
            ;;
        failure)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          ALERTA: FALLO EN PIPELINE CI/CD                      #
#                                                               #
#   Tag: $tag
#   Error: $details
#   Fecha: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#   Revisa los logs en:                                         #
#   /home/YOUR_USER/cicd/logs/                                      #
#                                                               #
#################################################################

EOF
)
            ;;
        sonar_failed)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          ALERTA: QUALITY GATE NO SUPERADO                     #
#                                                               #
#   Tag: $tag
#   Proyecto: $(config_get "sonarqube.project_key" "GALTTCMC")
#   Fecha: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#   El analisis de codigo no cumple los umbrales.               #
#   Revisa el informe en: $(config_get "sonarqube.url")
#                                                               #
#################################################################

EOF
)
            ;;
        compiling)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          COMPILACION EN PROGRESO                              #
#                                                               #
#   Tag: $tag
#   Inicio: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#   Se ha detectado un nuevo tag y se esta compilando.          #
#   Recibiras otra notificacion cuando termine.                 #
#                                                               #
#################################################################

EOF
)
            ;;
        deploying)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          DESPLIEGUE EN PROGRESO                               #
#                                                               #
#   Tag: $tag
#   Fecha: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#   Compilacion exitosa. Iniciando despliegue en VM Releases.   #
#                                                               #
#################################################################

EOF
)
            ;;
        *)
            message=$(cat <<EOF

#################################################################
#                                                               #
#          NOTIFICACION CI/CD                                   #
#                                                               #
#   Tipo: $message_type
#   Tag: $tag
#   Detalles: $details
#   Fecha: $(date '+%Y-%m-%d %H:%M:%S')
#                                                               #
#################################################################

EOF
)
            ;;
    esac
    
    # Enviar via wall
    if command -v wall &>/dev/null; then
        echo "$message" | sudo wall 2>/dev/null || echo "$message" | wall 2>/dev/null || {
            log_warn "No se pudo enviar notificación wall"
            echo "$message"
        }
        log_ok "Notificación wall enviada"
    else
        log_warn "Comando 'wall' no disponible"
        echo "$message"
    fi
}

#===============================================================================
# Actualizar script de profile.d
#===============================================================================

update_profile_script() {
    local tag=$1
    local status=$2
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_info "Actualizando script de información: $PROFILE_SCRIPT"
    
    # Obtener información adicional
    local sonar_url
    sonar_url=$(config_get "sonarqube.url" "https://YOUR_SONARQUBE_SERVER")
    local project_key
    project_key=$(config_get "sonarqube.project_key" "GALTTCMC")
    
    # Determinar color del estado
    local status_color=""
    local status_icon=""
    case $status in
        DESPLEGADO|OK|SUCCESS)
            status_color='\033[0;32m'  # Verde
            status_icon="[OK]"
            ;;
        FALLIDO|ERROR|FAILED)
            status_color='\033[0;31m'  # Rojo
            status_icon="[!!]"
            ;;
        COMPILANDO|EN_PROCESO)
            status_color='\033[0;33m'  # Amarillo
            status_icon="[..]"
            ;;
        *)
            status_color='\033[0;34m'  # Azul
            status_icon="[--]"
            ;;
    esac
    
    # Crear contenido del script
    local script_content
    script_content=$(cat <<'SCRIPT_HEADER'
#!/bin/bash
# =============================================================================
# Información de última versión CI/CD
# Generado automáticamente por el pipeline CI/CD
# =============================================================================

# Solo mostrar en sesiones interactivas
[[ $- == *i* ]] || return

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_HEADER
)

    script_content+=$(cat <<EOF

# Datos de la última versión
LAST_TAG="$tag"
LAST_STATUS="$status"
LAST_TIMESTAMP="$timestamp"
SONAR_URL="$sonar_url"
PROJECT_KEY="$project_key"

# Mostrar información
echo ""
echo -e "\${CYAN}+==============================================================+\${NC}"
echo -e "\${CYAN}|           INFORMACIÓN DE ÚLTIMA VERSIÓN CI/CD                |\${NC}"
echo -e "\${CYAN}+==============================================================+\${NC}"
echo -e "\${CYAN}|\${NC}  Tag desplegado:  \${GREEN}\${LAST_TAG}\${NC}"
echo -e "\${CYAN}|\${NC}  Estado:          $status_icon \${LAST_STATUS}"
echo -e "\${CYAN}|\${NC}  Fecha:           \${LAST_TIMESTAMP}"
echo -e "\${CYAN}|\${NC}"
echo -e "\${CYAN}|\${NC}  Logs:      /home/YOUR_USER/cicd/logs/"
echo -e "\${CYAN}|\${NC}  SonarQube: \${SONAR_URL}/dashboard?id=\${PROJECT_KEY}"
echo -e "\${CYAN}+==============================================================+\${NC}"
echo ""
EOF
)

    # Escribir el script
    if echo "$script_content" | sudo tee "$PROFILE_SCRIPT" > /dev/null 2>&1; then
        sudo chmod +x "$PROFILE_SCRIPT" 2>/dev/null || true
        log_ok "Script de información actualizado: $PROFILE_SCRIPT"
    else
        # Intentar sin sudo
        if echo "$script_content" > "$PROFILE_SCRIPT" 2>/dev/null; then
            chmod +x "$PROFILE_SCRIPT" 2>/dev/null || true
            log_ok "Script de información actualizado: $PROFILE_SCRIPT"
        else
            log_warn "No se pudo actualizar $PROFILE_SCRIPT (permisos insuficientes)"
            log_info "Contenido que se intentó escribir:"
            echo "$script_content"
        fi
    fi
}

#===============================================================================
# Notificación completa (wall + profile)
#===============================================================================

notify_both() {
    local message_type=$1
    local tag=$2
    local details=${3:-""}
    
    # Mapear tipo de mensaje a estado para profile
    local status
    case $message_type in
        success)
            status="DESPLEGADO"
            ;;
        failure)
            status="FALLIDO"
            ;;
        sonar_failed)
            status="QUALITY_GATE_FALLIDO"
            ;;
        compiling)
            status="COMPILANDO"
            ;;
        deploying)
            status="DESPLEGANDO"
            ;;
        *)
            status="$message_type"
            ;;
    esac
    
    # Enviar wall
    notify_wall "$message_type" "$tag" "$details"
    
    # Actualizar profile (solo en success/failure)
    local notify_on_success
    local notify_on_failure
    notify_on_success=$(config_get "notifications.notify_on_success" "true")
    notify_on_failure=$(config_get "notifications.notify_on_failure" "true")
    
    if [[ "$message_type" == "success" && "$notify_on_success" == "true" ]]; then
        update_profile_script "$tag" "$status"
    elif [[ "$message_type" == "failure" && "$notify_on_failure" == "true" ]]; then
        update_profile_script "$tag" "$status"
    elif [[ "$message_type" == "sonar_failed" ]]; then
        update_profile_script "$tag" "$status"
    fi
}

#===============================================================================
# Registro en base de datos
#===============================================================================

log_notification() {
    local message_type=$1
    local tag=$2
    local details=${3:-""}
    
    # Obtener deployment_id si existe
    local deployment_id
    deployment_id=$(db_query "SELECT id FROM deployments WHERE tag_name='$tag' ORDER BY id DESC LIMIT 1" 2>/dev/null || echo "")
    
    if [[ -n "$deployment_id" ]]; then
        db_log_execution "$deployment_id" "notify" "Notificación enviada: $message_type" "INFO"
    fi
}

#===============================================================================
# Utilidades
#===============================================================================

show_last_notification() {
    log_info "Última información de versión:"
    echo ""
    
    if [[ -f "$PROFILE_SCRIPT" ]]; then
        source "$PROFILE_SCRIPT" 2>/dev/null || cat "$PROFILE_SCRIPT"
    else
        echo "No hay información de versión disponible"
        echo "Archivo: $PROFILE_SCRIPT no existe"
    fi
}

test_notifications() {
    log_info "Probando sistema de notificaciones..."
    
    echo ""
    log_info "1. Probando notificación wall (success):"
    notify_wall "success" "V_TEST_01_02_03_04"
    
    echo ""
    log_info "2. Probando actualización de profile:"
    update_profile_script "V_TEST_01_02_03_04" "TEST"
    
    echo ""
    log_ok "Pruebas completadas"
}

#===============================================================================
# Main
#===============================================================================

usage() {
    cat <<EOF
Uso: $(basename "$0") <comando> [argumentos]

Comandos:
  wall <tipo> <tag> [detalles]      Enviar notificación wall
  profile <tag> <estado>            Actualizar script profile.d
  both <tipo> <tag> [detalles]      Enviar wall + actualizar profile
  show                              Mostrar última información
  test                              Probar notificaciones

Tipos de mensaje (wall):
  success       - Despliegue exitoso
  failure       - Fallo en pipeline
  sonar_failed  - Quality Gate no superado
  compiling     - Compilación en progreso
  deploying     - Despliegue en progreso

Ejemplos:
  $(basename "$0") wall success V01_02_03_04
  $(basename "$0") wall failure V01_02_03_04 "Error en compilación"
  $(basename "$0") profile V01_02_03_04 DESPLEGADO
  $(basename "$0") both success V01_02_03_04

EOF
}

main() {
    local cmd="${1:-}"
    
    case "$cmd" in
        wall)
            if [[ -z "${2:-}" || -z "${3:-}" ]]; then
                log_error "Uso: $0 wall <tipo> <tag> [detalles]"
                exit 1
            fi
            notify_wall "$2" "$3" "${4:-}"
            log_notification "$2" "$3" "${4:-}"
            ;;
        profile)
            if [[ -z "${2:-}" || -z "${3:-}" ]]; then
                log_error "Uso: $0 profile <tag> <estado>"
                exit 1
            fi
            update_profile_script "$2" "$3"
            ;;
        both)
            if [[ -z "${2:-}" || -z "${3:-}" ]]; then
                log_error "Uso: $0 both <tipo> <tag> [detalles]"
                exit 1
            fi
            notify_both "$2" "$3" "${4:-}"
            log_notification "$2" "$3" "${4:-}"
            ;;
        show)
            show_last_notification
            ;;
        test)
            test_notifications
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
