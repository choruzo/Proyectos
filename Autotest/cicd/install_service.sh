#!/bin/bash
#===============================================================================
# install_service.sh - Instalar servicio systemd del pipeline CI/CD
#===============================================================================
# Ejecutar como root o con sudo
#
# Uso:
#   sudo ./install_service.sh install    # Instalar y habilitar servicio
#   sudo ./install_service.sh uninstall  # Desinstalar servicio
#   sudo ./install_service.sh status     # Ver estado del servicio
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="cicd"
SERVICE_FILE="$SCRIPT_DIR/cicd.service"
SYSTEMD_DIR="/etc/systemd/system"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Este script debe ejecutarse como root o con sudo"
        exit 1
    fi
}

install_service() {
    log_info "Instalando servicio $SERVICE_NAME..."
    
    # Verificar que existe el fichero de servicio
    if [[ ! -f "$SERVICE_FILE" ]]; then
        log_error "Fichero de servicio no encontrado: $SERVICE_FILE"
        exit 1
    fi
    
    # Verificar que existe el script principal
    if [[ ! -f "$SCRIPT_DIR/ci_cd.sh" ]]; then
        log_error "Script principal no encontrado: $SCRIPT_DIR/ci_cd.sh"
        exit 1
    fi
    
    # Asignar permisos de ejecución a los scripts
    log_info "Asignando permisos de ejecución..."
    chmod +x "$SCRIPT_DIR/ci_cd.sh"
    chmod +x "$SCRIPT_DIR/scripts/"*.sh 2>/dev/null || true
    chmod +x "$SCRIPT_DIR/python/"*.py 2>/dev/null || true
    
    # Copiar fichero de servicio
    log_info "Copiando fichero de servicio a $SYSTEMD_DIR..."
    cp "$SERVICE_FILE" "$SYSTEMD_DIR/${SERVICE_NAME}.service"
    
    # Recargar systemd
    log_info "Recargando systemd..."
    systemctl daemon-reload
    
    # Habilitar servicio
    log_info "Habilitando servicio para inicio automático..."
    systemctl enable "$SERVICE_NAME"
    
    # Crear directorios necesarios
    log_info "Creando directorios necesarios..."
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/db"
    
    # Ajustar propietario (asumiendo usuario 'agent')
    if id "agent" &>/dev/null; then
        chown -R agent:agent "$SCRIPT_DIR"
        log_info "Propietario ajustado a usuario 'agent'"
    else
        log_warn "Usuario 'agent' no existe, ajusta los permisos manualmente"
    fi
    
    # Inicializar base de datos si no existe
    if [[ ! -f "$SCRIPT_DIR/db/pipeline.db" ]]; then
        log_info "Inicializando base de datos..."
        if [[ -f "$SCRIPT_DIR/db/init_db.sql" ]]; then
            sqlite3 "$SCRIPT_DIR/db/pipeline.db" < "$SCRIPT_DIR/db/init_db.sql"
            log_info "Base de datos creada"
        fi
    fi
    
    log_info "════════════════════════════════════════════════════════════"
    log_info "Servicio instalado correctamente"
    log_info "════════════════════════════════════════════════════════════"
    echo ""
    echo "Comandos útiles:"
    echo "  sudo systemctl start $SERVICE_NAME     # Iniciar servicio"
    echo "  sudo systemctl stop $SERVICE_NAME      # Detener servicio"
    echo "  sudo systemctl restart $SERVICE_NAME   # Reiniciar servicio"
    echo "  sudo systemctl status $SERVICE_NAME    # Ver estado"
    echo "  journalctl -u $SERVICE_NAME -f         # Ver logs en tiempo real"
    echo ""
    echo "IMPORTANTE: Configura las credenciales en:"
    echo "  $SCRIPT_DIR/config/.env"
    echo ""
}

uninstall_service() {
    log_info "Desinstalando servicio $SERVICE_NAME..."
    
    # Detener servicio si está corriendo
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_info "Deteniendo servicio..."
        systemctl stop "$SERVICE_NAME"
    fi
    
    # Deshabilitar servicio
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_info "Deshabilitando servicio..."
        systemctl disable "$SERVICE_NAME"
    fi
    
    # Eliminar fichero de servicio
    if [[ -f "$SYSTEMD_DIR/${SERVICE_NAME}.service" ]]; then
        log_info "Eliminando fichero de servicio..."
        rm -f "$SYSTEMD_DIR/${SERVICE_NAME}.service"
    fi
    
    # Recargar systemd
    systemctl daemon-reload
    
    log_info "Servicio desinstalado correctamente"
    log_warn "Los logs y base de datos NO se han eliminado"
}

show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "              Estado del servicio $SERVICE_NAME"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo -e "Estado: ${GREEN}ACTIVO${NC}"
    else
        echo -e "Estado: ${RED}INACTIVO${NC}"
    fi
    
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo -e "Inicio automático: ${GREEN}HABILITADO${NC}"
    else
        echo -e "Inicio automático: ${YELLOW}DESHABILITADO${NC}"
    fi
    
    echo ""
    echo "─── Información del servicio ───────────────────────────────"
    systemctl status "$SERVICE_NAME" --no-pager 2>/dev/null || echo "(servicio no instalado)"
    
    echo ""
    echo "─── Últimos logs ───────────────────────────────────────────"
    journalctl -u "$SERVICE_NAME" -n 10 --no-pager 2>/dev/null || echo "(sin logs)"
    echo ""
}

usage() {
    cat <<EOF
Uso: $(basename "$0") <comando>

Comandos:
  install     Instalar y habilitar servicio systemd
  uninstall   Desinstalar servicio
  status      Ver estado del servicio
  help        Mostrar esta ayuda

Ejemplos:
  sudo $(basename "$0") install
  sudo $(basename "$0") status

EOF
}

main() {
    local cmd="${1:-}"
    
    case "$cmd" in
        install)
            check_root
            install_service
            ;;
        uninstall)
            check_root
            uninstall_service
            ;;
        status)
            show_status
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
