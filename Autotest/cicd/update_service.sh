#!/bin/bash
#===============================================================================
# update_service.sh - Actualizar y reiniciar el servicio cicd
#===============================================================================
# Script para aplicar cambios al servicio systemd después de modificaciones
#
# Uso:
#   ./update_service.sh         # Actualizar y reiniciar
#   ./update_service.sh status  # Solo ver estado
#   ./update_service.sh logs    # Ver logs en tiempo real
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/cicd.service"
SERVICE_NAME="cicd.service"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

#===============================================================================
# Funciones
#===============================================================================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Este script debe ejecutarse con sudo"
        exit 1
    fi
}

update_service() {
    log_info "Actualizando servicio $SERVICE_NAME..."
    
    # Verificar que existe el archivo de servicio
    if [[ ! -f "$SERVICE_FILE" ]]; then
        log_error "No se encuentra: $SERVICE_FILE"
        exit 1
    fi
    
    # Detener el servicio si está corriendo
    log_info "Deteniendo servicio..."
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        log_ok "Servicio detenido"
    else
        log_info "Servicio ya estaba detenido"
    fi
    
    # Copiar nuevo archivo de servicio
    log_info "Copiando archivo de servicio a /etc/systemd/system/..."
    cp "$SERVICE_FILE" /etc/systemd/system/
    log_ok "Archivo copiado"
    
    # Recargar configuración de systemd
    log_info "Recargando configuración de systemd..."
    systemctl daemon-reload
    log_ok "Configuración recargada"
    
    # Reiniciar el servicio
    log_info "Iniciando servicio..."
    systemctl start "$SERVICE_NAME"
    
    # Esperar un momento para que inicie
    sleep 2
    
    # Verificar estado
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_ok "✓ Servicio iniciado correctamente"
        echo ""
        systemctl status "$SERVICE_NAME" --no-pager -l
        echo ""
        log_ok "Servicio actualizado y funcionando"
    else
        log_error "✗ Servicio falló al iniciar"
        echo ""
        systemctl status "$SERVICE_NAME" --no-pager -l
        echo ""
        log_error "Ver logs con: journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi
}

show_status() {
    log_info "Estado del servicio $SERVICE_NAME:"
    echo ""
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_ok "Servicio está ACTIVO"
    else
        log_error "Servicio está INACTIVO"
    fi
    
    # Mostrar última línea del log
    echo ""
    log_info "Última entrada del log:"
    journalctl -u "$SERVICE_NAME" -n 1 --no-pager 2>/dev/null || true
}

show_logs() {
    log_info "Mostrando logs en tiempo real (Ctrl+C para salir)..."
    echo ""
    journalctl -u "$SERVICE_NAME" -f
}

show_logs_errors() {
    log_info "Últimos 50 errores:"
    echo ""
    journalctl -u "$SERVICE_NAME" -p err -n 50 --no-pager
}

#===============================================================================
# Main
#===============================================================================

main() {
    local cmd="${1:-update}"
    
    case "$cmd" in
        update)
            check_root
            update_service
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        errors)
            show_logs_errors
            ;;
        restart)
            check_root
            log_info "Reiniciando servicio..."
            systemctl restart "$SERVICE_NAME"
            sleep 2
            show_status
            ;;
        stop)
            check_root
            log_info "Deteniendo servicio..."
            systemctl stop "$SERVICE_NAME"
            log_ok "Servicio detenido"
            ;;
        start)
            check_root
            log_info "Iniciando servicio..."
            systemctl start "$SERVICE_NAME"
            sleep 2
            show_status
            ;;
        enable)
            check_root
            log_info "Habilitando servicio para inicio automático..."
            systemctl enable "$SERVICE_NAME"
            log_ok "Servicio habilitado"
            ;;
        disable)
            check_root
            log_info "Deshabilitando inicio automático..."
            systemctl disable "$SERVICE_NAME"
            log_ok "Servicio deshabilitado"
            ;;
        *)
            cat <<EOF
Uso: $0 <comando>

Comandos:
  update      Actualizar servicio y reiniciar (requiere sudo)
  status      Ver estado del servicio
  logs        Ver logs en tiempo real
  errors      Ver últimos errores
  restart     Reiniciar servicio (requiere sudo)
  stop        Detener servicio (requiere sudo)
  start       Iniciar servicio (requiere sudo)
  enable      Habilitar inicio automático (requiere sudo)
  disable     Deshabilitar inicio automático (requiere sudo)

Ejemplos:
  sudo $0 update          # Actualizar y reiniciar
  $0 status               # Ver estado
  $0 logs                 # Ver logs en tiempo real
  $0 errors               # Ver errores

EOF
            ;;
    esac
}

main "$@"
