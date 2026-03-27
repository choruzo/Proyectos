#!/bin/bash
#===============================================================================
# build_rpm.sh - Script para construir el RPM del pipeline CI/CD
#===============================================================================
# Uso:
#   ./build_rpm.sh [version]
#
# Ejemplo:
#   ./build_rpm.sh 1.0.0
#
# Este script:
#   1. Crea la estructura de directorios para rpmbuild
#   2. Copia todos los archivos necesarios a SOURCES
#   3. Ejecuta rpmbuild con el spec file
#   4. Muestra la ubicación del RPM generado
#===============================================================================

set -euo pipefail

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Directorio base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPM_VERSION="${1:-1.0.0}"

# Directorios de rpmbuild
RPMBUILD_DIR="${HOME}/rpmbuild"
SOURCES_DIR="${RPMBUILD_DIR}/SOURCES"
SPECS_DIR="${RPMBUILD_DIR}/SPECS"
RPMS_DIR="${RPMBUILD_DIR}/RPMS"
SRPMS_DIR="${RPMBUILD_DIR}/SRPMS"
BUILD_DIR="${RPMBUILD_DIR}/BUILD"

#-------------------------------------------------------------------------------
# Funciones
#-------------------------------------------------------------------------------

check_requirements() {
    log_info "Verificando requisitos..."
    
    if ! command -v rpmbuild >/dev/null 2>&1; then
        log_error "rpmbuild no está instalado"
        log_info "En SUSE/openSUSE: sudo zypper install rpm-build"
        log_info "En RHEL/CentOS: sudo yum install rpm-build"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/cicd-galttcmc.spec" ]]; then
        log_error "Spec file no encontrado: $SCRIPT_DIR/cicd-galttcmc.spec"
        exit 1
    fi
    
    log_ok "Requisitos verificados"
}

create_rpmbuild_structure() {
    log_info "Creando estructura de directorios rpmbuild..."
    
    mkdir -p "$SOURCES_DIR"
    mkdir -p "$SPECS_DIR"
    mkdir -p "$RPMS_DIR"
    mkdir -p "$SRPMS_DIR"
    mkdir -p "$BUILD_DIR"
    
    # Limpiar SOURCES anterior
    rm -rf "${SOURCES_DIR:?}"/*
    
    log_ok "Estructura creada en: $RPMBUILD_DIR"
}

copy_sources() {
    log_info "Copiando archivos fuente a $SOURCES_DIR..."
    
    # Copiar script principal
    cp "$SCRIPT_DIR/ci_cd.sh" "$SOURCES_DIR/"
    
    # Copiar scripts Bash
    mkdir -p "$SOURCES_DIR/scripts"
    cp "$SCRIPT_DIR/scripts"/*.sh "$SOURCES_DIR/scripts/" 2>/dev/null || true
    
    # Copiar scripts Python
    mkdir -p "$SOURCES_DIR/python/librerias_offline"
    cp "$SCRIPT_DIR/python"/*.py "$SOURCES_DIR/python/" 2>/dev/null || true
    cp "$SCRIPT_DIR/python/requirements.txt" "$SOURCES_DIR/python/" 2>/dev/null || true
    cp "$SCRIPT_DIR/python/librerias_offline"/*.whl "$SOURCES_DIR/python/librerias_offline/" 2>/dev/null || true
    
    # Copiar configuración
    mkdir -p "$SOURCES_DIR/config"
    cp "$SCRIPT_DIR/config/ci_cd_config.yaml" "$SOURCES_DIR/config/"
    cp "$SCRIPT_DIR/config/sonar-project.properties" "$SOURCES_DIR/config/"
    
    # Copiar base de datos
    mkdir -p "$SOURCES_DIR/db"
    cp "$SCRIPT_DIR/db/init_db.sql" "$SOURCES_DIR/db/"
    
    # Copiar scripts de setup
    cp "$SCRIPT_DIR/setup_phase0.sh" "$SOURCES_DIR/"
    cp "$SCRIPT_DIR/install_service.sh" "$SOURCES_DIR/"
    
    # Copiar unit file de systemd
    cp "$SCRIPT_DIR/cicd.service" "$SOURCES_DIR/"
    
    # Copiar documentación
    cp "$SCRIPT_DIR/README.md" "$SOURCES_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/CLAUDE.md" "$SOURCES_DIR/" 2>/dev/null || true
    
    # Copiar utilidades si existen
    if [[ -d "$SCRIPT_DIR/utils" ]]; then
        mkdir -p "$SOURCES_DIR/utils"
        cp -r "$SCRIPT_DIR/utils"/* "$SOURCES_DIR/utils/" 2>/dev/null || true
    fi
    
    log_ok "Archivos copiados: $(du -sh "$SOURCES_DIR" | cut -f1)"
}

copy_spec() {
    log_info "Copiando spec file..."
    
    # Modificar versión en el spec file si se proporciona
    if [[ "$RPM_VERSION" != "1.0.0" ]]; then
        sed "s/^Version:.*/Version:        $RPM_VERSION/" \
            "$SCRIPT_DIR/cicd-galttcmc.spec" > "$SPECS_DIR/cicd-galttcmc.spec"
        log_ok "Spec file modificado con versión: $RPM_VERSION"
    else
        cp "$SCRIPT_DIR/cicd-galttcmc.spec" "$SPECS_DIR/"
        log_ok "Spec file copiado"
    fi
}

build_rpm() {
    log_info "Construyendo RPM..."
    echo ""
    
    # Ejecutar rpmbuild
    if rpmbuild -ba "$SPECS_DIR/cicd-galttcmc.spec"; then
        echo ""
        log_ok "RPM construido exitosamente"
        echo ""
        log_info "Archivos generados:"
        find "$RPMBUILD_DIR" -name "cicd-galttcmc*.rpm" -exec ls -lh {} \;
        echo ""
        log_ok "RPMs disponibles en:"
        echo "  - Binary RPM: $RPMS_DIR/noarch/"
        echo "  - Source RPM: $SRPMS_DIR/"
        return 0
    else
        log_error "Error al construir RPM"
        return 1
    fi
}

show_install_instructions() {
    echo ""
    log_info "Para instalar el RPM:"
    echo ""
    echo "  # Instalación inicial:"
    echo "  sudo rpm -ivh $RPMS_DIR/noarch/cicd-galttcmc-${RPM_VERSION}-*.noarch.rpm"
    echo ""
    echo "  # Actualización:"
    echo "  sudo rpm -Uvh $RPMS_DIR/noarch/cicd-galttcmc-${RPM_VERSION}-*.noarch.rpm"
    echo ""
    echo "  # Verificar instalación:"
    echo "  rpm -qi cicd-galttcmc"
    echo "  rpm -ql cicd-galttcmc"
    echo ""
    echo "  # Desinstalar:"
    echo "  sudo rpm -e cicd-galttcmc"
    echo ""
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

main() {
    log_info "==================================================================="
    log_info "  Construcción de RPM - CI/CD Pipeline GALTTCMC"
    log_info "  Versión: $RPM_VERSION"
    log_info "==================================================================="
    echo ""
    
    check_requirements
    create_rpmbuild_structure
    copy_sources
    copy_spec
    
    if build_rpm; then
        show_install_instructions
        log_ok "Proceso completado exitosamente"
        exit 0
    else
        log_error "Proceso fallido"
        exit 1
    fi
}

# Ejecutar
main "$@"
