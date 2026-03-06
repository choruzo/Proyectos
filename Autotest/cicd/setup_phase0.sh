#!/bin/bash
#===============================================================================
# setup_phase0.sh - Script de instalación Fase 0: Preparación del Entorno
#
# Ejecutar como usuario 'agent' en la máquina de desarrollo (172.30.188.137)
# Uso: ./setup_phase0.sh
#===============================================================================

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CICD_HOME="/home/agent/cicd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#-------------------------------------------------------------------------------
# Funciones de utilidad
#-------------------------------------------------------------------------------
log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_user() {
    if [[ "$(whoami)" != "agent" ]]; then
        log_error "Este script debe ejecutarse como usuario 'agent'"
        log_info "Ejecuta: sudo su - agent"
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# 0.1 - Crear estructura de directorios
#-------------------------------------------------------------------------------
create_directories() {
    log_info "Creando estructura de directorios en $CICD_HOME..."
    
    mkdir -p "$CICD_HOME"/{scripts,python,config,db,logs}
    
    log_ok "Estructura de directorios creada:"
    tree "$CICD_HOME" 2>/dev/null || ls -la "$CICD_HOME"
}

#-------------------------------------------------------------------------------
# 0.2 - Copiar ficheros de configuración
#-------------------------------------------------------------------------------
copy_config_files() {
    log_info "Copiando ficheros de configuración..."
    
    # Si estamos ejecutando desde CICD_HOME, no hay nada que copiar
    if [[ "$SCRIPT_DIR" == "$CICD_HOME" ]]; then
        log_ok "Ejecutando desde directorio destino - ficheros ya en su lugar"
        
        # Solo asegurar permisos de ejecución
        chmod +x "$CICD_HOME/scripts/"*.sh 2>/dev/null || true
        chmod +x "$CICD_HOME/python/"*.py 2>/dev/null || true
        return 0
    fi
    
    # Copiar config YAML
    if [[ -f "$SCRIPT_DIR/config/ci_cd_config.yaml" ]]; then
        cp "$SCRIPT_DIR/config/ci_cd_config.yaml" "$CICD_HOME/config/"
        log_ok "ci_cd_config.yaml copiado"
    else
        log_warn "ci_cd_config.yaml no encontrado en $SCRIPT_DIR/config/"
    fi
    
    # Copiar scripts Python
    if [[ -d "$SCRIPT_DIR/python" ]]; then
        cp "$SCRIPT_DIR/python/"*.py "$CICD_HOME/python/" 2>/dev/null || true
        cp "$SCRIPT_DIR/python/requirements.txt" "$CICD_HOME/python/" 2>/dev/null || true
        log_ok "Scripts Python copiados"
    fi
    
    # Copiar scripts shell
    if [[ -d "$SCRIPT_DIR/scripts" ]]; then
        cp "$SCRIPT_DIR/scripts/"*.sh "$CICD_HOME/scripts/" 2>/dev/null || true
        chmod +x "$CICD_HOME/scripts/"*.sh 2>/dev/null || true
        log_ok "Scripts shell copiados"
    fi
    
    # Copiar script SQL
    if [[ -f "$SCRIPT_DIR/db/init_db.sql" ]]; then
        cp "$SCRIPT_DIR/db/init_db.sql" "$CICD_HOME/db/"
        log_ok "init_db.sql copiado"
    fi
    
    # Copiar orquestador principal
    if [[ -f "$SCRIPT_DIR/ci_cd.sh" ]]; then
        cp "$SCRIPT_DIR/ci_cd.sh" "$CICD_HOME/"
        chmod +x "$CICD_HOME/ci_cd.sh"
        log_ok "ci_cd.sh copiado"
    fi
}

#-------------------------------------------------------------------------------
# 0.3 - Verificar e instalar dependencias Python
#-------------------------------------------------------------------------------
setup_python() {
    log_info "Verificando Python 3.6..."
    
    if command -v python3.6 &>/dev/null; then
        local py_version=$(python3.6 --version 2>&1)
        log_ok "Python encontrado: $py_version"
    else
        log_error "Python 3.6 no encontrado"
        log_info "Instalar con: sudo zypper install python36 python36-pip"
        return 1
    fi
    
    # Añadir ~/.local/bin al PATH si no está
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        log_info "Añadido ~/.local/bin al PATH"
    fi
    
    log_info "Instalando dependencias Python..."
    
    if [[ -f "$CICD_HOME/python/requirements.txt" ]]; then
        python3.6 -m pip install --user -r "$CICD_HOME/python/requirements.txt"
        log_ok "Dependencias Python instaladas"
    else
        log_warn "requirements.txt no encontrado, instalando manualmente..."
        # Sin pyvmomi - usamos API REST directa de vCenter
        python3.6 -m pip install --user 'requests>=2.20.0,<3.0.0' 'PyYAML>=5.0,<6.1'
    fi
    
    # Verificar instalación (sin pyvmomi - usamos API REST)
    log_info "Verificando módulos instalados..."
    python3.6 -c "import requests; print('  - requests:', requests.__version__)"
    python3.6 -c "import yaml; print('  - PyYAML: OK')"
    log_ok "Módulos Python verificados"
    log_info "NOTA: Se usa API REST de vCenter directamente (sin pyvmomi)"
}

#-------------------------------------------------------------------------------
# 0.4 - Configurar claves SSH
#-------------------------------------------------------------------------------
setup_ssh_keys() {
    log_info "Configurando claves SSH..."
    
    local ssh_key="$HOME/.ssh/id_rsa"
    local target_vm="172.30.188.147"
    local target_user="root"
    
    # Crear directorio .ssh si no existe
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    
    # Generar clave si no existe
    if [[ ! -f "$ssh_key" ]]; then
        log_info "Generando par de claves SSH..."
        ssh-keygen -t rsa -b 4096 -f "$ssh_key" -N "" -C "agent@cicd-pipeline"
        log_ok "Claves SSH generadas"
    else
        log_ok "Clave SSH existente encontrada: $ssh_key"
    fi
    
    # Verificar conectividad antes de intentar copiar
    log_info "Verificando conectividad con $target_vm..."
    if ! ping -c 1 -W 3 "$target_vm" &>/dev/null; then
        log_warn "No hay conectividad con $target_vm"
        log_warn "Omitiendo copia de clave SSH. Configura manualmente después:"
        echo "  ssh-copy-id -i ${ssh_key}.pub ${target_user}@${target_vm}"
        return 0
    fi
    log_ok "Conectividad verificada"
    
    # Preguntar si desea copiar la clave
    echo ""
    read -p "¿Deseas copiar la clave SSH a $target_vm ahora? (s/N): " -t 30 response || response="n"
    
    if [[ "$response" =~ ^[Ss]$ ]]; then
        log_info "Copiando clave pública a $target_vm..."
        log_warn "Introduce la contraseña de ${target_user}@${target_vm}"
        
        if ssh-copy-id -i "${ssh_key}.pub" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${target_user}@${target_vm}"; then
            log_ok "Clave pública copiada a la VM destino"
            
            # Verificar conexión
            log_info "Verificando conexión SSH sin contraseña..."
            if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$ssh_key" "${target_user}@${target_vm}" "echo 'SSH OK'" 2>/dev/null; then
                log_ok "Conexión SSH sin contraseña: FUNCIONAL"
            else
                log_warn "Conexión SSH sin contraseña aún no funciona"
            fi
        else
            log_warn "No se pudo copiar la clave. Ejecuta manualmente después:"
            echo "  ssh-copy-id -i ${ssh_key}.pub ${target_user}@${target_vm}"
        fi
    else
        log_info "Omitido. Para configurar SSH después, ejecuta:"
        echo "  ssh-copy-id -i ${ssh_key}.pub ${target_user}@${target_vm}"
    fi
}

#-------------------------------------------------------------------------------
# 0.5 - Inicializar base de datos SQLite
#-------------------------------------------------------------------------------
setup_database() {
    log_info "Inicializando base de datos SQLite..."
    
    local db_path="$CICD_HOME/db/pipeline.db"
    local sql_file="$CICD_HOME/db/init_db.sql"
    
    if [[ -f "$sql_file" ]]; then
        sqlite3 "$db_path" < "$sql_file"
        log_ok "Base de datos creada: $db_path"
        
        # Mostrar tablas creadas
        log_info "Tablas creadas:"
        sqlite3 "$db_path" ".tables" | sed 's/^/  /'
    else
        log_error "Fichero SQL no encontrado: $sql_file"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# Crear fichero .env de ejemplo
#-------------------------------------------------------------------------------
create_env_file() {
    log_info "Creando fichero .env de ejemplo..."
    
    local env_file="$CICD_HOME/config/.env"
    
    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" << 'EOF'
# =============================================================================
# Variables de entorno para CI/CD Pipeline
# =============================================================================
# IMPORTANTE: Este fichero contiene credenciales sensibles
# NO commitear a Git - añadir a .gitignore

# Credenciales Git (si se usa autenticación por token)
GIT_PASSWORD=your_git_token_here

# Token de SonarQube (usar el de sonar.login)
SONAR_TOKEN=your_sonar_token_here

# Credenciales vCenter
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=your_vcenter_password_here
EOF
        chmod 600 "$env_file"
        log_ok "Fichero .env creado: $env_file"
        log_warn "IMPORTANTE: Edita $env_file con las credenciales reales"
    else
        log_ok "Fichero .env ya existe"
    fi
}

#-------------------------------------------------------------------------------
# Verificar herramientas adicionales
#-------------------------------------------------------------------------------
check_tools() {
    log_info "Verificando herramientas adicionales..."
    
    local tools=("git" "sqlite3" "yq" "jq")
    local missing=()
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" &>/dev/null; then
            log_ok "$tool: $(command -v "$tool")"
        else
            missing+=("$tool")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_warn "Herramientas no encontradas: ${missing[*]}"
        log_info "Instalar con: sudo zypper install ${missing[*]}"
    fi
}

#-------------------------------------------------------------------------------
# Mostrar resumen
#-------------------------------------------------------------------------------
show_summary() {
    echo ""
    echo "==============================================================================="
    echo "                    FASE 0: PREPARACIÓN COMPLETADA"
    echo "==============================================================================="
    echo ""
    echo "Estructura creada:"
    tree "$CICD_HOME" 2>/dev/null || find "$CICD_HOME" -type f | head -20
    echo ""
    echo "Próximos pasos:"
    echo "  1. Editar credenciales en: $CICD_HOME/config/.env"
    echo "  2. Ajustar configuración en: $CICD_HOME/config/ci_cd_config.yaml"
    echo "  3. Verificar SSH a la VM: ssh -i ~/.ssh/id_rsa root@172.30.188.147"
    echo "  4. Probar con: $CICD_HOME/ci_cd.sh status"
    echo ""
    echo "==============================================================================="
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
main() {
    echo "==============================================================================="
    echo "           SETUP FASE 0: Preparación del Entorno CI/CD"
    echo "==============================================================================="
    echo ""
    
    check_user
    
    create_directories
    copy_config_files
    setup_python
    setup_ssh_keys
    setup_database
    create_env_file
    check_tools
    
    show_summary
}

# Ejecutar si es el script principal
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
