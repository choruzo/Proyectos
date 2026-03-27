# =============================================================================
# RPM SPEC File - CI/CD Pipeline GALTTCMC
# =============================================================================
# Genera un RPM que instala el pipeline CI/CD en /home/YOUR_USER/cicd
#
# Build: rpmbuild -ba cicd-galttcmc.spec
# Install: sudo rpm -ivh cicd-galttcmc-*.rpm
# Upgrade: sudo rpm -Uvh cicd-galttcmc-*.rpm
# Remove: sudo rpm -e cicd-galttcmc
# =============================================================================

%define _app_name cicd
%define _app_dir /home/YOUR_USER/cicd
%define _app_user agent
%define _app_group users
%define _build_id_links none
%define _python_version 3.6

Summary:        CI/CD Pipeline automatizado para GALTTCMC
Name:           cicd-galttcmc
Version:        1.0.0
Release:        1%{?dist}
License:        Proprietary - Indra
Group:          Development/Tools
BuildArch:      noarch
URL:            https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
Vendor:         Indra Sistemas
Packager:       GALTTCMC Team <your-team@your-company.com>

# Dependencias del sistema
Requires:       bash >= 4.0
Requires:       git >= 2.0
Requires:       sqlite3
Requires:       python3 >= 3.6
Requires:       sudo
Requires:       util-linux
Requires:       coreutils
Requires:       findutils
Requires:       grep
Requires:       sed
Requires:       gawk
# yq para parsing YAML en bash (puede instalarse manual si no está en repos)
# Requires:       yq

%description
Pipeline CI/CD automatizado para el proyecto GALTTCMC que realiza:
- Monitorización de tags Git (patrón MAC_X_VXX_XX_XX_XX)
- Compilación automática de DVDs/ISOs
- Análisis de calidad con SonarQube
- Despliegue a vCenter y VMs objetivo vía SSH
- Auditoría completa en base de datos SQLite
- Ejecución como servicio systemd

Instalación en: %{_app_dir}
Usuario: %{_app_user}

%prep
# No hay preparación necesaria - archivos ya en SOURCE

%build
# No hay compilación - solo scripts

%install
rm -rf %{buildroot}

# Crear estructura de directorios
install -d %{buildroot}%{_app_dir}
install -d %{buildroot}%{_app_dir}/scripts
install -d %{buildroot}%{_app_dir}/python
install -d %{buildroot}%{_app_dir}/python/librerias_offline
install -d %{buildroot}%{_app_dir}/config
install -d %{buildroot}%{_app_dir}/db
install -d %{buildroot}%{_app_dir}/logs
install -d %{buildroot}%{_app_dir}/utils

# Copiar script principal
install -m 0755 %{_sourcedir}/ci_cd.sh %{buildroot}%{_app_dir}/

# Copiar scripts Bash
install -m 0755 %{_sourcedir}/scripts/common.sh %{buildroot}%{_app_dir}/scripts/
install -m 0755 %{_sourcedir}/scripts/git_monitor.sh %{buildroot}%{_app_dir}/scripts/
install -m 0755 %{_sourcedir}/scripts/compile.sh %{buildroot}%{_app_dir}/scripts/
install -m 0755 %{_sourcedir}/scripts/deploy.sh %{buildroot}%{_app_dir}/scripts/
install -m 0755 %{_sourcedir}/scripts/notify.sh %{buildroot}%{_app_dir}/scripts/
install -m 0755 %{_sourcedir}/scripts/db_viewer.sh %{buildroot}%{_app_dir}/scripts/

# Copiar scripts Python
install -m 0755 %{_sourcedir}/python/sonar_check.py %{buildroot}%{_app_dir}/python/
install -m 0755 %{_sourcedir}/python/vcenter_api.py %{buildroot}%{_app_dir}/python/
install -m 0644 %{_sourcedir}/python/requirements.txt %{buildroot}%{_app_dir}/python/

# Copiar librerías Python offline (wheels)
install -m 0644 %{_sourcedir}/python/librerias_offline/*.whl %{buildroot}%{_app_dir}/python/librerias_offline/

# Copiar ficheros de configuración
install -m 0644 %{_sourcedir}/config/ci_cd_config.yaml %{buildroot}%{_app_dir}/config/
install -m 0644 %{_sourcedir}/config/sonar-project.properties %{buildroot}%{_app_dir}/config/

# Copiar schema de base de datos
install -m 0644 %{_sourcedir}/db/init_db.sql %{buildroot}%{_app_dir}/db/

# Copiar scripts de setup y mantenimiento
install -m 0755 %{_sourcedir}/setup_phase0.sh %{buildroot}%{_app_dir}/
install -m 0755 %{_sourcedir}/install_service.sh %{buildroot}%{_app_dir}/

# Copiar unit file de systemd
install -d %{buildroot}%{_unitdir}
install -m 0644 %{_sourcedir}/cicd.service %{buildroot}%{_unitdir}/cicd.service

# Copiar documentación
install -d %{buildroot}%{_app_dir}/docs
install -m 0644 %{_sourcedir}/README.md %{buildroot}%{_app_dir}/docs/
install -m 0644 %{_sourcedir}/CLAUDE.md %{buildroot}%{_app_dir}/docs/ 2>/dev/null || true

# Copiar utilidades (sonar-scanner, build-wrapper)
if [ -d %{_sourcedir}/utils/sonar-scanner-* ]; then
    cp -a %{_sourcedir}/utils/sonar-scanner-* %{buildroot}%{_app_dir}/utils/
fi
if [ -d %{_sourcedir}/utils/build-wrapper-* ]; then
    cp -a %{_sourcedir}/utils/build-wrapper-* %{buildroot}%{_app_dir}/utils/
fi

# Crear placeholder para .env (no incluir el real por seguridad)
cat > %{buildroot}%{_app_dir}/config/.env.example <<'EOF'
# =============================================================================
# Variables de entorno para CI/CD Pipeline
# =============================================================================
# Copiar este fichero a .env y rellenar con credenciales reales
# NUNCA commitear .env al control de versiones

# Git credentials
GIT_USERNAME=agent
GIT_PASSWORD=tu_password_o_token_git

# SonarQube token
SONAR_TOKEN=tu_token_sonarqube

# vCenter credentials
VCENTER_USER=usuario_vcenter
VCENTER_PASSWORD=password_vcenter

# Target VM SSH key (ruta absoluta)
TARGET_VM_KEY=/home/YOUR_USER/.ssh/id_rsa
TARGET_VM_USER=root
TARGET_VM_HOST=YOUR_TARGET_VM_IP
EOF

%files
%defattr(-,%{_app_user},%{_app_group},-)
%dir %{_app_dir}
%dir %{_app_dir}/scripts
%dir %{_app_dir}/python
%dir %{_app_dir}/python/librerias_offline
%dir %{_app_dir}/config
%dir %{_app_dir}/db
%dir %{_app_dir}/logs
%dir %{_app_dir}/utils
%dir %{_app_dir}/docs

# Scripts ejecutables
%attr(0755,%{_app_user},%{_app_group}) %{_app_dir}/ci_cd.sh
%attr(0755,%{_app_user},%{_app_group}) %{_app_dir}/setup_phase0.sh
%attr(0755,%{_app_user},%{_app_group}) %{_app_dir}/install_service.sh

# Scripts Bash
%attr(0755,%{_app_user},%{_app_group}) %{_app_dir}/scripts/*.sh

# Scripts Python
%attr(0755,%{_app_user},%{_app_group}) %{_app_dir}/python/*.py
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/python/requirements.txt
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/python/librerias_offline/*.whl

# Configuración
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/config/ci_cd_config.yaml
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/config/sonar-project.properties
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/config/.env.example

# Base de datos
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/db/init_db.sql

# Systemd service (owned by root)
%defattr(-,root,root,-)
%{_unitdir}/cicd.service

# Utilidades (si existen)
%defattr(-,%{_app_user},%{_app_group},-)
%{_app_dir}/utils/*

# Documentación
%attr(0644,%{_app_user},%{_app_group}) %{_app_dir}/docs/*.md

%pre
# Verificar que existe el usuario agent
if ! id -u %{_app_user} >/dev/null 2>&1; then
    echo "ADVERTENCIA: El usuario '%{_app_user}' no existe en el sistema."
    echo "Creando usuario '%{_app_user}'..."
    useradd -m -s /bin/bash %{_app_user} 2>/dev/null || true
fi

%post
echo "=============================================================================="
echo "  CI/CD Pipeline GALTTCMC instalado en %{_app_dir}"
echo "=============================================================================="
echo ""

# Instalar dependencias Python offline si pip está disponible
if command -v pip%{_python_version} >/dev/null 2>&1; then
    echo "Instalando dependencias Python offline..."
    su - %{_app_user} -c "pip%{_python_version} install --user --no-index --find-links=%{_app_dir}/python/librerias_offline -r %{_app_dir}/python/requirements.txt" 2>/dev/null || {
        echo "ADVERTENCIA: No se pudieron instalar algunas dependencias Python automáticamente"
        echo "Ejecuta manualmente: pip%{_python_version} install --user -r %{_app_dir}/python/requirements.txt"
    }
else
    echo "ADVERTENCIA: pip%{_python_version} no encontrado - instalar dependencias manualmente"
fi

# Inicializar base de datos
echo "Inicializando base de datos SQLite..."
su - %{_app_user} -c "cd %{_app_dir} && ./ci_cd.sh init" 2>/dev/null || {
    echo "ADVERTENCIA: No se pudo inicializar la BD automáticamente"
}

# Generar claves SSH si no existen
if [ ! -f /home/%{_app_user}/.ssh/id_rsa ]; then
    echo "Generando claves SSH para usuario %{_app_user}..."
    su - %{_app_user} -c "ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N '' -C 'cicd-pipeline@%{_app_user}'" 2>/dev/null || true
    echo "Clave pública generada en: /home/%{_app_user}/.ssh/id_rsa.pub"
    echo "Recuerda copiarla a las VMs objetivo con: ssh-copy-id YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP"
fi

# Reload systemd si el servicio está instalado
systemctl daemon-reload 2>/dev/null || true

echo ""
echo "PASOS SIGUIENTES:"
echo "  1. Copiar y configurar credenciales:"
echo "     $ cp %{_app_dir}/config/.env.example %{_app_dir}/config/.env"
echo "     $ nano %{_app_dir}/config/.env"
echo ""
echo "  2. Copiar clave SSH pública a VM objetivo:"
echo "     $ cat /home/%{_app_user}/.ssh/id_rsa.pub"
echo "     $ ssh-copy-id YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP"
echo ""
echo "  3. Verificar instalación de yq (parser YAML para bash):"
echo "     $ command -v yq || echo 'Instalar yq manualmente desde https://github.com/mikefarah/yq'"
echo ""
echo "  4. Habilitar y arrancar servicio systemd:"
echo "     $ sudo systemctl enable cicd.service"
echo "     $ sudo systemctl start cicd.service"
echo "     $ sudo systemctl status cicd.service"
echo ""
echo "  5. O ejecutar en modo manual:"
echo "     $ cd %{_app_dir}"
echo "     $ ./ci_cd.sh daemon        # Modo demonio (polling continuo)"
echo "     $ ./ci_cd.sh --tag TAG     # Procesar un tag específico"
echo "     $ ./ci_cd.sh status        # Ver estado del pipeline"
echo ""
echo "Para más información, consulta: %{_app_dir}/docs/README.md"
echo "=============================================================================="

%preun
# Si se está desinstalando (no actualizando), parar el servicio
if [ $1 -eq 0 ]; then
    systemctl stop cicd.service 2>/dev/null || true
    systemctl disable cicd.service 2>/dev/null || true
fi

%postun
# Si se ha desinstalado completamente (no actualización)
if [ $1 -eq 0 ]; then
    echo "Pipeline CI/CD desinstalado."
    echo "Los logs y base de datos en %{_app_dir} se han eliminado."
    echo "Backups de configuración (.env) deben restaurarse manualmente si es necesario."
    systemctl daemon-reload 2>/dev/null || true
fi

%clean
rm -rf %{buildroot}

%changelog
* Mon Feb 19 2026 GALTTCMC Team <your-team@your-company.com> - 1.0.0-1
- Versión inicial del RPM
- Pipeline completo con monitorización Git, compilación, SonarQube y despliegue
- Incluye dependencias Python offline (wheels)
- Servicio systemd para ejecución automática
- Base de datos SQLite para auditoría
- Integración con vCenter REST API
- Scripts de setup y configuración

