# Guía de Construcción y Distribución del RPM CI/CD GALTTCMC

## Descripción

Este directorio contiene los archivos necesarios para empaquetar el pipeline CI/CD como un RPM que puede instalarse fácilmente en sistemas SUSE Linux Enterprise Server (SLES) 15.

## Archivos

- **cicd-galttcmc.spec**: Especificación RPM que define cómo empaquetar e instalar el pipeline
- **build_rpm.sh**: Script automático para construir el RPM
- **RPM-BUILD-GUIDE.md**: Este documento

## Requisitos Previos

### En la máquina de construcción:

```bash
# SUSE/openSUSE
sudo zypper install rpm-build

# RHEL/CentOS/Fedora
sudo yum install rpm-build
```

### Verificar que todos los archivos del proyecto están presentes:

```
cicd/
├── ci_cd.sh
├── cicd.service
├── install_service.sh
├── setup_phase0.sh
├── scripts/
│   ├── common.sh
│   ├── compile.sh
│   ├── db_viewer.sh
│   ├── deploy.sh
│   ├── git_monitor.sh
│   └── notify.sh
├── python/
│   ├── sonar_check.py
│   ├── vcenter_api.py
│   ├── requirements.txt
│   └── librerias_offline/
│       ├── certifi-*.whl
│       ├── charset_normalizer-*.whl
│       ├── idna-*.whl
│       ├── PyYAML-*.whl
│       ├── requests-*.whl
│       └── urllib3-*.whl
├── config/
│   ├── ci_cd_config.yaml
│   └── sonar-project.properties
├── db/
│   └── init_db.sql
├── utils/
│   ├── sonar-scanner-*/
│   └── build-wrapper-*/
└── README.md
```

## Construcción del RPM

### Opción 1: Usando el script automático (RECOMENDADO)

```bash
cd /ruta/al/proyecto/cicd

# Construir con versión por defecto (1.0.0)
chmod +x build_rpm.sh
./build_rpm.sh

# O especificar una versión personalizada
./build_rpm.sh 1.2.3
```

El script:
1. Verifica que rpmbuild esté instalado
2. Crea la estructura de directorios ~/rpmbuild
3. Copia todos los archivos necesarios
4. Ejecuta rpmbuild
5. Muestra la ubicación del RPM generado

### Opción 2: Construcción manual

```bash
# Crear estructura de directorios
mkdir -p ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Copiar el spec file
cp cicd-galttcmc.spec ~/rpmbuild/SPECS/

# Copiar todos los archivos fuente a SOURCES
cd ~/rpmbuild/SOURCES
cp -r /ruta/al/proyecto/cicd/* .

# Construir el RPM
cd ~/rpmbuild/SPECS
rpmbuild -ba cicd-galttcmc.spec
```

### Resultado

El RPM generado estará en:
- **Binary RPM**: `~/rpmbuild/RPMS/noarch/cicd-galttcmc-1.0.0-1.noarch.rpm`
- **Source RPM**: `~/rpmbuild/SRPMS/cicd-galttcmc-1.0.0-1.src.rpm`

## Instalación del RPM

### En la máquina objetivo (172.30.188.137)

```bash
# Copiar el RPM a la máquina objetivo
scp ~/rpmbuild/RPMS/noarch/cicd-galttcmc-*.rpm agent@172.30.188.137:/tmp/

# Conectarse a la máquina
ssh agent@172.30.188.137

# Instalar como root
sudo rpm -ivh /tmp/cicd-galttcmc-*.rpm
```

### Verificar instalación

```bash
# Ver información del paquete
rpm -qi cicd-galttcmc

# Listar archivos instalados
rpm -ql cicd-galttcmc

# Verificar archivos
rpm -V cicd-galttcmc
```

## Configuración Post-Instalación

El RPM instalará todo en `/home/agent/cicd`, pero requiere configuración manual:

### 1. Configurar credenciales

```bash
cd /home/agent/cicd
cp config/.env.example config/.env
nano config/.env
```

Rellenar con los valores reales:
```bash
GIT_USERNAME=agent
GIT_PASSWORD=tu_password_o_token_git
SONAR_TOKEN=tu_token_sonarqube
VCENTER_USER=usuario_vcenter
VCENTER_PASSWORD=password_vcenter
TARGET_VM_KEY=/home/agent/.ssh/id_rsa
TARGET_VM_USER=root
TARGET_VM_HOST=172.30.188.147
```

### 2. Copiar clave SSH a VM objetivo

```bash
# Verificar que se generó la clave (el RPM lo hace automáticamente)
ls -l ~/.ssh/id_rsa*

# Copiar a la VM objetivo
ssh-copy-id root@172.30.188.147
```

### 3. Instalar yq (parser YAML para bash)

```bash
# Opción A: Desde repositorios (si está disponible)
sudo zypper install yq

# Opción B: Instalación manual
sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
sudo chmod +x /usr/local/bin/yq

# Verificar instalación
yq --version
```

### 4. Habilitar el servicio systemd

```bash
sudo systemctl daemon-reload
sudo systemctl enable cicd.service
sudo systemctl start cicd.service
sudo systemctl status cicd.service
```

### 5. Verificar funcionamiento

```bash
cd /home/agent/cicd

# Ver estado del pipeline
./ci_cd.sh status

# Procesar un tag manualmente (prueba)
./ci_cd.sh --tag V24_02_15_01

# Ver logs
tail -f logs/pipeline_*.log

# Ver base de datos
./scripts/db_viewer.sh list
```

## Actualización del RPM

Para actualizar a una nueva versión:

```bash
# Construir nueva versión
./build_rpm.sh 1.1.0

# Instalar actualización (preserva config y datos)
sudo rpm -Uvh ~/rpmbuild/RPMS/noarch/cicd-galttcmc-1.1.0-*.rpm

# Reiniciar servicio
sudo systemctl restart cicd.service
```

## Desinstalación

```bash
# Parar servicio
sudo systemctl stop cicd.service
sudo systemctl disable cicd.service

# Desinstalar RPM
sudo rpm -e cicd-galttcmc

# NOTA: Esto eliminará /home/agent/cicd y todos sus contenidos
# Hacer backup de config/.env y db/pipeline.db si es necesario
```

## Personalización del SPEC

### Cambiar ubicación de instalación

Editar en `cicd-galttcmc.spec`:
```spec
%define _app_dir /opt/cicd_pipeline    # En lugar de /home/agent/cicd
%define _app_user cicd                 # En lugar de agent
```

### Añadir dependencias adicionales

```spec
Requires:       yq >= 4.0
Requires:       python3-requests
```

### Modificar scripts post-instalación

Editar secciones `%post`, `%postun`, `%preun` según necesidades.

## Distribución del RPM

### Crear repositorio YUM/Zypper propio

```bash
# Crear estructura de repositorio
mkdir -p /var/www/html/repos/cicd/{noarch,SRPMS}

# Copiar RPMs
cp ~/rpmbuild/RPMS/noarch/*.rpm /var/www/html/repos/cicd/noarch/
cp ~/rpmbuild/SRPMS/*.rpm /var/www/html/repos/cicd/SRPMS/

# Crear metadata
cd /var/www/html/repos/cicd
createrepo .

# Configurar en clientes
cat > /etc/zypp/repos.d/cicd.repo <<EOF
[cicd]
name=CI/CD Pipeline Repository
baseurl=http://tu-servidor/repos/cicd
enabled=1
gpgcheck=0
EOF

# Instalar desde repo
sudo zypper refresh
sudo zypper install cicd-galttcmc
```

### Firmar el RPM (opcional pero recomendado)

```bash
# Generar clave GPG si no existe
gpg --gen-key

# Importar al sistema RPM
gpg --export -a 'Tu Nombre' > RPM-GPG-KEY-cicd
sudo rpm --import RPM-GPG-KEY-cicd

# Firmar RPM
rpm --addsign ~/rpmbuild/RPMS/noarch/cicd-galttcmc-*.rpm

# Verificar firma
rpm --checksig ~/rpmbuild/RPMS/noarch/cicd-galttcmc-*.rpm
```

## Solución de Problemas

### Error: "User agent does not exist"

El RPM crea el usuario automáticamente en `%pre`, pero si falla:
```bash
sudo useradd -m -s /bin/bash agent
```

### Error: "python3.6 not found"

```bash
sudo zypper install python3 python3-pip
```

### Error: "rpmbuild command not found"

```bash
sudo zypper install rpm-build
```

### El servicio no arranca

```bash
# Ver logs completos
sudo journalctl -u cicd.service -xe

# Verificar permisos
ls -la /home/agent/cicd

# Verificar configuración
cat /etc/systemd/system/cicd.service
```

### Dependencias Python no se instalan

```bash
cd /home/agent/cicd
pip3.6 install --user --no-index \
  --find-links=python/librerias_offline \
  -r python/requirements.txt
```

## Mejoras Futuras

- [ ] Añadir soporte para múltiples versiones de SLES
- [ ] Incluir yq binario en el RPM
- [ ] Crear RPM firmado automáticamente
- [ ] Añadir scripts de validación pre-instalación
- [ ] Crear DEB package para Ubuntu/Debian
- [ ] Generar changelog automático desde Git
- [ ] Incluir tests de integración en el RPM

## Referencias

- [RPM Packaging Guide](https://rpm-packaging-guide.github.io/)
- [Fedora RPM Guide](https://docs.fedoraproject.org/en-US/package-maintainers/Packaging_Tutorial_GNU_Hello/)
- [OpenSUSE Build Service](https://build.opensuse.org/project/show/home)

---

**Última actualización**: 19 de febrero de 2026  
**Mantenedor**: GALTTCMC Team <galttcmc@indra.es>
