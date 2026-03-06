# CI/CD Pipeline - GALTTCMC

Pipeline automatizado para monitorización de tags Git, compilación, análisis SonarQube y despliegue en vCenter/VM.

## Descripción General

Este pipeline CI/CD automatiza el ciclo completo **Dev → QA → Release → Despliegue**:

1. **Monitorización Git** - Polling cada 5 minutos detectando nuevos tags
2. **Compilación** - Checkout del tag, build de DVDs/ISOs
3. **Análisis SonarQube** - Verificación de quality gate
4. **Despliegue** - Upload a datastore vCenter, configuración VM, instalación vía SSH
5. **Notificaciones** - Mensajes wall y actualización de `/etc/profile.d/informacion.sh`
6. **Auditoría** - Registro completo en SQLite

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MÁQUINA DE DESARROLLO (SUSE 15)                       │
│                           172.30.188.137/26                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   ci_cd.sh      │────▶│  compile.sh     │────▶│  sonar_check.py │       │
│  │  (orquestador)  │     │  (compilación)  │     │  (API SonarQube)│       │
│  └────────┬────────┘     └─────────────────┘     └────────┬────────┘       │
│           │                                               │                 │
│           ▼                                               ▼                 │
│  ┌─────────────────┐                            ┌─────────────────┐        │
│  │ git_monitor.sh  │                            │ vcenter_api.py  │        │
│  │ (polling tags)  │                            │ (gestión VM)    │        │
│  └─────────────────┘                            └────────┬────────┘        │
│                                                          │                  │
│  ┌─────────────────┐     ┌─────────────────┐            │                  │
│  │ notify.sh       │     │ pipeline.db     │            │                  │
│  │ (wall + profile)│     │ (SQLite audit)  │            │                  │
│  └─────────────────┘     └─────────────────┘            │                  │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                    ┌──────────────────────────────────────┼───────────────┐
                    │            vCenter API               │               │
                    │  (Subir ISO, configurar CD, power)   │               │
                    └──────────────────────────────────────┼───────────────┘
                                                           │
                    ┌──────────────────────────────────────┼───────────────┐
                    │                   VM DESTINO "Releases"              │
                    │                   172.30.188.147/26                  │
                    │  SSH: mount ISO, copy, ejecutar install.sh           │
                    └──────────────────────────────────────────────────────┘
```

## Requisitos Previos

- SUSE Linux Enterprise 15
- Python 3.6+
- SQLite3
- Git
- yq (parser YAML)
- Acceso SSH a la VM destino (172.30.188.147)
- Acceso a Git (https://git.indra.es/git/GALTTCMC/GALTTCMC)
- Acceso a SonarQube (https://sonarqube.indra.es)
- Acceso a vCenter REST API

## Instalación

### Opción A: Instalación vía RPM (RECOMENDADA)

Para facilitar la instalación y distribución, se proporciona un archivo SPEC para generar un RPM:

```bash
# 1. Construir el RPM (desde máquina con acceso al código)
cd /ruta/al/proyecto/cicd
chmod +x build_rpm.sh
./build_rpm.sh

# 2. Copiar RPM a la máquina objetivo
scp ~/rpmbuild/RPMS/noarch/cicd-galttcmc-*.rpm agent@172.30.188.137:/tmp/

# 3. Instalar en la máquina objetivo
ssh agent@172.30.188.137
sudo rpm -ivh /tmp/cicd-galttcmc-*.rpm

# 4. Configurar credenciales
cd /home/agent/cicd
cp config/.env.example config/.env
nano config/.env  # Rellenar con credenciales reales

# 5. Copiar clave SSH a VM destino
ssh-copy-id root@172.30.188.147

# 6. Habilitar servicio
sudo systemctl enable cicd.service
sudo systemctl start cicd.service
```

**Ventajas del RPM:**
- ✅ Instalación automática de dependencias Python offline
- ✅ Configuración automática de permisos y estructura de directorios
- ✅ Inicialización automática de base de datos
- ✅ Generación automática de claves SSH
- ✅ Instalación del servicio systemd
- ✅ Facilita actualizaciones y desinstalación
- ✅ Auditoría completa de lo instalado (`rpm -ql cicd-galttcmc`)

**Más información:** Ver [RPM-BUILD-GUIDE.md](RPM-BUILD-GUIDE.md)

### Opción B: Instalación Manual (Fase 0)

### 1. Copiar ficheros a la máquina de desarrollo

```bash
# Desde tu máquina local (Windows)
scp -r cicd/ agent@172.30.188.137:/home/agent/
```

### 2. Conectar a la máquina de desarrollo

```bash
ssh agent@172.30.188.137
# Password: gal1$LEO
```

### 3. Ejecutar script de setup

```bash
cd /home/agent/cicd
chmod +x setup_phase0.sh
./setup_phase0.sh
```

El script `setup_phase0.sh` automáticamente:
- Instala dependencias Python (requests, PyYAML)
- Crea estructura de directorios
- Inicializa la base de datos SQLite
- Configura permisos de ejecución
- Verifica yq y git

### 4. Configurar credenciales

```bash
# Copiar y editar fichero de credenciales
cp config/.env.example config/.env
chmod 600 config/.env  # Proteger credenciales
nano config/.env

# Editar:
# - GIT_PASSWORD (token o contraseña de Git)
# - SONAR_TOKEN (obtener de sonarqube.indra.es)
# - VCENTER_USER / VCENTER_PASSWORD
```

### 5. Configurar clave SSH para la VM destino

```bash
# Generar clave SSH (si no existe)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copiar clave pública a la VM destino
ssh-copy-id -i ~/.ssh/id_rsa.pub root@172.30.188.147

# Probar conexión
ssh -i ~/.ssh/id_rsa root@172.30.188.147 "hostname"
```

### 6. Ajustar configuración (opcional)

```bash
nano config/ci_cd_config.yaml

# Verificar y ajustar:
# - general.polling_interval_seconds
# - git.tag_pattern
# - sonarqube.thresholds
# - vcenter.url / datacenter / datastore
# - target_vm.ip
```

### 7. Verificar instalación

```bash
# Verificar entorno completo con el orquestador
./ci_cd.sh verify

# O verificar componentes individuales:

# Verificar Python y módulos
python3.6 --version
python3.6 -c "import requests, yaml; print('OK')"

# Verificar SSH a VM destino
ssh -i ~/.ssh/id_rsa root@172.30.188.147 "hostname"

# Verificar base de datos
sqlite3 db/pipeline.db ".tables"

# Verificar Git
./scripts/git_monitor.sh verify

# Verificar compilación
./scripts/compile.sh verify

# Verificar despliegue
./scripts/deploy.sh verify
```

## Estructura de Ficheros

```
/home/agent/cicd/
├── ci_cd.sh                    # Orquestador principal
├── setup_phase0.sh             # Script de instalación
├── scripts/
│   ├── common.sh               # Funciones compartidas
│   ├── git_monitor.sh          # Monitorización de tags
│   ├── compile.sh              # Gestión de compilación
│   ├── deploy.sh               # Despliegue SSH
│   └── notify.sh               # Notificaciones
├── python/
│   ├── sonar_check.py          # API SonarQube
│   ├── vcenter_api.py          # API vCenter
│   └── requirements.txt        # Dependencias Python
├── config/
│   ├── ci_cd_config.yaml       # Configuración principal
│   ├── .env                    # Credenciales (NO COMMITEAR)
│   └── .env.example            # Ejemplo de credenciales
├── db/
│   ├── init_db.sql             # Esquema SQLite
│   └── pipeline.db             # Base de datos
└── logs/
    └── pipeline_YYYYMMDD.log   # Logs diarios
```

## Uso Básico

### Orquestador Principal (ci_cd.sh)

```bash
# Modo daemon (polling continuo cada 5 min)
./ci_cd.sh daemon

# Procesar tag específico manualmente
./ci_cd.sh --tag V01_02_03_04

# Ver estado de últimos despliegues
./ci_cd.sh status

# Ver últimos logs
./ci_cd.sh logs 100

# Inicializar base de datos
./ci_cd.sh init

# Verificar entorno
./ci_cd.sh verify
```

### Scripts Individuales

```bash
# Monitorización Git
./scripts/git_monitor.sh detect              # Detectar nuevo tag
./scripts/git_monitor.sh checkout V01_02_03  # Checkout de tag
./scripts/git_monitor.sh list                # Listar tags remotos
./scripts/git_monitor.sh status              # Estado de procesados
./scripts/git_monitor.sh verify              # Verificar configuración

# Compilación
./scripts/compile.sh                  # Pipeline completo (full)
./scripts/compile.sh prepare          # Solo preparar workspace
./scripts/compile.sh build            # Solo compilar
./scripts/compile.sh validate         # Solo validar ISO
./scripts/compile.sh clean            # Limpiar directorio de compilación
./scripts/compile.sh status           # Ver estado de compilación
./scripts/compile.sh verify           # Verificar prerequisitos

# Despliegue SSH
./scripts/deploy.sh                   # Despliegue completo (full)
./scripts/deploy.sh wait_ssh          # Esperar conexión SSH
./scripts/deploy.sh mount             # Solo montar ISO
./scripts/deploy.sh copy              # Solo copiar contenido
./scripts/deploy.sh install           # Solo ejecutar install.sh
./scripts/deploy.sh cleanup           # Limpiar recursos (umount, rm)
./scripts/deploy.sh status            # Estado de la VM
./scripts/deploy.sh verify            # Verificar prerequisitos

# Notificaciones
./scripts/notify.sh wall success V01_02_03           # Notificación wall
./scripts/notify.sh wall failure V01_02_03 "Error"   # Notificación de fallo
./scripts/notify.sh profile V01_02_03 DESPLEGADO     # Actualizar profile.d
./scripts/notify.sh both success V01_02_03           # Ambas notificaciones
./scripts/notify.sh show                             # Mostrar última notificación
./scripts/notify.sh test                             # Probar notificaciones

# Python - SonarQube
python3.6 python/sonar_check.py config/ci_cd_config.yaml V01_02_03_04

# Python - vCenter
python3.6 python/vcenter_api.py config/ci_cd_config.yaml upload_iso /path/to.iso
python3.6 python/vcenter_api.py config/ci_cd_config.yaml configure_cdrom
python3.6 python/vcenter_api.py config/ci_cd_config.yaml power_on
python3.6 python/vcenter_api.py config/ci_cd_config.yaml power_off
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status
```

## Servicio systemd

### Instalación con script

```bash
# Instalar servicio (recomendado)
sudo ./install_service.sh install

# Iniciar
sudo systemctl start cicd

# Ver estado
sudo ./install_service.sh status
```

### Instalación manual

```bash
# Copiar fichero de servicio
sudo cp cicd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cicd
sudo systemctl start cicd

# Ver logs en tiempo real
sudo journalctl -u cicd -f

# Reiniciar servicio
sudo systemctl restart cicd
```

### Desinstalación

```bash
sudo ./install_service.sh uninstall
```

## Web UI - Interfaz de Monitorización

### 🌐 Descripción

Interfaz web moderna para visualizar en tiempo real el estado del pipeline CI/CD:

- **Dashboard**: Métricas, gráficos de tendencias, últimos despliegues
- **Pipeline Runs**: Historial completo con filtros por estado
- **Logs Viewer**: Visualización de logs con búsqueda en tiempo real
- **SonarQube Results**: Análisis de calidad de código y tendencias
- **Dark Mode**: Tema oscuro/claro
- **Auto-refresh**: Actualización automática opcional

### Instalación de la Web UI

```bash
cd /home/agent/cicd
sudo ./install_web.sh install
```

El script automáticamente:
- ✅ Instala dependencias Python (Flask, Gunicorn)
- ✅ Configura firewall (puerto 8080)
- ✅ Instala servicio systemd `cicd-web`
- ✅ Inicia la aplicación web

### Acceso

Una vez instalado, accede desde cualquier navegador:

```
http://172.30.188.137:8080
```

### Gestión del Servicio Web

```bash
# Ver estado
sudo systemctl status cicd-web

# Reiniciar
sudo systemctl restart cicd-web

# Ver logs
sudo journalctl -u cicd-web -f

# Logs de aplicación
tail -f /home/agent/cicd/logs/web_access.log
tail -f /home/agent/cicd/logs/web_error.log
```

### API REST

La Web UI expone endpoints REST para integración:

```bash
# Estadísticas del dashboard
curl http://localhost:8080/api/dashboard/stats

# Lista de deployments (paginado)
curl http://localhost:8080/api/deployments?page=1&per_page=20&status=all

# Detalle de un deployment
curl http://localhost:8080/api/deployment/123

# Lista de archivos de log
curl http://localhost:8080/api/logs/list

# Ver contenido de un log
curl http://localhost:8080/api/logs/view/pipeline_20260226.log?lines=500

# Resultados SonarQube
curl http://localhost:8080/api/sonar/results

# Tendencias SonarQube (últimos 10 deployments)
curl http://localhost:8080/api/sonar/trends
```

**Documentación completa:** Ver [web/README.md](web/README.md)

## Configuración

### Fichero principal: config/ci_cd_config.yaml

```yaml
general:
  polling_interval_seconds: 300    # 5 minutos

git:
  repo_url: https://git.indra.es/git/GALTTCMC/GALTTCMC
  branch: WORKING_G2G_DEVELOPMENT
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"

sonarqube:
  url: https://sonarqube.indra.es
  project_key: GALTTCMC
  thresholds:
    coverage: 80.0
    bugs: 0
    vulnerabilities: 0
  allow_override: false    # Si true, continúa aunque falle quality gate

vcenter:
  url: https://vcenter.example.com
  vm_name: "Releases"

target_vm:
  ip: 172.30.188.147
  ssh_user: root
```

### Credenciales: config/.env

```bash
# Copiar plantilla
cp config/.env.example config/.env

# Editar con tus credenciales
GIT_PASSWORD=tu_password_git
SONAR_TOKEN=tu_token_sonar
VCENTER_USER=tu_usuario_vcenter
VCENTER_PASSWORD=tu_password_vcenter
```

## Base de Datos SQLite

### Tablas principales

| Tabla | Descripción |
|-------|-------------|
| `deployments` | Registro de despliegues |
| `build_logs` | Logs de compilación por fase |
| `sonar_results` | Resultados de análisis SonarQube |
| `execution_log` | Log detallado de ejecución |
| `processed_tags` | Tags procesados |

### Consultas útiles

```bash
# Últimos 5 despliegues
sqlite3 db/pipeline.db "SELECT tag_name, status, started_at FROM deployments ORDER BY id DESC LIMIT 5"

# Estadísticas (si existe la vista)
sqlite3 db/pipeline.db "SELECT * FROM v_deployment_stats" 2>/dev/null

# Resultados SonarQube
sqlite3 db/pipeline.db "SELECT tag, coverage, bugs, passed FROM sonar_results ORDER BY id DESC LIMIT 5"

# Ver todos los tags procesados
sqlite3 db/pipeline.db "SELECT * FROM processed_tags ORDER BY id DESC LIMIT 10"

# Ver logs de build
sqlite3 db/pipeline.db "SELECT deployment_id, phase, exit_code, duration FROM build_logs ORDER BY id DESC LIMIT 10"

# Despliegues fallidos
sqlite3 db/pipeline.db "SELECT tag_name, status, error_message, started_at FROM deployments WHERE status='failed' ORDER BY id DESC LIMIT 5"
```

## Troubleshooting

### Python 3.6 - Módulos no encontrados
```bash
# Añadir ~/.local/bin al PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Reinstalar dependencias
pip3.6 install --user requests PyYAML
```

### yq no encontrado
```bash
sudo zypper install yq
# o descargar binario
wget https://github.com/mikefarah/yq/releases/download/v4.35.1/yq_linux_amd64 -O ~/bin/yq
chmod +x ~/bin/yq
```

### Error de conexión SSH
```bash
# Regenerar claves
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
ssh-copy-id -i ~/.ssh/id_rsa.pub root@172.30.188.147
```

### Error en SQLite
```bash
# Reinicializar base de datos
rm db/pipeline.db
sqlite3 db/pipeline.db < db/init_db.sql
```

### El daemon no detecta nuevos tags
```bash
# Verificar configuración de repositorio
./scripts/git_monitor.sh verify

# Verificar patrón de tags
./scripts/git_monitor.sh list

# Comprobar credenciales Git
git ls-remote https://git.indra.es/git/GALTTCMC/GALTTCMC
```

### Error de conexión a vCenter
```bash
# Verificar credenciales
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status

# Verificar URL en config
grep -A5 "vcenter:" config/ci_cd_config.yaml

# Verificar conectividad
ping -c 4 172.30.188.136

# Probar conexión HTTPS
curl -k https://172.30.188.136/
```

### Compilación falla
```bash
# Ver últimos logs de compilación
ls -lt logs/compile_*.log | head -1 | awk '{print $NF}' | xargs tail -100

# Ver estado de la última compilación
./scripts/compile.sh status

# Ejecutar compilación manual paso a paso
./scripts/compile.sh verify        # Verificar prerequisitos
./scripts/compile.sh clean         # Limpiar workspace
./scripts/compile.sh prepare       # Preparar código
./scripts/compile.sh build         # Compilar
./scripts/compile.sh validate      # Validar ISO generado

# Verificar que existe el script de build
ls -la /home/agent/GALTTCMC/Development_TTCF/ttcf/utils/dvds/build_DVDs.sh
```

### SonarQube rechaza el análisis
```bash
# Verificar token
curl -u "$SONAR_TOKEN:" "https://sonarqube.indra.es/api/authentication/validate"

# Ejecutar análisis manual
python3.6 python/sonar_check.py config/ci_cd_config.yaml V01_02_03_04

# Ver configuración de thresholds
grep -A10 "thresholds:" config/ci_cd_config.yaml

# Si necesitas continuar sin quality gate, edita config:
# sonarqube.allow_override: true
```

### La VM no es accesible por SSH
```bash
# Verificar que la VM está encendida
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status
python3.6 python/vcenter_api.py config/ci_cd_config.yaml power_on

# Probar conexión manual
ssh -v -i ~/.ssh/id_rsa root@172.30.188.147

# Ping a la VM
ping -c 4 172.30.188.147

# Si falla, verificar script de deploy
./scripts/deploy.sh verify

# Si falla autenticación, regenerar claves
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
ssh-copy-id -i ~/.ssh/id_rsa.pub root@172.30.188.147

# Eliminar entrada conocida antigua si cambió la VM
ssh-keygen -R 172.30.188.147
```

### Ya hay una instancia del daemon corriendo
```bash
# Verificar PID
cat .cicd.pid

# Matar proceso antiguo si es necesario
kill $(cat .cicd.pid)
rm .cicd.pid

# O usar systemctl si es servicio
sudo systemctl stop cicd
```

## Logs

### Ver logs en tiempo real

```bash
# Log principal del daemon
tail -f logs/pipeline_$(date +%Y%m%d).log

# Logs del servicio systemd
sudo journalctl -u cicd -f

# Últimas líneas con el orquestador
./ci_cd.sh logs 100

# Ver todos los logs del día
cat logs/pipeline_$(date +%Y%m%d).log
```

### Archivos de log

| Fichero | Contenido |
|---------|-----------|
| `logs/pipeline_YYYYMMDD.log` | Log principal del día |
| `logs/compile_*.log` | Logs de compilación |
| `logs/deploy_*.log` | Logs de despliegue |
| `logs/service.log` | Stdout del servicio systemd |
| `logs/service_error.log` | Stderr del servicio systemd |

## Desarrollo y Pruebas

### Probar el pipeline en "seco"

```bash
# 1. Probar detección de tags sin procesar
./scripts/git_monitor.sh detect

# 2. Hacer checkout manual de un tag de prueba
./scripts/git_monitor.sh checkout V01_02_03_04

# 3. Probar compilación (sin ejecutar)
./scripts/compile.sh verify
./scripts/compile.sh status

# 4. Probar notificaciones
./scripts/notify.sh test

# 5. Ver estado general
./ci_cd.sh status
```

### Ejecutar pipeline completo manualmente

```bash
# Procesar un tag específico sin daemon
./ci_cd.sh --tag V01_02_03_04

# Esto ejecutará todas las fases:
# - Checkout del tag
# - Compilación
# - Análisis SonarQube
# - Despliegue en vCenter
# - Despliegue SSH
# - Notificaciones
```

## Contribuir

1. Los scripts bash deben:
   - Usar `set -euo pipefail` al inicio
   - Pasar shellcheck sin errores
   - Usar funciones de `common.sh` (log_info, log_error, etc.)
   - Documentar parámetros y comportamiento
2. Los scripts Python deben:
   - Ser compatibles con Python 3.6+
   - Evitar f-strings (usar `.format()` o `%`)
   - Incluir `from __future__ import print_function`
   - Usar `requests` en lugar de `pyvmomi`
3. Documentar cambios en este README
4. No commitear credenciales (usar `.env` y `.gitignore`)
5. Probar cambios localmente antes de commitear

## Licencia

Uso interno - Indra
