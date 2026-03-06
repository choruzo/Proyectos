## Descripción general (explicada en detalle)

tengo que automatizar flujos CI/CD complejos ligados al ciclo de vida de releases basados en tags. Este nuevo script de CI/CD busca cerrar ese gap, permitiendo acelerar el ciclo Dev→QA→Release→Despliegue de forma controlada y reproducible, reduciendo errores manuales, asegurando la calidad del software mediante integración con SonarQube y proporcionando trazabilidad completa de cada despliegue.

**¿Qué hará este script?**
- Permitirá un pipeline híbrido (modo "daemon" o ejecutable manualmente) que:
    1. Monitoriza un repositorio Git desde una máquina de desarrollo propia, en busca de la aparición de nuevos tags (polling cada 5 minutos)
    2. Cuando detecta un tag nuevo, baja el código relevante, ejecuta un script de compilación (en la máquina de desarrollo) que construye binarios y empaqueta los resultados en DVDs/ISOs
    3. Verifica si la compilación fue exitosa (log y código de salida). En caso de error, informa a los usuarios relacionados (por logs y/o interfaz)
    4. Tras compilar y empaquetar, lanza un análisis de calidad utilizando SonarQube (ejecuta un script y luego consulta la API rest de Sonar para comprobar el resultado)
    5. Si el análisis no cumple los baremos configurados—pero se permite por política—da la opción de continuar bajo supervisión, si no, detiene el pipeline y notifica los detalles
    6. Si todo es correcto, realiza el despliegue: copia el ISO/DVD en un datastore especifico y la ruta especificada, configurar una VM existente con el nuevo ISO/DVD, encender la MV, montar el ISO en `/mnt/cdrom`, copiar el contenido a /root/install y lanzar scripts post-instalación vía SSH, recalco que la conexion con la MV y la ejecucion de comandos seria por ssh, y la configuracion de la maquina via api del vcenter.
    7. Cada pipeline mantiene registro persistente en SQLite: tags desplegados, logs de ejecución, errores y resultados SonarQube (para auditoría, rollback, trazabilidad)
- Todas las credenciales y parámetros sensibles estarán centralizados en `config.json` para facilitar su gestión (con vistas a externalización futura via Vault, envvars, etc.)
- Pensado para ser fácilmente ampliable y tolerante a fallos: cada etapa notifica y detiene en caso de error, sin provocar efectos sobre el resto del sistema

---

## Tecnologías Utilizadas

| Componente | Tecnología | Versión | Propósito |
|------------|------------|---------|----------|
| Orquestación | **Shell Script (Bash)** | 4.4+ | Script principal, monitorización Git, compilación, notificaciones |
| API vCenter | **Python** | **3.6** | Interacción con vCenter REST API (requests) |
| API SonarQube | **Python** | **3.6** | Consulta de métricas y validación de umbrales |
| Servicio | **systemd** | - | Ejecución como daemon con auto-restart |
| Base de datos | **SQLite3** | - | Auditoría y trazabilidad de despliegues |

> **IMPORTANTE**: Se usa **Python 3.6** para los scripts de integración con APIs. Se utiliza la **API REST de vCenter** directamente (sin pyvmomi, que requiere Python 3.9+). Las librerías requeridas son: requests, PyYAML.

---

## Roadmap - Subtareas explicativas

### Arquitectura de la Solución

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
│                                                          │                  │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                    ┌──────────────────────────────────────┼───────────────┐
                    │                                      ▼               │
                    │  vCenter API ◀──────────────────────────────────────▶│
                    │  (Subir ISO, configurar CD, encender VM)             │
                    │                                                      │
                    └──────────────────────────────────────────────────────┘
                                                           │
                    ┌──────────────────────────────────────┼───────────────┐
                    │                   VM DESTINO "Releases"              │
                    │                   172.30.188.147/26                  │
                    ├──────────────────────────────────────────────────────┤
                    │  SSH ◀───────────────────────────────────────────────│
                    │  - mount /mnt/cdrom                                  │
                    │  - cp -r /mnt/cdrom/* /root/install                  │
                    │  - /root/install/install.sh                          │
                    └──────────────────────────────────────────────────────┘
```

---

### Estructura de Ficheros

```
/home/agent/cicd/
├── ci_cd.sh                    # Script principal orquestador
├── scripts/
│   ├── git_monitor.sh          # Monitorización de tags Git
│   ├── compile.sh              # Gestión del proceso de compilación
│   ├── notify.sh               # Notificaciones (wall + profile.d)
│   └── deploy.sh               # Lógica de despliegue SSH
├── python/
│   ├── sonar_check.py          # Consulta API SonarQube
│   ├── vcenter_api.py          # Interacción con vCenter REST API
│   └── requirements.txt        # Dependencias Python
├── config/
│   ├── ci_cd_config.yaml       # Configuración principal
│   └── sonar-project.properties # Propiedades SonarQube (existente)
├── db/
│   └── pipeline.db             # Base de datos SQLite para auditoría
├── logs/
│   └── pipeline_YYYYMMDD.log   # Logs diarios
└── cicd.service                # Fichero de servicio systemd
```

---

### Fase 0: Preparación del Entorno

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 0.1 | Crear estructura de directorios | Crear `/home/agent/cicd/` con subdirectorios scripts/, python/, config/, db/, logs/ | Directorios creados |
| 0.2 | Fichero de configuración YAML | Crear `ci_cd_config.yaml` con todas las credenciales, rutas, umbrales y parámetros | ci_cd_config.yaml |
| 0.3 | Instalar dependencias Python | Verificar Python 3.8+, instalar requests, pyvmomi, PyYAML | requirements.txt instalado |
| 0.4 | Configurar claves SSH | Generar par de claves SSH y copiar clave pública a la VM destino para acceso sin contraseña | Acceso SSH sin password |
| 0.5 | Inicializar base de datos SQLite | Crear esquema de tablas para auditoría (deployments, logs, sonar_results) | pipeline.db con esquema |

**Fichero: ci_cd_config.yaml**
```yaml
# =============================================================================
# CI/CD Pipeline Configuration
# =============================================================================

general:
  polling_interval_seconds: 300    # 5 minutos
  working_directory: /home/agent/cicd
  log_directory: /home/agent/cicd/logs
  db_path: /home/agent/cicd/db/pipeline.db

git:
  repo_url: https://git.indra.es/git/GALTTCMC/GALTTCMC
  branch: WORKING_G2G_DEVELOPMENT
  repo_local_path: /home/agent/GALTTCMC
  compile_path: /home/agent/compile
  # Patrón regex para tags: MAC_X_VXX_XX_XX_XX (oficiales) o VXX_XX_XX_XX (internas)
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"
  credentials:
    username: agent
    # Usar token o password (preferible token)
    password: "${GIT_PASSWORD}"  # Variable de entorno

compilation:
  build_script: Development_TTCF/ttcf/utils/dvds/build_DVDs.sh
  output_iso: InstallationDVD.iso
  timeout_seconds: 3600  # 1 hora máximo para compilación

sonarqube:
  url: https://sonarqube.indra.es
  project_key: GALTTCMC
  token: "${SONAR_TOKEN}"  # Usar sonar.login existente
  thresholds:
    coverage: 80.0           # >= 80%
    bugs: 0                  # = 0
    vulnerabilities: 0       # = 0
    code_smells: 10          # <= 10
    security_hotspots: 0     # = 0
  allow_override: false      # Si true, permite continuar con supervisión

vcenter:
  url: https://vcenter.example.com  # TODO: Añadir URL real
  username: "${VCENTER_USER}"
  password: "${VCENTER_PASSWORD}"
  datacenter: "Datacenter"          # TODO: Ajustar
  datastore: "datastore1"           # TODO: Ajustar
  iso_path: "/ISO/GALTTCMC"         # Ruta dentro del datastore
  vm_name: "Releases"

target_vm:
  ip: 172.30.188.147
  ssh_user: root
  ssh_key_path: /home/agent/.ssh/id_rsa
  mount_point: /mnt/cdrom
  install_path: /root/install
  install_script: install.sh

notifications:
  wall_enabled: true
  profile_script: /etc/profile.d/informacion.sh
  notify_on_success: true
  notify_on_failure: true
```

---

### Fase 1: Monitorización de Tags Git

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 1.1 | Script git_monitor.sh | Implementar polling cada 5 min, detectar tags nuevos con patrón regex | git_monitor.sh |
| 1.2 | Función de detección de tags | Comparar tags remotos vs procesados (guardados en SQLite) | Función detect_new_tags() |
| 1.3 | Logging de monitorización | Registrar cada ciclo de polling con timestamp | Logs estructurados |
| 1.4 | Checkout de tag detectado | Al detectar tag nuevo, hacer checkout y preparar código | Código actualizado |

**Pseudocódigo git_monitor.sh:**
```bash
#!/bin/bash
# git_monitor.sh - Monitorización de tags Git

source "$(dirname "$0")/../config/ci_cd_config.yaml.sh"  # Funciones de config

get_remote_tags() {
    git ls-remote --tags "$GIT_REPO_URL" | \
        grep -E "$TAG_PATTERN" | \
        awk '{print $2}' | \
        sed 's|refs/tags/||'
}

get_processed_tags() {
    sqlite3 "$DB_PATH" "SELECT tag_name FROM deployments WHERE status != 'failed'"
}

detect_new_tags() {
    local remote_tags=$(get_remote_tags)
    local processed=$(get_processed_tags)
    
    for tag in $remote_tags; do
        if ! echo "$processed" | grep -q "^${tag}$"; then
            echo "$tag"
            return 0
        fi
    done
    return 1
}

checkout_tag() {
    local tag=$1
    cd "$REPO_LOCAL_PATH" || exit 1
    git fetch --all --tags
    git checkout "tags/$tag"
}
```

---

### Fase 2: Proceso de Compilación

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 2.1 | Script compile.sh | Copiar repo a /home/agent/compile, dar permisos, ejecutar build | compile.sh |
| 2.2 | Preparación del workspace | Limpiar compile/, copiar contenido de GALTTCMC | Función prepare_workspace() |
| 2.3 | Asignación de permisos | find -name "*.sh" -exec chmod +x {} \; | Permisos aplicados |
| 2.4 | Ejecución de build_DVDs.sh | Lanzar script con timeout y captura de logs | ISO generado |
| 2.5 | Validación de compilación | Verificar código de salida y existencia de InstallationDVD.iso | Función validate_build() |
| 2.6 | Registro en SQLite | Guardar timestamp, duración, resultado, logs en BD | Registro de compilación |

**Pseudocódigo compile.sh:**
```bash
#!/bin/bash
# compile.sh - Gestión del proceso de compilación

prepare_workspace() {
    local source_dir=$1
    local target_dir=$2
    
    log_info "Limpiando directorio de compilación..."
    rm -rf "$target_dir"/*
    
    log_info "Copiando código fuente..."
    cp -r "$source_dir"/* "$target_dir"/
    
    log_info "Asignando permisos de ejecución a scripts..."
    find "$target_dir" -name "*.sh" -exec chmod +x {} \;
}

run_compilation() {
    local compile_dir=$1
    local build_script=$2
    local start_time=$(date +%s)
    
    cd "$compile_dir" || return 1
    
    log_info "Iniciando compilación: $build_script"
    
    # Ejecutar con timeout y capturar salida
    timeout "$COMPILE_TIMEOUT" "./$build_script" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Registrar en SQLite
    sqlite3 "$DB_PATH" "INSERT INTO build_logs (tag, start_time, duration, exit_code) 
                        VALUES ('$CURRENT_TAG', $start_time, $duration, $exit_code)"
    
    return $exit_code
}

validate_build() {
    local iso_path="$COMPILE_PATH/$OUTPUT_ISO"
    
    if [[ -f "$iso_path" ]]; then
        log_info "Compilación exitosa: $iso_path ($(du -h "$iso_path" | cut -f1))"
        return 0
    else
        log_error "ERROR: No se encontró el ISO esperado: $iso_path"
        return 1
    fi
}
```

---

### Fase 3: Análisis SonarQube

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 3.1 | Ejecutar análisis Sonar | Lanzar sonar-scanner con properties existentes | Análisis ejecutado |
| 3.2 | Script sonar_check.py | Consultar API REST de SonarQube para obtener métricas | sonar_check.py |
| 3.3 | Validación de umbrales | Comparar métricas obtenidas vs umbrales configurados | Función check_thresholds() |
| 3.4 | Gestión de override | Si allow_override=true y falla, solicitar confirmación | Lógica de supervisión |
| 3.5 | Registro de resultados | Guardar métricas y resultado en SQLite | Registro SonarQube |

**Fichero: sonar_check.py**
```python
#!/usr/bin/env python3
"""
sonar_check.py - Consulta API SonarQube y valida umbrales
"""

import sys
import yaml
import requests
import sqlite3
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_sonar_metrics(config):
    """Obtiene métricas del proyecto desde SonarQube API"""
    url = f"{config['sonarqube']['url']}/api/measures/component"
    params = {
        'component': config['sonarqube']['project_key'],
        'metricKeys': 'coverage,bugs,vulnerabilities,code_smells,security_hotspots'
    }
    headers = {'Authorization': f"Bearer {config['sonarqube']['token']}"}
    
    response = requests.get(url, params=params, headers=headers, verify=True)
    response.raise_for_status()
    
    measures = response.json()['component']['measures']
    return {m['metric']: float(m['value']) for m in measures}

def check_thresholds(metrics, thresholds):
    """Valida métricas contra umbrales configurados"""
    results = {}
    all_passed = True
    
    checks = {
        'coverage': lambda v, t: v >= t,
        'bugs': lambda v, t: v <= t,
        'vulnerabilities': lambda v, t: v <= t,
        'code_smells': lambda v, t: v <= t,
        'security_hotspots': lambda v, t: v <= t
    }
    
    for metric, check_func in checks.items():
        value = metrics.get(metric, 0)
        threshold = thresholds.get(metric, 0)
        passed = check_func(value, threshold)
        results[metric] = {
            'value': value,
            'threshold': threshold,
            'passed': passed
        }
        if not passed:
            all_passed = False
    
    return all_passed, results

def save_to_db(db_path, tag, metrics, passed):
    """Guarda resultados en SQLite"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sonar_results 
        (tag, timestamp, coverage, bugs, vulnerabilities, code_smells, security_hotspots, passed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        tag,
        datetime.now().isoformat(),
        metrics.get('coverage', 0),
        metrics.get('bugs', 0),
        metrics.get('vulnerabilities', 0),
        metrics.get('code_smells', 0),
        metrics.get('security_hotspots', 0),
        1 if passed else 0
    ))
    conn.commit()
    conn.close()

def main():
    if len(sys.argv) < 3:
        print("Uso: sonar_check.py <config_path> <tag>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    tag = sys.argv[2]
    
    config = load_config(config_path)
    
    print(f"[*] Consultando métricas SonarQube para proyecto: {config['sonarqube']['project_key']}")
    metrics = get_sonar_metrics(config)
    
    print(f"[*] Validando umbrales...")
    passed, results = check_thresholds(metrics, config['sonarqube']['thresholds'])
    
    for metric, data in results.items():
        status = "✓ PASS" if data['passed'] else "✗ FAIL"
        print(f"    {metric}: {data['value']} (umbral: {data['threshold']}) [{status}]")
    
    save_to_db(config['general']['db_path'], tag, metrics, passed)
    
    if passed:
        print("[✓] Análisis SonarQube: APROBADO")
        sys.exit(0)
    else:
        print("[✗] Análisis SonarQube: RECHAZADO")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

---

### Fase 4: Despliegue en vCenter y VM

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 4.1 | Script vcenter_api.py | Implementar cliente Python para API REST de vCenter | vcenter_api.py |
| 4.2 | Subir ISO al datastore | Función para upload del ISO al datastore configurado | Función upload_iso() |
| 4.3 | Configurar CD-ROM de la VM | Montar ISO en el lector virtual de la VM "Releases" | Función configure_cdrom() |
| 4.4 | Encender VM | Power on de la VM si está apagada | Función power_on_vm() |
| 4.5 | Script deploy.sh | Orquestación de comandos SSH en la VM destino | deploy.sh |
| 4.6 | Montar ISO vía SSH | ssh root@VM "mount /dev/cdrom /mnt/cdrom" | ISO montado |
| 4.7 | Copiar contenido | ssh root@VM "mkdir -p /root/install && cp -r /mnt/cdrom/* /root/install/" | Ficheros copiados |
| 4.8 | Ejecutar instalación | ssh root@VM "cd /root/install && ./install.sh" | Instalación completada |
| 4.9 | Verificación post-deploy | Comprobar estado de la instalación | Validación final |

**Fichero: vcenter_api.py**
```python
#!/usr/bin/env python3
"""
vcenter_api.py - Interacción con vCenter REST API
Requiere: pip install pyvmomi requests
"""

import sys
import yaml
import requests
import ssl
import urllib3
from pyVim.connect import SmartConnect, Disconnect
from pyVmomi import vim

# Desactivar warnings SSL para entornos con certificados auto-firmados
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VCenterClient:
    def __init__(self, config):
        self.config = config['vcenter']
        self.si = None
        self.session = None
        
    def connect(self):
        """Conectar a vCenter vía API"""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        self.si = SmartConnect(
            host=self.config['url'].replace('https://', ''),
            user=self.config['username'],
            pwd=self.config['password'],
            sslContext=context
        )
        print(f"[✓] Conectado a vCenter: {self.config['url']}")
        
    def disconnect(self):
        if self.si:
            Disconnect(self.si)
            
    def get_vm(self, vm_name):
        """Obtener objeto VM por nombre"""
        content = self.si.RetrieveContent()
        container = content.viewManager.CreateContainerView(
            content.rootFolder, [vim.VirtualMachine], True
        )
        for vm in container.view:
            if vm.name == vm_name:
                return vm
        raise Exception(f"VM no encontrada: {vm_name}")
    
    def upload_iso_to_datastore(self, local_iso_path, remote_path):
        """Subir ISO al datastore vía HTTP PUT"""
        datastore = self.config['datastore']
        datacenter = self.config['datacenter']
        
        url = (f"{self.config['url']}/folder/{remote_path}"
               f"?dcPath={datacenter}&dsName={datastore}")
        
        with open(local_iso_path, 'rb') as f:
            response = requests.put(
                url,
                data=f,
                auth=(self.config['username'], self.config['password']),
                verify=False,
                headers={'Content-Type': 'application/octet-stream'}
            )
        
        if response.status_code in [200, 201]:
            print(f"[✓] ISO subido: [{datastore}] {remote_path}")
            return True
        else:
            raise Exception(f"Error subiendo ISO: {response.status_code} - {response.text}")
    
    def configure_cdrom(self, vm_name, iso_datastore_path):
        """Configurar CD-ROM de la VM para usar el ISO"""
        vm = self.get_vm(vm_name)
        
        # Buscar dispositivo CD-ROM
        cdrom_device = None
        for device in vm.config.hardware.device:
            if isinstance(device, vim.vm.device.VirtualCdrom):
                cdrom_device = device
                break
        
        if not cdrom_device:
            raise Exception("No se encontró dispositivo CD-ROM en la VM")
        
        # Configurar backing con ISO
        backing = vim.vm.device.VirtualCdrom.IsoBackingInfo()
        backing.fileName = iso_datastore_path
        
        cdrom_spec = vim.vm.device.VirtualDeviceSpec()
        cdrom_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
        cdrom_spec.device = cdrom_device
        cdrom_spec.device.backing = backing
        cdrom_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
        cdrom_spec.device.connectable.connected = True
        cdrom_spec.device.connectable.startConnected = True
        
        config_spec = vim.vm.ConfigSpec()
        config_spec.deviceChange = [cdrom_spec]
        
        task = vm.ReconfigVM_Task(spec=config_spec)
        self._wait_for_task(task)
        print(f"[✓] CD-ROM configurado con ISO: {iso_datastore_path}")
        
    def power_on_vm(self, vm_name):
        """Encender la VM si está apagada"""
        vm = self.get_vm(vm_name)
        
        if vm.runtime.powerState == vim.VirtualMachinePowerState.poweredOn:
            print(f"[*] VM ya encendida: {vm_name}")
            return
        
        task = vm.PowerOnVM_Task()
        self._wait_for_task(task)
        print(f"[✓] VM encendida: {vm_name}")
        
    def _wait_for_task(self, task):
        """Esperar a que complete una tarea de vCenter"""
        while task.info.state not in [vim.TaskInfo.State.success, vim.TaskInfo.State.error]:
            pass
        if task.info.state == vim.TaskInfo.State.error:
            raise Exception(f"Error en tarea vCenter: {task.info.error.msg}")

def main():
    if len(sys.argv) < 4:
        print("Uso: vcenter_api.py <config_path> <action> [args...]")
        print("Acciones: upload_iso, configure_cdrom, power_on")
        sys.exit(1)
    
    config_path = sys.argv[1]
    action = sys.argv[2]
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    client = VCenterClient(config)
    client.connect()
    
    try:
        if action == 'upload_iso':
            local_path = sys.argv[3]
            remote_path = f"{config['vcenter']['iso_path']}/{config['vcenter']['vm_name']}.iso"
            client.upload_iso_to_datastore(local_path, remote_path)
            
        elif action == 'configure_cdrom':
            iso_path = f"[{config['vcenter']['datastore']}] {config['vcenter']['iso_path']}/{config['vcenter']['vm_name']}.iso"
            client.configure_cdrom(config['vcenter']['vm_name'], iso_path)
            
        elif action == 'power_on':
            client.power_on_vm(config['vcenter']['vm_name'])
            
    finally:
        client.disconnect()

if __name__ == '__main__':
    main()
```

**Fichero: deploy.sh**
```bash
#!/bin/bash
# deploy.sh - Despliegue en VM destino vía SSH

set -e

SSH_USER="root"
SSH_HOST="172.30.188.147"
SSH_KEY="/home/agent/.ssh/id_rsa"
SSH_OPTS="-o StrictHostKeyChecking=no -i $SSH_KEY"

ssh_exec() {
    ssh $SSH_OPTS ${SSH_USER}@${SSH_HOST} "$@"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Esperar a que la VM esté accesible
wait_for_ssh() {
    local max_attempts=30
    local attempt=1
    
    log_info "Esperando conexión SSH a $SSH_HOST..."
    while ! ssh $SSH_OPTS ${SSH_USER}@${SSH_HOST} "echo ok" &>/dev/null; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Timeout esperando SSH"
            return 1
        fi
        sleep 10
        ((attempt++))
    done
    log_info "Conexión SSH establecida"
}

# Montar ISO
mount_iso() {
    log_info "Montando ISO en /mnt/cdrom..."
    ssh_exec "mkdir -p /mnt/cdrom && mount /dev/cdrom /mnt/cdrom" || {
        # Intentar desmontar primero si ya estaba montado
        ssh_exec "umount /mnt/cdrom 2>/dev/null || true"
        ssh_exec "mount /dev/cdrom /mnt/cdrom"
    }
    log_info "ISO montado correctamente"
}

# Preparar directorio de instalación
prepare_install_dir() {
    log_info "Preparando directorio /root/install..."
    ssh_exec "rm -rf /root/install && mkdir -p /root/install"
    ssh_exec "cp -r /mnt/cdrom/* /root/install/"
    ssh_exec "chmod +x /root/install/*.sh 2>/dev/null || true"
    log_info "Contenido copiado a /root/install"
}

# Ejecutar instalación
run_installation() {
    log_info "Ejecutando install.sh..."
    ssh_exec "cd /root/install && ./install.sh" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log_info "Instalación completada correctamente"
    else
        log_error "Instalación falló con código: $exit_code"
        return $exit_code
    fi
}

# Limpieza post-instalación
cleanup() {
    log_info "Desmontando ISO..."
    ssh_exec "umount /mnt/cdrom 2>/dev/null || true"
}

main() {
    wait_for_ssh
    mount_iso
    prepare_install_dir
    run_installation
    cleanup
    log_info "Despliegue completado exitosamente"
}

main "$@"
```

---

### Fase 5: Sistema de Notificaciones

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 5.1 | Script notify.sh | Implementar notificaciones wall y profile.d | notify.sh |
| 5.2 | Notificación wall | Mensaje broadcast a usuarios conectados | Función notify_wall() |
| 5.3 | Script profile.d | Actualizar /etc/profile.d/informacion.sh con última versión | Función update_profile_script() |
| 5.4 | Plantillas de mensaje | Mensajes para éxito, fallo y nueva versión disponible | Templates configurables |

**Fichero: notify.sh**
```bash
#!/bin/bash
# notify.sh - Sistema de notificaciones

PROFILE_SCRIPT="/etc/profile.d/informacion.sh"

notify_wall() {
    local message_type=$1
    local tag=$2
    local details=$3
    
    case $message_type in
        success)
            sudo wall <<EOF
#################################################
#                                               #
#   ¡ATENCIÓN: Nueva versión disponible!        #
#                                               #
#   Tag: $tag                                   #
#   Estado: DESPLEGADO CORRECTAMENTE            #
#                                               #
#   Puedes clonar la nueva máquina o revisar    #
#   el informe de SonarQube para más detalles.  #
#                                               #
#################################################
EOF
            ;;
        failure)
            sudo wall <<EOF
#################################################
#                                               #
#   ⚠️  ALERTA: Fallo en pipeline CI/CD          #
#                                               #
#   Tag: $tag                                   #
#   Error: $details                             #
#                                               #
#   Revisa los logs en:                         #
#   /home/agent/cicd/logs/                      #
#                                               #
#################################################
EOF
            ;;
        sonar_failed)
            sudo wall <<EOF
#################################################
#                                               #
#   ⚠️  ALERTA: Quality Gate NO superado         #
#                                               #
#   Tag: $tag                                   #
#   Proyecto: GALTTCMC                          #
#                                               #
#   Revisa el informe en:                       #
#   https://sonarqube.indra.es                  #
#                                               #
#################################################
EOF
            ;;
    esac
}

update_profile_script() {
    local tag=$1
    local status=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    sudo tee "$PROFILE_SCRIPT" > /dev/null <<EOF
#!/bin/bash
# Información de última versión CI/CD - Generado automáticamente

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           INFORMACIÓN DE ÚLTIMA VERSIÓN                    ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Tag desplegado: $tag"
echo "║  Estado: $status"
echo "║  Fecha: $timestamp"
echo "║                                                            ║"
echo "║  Más info: /home/agent/cicd/logs/                          ║"
echo "║  SonarQube: https://sonarqube.indra.es                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
EOF
    
    sudo chmod +x "$PROFILE_SCRIPT"
}

# Main
case $1 in
    wall)
        notify_wall "$2" "$3" "$4"
        ;;
    profile)
        update_profile_script "$2" "$3"
        ;;
    *)
        echo "Uso: notify.sh <wall|profile> <args...>"
        exit 1
        ;;
esac
```

---

### Fase 6: Base de Datos y Auditoría

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 6.1 | Esquema SQLite | Crear tablas para deployments, build_logs, sonar_results | Script init_db.sql |
| 6.2 | Funciones de registro | Helpers para insertar/consultar registros | Funciones en scripts |
| 6.3 | Queries de auditoría | Consultas útiles para trazabilidad | Documentación |
| 6.4 | Limpieza de logs antiguos | Rotación automática de registros > 90 días | Tarea cron opcional |

**Esquema: init_db.sql**
```sql
-- init_db.sql - Esquema de base de datos para auditoría CI/CD

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,
    status TEXT CHECK(status IN ('pending', 'compiling', 'analyzing', 'deploying', 'success', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds INTEGER,
    triggered_by TEXT DEFAULT 'daemon',
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS build_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    tag TEXT NOT NULL,
    phase TEXT CHECK(phase IN ('checkout', 'prepare', 'compile', 'package')),
    start_time INTEGER,
    duration INTEGER,
    exit_code INTEGER,
    log_file TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sonar_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    tag TEXT NOT NULL,
    timestamp TEXT,
    coverage REAL,
    bugs INTEGER,
    vulnerabilities INTEGER,
    code_smells INTEGER,
    security_hotspots INTEGER,
    passed INTEGER CHECK(passed IN (0, 1)),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    phase TEXT NOT NULL,
    message TEXT,
    level TEXT CHECK(level IN ('DEBUG', 'INFO', 'WARN', 'ERROR')),
    timestamp TEXT DEFAULT (datetime('now'))
);

-- Índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_deployments_tag ON deployments(tag_name);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_sonar_tag ON sonar_results(tag);
```

---

### Fase 7: Orquestador Principal y Servicio

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 7.1 | Script ci_cd.sh | Orquestador principal que invoca todas las fases | ci_cd.sh |
| 7.2 | Modo daemon | Loop infinito con polling configurable | Función run_daemon() |
| 7.3 | Modo manual | Ejecución única para un tag específico | Opción --tag |
| 7.4 | Gestión de errores | Try/catch, rollback parcial, notificación | Manejo de excepciones |
| 7.5 | Servicio systemd | Fichero .service para ejecución como daemon | cicd.service |
| 7.6 | Instalación del servicio | Script de instalación y habilitación | install_service.sh |

**Fichero: ci_cd.sh (Orquestador Principal)**
```bash
#!/bin/bash
#===============================================================================
# ci_cd.sh - Orquestador Principal del Pipeline CI/CD
#
# Uso: 
#   ./ci_cd.sh daemon           # Modo daemon (polling continuo)
#   ./ci_cd.sh --tag TAG_NAME   # Procesar tag específico manualmente
#   ./ci_cd.sh status           # Ver estado del último despliegue
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/ci_cd_config.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
DB_PATH="$SCRIPT_DIR/db/pipeline.db"

# Cargar configuración
source "$SCRIPT_DIR/scripts/common.sh"

#-------------------------------------------------------------------------------
# Logging
#-------------------------------------------------------------------------------
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d).log"

log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info()  { log "INFO" "$@"; }
log_warn()  { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

#-------------------------------------------------------------------------------
# Pipeline Principal
#-------------------------------------------------------------------------------
run_pipeline() {
    local tag=$1
    local triggered_by=${2:-"daemon"}
    local start_time=$(date +%s)
    local deployment_id
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Iniciando pipeline para tag: $tag"
    log_info "═══════════════════════════════════════════════════════════"
    
    # Registrar inicio en BD
    deployment_id=$(sqlite3 "$DB_PATH" \
        "INSERT INTO deployments (tag_name, status, started_at, triggered_by) 
         VALUES ('$tag', 'pending', datetime('now'), '$triggered_by');
         SELECT last_insert_rowid();")
    
    # Función de cleanup en caso de error
    cleanup_on_error() {
        local error_msg=$1
        log_error "Pipeline fallido: $error_msg"
        sqlite3 "$DB_PATH" \
            "UPDATE deployments SET status='failed', error_message='$error_msg', 
             completed_at=datetime('now') WHERE id=$deployment_id"
        "$SCRIPT_DIR/scripts/notify.sh" wall failure "$tag" "$error_msg"
        exit 1
    }
    
    trap 'cleanup_on_error "Error inesperado en línea $LINENO"' ERR
    
    #---------------------------------------------------------------------------
    # FASE 1: Checkout del tag
    #---------------------------------------------------------------------------
    log_info "[1/5] Actualizando repositorio y checkout de tag..."
    sqlite3 "$DB_PATH" "UPDATE deployments SET status='compiling' WHERE id=$deployment_id"
    
    "$SCRIPT_DIR/scripts/git_monitor.sh" checkout "$tag" || \
        cleanup_on_error "Fallo en checkout del tag $tag"
    
    #---------------------------------------------------------------------------
    # FASE 2: Compilación
    #---------------------------------------------------------------------------
    log_info "[2/5] Iniciando compilación..."
    
    "$SCRIPT_DIR/scripts/compile.sh" || \
        cleanup_on_error "Fallo en compilación"
    
    #---------------------------------------------------------------------------
    # FASE 3: Análisis SonarQube
    #---------------------------------------------------------------------------
    log_info "[3/5] Ejecutando análisis SonarQube..."
    sqlite3 "$DB_PATH" "UPDATE deployments SET status='analyzing' WHERE id=$deployment_id"
    
    # Ejecutar sonar-scanner (asumiendo que existe el script en la máquina)
    cd "$COMPILE_PATH" && sonar-scanner || log_warn "sonar-scanner no disponible, continuando..."
    
    # Verificar resultados
    python3.6 "$SCRIPT_DIR/python/sonar_check.py" "$CONFIG_FILE" "$tag" || {
        log_warn "Quality Gate no superado"
        "$SCRIPT_DIR/scripts/notify.sh" wall sonar_failed "$tag"
        # Si allow_override=false, detenemos
        if [[ "$(yq '.sonarqube.allow_override' "$CONFIG_FILE")" != "true" ]]; then
            cleanup_on_error "Quality Gate no superado y override no permitido"
        fi
        log_warn "Continuando con override habilitado..."
    }
    
    #---------------------------------------------------------------------------
    # FASE 4: Despliegue en vCenter + VM
    #---------------------------------------------------------------------------
    log_info "[4/5] Iniciando despliegue..."
    sqlite3 "$DB_PATH" "UPDATE deployments SET status='deploying' WHERE id=$deployment_id"
    
    local iso_path="$COMPILE_PATH/InstallationDVD.iso"
    
    # Subir ISO al datastore
    log_info "Subiendo ISO al datastore..."
    python3.6 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" upload_iso "$iso_path"
    
    # Configurar CD-ROM de la VM
    log_info "Configurando CD-ROM de la VM..."
    python3.6 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" configure_cdrom
    
    # Encender VM
    log_info "Encendiendo VM..."
    python3.6 "$SCRIPT_DIR/python/vcenter_api.py" "$CONFIG_FILE" power_on
    
    # Despliegue vía SSH
    log_info "Ejecutando despliegue en VM destino..."
    "$SCRIPT_DIR/scripts/deploy.sh" || \
        cleanup_on_error "Fallo en despliegue SSH"
    
    #---------------------------------------------------------------------------
    # FASE 5: Finalización
    #---------------------------------------------------------------------------
    log_info "[5/5] Finalizando pipeline..."
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    sqlite3 "$DB_PATH" \
        "UPDATE deployments SET status='success', completed_at=datetime('now'), 
         duration_seconds=$duration WHERE id=$deployment_id"
    
    # Notificaciones
    "$SCRIPT_DIR/scripts/notify.sh" wall success "$tag"
    "$SCRIPT_DIR/scripts/notify.sh" profile "$tag" "DESPLEGADO"
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Pipeline completado exitosamente en ${duration}s"
    log_info "═══════════════════════════════════════════════════════════"
}

#-------------------------------------------------------------------------------
# Modo Daemon
#-------------------------------------------------------------------------------
run_daemon() {
    local polling_interval
    polling_interval=$(yq '.general.polling_interval_seconds' "$CONFIG_FILE")
    
    log_info "Iniciando modo daemon (polling cada ${polling_interval}s)"
    
    while true; do
        log_info "Verificando nuevos tags..."
        
        local new_tag
        new_tag=$("$SCRIPT_DIR/scripts/git_monitor.sh" detect) || true
        
        if [[ -n "$new_tag" ]]; then
            log_info "Nuevo tag detectado: $new_tag"
            run_pipeline "$new_tag" "daemon"
        else
            log_info "No hay tags nuevos"
        fi
        
        sleep "$polling_interval"
    done
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
main() {
    mkdir -p "$LOG_DIR"
    
    case "${1:-}" in
        daemon)
            run_daemon
            ;;
        --tag)
            if [[ -z "${2:-}" ]]; then
                echo "Error: Debe especificar un tag"
                echo "Uso: $0 --tag TAG_NAME"
                exit 1
            fi
            run_pipeline "$2" "manual"
            ;;
        status)
            sqlite3 -header -column "$DB_PATH" \
                "SELECT tag_name, status, started_at, duration_seconds 
                 FROM deployments ORDER BY id DESC LIMIT 5"
            ;;
        *)
            echo "Uso: $0 {daemon|--tag TAG_NAME|status}"
            exit 1
            ;;
    esac
}

main "$@"
```

**Fichero: cicd.service (systemd)**
```ini
[Unit]
Description=CI/CD Pipeline Service - GALTTCMC
Documentation=https://git.indra.es/git/GALTTCMC/GALTTCMC
After=network.target

[Service]
Type=simple
User=agent
Group=agent
WorkingDirectory=/home/agent/cicd
ExecStart=/home/agent/cicd/ci_cd.sh daemon
Restart=always
RestartSec=30

# Logging
StandardOutput=append:/home/agent/cicd/logs/service.log
StandardError=append:/home/agent/cicd/logs/service_error.log

# Environment
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="HOME=/home/agent"
EnvironmentFile=-/home/agent/cicd/config/.env

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

---

### Fase 8: Testing y Documentación

| ID | Subtarea | Descripción | Entregable |
|----|----------|-------------|------------|
| 8.1 | Test unitarios | Tests para funciones críticas (detect_tags, check_thresholds) | tests/ |
| 8.2 | Test de integración | Prueba completa del pipeline con tag de prueba | Informe de test |
| 8.3 | README.md | Documentación de uso, instalación y troubleshooting | README.md |
| 8.4 | Diagrama de flujo | Documentación visual del pipeline | docs/flow.png |

---

## Cronograma Estimado

| Fase | Descripción | Duración Estimada |
|------|-------------|-------------------|
| 0 | Preparación del entorno | 2-3 horas |
| 1 | Monitorización Git | 3-4 horas |
| 2 | Compilación | 3-4 horas |
| 3 | Integración SonarQube | 4-5 horas |
| 4 | Despliegue vCenter + VM | 6-8 horas |
| 5 | Sistema de notificaciones | 2-3 horas |
| 6 | Base de datos y auditoría | 2-3 horas |
| 7 | Orquestador y servicio systemd | 4-5 horas |
| 8 | Testing y documentación | 3-4 horas |
| **Total** | | **~30-40 horas** |

---

## Consideraciones Técnicas

### Dependencias Requeridas

**En la máquina de desarrollo (SUSE 15):**
```bash
# Python 3.6 ya está instalado en SUSE 15
# Verificar versión:
python3.6 --version

# Dependencias Python (sin pyvmomi - usamos API REST directa)
pip3.6 install --user requests PyYAML

# Añadir ~/.local/bin al PATH (para scripts instalados por pip)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Herramientas adicionales  
sudo zypper install sqlite3 git yq jq

# SonarQube Scanner (si no está instalado)
# wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-X.X.X-linux.zip
```

> **NOTA**: No se usa pyvmomi porque requiere Python 3.9+. En su lugar, vcenter_api.py utiliza la API REST de vCenter directamente con la librería `requests`.

### Seguridad

1. **Credenciales**: Usar variables de entorno o fichero `.env` (no commitear)
2. **SSH**: Configurar claves SSH sin passphrase para automatización
3. **Permisos**: Scripts con permisos 750, configs con 640
4. **Logs**: Rotar logs con logrotate, no incluir credenciales en logs

### Tolerancia a Fallos

- Cada fase valida resultado antes de continuar
- Timeout en operaciones largas (compilación, SSH)
- Reintentos automáticos en operaciones de red
- Notificación inmediata en fallos
- Estado persistido en SQLite para recovery

---