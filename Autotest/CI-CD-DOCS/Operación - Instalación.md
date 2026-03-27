# 🛠️ Operación - Instalación

## Visión General

Guía completa de instalación del pipeline CI/CD GALTTCMC desde cero.

**Relacionado con**:
- [[01 - Quick Start#Inicialización y Setup]]
- [[Operación - Mantenimiento]] - Tareas posteriores
- [[Arquitectura del Pipeline]] - Sistema a instalar

---

## Prerequisites

### Hardware

- **CPU**: 4+ cores recomendado
- **RAM**: 8 GB mínimo, 16 GB recomendado
- **Disco**: 100 GB espacio libre
- **Red**: Acceso a YOUR_GIT_SERVER, YOUR_SONARQUBE_SERVER, vCenter

### Software Base

```bash
# SUSE Linux 15
# Python 3.6+
# Git 2.x
# SQLite 3.x
# SSH client
```

---

## Instalación Paso a Paso

### 1. Preparación del Entorno

```bash
# Crear usuario agent si no existe
sudo useradd -m -s /bin/bash agent
sudo passwd agent

# Cambiar a usuario agent
su - agent
cd ~
```

### 2. Clonar/Copiar Pipeline

```bash
# Opción A: Desde repositorio
git clone https://internal-repo/cicd-galttcmc.git cicd

# Opción B: Copiar archivos
mkdir -p /home/YOUR_USER/cicd
# Copiar archivos del pipeline
```

### 3. Instalar Dependencias del Sistema

```bash
# Como root
sudo zypper refresh
sudo zypper install -y git python3 python3-pip sqlite3 yq

# Verificar versiones
git --version
python3 --version
sqlite3 --version
```

### 4. Instalar Dependencias Python

```bash
cd /home/YOUR_USER/cicd

# Para el pipeline
pip3 install --user requests PyYAML urllib3

# Para la Web UI
cd web
pip3 install --user -r requirements.txt
cd ..
```

### 5. Configurar Credenciales

```bash
# Copiar template
cp config/.env.example config/.env

# Editar con credenciales reales
nano config/.env
```

**Contenido de `.env`**:
```bash
GIT_USER=automation_user
GIT_PASSWORD=ghp_xxxxxxxxxxxxxxxxxx
SONAR_TOKEN=squ_yyyyyyyyyyyyyyyy
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=vcenter_pass_here
```

**⚠️ Importante**: Nunca commitear `.env`

### 6. Configurar SSH Key

```bash
# Generar key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copiar public key a target VM
ssh-copy-id YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP

# Verificar acceso
ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP 'echo OK'
```

### 7. Inicializar Base de Datos

```bash
cd /home/YOUR_USER/cicd
./ci_cd.sh init

# Verificar
sqlite3 db/pipeline.db ".tables"
# Output: build_logs  deployments  execution_log  processed_tags  sonar_results
```

### 8. Verificar Configuración

```bash
./ci_cd.sh verify

# Output esperado:
# ✓ Git disponible
# ✓ Python 3.6+ disponible
# ✓ SQLite disponible
# ✓ SSH key configurada
# ✓ Base de datos inicializada
# ✓ Configuración válida
```

### 9. Instalar Servicio systemd (Pipeline)

```bash
sudo ./install_service.sh install

# Verificar instalación
sudo systemctl status cicd

# Habilitar auto-start
sudo systemctl enable cicd

# Iniciar servicio
sudo systemctl start cicd
```

### 10. Instalar Servicio Web UI

```bash
sudo ./install_web.sh install

# Verificar
sudo systemctl status cicd-web

# Habilitar y arrancar
sudo systemctl enable cicd-web
sudo systemctl start cicd-web
```

### 11. Verificar Instalación Completa

```bash
# Ver logs del pipeline
sudo journalctl -u cicd -f

# Acceder a Web UI
curl http://localhost:8080
# O desde navegador: http://YOUR_PIPELINE_HOST_IP:8080

# Test manual de tag
./ci_cd.sh --tag TEST_V99_99_99_99
```

---

## Configuración Avanzada

### Firewall (si aplica)

```bash
# Abrir puerto 8080 para Web UI
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### Rotación de Logs

```bash
sudo nano /etc/logrotate.d/cicd
```

**Contenido**:
```
/home/YOUR_USER/cicd/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 agent agent
}
```

### Backup Automático

```bash
# Añadir a crontab
crontab -e

# Backup DB diario a las 3 AM
0 3 * * * /home/YOUR_USER/cicd/scripts/backup_db.sh
```

---

## Troubleshooting Instalación

### Error: Python module not found

```bash
# Reinstalar con usuario correcto
pip3 install --user requests PyYAML urllib3
```

### Error: systemd service fails to start

```bash
# Ver logs detallados
sudo journalctl -u cicd -n 50 --no-pager

# Verificar permisos
ls -la /home/YOUR_USER/cicd/ci_cd.sh
chmod +x /home/YOUR_USER/cicd/ci_cd.sh
```

### Error: Database locked

```bash
# Activar WAL mode
sqlite3 db/pipeline.db "PRAGMA journal_mode = WAL;"
```

---

## Desinstalación

```bash
# Detener servicios
sudo systemctl stop cicd cicd-web
sudo systemctl disable cicd cicd-web

# Remover servicios
sudo rm /etc/systemd/system/cicd.service
sudo rm /etc/systemd/system/cicd-web.service
sudo systemctl daemon-reload

# Opcional: Eliminar datos
rm -rf /home/YOUR_USER/cicd
```

---

## Enlaces Relacionados

- [[01 - Quick Start]]
- [[Operación - Mantenimiento]]
- [[Operación - Troubleshooting]]
