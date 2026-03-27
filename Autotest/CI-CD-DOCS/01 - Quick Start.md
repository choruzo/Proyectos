# 🚀 Quick Start - Comandos Esenciales

## Prerequisitos

- Acceso SSH a `YOUR_PIPELINE_HOST_IP` como usuario `agent`
- Credenciales configuradas en `config/.env`
- Servicios systemd instalados

**Guía completa**: [[Operación - Instalación]]

---

## Comandos Básicos del Pipeline

### Verificar Estado del Sistema

```bash
cd /home/YOUR_USER/cicd

# Verificar prerequisites
./ci_cd.sh verify

# Ver estado de deployments recientes
./ci_cd.sh status

# Ver logs en tiempo real
./ci_cd.sh logs 100
```

### Ejecutar Pipeline

```bash
# Modo daemon (polling continuo cada 5 min)
./ci_cd.sh daemon

# Procesar un tag específico manualmente
./ci_cd.sh --tag MAC_1_V24_02_15_01

# Solo detectar nuevos tags sin procesar
./scripts/git_monitor.sh detect
```

**Referencia completa**: [[Arquitectura del Pipeline#Comandos]]

---

## Gestión del Servicio systemd

### Pipeline Principal

```bash
# Ver estado
sudo systemctl status cicd

# Iniciar/detener/reiniciar
sudo systemctl start cicd
sudo systemctl stop cicd
sudo systemctl restart cicd

# Ver logs en tiempo real
sudo journalctl -u cicd -f

# Ver últimas 50 líneas
sudo journalctl -u cicd -n 50 --no-pager
```

### Web UI

```bash
# Ver estado
sudo systemctl status cicd-web

# Controlar servicio
sudo systemctl start cicd-web
sudo systemctl restart cicd-web

# Ver logs
sudo journalctl -u cicd-web -f
tail -f logs/web_access.log
tail -f logs/web_error.log
```

**Referencia completa**: [[Operación - Monitorización#Systemd]]

---

## Consultas a la Base de Datos

```bash
cd /home/YOUR_USER/cicd

# Ver deployments recientes
sqlite3 db/pipeline.db "SELECT * FROM v_recent_deployments LIMIT 10"

# Estadísticas generales
sqlite3 db/pipeline.db "SELECT * FROM v_deployment_stats"

# Deployments fallidos
sqlite3 db/pipeline.db "SELECT tag_name, status, error_message FROM deployments WHERE status='failed' ORDER BY started_at DESC LIMIT 5"

# Tags procesados
sqlite3 db/pipeline.db "SELECT * FROM processed_tags ORDER BY detected_at DESC LIMIT 10"
```

**Referencia completa**: [[Modelo de Datos]]

---

## Web UI - Acceso Rápido

### URL Principal
```
http://YOUR_PIPELINE_HOST_IP:8080
```

### Páginas Principales

| Página | URL | Función |
|--------|-----|---------|
| **Dashboard** | `/` | Métricas y estado general |
| **Pipeline Runs** | `/pipeline_runs` | Historial de deployments |
| **Logs** | `/logs` | Visor de logs con búsqueda |
| **SonarQube** | `/sonar_results` | Resultados análisis calidad |

### API Endpoints (JSON)

```bash
# Estadísticas
curl http://YOUR_PIPELINE_HOST_IP:8080/api/dashboard/stats | jq

# Deployments recientes
curl http://YOUR_PIPELINE_HOST_IP:8080/api/dashboard/recent-deployments | jq

# Detalles de un deployment
curl http://YOUR_PIPELINE_HOST_IP:8080/api/deployment/1 | jq

# Listar logs disponibles
curl http://YOUR_PIPELINE_HOST_IP:8080/api/logs/list | jq

# Ver log específico (últimas 500 líneas)
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/logs/view/pipeline_20260320.log?lines=500" | jq
```

**Referencia completa**: [[Web - API Endpoints]]

---

## Testing de Fases Individuales

### Fase 1 - Git Monitor

```bash
cd /home/YOUR_USER/cicd

# Detectar nuevos tags
./scripts/git_monitor.sh detect

# Hacer checkout de un tag específico
./scripts/git_monitor.sh checkout MAC_1_V24_02_15_01
```

**Ver detalles**: [[Pipeline - Git Monitor]]

### Fase 2 - Compilación

```bash
# Ejecutar compilación completa
./scripts/compile.sh

# La compilación crea:
# - /home/YOUR_USER/compile/ (directorio temporal)
# - InstallationDVD.iso (output)
```

**Ver detalles**: [[Pipeline - Compilación]]

### Fase 3 - SonarQube

```bash
cd /home/YOUR_USER/cicd

# Análisis completo de calidad
python3.6 python/sonar_check.py config/ci_cd_config.yaml V24_02_15_01

# Solo consultar resultado de análisis previo
python3.6 python/sonar_check.py config/ci_cd_config.yaml V24_02_15_01 --query-only
```

**Ver detalles**: [[Pipeline - SonarQube]]

### Fase 4 - vCenter

```bash
# Ver estado de la VM
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status

# Subir ISO al datastore
python3.6 python/vcenter_api.py config/ci_cd_config.yaml upload_iso /path/to/V24_02_15_01.iso

# Configurar CD-ROM con el ISO
python3.6 python/vcenter_api.py config/ci_cd_config.yaml configure_cdrom V24_02_15_01.iso

# Encender VM
python3.6 python/vcenter_api.py config/ci_cd_config.yaml power_on_vm
```

**Ver detalles**: [[Pipeline - vCenter]]

### Fase 5 - SSH Deploy

```bash
# Despliegue completo a VM destino
./scripts/deploy.sh

# El script espera que el ISO ya esté montado en la VM
```

**Ver detalles**: [[Pipeline - SSH Deploy]]

---

## Inicialización y Setup

### Primera Instalación

```bash
# 1. Setup inicial del entorno
cd /home/YOUR_USER/cicd
./setup_phase0.sh

# 2. Configurar credenciales
cp config/.env.example config/.env
nano config/.env  # Rellenar GIT_PASSWORD, SONAR_TOKEN, etc.

# 3. Configurar SSH key
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
ssh-copy-id YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP

# 4. Inicializar base de datos
./ci_cd.sh init

# 5. Instalar servicio systemd
sudo ./install_service.sh install

# 6. Instalar Web UI
sudo ./install_web.sh install
```

**Guía completa**: [[Operación - Instalación]]

### Verificar Instalación

```bash
# Verificar todos los prerequisites
./ci_cd.sh verify

# Debería mostrar:
# ✓ Git disponible
# ✓ Python 3.6+ disponible
# ✓ SQLite disponible
# ✓ SSH key configurada
# ✓ Base de datos inicializada
# ✓ Configuración válida
```

---

## Troubleshooting Rápido

### Pipeline no detecta tags

```bash
# Verificar conectividad Git
./scripts/git_monitor.sh detect

# Ver credenciales Git
grep GIT_ config/.env

# Verificar tags en remoto
cd /home/YOUR_USER/compile && git ls-remote --tags https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
```

**Solución**: [[Operación - Troubleshooting#Git Monitor]]

### Compilación falla

```bash
# Ver logs de compilación
ls -lh logs/compile_*.log | tail -1
tail -100 logs/compile_*.log  # último log

# Verificar permisos
./scripts/compile.sh
```

**Solución**: [[Operación - Troubleshooting#Compilación]]

### Web UI no responde

```bash
# Ver estado del servicio
sudo systemctl status cicd-web

# Ver logs de error
sudo journalctl -u cicd-web -n 50 --no-pager

# Verificar puerto
sudo lsof -i :8080
```

**Solución**: [[Operación - Troubleshooting#Web UI]]

### Base de datos corrupta

```bash
# Backup
cp db/pipeline.db db/pipeline.db.backup

# Re-inicializar
./ci_cd.sh init
```

**Solución**: [[Operación - Troubleshooting#Base de Datos]]

---

## Como Desarrollador

### 1. Crear Nueva Versión

```bash
# En tu máquina de desarrollo
cd /path/to/GALTTCMC
git checkout YOUR_GIT_BRANCH
git pull origin YOUR_GIT_BRANCH

# Hacer cambios, commit
git add .
git commit -m "Nueva feature X"

# Crear tag (formato importante)
git tag MAC_1_V24_03_20_01
git push origin YOUR_GIT_BRANCH --tags
```

### 2. Monitorizar Deployment

- Abrir Web UI: http://YOUR_PIPELINE_HOST_IP:8080
- Ver "Pipeline Runs" para seguimiento en tiempo real
- Esperar notificación de resultado (success/failed)

### 3. Si Falla el Quality Gate

```bash
# Ver resultados SonarQube en Web UI
# O consultar directamente
sqlite3 /home/YOUR_USER/cicd/db/pipeline.db "SELECT * FROM sonar_results WHERE tag_name='MAC_1_V24_03_20_01'"

# Ver umbrales configurados
grep -A 10 "thresholds:" /home/YOUR_USER/cicd/config/ci_cd_config.yaml
```

**Referencia**: [[Pipeline - SonarQube#Quality Gates]]

---

## Como Operador

### Monitorización Diaria

1. Abrir dashboard Web UI: http://YOUR_PIPELINE_HOST_IP:8080
2. Verificar "Success Rate" > 80%
3. Revisar "Recent Deployments" en busca de fallos
4. Consultar logs si hay errores

**Guía completa**: [[Operación - Monitorización]]

### Despliegue Manual Urgente

```bash
# Si el daemon está detenido o hay un tag urgente
cd /home/YOUR_USER/cicd
./ci_cd.sh --tag MAC_1_V24_03_20_02

# Seguir progreso
tail -f logs/pipeline_$(date +%Y%m%d).log
```

### Limpieza de Logs Antiguos

```bash
# Logs mayores a 30 días
find logs/ -name "*.log" -mtime +30 -delete

# Compactar base de datos
sqlite3 db/pipeline.db "VACUUM;"
```

**Guía completa**: [[Operación - Mantenimiento]]

---

## Enlaces Útiles

- [[Arquitectura del Pipeline]] - Entender el flujo completo
- [[Operación - Troubleshooting]] - Problemas comunes
- [[Referencia - Configuración]] - Opciones YAML y .env
- [[Modelo de Datos]] - Queries SQL útiles
- [[Web - API Endpoints]] - Integración con otros sistemas
