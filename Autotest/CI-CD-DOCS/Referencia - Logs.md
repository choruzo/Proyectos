# 📋 Referencia - Logs

## Visión General

Sistema de logging del pipeline CI/CD: estructura, ubicación, formatos y queries.

**Relacionado con**:
- [[Pipeline - Common Functions#Funciones de Logging]]
- [[Operación - Monitorización#Logs de Archivos]]
- [[Arquitectura del Pipeline#Sistema de Logs]]

---

## Ubicación de Logs

### Directorio Principal

**Path**: `/home/YOUR_USER/cicd/logs/`

**Estructura**:
```
logs/
├── pipeline_YYYYMMDD.log          # Log general del orquestador
├── compile_YYYYMMDD_HHMMSS.log    # Logs de compilación
├── deploy_YYYYMMDD_HHMMSS.log     # Logs de deployment
├── web_access.log                 # Web UI access log
└── web_error.log                  # Web UI error log
```

---

## Tipos de Logs

### 1. Pipeline General

**Archivo**: `pipeline_YYYYMMDD.log`

**Contenido**: Eventos generales del daemon, Git monitoring, orquestación.

**Formato**:
```
[2026-03-20 10:05:33] [INFO] Git monitor: Checking for new tags...
[2026-03-20 10:05:34] [OK] Found 1 new tag: MAC_1_V24_02_15_01
[2026-03-20 10:05:35] [INFO] Starting pipeline for tag: MAC_1_V24_02_15_01
```

**Ver en tiempo real**:
```bash
tail -f logs/pipeline_$(date +%Y%m%d).log
```

---

### 2. Compilation Logs

**Archivo**: `compile_YYYYMMDD_HHMMSS.log`

**Contenido**: Output de `build_DVDs.sh`, errores de compilación.

**Tamaño típico**: 5-10 MB

**Ejemplo**:
```
[2026-03-20 10:10:00] Starting compilation...
make[1]: Entering directory '/home/YOUR_USER/compile/Development_TTCF/ttcf/core'
gcc -o ttcf_core main.c utils.c
make[1]: Leaving directory '/home/YOUR_USER/compile/Development_TTCF/ttcf/core'
Creating ISO image...
InstallationDVD.iso created successfully (3.5 GB)
```

**Buscar errores**:
```bash
grep -i error logs/compile_*.log | tail -20
```

---

### 3. Deployment Logs

**Archivo**: `deploy_YYYYMMDD_HHMMSS.log`

**Contenido**: SSH operations, instalación en target VM.

**Ejemplo**:
```
[2026-03-20 11:00:00] [INFO] Testing SSH connection...
[2026-03-20 11:00:02] [OK] SSH connection successful
[2026-03-20 11:00:05] [OK] ISO mounted at /mnt/cdrom
[2026-03-20 11:02:45] [OK] Installation completed
```

**Ver último deployment**:
```bash
ls -t logs/deploy_*.log | head -1 | xargs tail -50
```

---

### 4. Web UI Logs

#### Access Log

**Archivo**: `web_access.log`

**Formato**: Apache combined log format

**Ejemplo**:
```
10.0.0.100 - - [20/Mar/2026:11:30:45 +0000] "GET /api/dashboard/stats HTTP/1.1" 200 542
10.0.0.101 - - [20/Mar/2026:11:30:50 +0000] "GET /pipeline_runs HTTP/1.1" 200 8192
```

#### Error Log

**Archivo**: `web_error.log`

**Contenido**: Errores de Flask, excepciones Python.

**Ejemplo**:
```
[2026-03-20 11:35:12] ERROR in app: Database connection failed
Traceback (most recent call last):
  File "app.py", line 45, in get_deployments
    conn = sqlite3.connect(DB_PATH)
sqlite3.OperationalError: unable to open database file
```

---

## Formato de Log

### Estructura

```
[TIMESTAMP] [LEVEL] Message
```

**Componentes**:
- `TIMESTAMP`: `YYYY-MM-DD HH:MM:SS`
- `LEVEL`: `DEBUG`, `INFO`, `WARN`, `ERROR`, `OK`
- `Message`: Descripción del evento

### Niveles de Log

| Nivel | Descripción | Color | Ejemplo |
|-------|-------------|-------|---------|
| **DEBUG** | Información detallada | Gris | Variables, estados internos |
| **INFO** | Eventos normales | Blanco | "Starting compilation..." |
| **WARN** | Advertencias no críticas | Amarillo | "Cleanup failed (non-blocking)" |
| **ERROR** | Errores que detienen operación | Rojo | "Compilation failed" |
| **OK** | Operación exitosa | Verde | "ISO created successfully" |

---

## Funciones de Logging

### Desde Bash

```bash
# Source common.sh
source scripts/common.sh

# Usar funciones de logging
log_info "Starting operation..."
log_ok "Operation completed"
log_warn "Non-critical issue detected"
log_error "Critical error occurred"
log_debug "Debug info: variable=$value"
```

**Implementación** en `common.sh`:
```bash
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" >&2
}

log_ok() {
    echo -e "\033[32m[$(date +'%Y-%m-%d %H:%M:%S')] [OK] $*\033[0m" >&2
}

log_error() {
    echo -e "\033[31m[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*\033[0m" >&2
}
```

**⚠️ Importante**: Los logs van a **stderr** (`>&2`), no stdout.

### Desde Python

```python
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)

# Usar logger
logging.info("Starting SonarQube analysis...")
logging.error("API call failed: {}".format(error))
logging.debug("Response: {}".format(response.text))
```

---

## Rotación de Logs

### Configuración logrotate

**Archivo**: `/etc/logrotate.d/cicd`

```
/home/YOUR_USER/cicd/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 agent agent
    postrotate
        systemctl reload cicd > /dev/null 2>&1 || true
    endscript
}
```

**Parámetros**:
- `daily`: Rotar diariamente
- `rotate 30`: Mantener últimos 30 días
- `compress`: Comprimir logs antiguos (.gz)
- `delaycompress`: No comprimir último rotado (para lectura)
- `missingok`: No error si log no existe
- `notifempty`: No rotar si está vacío

### Limpieza Manual

```bash
# Eliminar logs mayores a 30 días
find logs/ -name "*.log" -mtime +30 -delete

# Comprimir logs antiguos
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# Ver tamaño de logs
du -sh logs/
du -h logs/*.log | sort -h | tail -10
```

---

## Queries de Logs

### Buscar Errores

```bash
# Errores en pipeline general
grep -i error logs/pipeline_$(date +%Y%m%d).log

# Errores en compilación (últimos 3 logs)
ls -t logs/compile_*.log | head -3 | xargs grep -i error

# Errores con contexto (5 líneas antes/después)
grep -C 5 -i error logs/pipeline_*.log
```

### Filtrar por Timestamp

```bash
# Logs entre 10:00 y 11:00
awk '/2026-03-20 10:/,/2026-03-20 11:/' logs/pipeline_20260320.log

# Logs de última hora
awk -v d="$(date -d '1 hour ago' '+%Y-%m-%d %H:')" '$0 ~ d' logs/pipeline_$(date +%Y%m%d).log
```

### Estadísticas

```bash
# Contar por nivel
grep -o '\[INFO\]\|\[ERROR\]\|\[WARN\]\|\[OK\]' logs/pipeline_$(date +%Y%m%d).log | sort | uniq -c

# Output ejemplo:
#  1250 [INFO]
#    45 [OK]
#     3 [WARN]
#     1 [ERROR]
```

---

## Logs en Base de Datos

### Tabla `build_logs`

Logs de compilación almacenados en SQLite para queries estructuradas.

```sql
SELECT 
    d.tag_name,
    bl.phase,
    bl.duration_seconds,
    LENGTH(bl.log_content) AS log_size,
    bl.created_at
FROM build_logs bl
JOIN deployments d ON bl.deployment_id = d.id
ORDER BY bl.created_at DESC
LIMIT 10;
```

**Ver schema**: [[Modelo de Datos#build_logs]]

---

## Logs de systemd

### journald

**Ver logs de servicios**:
```bash
# Pipeline daemon
sudo journalctl -u cicd -f

# Web UI
sudo journalctl -u cicd-web -f

# Últimas 50 líneas
sudo journalctl -u cicd -n 50 --no-pager

# Logs desde ayer
sudo journalctl -u cicd --since yesterday

# Logs entre fechas
sudo journalctl -u cicd --since "2026-03-20 10:00" --until "2026-03-20 12:00"
```

### Filtrar por Prioridad

```bash
# Solo errores
sudo journalctl -u cicd -p err

# Warning o superior
sudo journalctl -u cicd -p warning
```

---

## Análisis de Logs con Web UI

### Logs Viewer

**URL**: http://YOUR_PIPELINE_HOST_IP:8080/logs

**Funcionalidades**:
- Selector de archivo
- Búsqueda en contenido
- Filtro por líneas (últimas N)
- Scroll infinito
- Copy to clipboard

**API Endpoint**:
```bash
# Listar logs disponibles
curl http://YOUR_PIPELINE_HOST_IP:8080/api/logs/list | jq

# Ver log específico (últimas 500 líneas)
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/logs/view/pipeline_20260320.log?lines=500" | jq

# Buscar en log
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/logs/view/pipeline_20260320.log?search=error" | jq
```

**Ver**: [[Web - API Endpoints#Logs]]

---

## Debugging con Logs

### Activar Debug Mode

```bash
# Bash scripts
export DEBUG=1
./ci_cd.sh daemon

# O en config
nano config/ci_cd_config.yaml
# logging:
#   level: "DEBUG"
```

### Trace de Deployment

Para seguir un deployment específico:
```bash
# 1. Pipeline general
grep "MAC_1_V24_02_15_01" logs/pipeline_$(date +%Y%m%d).log

# 2. Compilation log (buscar timestamp)
ls -la logs/compile_20260320_*.log
tail -f logs/compile_20260320_100535.log

# 3. Deployment log
ls -la logs/deploy_20260320_*.log
tail -f logs/deploy_20260320_110000.log

# 4. DB audit trail
sqlite3 db/pipeline.db "
SELECT event_type, message, created_at
FROM execution_log
WHERE message LIKE '%MAC_1_V24_02_15_01%'
ORDER BY created_at
"
```

---

## Mejores Prácticas

### Desarrollo

1. **Log contexto suficiente**: Tag name, fase, operación
2. **Usar niveles apropiados**: INFO para eventos, ERROR para fallos
3. **No hacer log de secrets**: Credenciales, tokens, passwords
4. **Estructurar mensajes**: Incluir componente, acción, resultado

### Producción

1. **Monitorizar tamaño de logs**: `du -sh logs/`
2. **Configurar rotación**: logrotate cada 1-7 días
3. **Backup de logs críticos**: Deployments fallidos
4. **Agregar métricas**: Parsear logs para dashboards

### Performance

1. **Evitar logs excesivos en loops**: Usar DEBUG level
2. **Flush buffers en scripts críticos**: `sync` después de writes importantes
3. **Usar `tee` para dual output**: `script.sh 2>&1 | tee log.txt`

---

## Troubleshooting de Logs

### Log No Se Crea

**Causa**: Permisos, directorio no existe

**Solución**:
```bash
mkdir -p logs
chmod 755 logs
touch logs/test.log
```

### Log Crece Sin Control

**Causa**: Loop infinito, logging excesivo

**Solución**:
```bash
# Truncar log
> logs/problem.log

# O mover y comprimir
mv logs/problem.log logs/problem_$(date +%Y%m%d).log
gzip logs/problem_*.log
```

### No Puedo Leer Logs

**Causa**: Permisos restrictivos

**Solución**:
```bash
chmod 644 logs/*.log
```

---

## Enlaces Relacionados

- [[Pipeline - Common Functions#Funciones de Logging]]
- [[Operación - Monitorización#Logs de Archivos]]
- [[Operación - Mantenimiento#Limpieza de Logs]]
- [[Operación - Troubleshooting]]
- [[Web - API Endpoints#Logs]]
