# 🛠️ Operación - Mantenimiento

## Visión General

Tareas periódicas de mantenimiento del pipeline CI/CD.

**Relacionado con**:
- [[Operación - Monitorización]] - Supervisión
- [[Operación - Troubleshooting]] - Resolución de problemas

---

## Backup de Base de Datos

### Manual

```bash
# Backup con timestamp
DATE=$(date +%Y%m%d_%H%M%S)
cp db/pipeline.db "db/backups/pipeline_${DATE}.db"

# Comprimir
gzip "db/backups/pipeline_${DATE}.db"
```

### Automático (Cron)

```bash
# Añadir a crontab
crontab -e

# Backup diario a las 3 AM
0 3 * * * /home/YOUR_USER/cicd/scripts/backup_db.sh
```

**Script `backup_db.sh`**:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
cp /home/YOUR_USER/cicd/db/pipeline.db "/home/YOUR_USER/cicd/db/backups/pipeline_${DATE}.db"
gzip "/home/YOUR_USER/cicd/db/backups/pipeline_${DATE}.db"

# Mantener últimos 30 días
find /home/YOUR_USER/cicd/db/backups -name "pipeline_*.db.gz" -mtime +30 -delete
```

---

## Limpieza de Logs

### Manual

```bash
# Logs mayores a 30 días
find logs/ -name "*.log" -mtime +30 -delete

# Ver tamaño de logs
du -sh logs/
```

### Rotación de Logs

**Configurar logrotate**:
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

---

## Mantenimiento de Base de Datos

### Vacuum (Compactar)

```bash
# Recuperar espacio de registros eliminados
sqlite3 db/pipeline.db "VACUUM;"

# Ver tamaño antes/después
ls -lh db/pipeline.db
```

### Reoptimizar Índices

```bash
sqlite3 db/pipeline.db "ANALYZE;"
```

### Limpieza de Datos Antiguos

```sql
-- Deployments mayores a 6 meses
DELETE FROM deployments
WHERE started_at < datetime('now', '-6 months');

-- Execution log antiguo
DELETE FROM execution_log
WHERE created_at < datetime('now', '-3 months');

-- Después de limpiar
VACUUM;
```

---

## Limpieza de Archivos Temporales

```bash
# ISOs antiguos en compile dir
find /home/YOUR_USER/compile -name "*.iso" -mtime +7 -delete

# Archivos temporales de SonarQube
rm -rf /home/YOUR_USER/compile/build-wrapper-output
rm -rf /home/YOUR_USER/compile/.scannerwork
```

---

## Actualización de Herramientas

### SonarScanner

```bash
cd /home/YOUR_USER/cicd/utils
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-latest-linux.zip
unzip sonar-scanner-cli-latest-linux.zip
rm -rf sonar-scanner-old
mv sonar-scanner sonar-scanner-old
mv sonar-scanner-* sonar-scanner
```

### Python Dependencies

```bash
cd /home/YOUR_USER/cicd
pip3.6 install --upgrade -r requirements.txt
```

---

## Verificación de Salud Semanal

```bash
# Script de health check
./scripts/health_check.sh

# Verificar:
# - Servicios activos
# - Espacio en disco
# - Tamaño de DB
# - Logs recientes
# - Success rate último mes
```

---

## Monitoreo de Recursos

### Espacio en Disco

```bash
# Alerta si uso > 80%
df -h /home/agent | awk 'NR==2 {if ($5+0 > 80) print "WARNING: Disk usage at " $5}'
```

### CPU/Memory

```bash
# Ver top processes
top -bn1 | grep cicd
```

---

## Procedimiento de Upgrade

### Pipeline

```bash
# 1. Backup completo
tar -czf cicd_backup_$(date +%Y%m%d).tar.gz cicd/

# 2. Detener servicios
sudo systemctl stop cicd cicd-web

# 3. Aplicar cambios
git pull  # Si usa Git
# O copiar nuevos archivos

# 4. Reiniciar servicios
sudo systemctl start cicd cicd-web

# 5. Verificar
./ci_cd.sh status
```

---

## Enlaces Relacionados

- [[Operación - Monitorización]]
- [[Operación - Troubleshooting]]
- [[Modelo de Datos#Backup y Mantenimiento]]
