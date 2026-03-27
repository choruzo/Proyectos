# 👁️ Operación - Monitorización

## Visión General

Supervisión del pipeline CI/CD, dashboards y alertas.

**Relacionado con**:
- [[Arquitectura Web UI]] - Dashboard principal
- [[Operación - Troubleshooting]] - Resolución de problemas
- [[Modelo de Datos]] - Fuente de métricas

---

## Dashboard Web UI

### Acceso

**URL**: http://YOUR_PIPELINE_HOST_IP:8080

### Páginas Principales

#### 1. Dashboard Principal (`/`)

**Métricas mostradas**:
- Total deployments
- Success rate (%)
- Deployments últimas 24h
- Duración promedio
- Estado actual del pipeline
- Recent deployments (últimos 10)

**Gráficos**:
- Success rate últimos 7 días (line chart)
- Distribución por estado (pie chart)
- Duración de builds (bar chart)

#### 2. Pipeline Runs (`/pipeline_runs`)

**Tabla de deployments**:
- Tag name
- Status (badge con color)
- Started at
- Duration
- Actions (view details, view logs)

**Filtros**:
- Por status: all, success, failed, in_progress
- Paginación: 20 items por página

#### 3. Logs Viewer (`/logs`)

**Funcionalidad**:
- Selector de archivo de log
- Búsqueda en logs
- Filtro por líneas (últimas N)
- Scroll infinito
- Copy to clipboard

#### 4. SonarQube Results (`/sonar_results`)

**Tabla de análisis**:
- Tag name
- Coverage %
- Bugs, Vulnerabilities, Code Smells
- Quality Gate status
- Analyzed at

**Gráficos de tendencias**:
- Coverage evolution
- Issues over time

---

## Monitorización CLI

### Estado General

```bash
# Ver estado del pipeline
./ci_cd.sh status

# Ver logs en tiempo real
./ci_cd.sh logs 100

# Ver deployments recientes
sqlite3 db/pipeline.db "SELECT * FROM v_recent_deployments LIMIT 10"
```

### Servicios systemd

```bash
# Estado de servicios
sudo systemctl status cicd
sudo systemctl status cicd-web

# Logs de servicios
sudo journalctl -u cicd -f
sudo journalctl -u cicd-web -f

# Últimas 50 líneas
sudo journalctl -u cicd -n 50 --no-pager
```

### Logs de Archivos

```bash
# Pipeline general
tail -f logs/pipeline_$(date +%Y%m%d).log

# Compilación
tail -f logs/compile_*.log | grep -i error

# Deployment
tail -f logs/deploy_*.log

# Web UI
tail -f logs/web_access.log
tail -f logs/web_error.log
```

---

## Métricas Clave

### Success Rate

```sql
SELECT 
    ROUND(CAST(SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS success_rate
FROM deployments;
```

**Target**: > 80%

### Duración Promedio

```sql
SELECT AVG(duration_seconds) / 60.0 AS avg_minutes
FROM deployments
WHERE status = 'success' AND duration_seconds IS NOT NULL;
```

**Target**: < 70 minutos

### Deployments por Día

```sql
SELECT DATE(started_at) AS day, COUNT(*) AS count
FROM deployments
WHERE started_at >= datetime('now', '-7 days')
GROUP BY DATE(started_at)
ORDER BY day DESC;
```

---

## Alertas y Notificaciones

### Deployment Fallido

**Detección**: Status = 'failed'

**Acción**:
1. Revisar `error_message` en DB
2. Consultar logs específicos
3. Notificar a desarrollador

### Deployment Tardío

**Detección**: Duration > 90 minutos

**Consulta**:
```sql
SELECT tag_name, status, 
       ROUND((julianday('now') - julianday(started_at)) * 1440) AS minutes_elapsed
FROM deployments
WHERE status IN ('pending', 'compiling', 'analyzing', 'deploying')
  AND julianday('now') - julianday(started_at) > 0.0625;  -- > 90 min
```

### Servicio Caído

```bash
# Check si servicio está activo
systemctl is-active cicd || echo "⚠️ ALERT: cicd service is down"
systemctl is-active cicd-web || echo "⚠️ ALERT: cicd-web service is down"
```

---

## Monitoreo de Recursos

### Espacio en Disco

```bash
# Uso de disco
df -h /home/agent

# Tamaño de componentes
du -sh /home/YOUR_USER/cicd/db
du -sh /home/YOUR_USER/cicd/logs
du -sh /home/YOUR_USER/compile
```

### Procesos

```bash
# Procesos del pipeline
ps aux | grep -E '(ci_cd|compile|sonar|vcenter)'

# Memoria y CPU
top -bn1 | grep -E '(ci_cd|python|flask|gunicorn)'
```

---

## Queries SQL Útiles

### Deployments Fallidos Recientes

```sql
SELECT tag_name, error_message, started_at, finished_at
FROM deployments
WHERE status='failed'
ORDER BY started_at DESC
LIMIT 10;
```

### Success Rate por Semana

```sql
SELECT 
    strftime('%Y-W%W', started_at) AS week,
    COUNT(*) AS total,
    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success,
    ROUND(CAST(SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS success_rate
FROM deployments
WHERE started_at >= datetime('now', '-4 weeks')
GROUP BY week
ORDER BY week DESC;
```

### Quality Gate Failures

```sql
SELECT tag_name, coverage, bugs, vulnerabilities, code_smells
FROM sonar_results
WHERE quality_gate_status = 'FAILED'
ORDER BY analyzed_at DESC
LIMIT 10;
```

---

## Enlaces Relacionados

- [[Arquitectura Web UI]]
- [[Operación - Troubleshooting]]
- [[Modelo de Datos]]
- [[01 - Quick Start]]
