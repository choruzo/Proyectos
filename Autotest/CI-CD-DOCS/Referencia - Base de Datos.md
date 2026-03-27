# 📚 Referencia - Base de Datos

## Visión General

Queries útiles, ejemplos avanzados y optimización de la base de datos SQLite.

**Relacionado con**:
- [[Modelo de Datos]] - Schema completo
- [[Arquitectura del Pipeline#Persistencia de Datos]]

---

## Queries de Analytics

### Success Rate por Día de la Semana

```sql
SELECT 
    CASE CAST(strftime('%w', started_at) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_of_week,
    COUNT(*) AS total,
    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successful,
    ROUND(CAST(SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS success_rate
FROM deployments
WHERE status IN ('success', 'failed')
GROUP BY CAST(strftime('%w', started_at) AS INTEGER)
ORDER BY CAST(strftime('%w', started_at) AS INTEGER);
```

---

### Top 10 Deployments Más Lentos

```sql
SELECT 
    tag_name,
    duration_seconds / 60.0 AS duration_minutes,
    started_at,
    status
FROM deployments
WHERE status = 'success' AND duration_seconds IS NOT NULL
ORDER BY duration_seconds DESC
LIMIT 10;
```

---

### Evolución de Coverage

```sql
SELECT 
    d.tag_name,
    s.coverage,
    s.analyzed_at,
    LAG(s.coverage) OVER (ORDER BY s.analyzed_at) AS previous_coverage,
    s.coverage - LAG(s.coverage) OVER (ORDER BY s.analyzed_at) AS coverage_diff
FROM sonar_results s
JOIN deployments d ON s.deployment_id = d.id
WHERE d.status = 'success'
ORDER BY s.analyzed_at DESC
LIMIT 20;
```

---

### Fase Más Lenta del Pipeline

```sql
SELECT 
    'Compilation' AS phase,
    AVG(duration_seconds) / 60.0 AS avg_minutes,
    MIN(duration_seconds) / 60.0 AS min_minutes,
    MAX(duration_seconds) / 60.0 AS max_minutes
FROM build_logs
WHERE phase = 'compilation'

UNION ALL

SELECT 
    'Deployment' AS phase,
    AVG(d.duration_seconds - COALESCE(bl.duration_seconds, 0)) / 60.0,
    MIN(d.duration_seconds - COALESCE(bl.duration_seconds, 0)) / 60.0,
    MAX(d.duration_seconds - COALESCE(bl.duration_seconds, 0)) / 60.0
FROM deployments d
LEFT JOIN build_logs bl ON d.id = bl.deployment_id
WHERE d.status = 'success';
```

---

## Queries de Troubleshooting

### Timeline de un Deployment

```sql
SELECT 
    'Deployment Start' AS event,
    started_at AS timestamp
FROM deployments
WHERE tag_name = 'MAC_1_V24_02_15_01'

UNION ALL

SELECT 
    'Build: ' || phase AS event,
    created_at AS timestamp
FROM build_logs
WHERE deployment_id = (SELECT id FROM deployments WHERE tag_name = 'MAC_1_V24_02_15_01')

UNION ALL

SELECT 
    'SonarQube: ' || quality_gate_status AS event,
    analyzed_at AS timestamp
FROM sonar_results
WHERE tag_name = 'MAC_1_V24_02_15_01'

UNION ALL

SELECT 
    'Deployment End: ' || status AS event,
    finished_at AS timestamp
FROM deployments
WHERE tag_name = 'MAC_1_V24_02_15_01'

ORDER BY timestamp;
```

---

### Deployments Fallidos con Causas

```sql
SELECT 
    tag_name,
    error_message,
    started_at,
    finished_at,
    ROUND((julianday(finished_at) - julianday(started_at)) * 1440, 1) AS duration_minutes,
    CASE 
        WHEN error_message LIKE '%ompil%' THEN 'Compilation'
        WHEN error_message LIKE '%uality%' OR error_message LIKE '%onar%' THEN 'Quality Gate'
        WHEN error_message LIKE '%Center%' OR error_message LIKE '%upload%' THEN 'vCenter'
        WHEN error_message LIKE '%SSH%' OR error_message LIKE '%install%' THEN 'SSH Deploy'
        ELSE 'Other'
    END AS failure_type
FROM deployments
WHERE status = 'failed'
ORDER BY finished_at DESC
LIMIT 20;
```

---

## Optimización

### Indices Recomendados

```sql
-- Ya creados en init_db.sql, pero para referencia:
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_started_at ON deployments(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_build_logs_deployment ON build_logs(deployment_id);
CREATE INDEX IF NOT EXISTS idx_sonar_tag ON sonar_results(tag_name);
CREATE INDEX IF NOT EXISTS idx_sonar_analyzed_at ON sonar_results(analyzed_at DESC);
```

---

### Explicar Query Plan

```sql
EXPLAIN QUERY PLAN
SELECT * FROM deployments 
WHERE status = 'success' 
ORDER BY started_at DESC 
LIMIT 10;
```

---

## Vistas Útiles

### Deployments con Todas las Métricas

```sql
CREATE VIEW v_full_deployment_details AS
SELECT 
    d.id,
    d.tag_name,
    d.status,
    d.started_at,
    d.finished_at,
    d.duration_seconds,
    bl.duration_seconds AS build_duration,
    s.coverage,
    s.bugs,
    s.vulnerabilities,
    s.code_smells,
    s.quality_gate_status
FROM deployments d
LEFT JOIN build_logs bl ON d.id = bl.deployment_id AND bl.phase = 'compilation'
LEFT JOIN sonar_results s ON d.id = s.deployment_id;
```

---

## Export/Import

### Export a CSV

```bash
# Export deployments
sqlite3 -header -csv db/pipeline.db "SELECT * FROM deployments" > deployments.csv

# Export para análisis
sqlite3 -header -csv db/pipeline.db "SELECT * FROM v_deployment_stats" > stats.csv
```

---

### Import desde CSV

```bash
# Crear tabla temporal
sqlite3 db/pipeline.db <<EOF
CREATE TABLE temp_import (
    tag_name TEXT,
    status TEXT,
    started_at TEXT
);
.mode csv
.import data.csv temp_import
EOF
```

---

## Enlaces Relacionados

- [[Modelo de Datos]] - Schema completo
- [[Operación - Mantenimiento#Mantenimiento de Base de Datos]]
- [[Operación - Monitorización#Queries SQL]]
