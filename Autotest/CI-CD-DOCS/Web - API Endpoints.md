# 🌐 Web - API Endpoints

## Visión General

Especificación completa de la REST API del backend Flask de la Web UI.

**Relacionado con**:
- [[Arquitectura Web UI]] - Contexto de la Web UI
- [[Web - Frontend Components]] - Consumidores de la API
- [[Modelo de Datos]] - Fuente de datos

---

## Base URL

```
http://YOUR_PIPELINE_HOST_IP:8080/api
```

**Autenticación**: No requiere (asumido red local confiable)

**Content-Type**: `application/json`

---

## Endpoints - Dashboard

### GET /api/dashboard/stats

**Descripción**: Estadísticas generales del pipeline

**Response**:
```json
{
  "total_deployments": 156,
  "success_rate": 82.5,
  "last_24h": 8,
  "avg_duration_minutes": 72.3,
  "successful": 129,
  "failed": 27
}
```

**Implementación**:
```python
@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    conn = get_db_connection()
    result = conn.execute('SELECT * FROM v_deployment_stats').fetchone()
    conn.close()
    return jsonify(dict(result))
```

---

### GET /api/dashboard/recent-deployments

**Descripción**: Últimos 10 deployments

**Response**:
```json
[
  {
    "id": 156,
    "tag_name": "MAC_1_V24_02_15_01",
    "status": "success",
    "started_at": "2026-03-20T08:30:00",
    "finished_at": "2026-03-20T09:45:00",
    "duration_seconds": 4500,
    "coverage": 85.3,
    "bugs": 0
  }
]
```

---

### GET /api/dashboard/chart-data

**Descripción**: Datos para gráfico de deployments (últimos 7 días)

**Response**:
```json
{
  "labels": ["2026-03-14", "2026-03-15", "2026-03-16", ...],
  "datasets": [
    {
      "label": "Successful",
      "data": [3, 4, 2, 5, 3, 4, 3]
    },
    {
      "label": "Failed",
      "data": [1, 0, 1, 0, 2, 1, 0]
    }
  ]
}
```

---

## Endpoints - Deployments

### GET /api/deployments

**Descripción**: Lista paginada de deployments con filtros

**Query Parameters**:
- `page` (int, default: 1): Número de página
- `per_page` (int, default: 20, max: 100): Resultados por página
- `status` (string, optional): Filtro por estado (`success`, `failed`, `pending`, `compiling`, `analyzing`, `deploying`, `all`)

**Ejemplo**:
```bash
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/deployments?page=1&per_page=20&status=success" | jq
```

**Response**:
```json
{
  "data": [
    {
      "id": 156,
      "tag_name": "MAC_1_V24_02_15_01",
      "status": "success",
      "started_at": "2026-03-20T08:30:00",
      "finished_at": "2026-03-20T09:45:00",
      "duration_seconds": 4500
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 156,
    "pages": 8
  }
}
```

---

### GET /api/deployment/:id

**Descripción**: Detalles completos de un deployment específico

**Path Parameters**:
- `id` (int): ID del deployment

**Response**:
```json
{
  "id": 156,
  "tag_name": "MAC_1_V24_02_15_01",
  "status": "success",
  "started_at": "2026-03-20T08:30:00",
  "finished_at": "2026-03-20T09:45:00",
  "duration_seconds": 4500,
  "error_message": null,
  "build_logs": [
    {
      "id": 1,
      "phase": "compilation",
      "log_content": "...",
      "duration_seconds": 2700,
      "created_at": "2026-03-20T08:35:00"
    }
  ],
  "sonar_results": {
    "coverage": 85.3,
    "bugs": 0,
    "vulnerabilities": 0,
    "security_hotspots": 0,
    "code_smells": 5,
    "quality_gate_status": "PASSED",
    "analyzed_at": "2026-03-20T09:20:00"
  }
}
```

---

## Endpoints - Logs

### GET /api/logs/list

**Descripción**: Lista de archivos de log disponibles

**Response**:
```json
[
  {
    "filename": "pipeline_20260320.log",
    "size_bytes": 5242880,
    "size_human": "5.0 MB",
    "modified_at": "2026-03-20T10:30:00"
  },
  {
    "filename": "compile_20260320_083000.log",
    "size_bytes": 8388608,
    "size_human": "8.0 MB",
    "modified_at": "2026-03-20T09:15:00"
  }
]
```

---

### GET /api/logs/view/:filename

**Descripción**: Contenido de un archivo de log

**Path Parameters**:
- `filename` (string): Nombre del archivo

**Query Parameters**:
- `lines` (int, default: 500, max: 5000): Número de líneas desde el final
- `search` (string, optional): Filtrar líneas que contienen texto

**Ejemplos**:
```bash
# Últimas 1000 líneas
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/logs/view/pipeline_20260320.log?lines=1000" | jq

# Buscar errores
curl "http://YOUR_PIPELINE_HOST_IP:8080/api/logs/view/pipeline_20260320.log?search=error" | jq
```

**Response**:
```json
{
  "filename": "pipeline_20260320.log",
  "lines": [
    "[2026-03-20 10:05:33] [INFO] Git monitor: Checking for new tags...",
    "[2026-03-20 10:05:34] [OK] Found 1 new tag: MAC_1_V24_02_15_01"
  ],
  "total_lines": 1247,
  "filtered": false
}
```

---

## Endpoints - SonarQube

### GET /api/sonar/results

**Descripción**: Últimos 50 análisis de SonarQube

**Response**:
```json
[
  {
    "id": 45,
    "tag_name": "MAC_1_V24_02_15_01",
    "coverage": 85.3,
    "bugs": 0,
    "vulnerabilities": 0,
    "security_hotspots": 0,
    "code_smells": 5,
    "duplications": 2.1,
    "lines_of_code": 125340,
    "quality_gate_status": "PASSED",
    "analyzed_at": "2026-03-20T09:20:00"
  }
]
```

---

### GET /api/sonar/trends

**Descripción**: Tendencias de métricas (últimos 10 deployments exitosos)

**Response**:
```json
{
  "labels": ["V24_02_10_01", "V24_02_11_01", ...],
  "coverage": [82.1, 83.5, 85.3, ...],
  "bugs": [0, 0, 0, ...],
  "code_smells": [8, 7, 5, ...]
}
```

---

## Manejo de Errores

**Estructura de error**:
```json
{
  "status": "error",
  "message": "Deployment not found",
  "code": 404
}
```

**Códigos HTTP**:
- `200`: Success
- `400`: Bad request (parámetros inválidos)
- `404`: Resource not found
- `500`: Internal server error

---

## Rate Limiting

**No implementado** actualmente. Asumido red local confiable.

**Implementación futura**:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/deployments')
@limiter.limit("100 per minute")
def api_deployments():
    ...
```

---

## Enlaces Relacionados

- [[Arquitectura Web UI]] - Contexto backend
- [[Web - Frontend Components]] - Consumidores
- [[Modelo de Datos]] - Queries SQL usadas
