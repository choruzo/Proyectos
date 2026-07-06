# Guía de configuración: Prometheus + Grafana

**Versión**: 1.0
**Aplica a**: vCenter Agent System v1.2+
**Prerequisito**: App Flask corriendo en `http://localhost:5000`

---

## Arquitectura de la integración

```
Prometheus ──scrape /metrics──► Flask App (puerto 5000)
     │
     └──scrape :9091──► Pushgateway ◄── historical_data_collector.py (ESXi)
                                    ◄── truenas_snmp_collector.py
                                    ◄── cisco_catalyst_snmp_collector.py
                                         │
                                         ▼
                                      Grafana
```

| Componente | Patrón | Puerto |
|---|---|---|
| Flask App | Pull — Prometheus hace scrape cada 30s | 5000 |
| ESXi / TrueNAS / Cisco | Push — collectors envían cada ciclo | 9091 (Pushgateway) |

---

## Paso 1 — Activar el endpoint en la app

### 1.1 Desde la UI (recomendado)

1. Inicia sesión como **superuser**
2. Abre el panel de administración: `/admin/stats`
3. En la tarjeta **Exportación Prometheus**, activa el toggle
4. El badge cambia a **Activo** — el endpoint `/metrics` ya responde

### 1.2 Desde config.json (alternativa)

Edita `config/config.json`:

```json
"prometheus": {
  "enabled": true,
  "allowed_ips": ["127.0.0.1", "192.168.x.x"],
  "pushgateway": {
    "enabled": false,
    "url": "localhost:9091"
  }
}
```

> **`allowed_ips`**: lista de IPs desde las que Prometheus puede hacer scrape. Si Prometheus corre en la misma máquina, `127.0.0.1` es suficiente. Si corre en otro servidor, añade su IP.

### 1.3 Verificar que el endpoint responde

```powershell
# Desde PowerShell (en la misma máquina)
Invoke-WebRequest -Uri http://127.0.0.1:5000/metrics | Select-Object -ExpandProperty Content
```

Debe devolver texto con líneas como:
```
# HELP vcenter_agent_active_sessions_total Active user sessions
# TYPE vcenter_agent_active_sessions_total gauge
vcenter_agent_active_sessions_total 2.0
# HELP vcenter_agent_queries_total Total queries processed per agent
...
```

Si devuelve 404, el toggle no está activo. Si devuelve 403, tu IP no está en `allowed_ips`.

---

## Paso 2 — Instalar Prometheus

### 2.1 Descargar

Descarga desde: https://prometheus.io/download/

Versión recomendada: `prometheus-*.windows-amd64.zip`

Descomprime en, por ejemplo: `C:\prometheus\`

### 2.2 Crear el archivo de configuración

Crea `C:\prometheus\prometheus.yml`:

```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:

  # Métricas de la app Flask (sesiones, queries, RAG, CPU, RAM)
  - job_name: 'vcenter-agent-app'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
```

> Si Prometheus y la app Flask corren en **máquinas distintas**, sustituye `localhost:5000` por la IP del servidor Flask y asegúrate de que esa IP esté en `allowed_ips` del `config.json`.

### 2.3 Arrancar Prometheus

```powershell
cd C:\prometheus
.\prometheus.exe --config.file=prometheus.yml
```

Abre `http://localhost:9090` en el navegador.

### 2.4 Verificar que el target está UP

Ve a **Status → Targets**. Debe aparecer:

| Job | Endpoint | State |
|---|---|---|
| vcenter-agent-app | `http://localhost:5000/metrics` | **UP** |

Si aparece **DOWN**, revisa:
- Que la app Flask está corriendo
- Que el toggle de Prometheus está activo
- Que la IP de Prometheus está en `allowed_ips`

### 2.5 Probar una query en Prometheus

En **Graph**, escribe:
```
vcenter_agent_active_sessions_total
```
Pulsa **Execute** → debe devolver el número de sesiones activas.

---

## Paso 3 — Pushgateway (opcional — para ESXi, TrueNAS, Cisco)

Solo necesario si quieres métricas de los colectores de background. Si solo quieres métricas de la app Flask, salta este paso.

### 3.1 Instalar con Docker

```powershell
docker run -d -p 9091:9091 --name pushgateway prom/pushgateway
```

O descarga el binario desde: https://prometheus.io/download/#pushgateway

```powershell
cd C:\pushgateway
.\pushgateway.exe
```

Verifica en `http://localhost:9091`.

### 3.2 Activar en config.json

```json
"prometheus": {
  "enabled": true,
  "allowed_ips": ["127.0.0.1"],
  "pushgateway": {
    "enabled": true,
    "url": "localhost:9091"
  }
}
```

### 3.3 Añadir el Pushgateway a prometheus.yml

```yaml
scrape_configs:

  - job_name: 'vcenter-agent-app'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'

  # Métricas enviadas por los colectores de background (ESXi, TrueNAS, Cisco)
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['localhost:9091']
    honor_labels: true   # preserva los labels job/instance enviados por los collectors
```

Reinicia Prometheus para que cargue la nueva config.

### 3.4 Verificar que llegan datos

Tras el próximo ciclo de `historical_data_collector.py` (~10 min), verifica en `http://localhost:9091`:

Deben aparecer métricas como `esxi_cpu_usage_percent`, `truenas_accessible`, etc.

En Prometheus (http://localhost:9090), prueba:
```
esxi_cpu_usage_percent
esxi_memory_usage_percent
truenas_memory_usage_percent
```

---

## Paso 4 — Instalar Grafana

### 4.1 Descargar e instalar

Descarga desde: https://grafana.com/grafana/download?platform=windows

Instala el `.msi` y arranca el servicio:

```powershell
# Si se instaló como servicio Windows
Start-Service Grafana
```

Abre `http://localhost:3000`
Login por defecto: `admin` / `admin`

### 4.2 Añadir Prometheus como datasource

1. Ve a **Connections → Data Sources → Add data source**
2. Selecciona **Prometheus**
3. Configura:
   - **Name**: `Prometheus`
   - **URL**: `http://localhost:9090`
   - **Access**: `Server (default)`
4. Pulsa **Save & Test** → debe aparecer `Data source is working`

---

## Paso 5 — Importar el dashboard

El dashboard JSON está en `Other_docs/grafana_dashboard.json`.

1. En Grafana, ve a **Dashboards → Import**
2. Pulsa **Upload dashboard JSON file**
3. Selecciona `Other_docs/grafana_dashboard.json`
4. En **Prometheus**, selecciona el datasource creado en el paso anterior
5. Pulsa **Import**

### Paneles del dashboard

| Row | Paneles |
|---|---|
| **vCenter Agent App** | Sesiones activas, queries por agente (24h), latencia p95, RAG cache hit rate, pool de conexiones, CPU y RAM Flask |
| **ESXi Hosts** | CPU y memoria por host, latencia disco RD/WR, uso datastores, tráfico red, uptime, estado hosts |
| **TrueNAS** | Accesibilidad, carga CPU, uso RAM, uso por pool ZFS, tráfico red, temperatura sensores |
| **Cisco Catalyst 3850** | Accesibilidad, CPU switch, estado interfaces, tráfico por interfaz, temperatura, fans, PSUs |

> Los rows de ESXi, TrueNAS y Cisco aparecen **colapsados** por defecto. Haz clic en el título del row para expandirlos. Requieren el Pushgateway activo (Paso 3).

---

## Paso 6 — Configurar alertas (opcional)

Ejemplo de alerta en Prometheus para sesiones excesivas:

```yaml
# En prometheus.yml, añadir sección rule_files y crear alerts.yml
rule_files:
  - "alerts.yml"
```

Contenido de `alerts.yml`:
```yaml
groups:
  - name: vcenter_agent
    rules:
      - alert: SesionesAltas
        expr: vcenter_agent_active_sessions_total > 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Demasiadas sesiones activas ({{ $value }})"

      - alert: LatenciaAltaP95
        expr: histogram_quantile(0.95, rate(vcenter_agent_query_latency_seconds_bucket[5m])) > 30
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Latencia p95 superior a 30s ({{ $value }}s)"

      - alert: ESXiDesconectado
        expr: esxi_host_connection_state == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Host ESXi desconectado: {{ $labels.host }}"
```

---

## Resumen de URLs

| Servicio | URL | Notas |
|---|---|---|
| Flask `/metrics` | `http://localhost:5000/metrics` | Requiere IP en `allowed_ips` |
| Pushgateway | `http://localhost:9091` | Solo si colectores activos |
| Prometheus | `http://localhost:9090` | Targets en Status → Targets |
| Grafana | `http://localhost:3000` | Login: admin/admin |

## Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `/metrics` devuelve 404 | Toggle desactivado | Activar en `/admin/stats` |
| `/metrics` devuelve 403 | IP no permitida | Añadir IP a `allowed_ips` en `config.json` |
| Target DOWN en Prometheus | App Flask no corre o IP incorrecta | Verificar `prometheus.yml` y estado de la app |
| No hay métricas ESXi en Pushgateway | `pushgateway.enabled: false` | Activar en `config.json` y reiniciar collector |
| Dashboard vacío en Grafana | Datasource mal configurado | Verificar URL de Prometheus en Data Sources |
| `esxi_*` métricas no aparecen | Primer ciclo aún no completado | Esperar ~10 min o revisar logs del collector |
