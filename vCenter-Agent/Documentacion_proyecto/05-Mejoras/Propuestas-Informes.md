---
tipo: propuesta
estado: actual
relacionado:
  - "[[Agentes-Background]]"
  - "[[Changelog]]"
  - "[[Propuestas-Funcionales]]"
tags: [mejoras, background, reporting, propuestas]
version: 1.0
ultima_actualizacion: 2026-03-24
---

# 📊 Propuestas de Mejoras - Informes Background

> Extensiones propuestas para el sistema de reporting y colección de datos en agentes background.

***
## 📋 Resumen Ejecutivo

2 mejoras identificadas para el subsistema de informes y colección de métricas:

| Mejora | Prioridad | Esfuerzo | Impacto |
|--------|-----------|----------|---------|
| **Retención 7 días** | 🟡 Media | Bajo (1-2 días) | Datos completos para tendencias semanales |
| **Gráficas matplotlib** | 🟢 Baja | Medio (3-5 días) | Visualización de tendencias en PDFs |

***
## 🔄 Mejora 1: Extender Retención Histórica 24h → 7 días

### Problema Actual

La sección "tendencia semanal" del PDF usa datos incompletos porque el colector solo mantiene las últimas 24h.

**Flujo actual:**
```
historical_data_collector.py
  ↓
Collect cada 10 min → {host_id}_historical.json
  ↓
cleanup_old_data() elimina registros > 24h
  ↓
Al generar PDF (07:00) → collect_weekly_trend()
  ↓
Intenta leer 7 días pero solo hay 24h ❌
```

### Solución Propuesta

**Cambiar retención a 168 horas (7 días) configurable:**

```python
# historical_data_collector.py, línea 33
# Antes:
self.max_data_age_hours = 24

# Después:
self.max_data_age_hours = config.get('reporting', {}).get(
    'historical_retention_hours', 168
)
```

**Configuración en `config/config.json`:**
```json
{
  "reporting": {
    "historical_retention_hours": 168,
    "cleanup_interval_hours": 24
  }
}
```

### Análisis de Impacto

| Aspecto | Antes (24h) | Después (7 días) |
|---------|-------------|------------------|
| **Registros por host** | 144 (10min × 24h) | 1,008 (10min × 7 días) |
| **Tamaño por registro** | ~500 B | ~500 B |
| **Espacio disco por host** | 72 KB | 504 KB |
| **Total 10 hosts** | 720 KB | ~5 MB |
| **Lectura trend semanal** | ❌ Incompleta | ✅ Completa |

**Coste:** ~5 MB total para 10 hosts. **Asumible.**

### Implementación

**Archivos a modificar:**
1. `server/background_agents/historical_data_collector.py` (línea 33)
2. `config/config.json` (nueva sección `reporting`)

**Tiempo estimado:** 1-2 días (incluyendo testing)

***
## 📈 Mejora 2: Gráficas de Tendencia en PDF

### Idea

Agregar datos semanales en JSON acumulado y generar gráficas matplotlib en PDFs diarios.

**Arquitectura propuesta:**
```
Recolección cada 10 min
  ↓
historical.json (rolling 7 días)
  ↓
Cuando día D-7 expira → aggregate_completed_weeks()
  ↓
weekly_history.json (acumula semanas indefinidamente)
  ↓
Gráficas matplotlib en PDF diario
```

### Casos de Uso

| Caso | Beneficio |
|------|-----------|
| **Planificación capacidad** | Ver si CPU/RAM sube durante semanas → anticipar ampliación |
| **Contexto alertas** | Saber si pico de hoy es habitual vs semanas anteriores |
| **Informe ejecutivo** | Comparativa semana a semana en una imagen |
| **Degradación lenta** | Latencia storage subiendo gradualmente → disco cerca del límite |
| **Perfil horario** | "Los martes 14h siempre hay pico CPU" |

### Viabilidad

**✅ Viable sin dependencias críticas nuevas**

| Aspecto | Estado |
|---------|--------|
| Almacenamiento 7 días raw | ~500 KB/host. Asumible. |
| Agregación semanal | Python stdlib (`json`, `datetime`, `statistics`) |
| Gráficas PDF | `matplotlib` en requirements (solo descomentar) |
| Integración ReportLab | `reportlab.platypus.Image` acepta `BytesIO` |
| Impacto performance | Agregación solo al generar informe (07:00) |

### Arquitectura de Almacenamiento

```
historical_data/
├── {host_id}_historical.json        # Raw rolling 7 días (~1008 registros)
└── {host_id}_weekly_history.json    # Agregados semanales (~100 KB/año/host)
```

#### Estructura weekly_history.json

```json
{
  "2026-W10": {
    "week_start": "2026-03-02",
    "week_end": "2026-03-08",
    "cpu_avg": 44.1,
    "cpu_peak": 89.2,
    "cpu_peak_day": "2026-03-05",
    "cpu_by_hour": [32.1, 28.4, 25.0, ...],  // 24 valores (promedio por hora del día)
    "ram_avg": 65.3,
    "ram_peak": 91.8,
    "datastore_free_avg_gb": 238.5,
    "sample_count": 1008
  },
  "2026-W11": {
    // Siguiente semana...
  }
}
```

### Pseudocódigo de Agregación

```python
def aggregate_completed_weeks():
    """
    Ejecutado al generar informe diario (07:00).
    Detecta semanas completas en historical.json y las agrega.
    """
    raw_data = load_json(f'{host_id}_historical.json')
    weekly_data = load_json(f'{host_id}_weekly_history.json') or {}
    
    # Agrupar por semana ISO
    weeks = defaultdict(list)
    for record in raw_data:
        week_id = datetime.fromisoformat(record['timestamp']).strftime('%Y-W%W')
        weeks[week_id].append(record)
    
    # Agregar semanas completas (≥1008 registros = 7 días completos)
    for week_id, records in weeks.items():
        if len(records) >= 1008 and week_id not in weekly_data:
            weekly_data[week_id] = {
                'week_start': records[0]['timestamp'][:10],
                'week_end': records[-1]['timestamp'][:10],
                'cpu_avg': mean([r['cpu_usage'] for r in records]),
                'cpu_peak': max([r['cpu_usage'] for r in records]),
                'ram_avg': mean([r['ram_usage'] for r in records]),
                # ... más métricas
                'sample_count': len(records)
            }
    
    save_json(f'{host_id}_weekly_history.json', weekly_data)
```

### Generación de Gráficas

```python
def generate_trend_chart(host_id: str, metric: str) -> BytesIO:
    """
    Genera gráfica matplotlib de tendencia semanal.
    
    Args:
        host_id: Identificador del host ESXi
        metric: 'cpu_avg' | 'ram_avg' | 'datastore_free_avg_gb'
    
    Returns:
        BytesIO con imagen PNG de la gráfica
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    weekly_data = load_json(f'{host_id}_weekly_history.json')
    
    # Extraer últimas 12 semanas
    weeks = sorted(weekly_data.keys())[-12:]
    values = [weekly_data[w][metric] for w in weeks]
    labels = [f"W{w.split('-W')[1]}" for w in weeks]
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, values, marker='o', linewidth=2)
    ax.set_title(f'{host_id} - {metric} (últimas 12 semanas)')
    ax.set_xlabel('Semana')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    
    # Guardar en memoria
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf
```

### Integración en PDF

```python
from reportlab.platypus import Image

def build_pdf_with_charts(host_id: str):
    # ... setup ReportLab ...
    
    # Añadir gráfica CPU
    cpu_chart = generate_trend_chart(host_id, 'cpu_avg')
    story.append(Image(cpu_chart, width=400, height=160))
    
    # Añadir gráfica RAM
    ram_chart = generate_trend_chart(host_id, 'ram_avg')
    story.append(Image(ram_chart, width=400, height=160))
    
    # ... resto del PDF ...
```

### Ejemplos de Gráficas

**CPU Average - 12 Semanas:**
```
80% │                               ╭─╮
    │                          ╭────╯ ╰─╮
60% │                     ╭────╯        ╰─╮
    │                ╭────╯                ╰─╮
40% │           ╭────╯                      ╰─╮
    │      ╭────╯                              ╰─
20% │  ────╯
    └──────────────────────────────────────────────
     W08  W09  W10  W11  W12  W13  W14  W15  W16
```

**Datastore Free Space - 12 Semanas:**
```
300GB│ ╮
     │  ╰─╮
250GB│    ╰─╮
     │      ╰─╮
200GB│        ╰─╮
     │          ╰─╮
150GB│            ╰─╮  ⚠️ Tendencia descendente
     │              ╰─╮
100GB│                ╰─
     └──────────────────────────────────────────────
      W08  W09  W10  W11  W12  W13  W14  W15  W16
```

### Implementación

**Archivos a modificar:**
1. `server/background_agents/historical_data_collector.py` (nueva función `aggregate_completed_weeks()`)
2. `server/background_agents/performance_report_agent.py` (añadir `generate_trend_chart()` y llamadas)
3. `requirements_oficial.txt` (descomentar `matplotlib>=3.5.0`)

**Archivos nuevos:**
- `server/background_agents/chart_generator.py` (módulo de gráficas reutilizable)

**Tiempo estimado:** 3-5 días (incluyendo testing y ajustes estéticos)

***
## 🗺️ Roadmap de Implementación

### Fase 1: Retención 7 días (1-2 días)
1. Actualizar `historical_data_collector.py`
2. Añadir sección `reporting` en `config.json`
3. Testing: verificar cleanup respeta 7 días
4. Validar tamaño de archivos históricos

### Fase 2: Agregación semanal (2-3 días)
1. Implementar `aggregate_completed_weeks()`
2. Integrar en pipeline de generación de informes
3. Testing: verificar agregación correcta de 1008 registros
4. Validar JSON semanal generado

### Fase 3: Gráficas matplotlib (2-3 días)
1. Crear `chart_generator.py`
2. Implementar `generate_trend_chart()` para CPU, RAM, Datastore
3. Integrar con ReportLab via `Image(BytesIO)`
4. Testing: PDFs con gráficas correctamente renderizadas
5. Ajustes estéticos (colores, tamaños, leyendas)

**Total estimado:** 5-8 días de desarrollo + testing

***
## 📊 Beneficios Esperados

### Mejora 1: Retención 7 días

- ✅ Datos completos para análisis de tendencias semanales
- ✅ Contexto histórico en informes PDF
- ✅ Detección de patrones de uso semanales
- ✅ Planificación de capacidad informada

### Mejora 2: Gráficas matplotlib

- ✅ Visualización intuitiva de tendencias
- ✅ Identificación rápida de degradaciones lentas
- ✅ Informes ejecutivos más profesionales
- ✅ Comparativa visual semana a semana
- ✅ Detección de patrones temporales (picos horarios)

***
## 🔧 Consideraciones Técnicas

### Impacto en Disco

| Componente | Tamaño | Por Host | 10 Hosts |
|------------|--------|----------|----------|
| Historical 7 días | ~500 KB | 500 KB | 5 MB |
| Weekly agregado (1 año) | ~100 KB | 100 KB | 1 MB |
| Gráficas PNG (en PDFs) | ~50 KB/gráfica | 150 KB | 1.5 MB |
| **Total incremental** | - | ~750 KB | **7.5 MB** |

**Impacto:** Mínimo (< 10 MB para 10 hosts)

### Impacto en Performance

| Operación | Frecuencia | Latencia | Impacto |
|-----------|-----------|----------|---------|
| Colección 10 min | Continuo | +0ms | Ninguno (solo retención) |
| Agregación semanal | 1x por semana | ~2-5s | Despreciable (07:00 AM) |
| Generación gráficas | 1x por día | ~1-2s/gráfica | Despreciable (07:00 AM) |

**Impacto total:** < 10s adicionales en generación de informe diario

***
## 📚 Documentos Relacionados

- [[Agentes-Background]] - Arquitectura del subsistema
- [[Changelog]] - Historial de mejoras implementadas
- [[Propuestas-Funcionales]] - Otras mejoras del sistema
- [[Arquitectura-Sistema]] - Visión general

***
## 📝 Dependencias

### Nuevas

- `matplotlib>=3.5.0` (ya en requirements, solo descomentar)

### Existentes (sin cambios)

- `reportlab>=3.6.0` (ya usado para PDFs)
- Python stdlib: `json`, `datetime`, `statistics`, `io`

***
*Última actualización: 2026-03-24 | v1.0*
