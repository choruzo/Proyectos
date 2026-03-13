# Documentación Técnica — Agentes de Fondo (Background Agents)

> **Sistema de Monitorización y Reporting Automático**
> Versión: 1.0 | Última actualización: 2026-03-12

---

## Índice

1. [Visión General](#1-visión-general)
2. [Arquitectura del Subsistema](#2-arquitectura-del-subsistema)
3. [Agente: Report Scheduler](#3-agente-report-scheduler)
4. [Agente: Performance Report Agent](#4-agente-performance-report-agent)
   - 4.1 [Pipeline de generación del informe](#41-pipeline-de-generación-del-informe)
   - 4.2 [Recolección vCenter](#42-recolección-vcenter)
   - 4.3 [Análisis de logs](#43-análisis-de-logs)
   - 4.4 [Métricas históricas](#44-métricas-históricas)
   - 4.5 [Análisis LLM](#45-análisis-llm)
   - 4.6 [Generación PDF](#46-generación-pdf)
5. [Colector: TrueNAS SNMP](#5-colector-truenas-snmp)
   - 5.1 [Arquitectura SNMP](#51-arquitectura-snmp)
   - 5.2 [Secciones de recolección](#52-secciones-de-recolección)
   - 5.3 [OIDs por sección](#53-oids-por-sección)
   - 5.4 [Configuración](#54-configuración)
6. [Colector: Cisco Catalyst SNMP](#6-colector-cisco-catalyst-snmp)
   - 6.1 [Arquitectura SNMP](#61-arquitectura-snmp)
   - 6.2 [Secciones de recolección](#62-secciones-de-recolección)
   - 6.3 [Cálculo de tráfico VLAN (delta de contadores)](#63-cálculo-de-tráfico-vlan-delta-de-contadores)
   - 6.4 [Correlación VLAN → Usuario](#64-correlación-vlan--usuario)
   - 6.5 [OIDs por sección](#65-oids-por-sección)
   - 6.6 [Configuración](#66-configuración)
7. [Integración de colectores en el informe](#7-integración-de-colectores-en-el-informe)
8. [Tolerancia a fallos](#8-tolerancia-a-fallos)
9. [Observabilidad y logs](#9-observabilidad-y-logs)
10. [Parámetros de configuración](#10-parámetros-de-configuración)
11. [Referencia de archivos](#11-referencia-de-archivos)

---

## 1. Visión General

El subsistema de Background Agents ejecuta tareas autónomas en segundo plano:

| Agente / Colector | Tipo | Frecuencia | Propósito |
|-------------------|------|------------|-----------|
| `report_scheduler.py` | Scheduler APScheduler | — | Dispara el informe diario a las 07:00 |
| `performance_report_agent.py` | Orquestador | Diario 07:00 | Genera PDF con estado completo del entorno |
| `truenas_snmp_collector.py` | Colector SNMP v3 | En cada informe | Métricas sistema/CPU/memoria/ZFS/red/temp TrueNAS |
| `cisco_catalyst_snmp_collector.py` | Colector SNMP v2c | En cada informe | Métricas switch Cisco Catalyst 3850 |
| `historical_data_collector.py` | Colector histórico | Cada 10 min | Serie temporal CPU/RAM ESXi para tendencias |
| `advanced_esxi_collector.py` | Colector ESXi | En cada informe | Métricas avanzadas por host ESXi |

---

## 2. Arquitectura del Subsistema

```mermaid
graph TB
    subgraph BOOT["Arranque del sistema — run.py"]
        RUN[run.py]
        HC[historical_data_collector\nBackground thread - cada 10 min]
        SCH[report_scheduler\nAPScheduler daemon]
        RUN -->|start_historical_collection| HC
        RUN -->|start_report_scheduler| SCH
    end

    subgraph TRIGGER["Trigger diario 07:00"]
        JOB[_run_report_job]
        PRA[PerformanceReportAgent.run]
        SCH -->|CronTrigger 07:00| JOB
        JOB --> PRA
    end

    subgraph PIPELINE["Pipeline del informe"]
        CV[collect_vcenter_data]
        TN[collect_truenas_data\nSNMP v3]
        CC[collect_cisco_catalyst_data\nSNMP v2c]
        AL[analyze_logs]
        HM[collect_hourly_metrics]
        WT[collect_weekly_trend]
        LLM[generate_llm_analyses\ngpt-oss:20b]
        PDF[generate_pdf\nReportLab]
        PRA --> CV & TN & CC & AL & HM & WT & LLM
        CV & TN & CC & AL & HM & WT & LLM --> PDF
    end

    subgraph STORAGE["Almacenamiento"]
        REPORTS[reports/\nPDF diario]
        CISCO_JSONL[logs/cisco_catalyst_metrics.jsonl]
        CISCO_STATE[data/cisco_catalyst_state.json]
        HIST[historical_data/\nhost_historical.json]
    end

    PDF --> REPORTS
    CC --> CISCO_JSONL
    CC --> CISCO_STATE
    HC --> HIST
```

---

## 3. Agente: Report Scheduler

`background_agents/report_scheduler.py`

```mermaid
flowchart LR
    STOP([Detenido])
    IDLE([En espera])
    EXEC([Ejecutando job])
    DONE([PDF generado])

    STOP -->|start_report_scheduler| IDLE
    IDLE -->|CronTrigger 07:00| EXEC
    EXEC -->|_run_report_job| DONE
    DONE -->|volver a esperar| IDLE
    IDLE -->|stop_report_scheduler| STOP
```

### API pública

| Función | Descripción |
|---------|-------------|
| `start_report_scheduler()` | Arranca el scheduler daemon (idempotente) |
| `stop_report_scheduler()` | Detiene el scheduler limpiamente |
| `generate_report_now()` | Genera un informe inmediato (botón admin) |
| `list_reports(n=30)` | Lista los últimos N PDF en `reports/` |

### Parámetros del job

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `trigger` | `CronTrigger(hour=7, minute=0)` | Ejecución diaria a las 07:00 |
| `misfire_grace_time` | 3600 s | Tolerancia si el sistema estaba apagado |
| `daemon` | `True` | No bloquea el shutdown de Flask |
| `id` | `daily_performance_report` | Identificador único del job |

---

## 4. Agente: Performance Report Agent

`background_agents/performance_report_agent.py`

### 4.1 Pipeline de generación del informe

```mermaid
flowchart TD
    START([run]) --> CV[collect_vcenter_data<br/>VMs · hosts · alarmas · eventos]
    START --> TN[collect_truenas_data<br/>SNMP v3 — no bloqueante]
    START --> CC[collect_cisco_catalyst_data<br/>SNMP v2c — no bloqueante]
    START --> AL[analyze_logs<br/>últimas 24h ERROR/CRITICAL]
    START --> HM[collect_hourly_metrics<br/>serie temporal 24h]
    START --> HA[collect_hourly_user_activity<br/>audit.log por hora]
    START --> WT[collect_weekly_trend<br/>tendencia semanal]

    CV --> LLM[generate_llm_analyses<br/>5 párrafos LLM]
    TN --> LLM
    CC --> LLM
    AL --> LLM
    HM --> LLM
    WT --> LLM

    LLM --> PDF[generate_pdf<br/>ReportLab]
    CV --> PDF
    TN --> PDF
    CC --> PDF
    AL --> PDF
    HM --> PDF
    WT --> PDF

    PDF --> FILE[reports/informe_rendimiento_YYYY-MM-DD.pdf]

    style TN fill:#2980b9,color:#fff
    style CC fill:#27ae60,color:#fff
    style LLM fill:#8e44ad,color:#fff
    style PDF fill:#e74c3c,color:#fff
```

### 4.2 Recolección vCenter

```mermaid
flowchart LR
    SI[vCenter SI<br/>get_si] --> VMENUM[list_all_vms_in_vcenter<br/>total · on · off]
    SI --> ALARMS[get_active_alarms]
    SI --> EVENTS[collect_vcenter_events<br/>últimas 24h · 6 tipos de evento]
    SI --> TOPVMS[collect_top_vms<br/>top 10 CPU/RAM]
    SI --> SNAPS[collect_obsolete_snapshots<br/>antigüedad ≥ 7 días]
    SI --> HW[collect_host_hardware<br/>vendor · model · CPU · RAM]
    SI --> LICS[collect_license_info<br/>LicenseManager]
    SI --> DS[collect_datastore_details<br/>capacidad · uso · VMs]
    SI --> ANOMALIES[collect_vm_anomalies<br/>zombies · tools · templates]

    ESXI[AdvancedESXiCollector<br/>por host habilitado] --> HOSTMETRICS[CPU% · RAM% · DS%<br/>estado · uptime]
```

**Eventos vCenter capturados (últimas 24h):**
- `HostConnectionLostEvent`
- `GeneralHostErrorEvent`
- `GeneralVmErrorEvent`
- `AlarmStatusChangedEvent`
- `VmPoweredOnEvent`
- `VmPoweredOffEvent`

**Anomalías de VMs detectadas:**

| Tipo | Criterio |
|------|----------|
| Zombie VMs | `poweredOn` + `overallCpuUsage == 0 MHz` |
| VMware Tools no instaladas | `toolsStatus == toolsNotInstalled` |
| VMware Tools no ejecutando | `toolsNotRunning` + `poweredOn` |
| VMware Tools desactualizadas | `toolsStatus == toolsOld` |
| Plantillas | `config.template == True` |

### 4.3 Análisis de logs

```mermaid
flowchart LR
    LOGS["Archivos de log JSONL
    logs/system.log
    logs/api/api.log
    logs/audit/audit.log
    logs/security/security.log
    logs/performance/performance.log"]

    LOGS --> FILTER["Filtro:
    level == ERROR | CRITICAL
    timestamp >= now - 24h"]

    FILTER --> GROUP["Agrupado por categoría
    {system: [...], api: [...], ...}"]
    GROUP --> COUNT["_total_errors: N"]
```

### 4.4 Métricas históricas

```mermaid
flowchart LR
    HIST["historical_data/
    {host_id}_historical.json
    (escritos por historical_data_collector
    cada 10 minutos)"]

    HIST --> HOURLY["collect_hourly_metrics:
    avg_cpu_24h · peak_cpu
    peak_cpu_hour · avg_ram_24h"]

    HIST --> WEEKLY["collect_weekly_trend:
    tendencia semanal
    por hora del día"]

    AUDIT["logs/audit/audit.log"] --> UA["collect_hourly_user_activity:
    consultas por hora"]
```

### 4.5 Análisis LLM

El agente genera **5 párrafos de análisis narrativo** usando el LLM local (`gpt-oss:20b`):

```mermaid
flowchart TD
    DATA[Datos recolectados] --> P1["Prompt P1 — Estado general
    resumen ejecutivo del entorno vCenter"]
    DATA --> P2["Prompt P2 — Análisis de hosts ESXi
    CPU · RAM · datastores con umbrales"]
    DATA --> P3["Prompt P3 — Diagnóstico de errores
    clasificación de severidad por categoría"]
    DATA --> P4["Prompt P4 — Hora pico
    análisis de actividad horaria"]
    DATA --> P5["Prompt P5 — Eventos vCenter
    análisis de eventos de las 24h"]

    DATA2[TrueNAS + Cisco Catalyst] --> P6["Prompt P6 — Infraestructura de red/storage
    estado TrueNAS ZFS + tráfico VLAN Cisco"]

    P1 --> LLM["ChatOllama
    gpt-oss:20b
    timeout=30s por párrafo"]
    P2 --> LLM
    P3 --> LLM
    P4 --> LLM
    P5 --> LLM
    P6 --> LLM

    LLM --> BLOCK["Bloque LLM en PDF
    texto narrativo formatteado"]
```

**Comportamiento ante timeout:** Si el LLM no responde en 30s, el bloque correspondiente aparece como `None` y el PDF omite ese análisis sin abortar la generación.

### 4.6 Generación PDF

El PDF se genera con **ReportLab** y se guarda en `reports/informe_rendimiento_YYYY-MM-DD.pdf`.

```mermaid
flowchart TD
    PDF_START[generate_pdf] --> HDR[Cabecera con fecha y logo]
    HDR --> SEC1[Sección 1 — Resumen Ejecutivo
    estado vCenter · totales VM · alarmas · errores]

    SEC1 --> SEC2[Sección 2 — Análisis LLM P1
    párrafo narrativo estado general]

    SEC2 --> SEC3[Sección 3 — Hosts ESXi
    tabla CPU · RAM · DS · uptime · alertas color]

    SEC3 --> SEC4[Sección 4 — Hardware de hosts
    vendor · model · ESXi version · CPU specs]

    SEC4 --> SEC5[Sección 5 — Datastores
    capacidad · uso% · VMs por datastore]

    SEC5 --> SEC6[Sección 6 — Top VMs CPU/RAM]

    SEC6 --> SEC7[Sección 7 — Snapshots obsoletos
    > 7 días · overhead GB]

    SEC7 --> SEC8[Sección 8 — Anomalías de VMs
    zombies · tools · templates]

    SEC8 --> SEC9[Sección 9 — Eventos vCenter 24h]

    SEC9 --> SEC10[Sección 10 — Errores de logs 24h]

    SEC10 --> SEC11[Sección 11 — TrueNAS
    CPU · memoria · pools ZFS · red · temperatura]

    SEC11 --> SEC12[Sección 12 — Cisco Catalyst
    CPU · memoria · interfaces · VLAN traffic · HW]

    SEC12 --> SEC13[Sección 13 — Tendencia semanal
    gráfica CPU/RAM por hora]

    SEC13 --> SAVE[Guardar PDF
    reports/informe_rendimiento_YYYY-MM-DD.pdf]
```

---

## 5. Colector: TrueNAS SNMP

`background_agents/truenas_snmp_collector.py`

### 5.1 Arquitectura SNMP

```mermaid
flowchart TD
    INIT[TrueNASSnmpCollector.__init__
    config/config.json → truenas] --> RESOLVE[Resolver binarios
    snmpget · snmpwalk vía shutil.which + paths conocidos]

    RESOLVE --> COLLECT[collect]

    COLLECT --> CHECK{¿enabled?
    ¿host configurado?
    ¿binarios encontrados?}
    CHECK -->|No| ERR[Retornar accessible=False]
    CHECK -->|Sí| PROBE[snmpget sysName
    prueba de conectividad]

    PROBE -->|Sin respuesta| ERR
    PROBE -->|OK| SECTIONS

    subgraph SECTIONS["Recolección por secciones (tolerante a fallos)"]
        S1[_get_system_info
        timeout: section_timeouts.system]
        S2[_get_cpu_load
        timeout: section_timeouts.cpu_memory]
        S3[_get_memory
        timeout: section_timeouts.cpu_memory]
        S4[_get_storage
        timeout: section_timeouts.storage]
        S5[_get_network
        si collect_network=true]
        S6[_get_temperatures
        si collect_temperatures=true]
    end

    SECTIONS --> RESULT["dict resultado:
    accessible · system · cpu · memory
    storage · network · temperatures · errors"]
```

**Principios de diseño:**
- **Sin dependencias Python para SNMP**: usa `subprocess` + binarios del SO (`snmpget`/`snmpwalk`)
- **Sin estado entre llamadas**: cada `collect()` es independiente
- **Tolerante a fallos**: un OID no disponible no detiene la recolección — se añade a `errors[]`
- **Timeouts por sección**: cada sección tiene su propio timeout configurable

### 5.2 Secciones de recolección

```mermaid
flowchart LR
    subgraph "Sistema (MIB-II)"
        S1A[sysName] --> SI[system dict]
        S1B[sysDescr] --> SI
        S1C[sysUptime → horas] --> SI
        S1D[sysContact] --> SI
    end

    subgraph "CPU (UCD-SNMP laLoad)"
        S2A[laLoad.1 → 1min] --> CPU[cpu dict]
        S2B[laLoad.2 → 5min] --> CPU
        S2C[laLoad.3 → 15min] --> CPU
    end

    subgraph "Memoria (UCD-SNMP, kB→GB)"
        S3A[memTotalReal → total_gb] --> MEM[memory dict]
        S3B[memAvailReal → free_gb] --> MEM
        S3C[memTotalSwap → swap_total_gb] --> MEM
        S3D[memAvailSwap → swap_free_gb] --> MEM
        S3E[usage_percent calculado] --> MEM
    end

    subgraph "Almacenamiento (hrStorage walk)"
        S4A[hrStorageDescr] --> ST[storage list]
        S4B[hrStorageUnits] --> ST
        S4C[hrStorageSize × Units → GB] --> ST
        S4D[hrStorageUsed × Units → GB] --> ST
        S4E[Filtro: /mnt · /dev/zvol] --> ST
        S4F[is_container: path parent] --> ST
    end

    subgraph "Red (IF-MIB walk)"
        S5A[ifName] --> NET[network list]
        S5B[ifOperStatus: up/down] --> NET
        S5C[ifHighSpeed → Mbps] --> NET
        S5D[ifHCInOctets] --> NET
        S5E[ifHCOutOctets] --> NET
        S5F[ifInErrors / ifOutErrors] --> NET
    end

    subgraph "Temperatura (lm-sensors UCD-SNMP)"
        S6A[lmTempSensorsDescr] --> TEMP[temperatures list]
        S6B[lmTempSensorsValue / 1000 → °C] --> TEMP
    end
```

### 5.3 OIDs por sección

| Sección | OID | Descripción |
|---------|-----|-------------|
| **Sistema** | `1.3.6.1.2.1.1.5.0` | sysName |
| | `1.3.6.1.2.1.1.1.0` | sysDescr |
| | `1.3.6.1.2.1.1.3.0` | sysUpTime |
| | `1.3.6.1.2.1.1.4.0` | sysContact |
| **CPU** | `1.3.6.1.4.1.2021.10.1.3.1` | laLoad 1 min |
| | `1.3.6.1.4.1.2021.10.1.3.2` | laLoad 5 min |
| | `1.3.6.1.4.1.2021.10.1.3.3` | laLoad 15 min |
| **Memoria** | `1.3.6.1.4.1.2021.4.5.0` | memTotalReal (kB) |
| | `1.3.6.1.4.1.2021.4.6.0` | memAvailReal (kB) |
| | `1.3.6.1.4.1.2021.4.3.0` | memTotalSwap (kB) |
| | `1.3.6.1.4.1.2021.4.4.0` | memAvailSwap (kB) |
| **Almacenamiento** | `1.3.6.1.2.1.25.2.3.1.3` | hrStorageDescr (walk) |
| | `1.3.6.1.2.1.25.2.3.1.4` | hrStorageAllocationUnits |
| | `1.3.6.1.2.1.25.2.3.1.5` | hrStorageSize |
| | `1.3.6.1.2.1.25.2.3.1.6` | hrStorageUsed |
| **Red** | `1.3.6.1.2.1.31.1.1.1.1` | ifName (walk) |
| | `1.3.6.1.2.1.2.2.1.8` | ifOperStatus |
| | `1.3.6.1.2.1.31.1.1.1.15` | ifHighSpeed (Mbps) |
| | `1.3.6.1.2.1.31.1.1.1.6` | ifHCInOctets (64-bit) |
| | `1.3.6.1.2.1.31.1.1.1.10` | ifHCOutOctets (64-bit) |
| | `1.3.6.1.2.1.2.2.1.14` | ifInErrors |
| | `1.3.6.1.2.1.2.2.1.20` | ifOutErrors |
| **Temperatura** | `1.3.6.1.4.1.2021.13.16.2.1.2` | lmTempSensorsDescr (walk) |
| | `1.3.6.1.4.1.2021.13.16.2.1.3` | lmTempSensorsValue (milli°C) |

### 5.4 Configuración

Sección `truenas` en `config/config.json`:

```json
{
  "truenas": {
    "enabled": true,
    "host": "192.168.1.X",
    "port": 161,
    "timeout_seconds": 10,
    "retries": 2,
    "snmp_user": "agent",
    "snmp_auth_protocol": "SHA",
    "snmp_auth_password": "YOUR_AUTH_PASS",
    "snmp_priv_protocol": "AES",
    "snmp_priv_password": "YOUR_PRIV_PASS",
    "collect_temperatures": true,
    "collect_network": true,
    "storage_filter_prefixes": ["/mnt", "/dev/zvol"],
    "section_timeouts": {
      "system": 5,
      "cpu_memory": 8,
      "storage": 15,
      "network": 10,
      "temperatures": 8
    }
  }
}
```

**Requisito:** Binarios SNMP del SO instalados:
```bash
sudo apt-get install snmp
```

---

## 6. Colector: Cisco Catalyst SNMP

`background_agents/cisco_catalyst_snmp_collector.py`

### 6.1 Arquitectura SNMP

```mermaid
flowchart TD
    INIT[CiscoCatalystSnmpCollector.__init__
    config/config.json → cisco_catalyst] --> RESOLVE[Resolver binarios
    snmpget · snmpwalk]

    RESOLVE --> COLLECT[collect]

    COLLECT --> CHECK{¿enabled?
    ¿host?
    ¿binarios?}
    CHECK -->|No| ERR[accessible=False]
    CHECK -->|Sí| PROBE[snmpget sysName
    SNMPv2c + community]

    PROBE -->|Sin respuesta| ERR
    PROBE -->|OK| SECTIONS

    subgraph SECTIONS["Secciones"]
        S1[_get_system_info]
        S2[_get_cpu
        CISCO-PROCESS-MIB]
        S3[_get_memory
        CISCO-MEMORY-POOL-MIB]
        S4[_get_all_interfaces
        IF-MIB walk]
        S5[_compute_vlan_traffic
        si collect_vlan_traffic=true]
        S6[_get_temperatures
        si collect_hardware_health=true]
        S7[_get_fans]
        S8[_get_power_supplies]
    end

    SECTIONS --> JSONL[_append_metrics_jsonl
    logs/cisco_catalyst_metrics.jsonl]
    SECTIONS --> RESULT[dict resultado]
```

**Diferencia clave vs TrueNAS:** usa **SNMPv2c** (community string) en lugar de SNMPv3 (usuario + auth + priv).

### 6.2 Secciones de recolección

```mermaid
flowchart LR
    subgraph "Sistema (MIB-II)"
        C1A[sysName] --> CSYS[system dict]
        C1B[sysDescr — IOS version] --> CSYS
        C1C[sysUpTime → horas] --> CSYS
        C1D[sysLocation] --> CSYS
    end

    subgraph "CPU (CISCO-PROCESS-MIB)"
        C2A[cpmCPUTotal5sec → pct_5sec] --> CCPU[cpu dict]
        C2B[cpmCPUTotal1min → pct_1min] --> CCPU
        C2C[cpmCPUTotal5min → pct_5min] --> CCPU
    end

    subgraph "Memoria (CISCO-MEMORY-POOL-MIB)"
        C3A[ciscoMemoryPoolName] --> CMEM[memory list]
        C3B[ciscoMemoryPoolUsed → MB] --> CMEM
        C3C[ciscoMemoryPoolFree → MB] --> CMEM
        C3D[ciscoMemoryPoolUsedMax → MB] --> CMEM
        C3E[usage_percent calculado] --> CMEM
        NOTE["Pools: Processor, I/O"]
    end

    subgraph "Interfaces (IF-MIB walk)"
        C4A[ifName] --> CIFACE[interfaces/vlan_traffic]
        C4B[ifAlias — descripción admin] --> CIFACE
        C4C[ifOperStatus → up/down] --> CIFACE
        C4D[ifHighSpeed → Mbps] --> CIFACE
        C4E[ifHCIn/OutOctets 64-bit] --> CIFACE
        C4F[ifIn/OutErrors + Discards] --> CIFACE
    end

    subgraph "Hardware (CISCO-ENVMON-MIB)"
        C5A[ciscoEnvMonTempStatus] --> CHW[temps/fans/power]
        C5B[ciscoEnvMonFanStatus] --> CHW
        C5C[ciscoEnvMonSupplyStatus] --> CHW
        NOTE2["Estados: normal/warning/critical/shutdown"]
    end
```

### 6.3 Cálculo de tráfico VLAN (delta de contadores)

El tráfico por VLAN se calcula mediante **deltas de contadores HC de 64 bits** entre polls consecutivos. Los contadores se persisten entre ejecuciones.

```mermaid
sequenceDiagram
    participant POLL1 as Poll N (primer run)
    participant STATE as data/cisco_catalyst_state.json
    participant POLL2 as Poll N+1 (07:00 siguiente)

    POLL1 ->> STATE: Guardar {ts, counters: {idx: {in_octets, out_octets}}}
    Note over POLL1: in_mbps = None (sin previo)

    POLL2 ->> STATE: Cargar estado anterior
    STATE -->> POLL2: {collected_at_ts, counters}

    Note over POLL2: delta_secs = ts_now - ts_prev
    Note over POLL2: d_in = in_now - in_prev (max 0)
    Note over POLL2: in_mbps = d_in × 8 / delta_secs / 1e6

    POLL2 ->> STATE: Actualizar estado con contadores actuales
```

**Fórmula de conversión:**

```
in_mbps  = (in_octets_now  - in_octets_prev)  × 8 / delta_seconds / 1_000_000
out_mbps = (out_octets_now - out_octets_prev) × 8 / delta_seconds / 1_000_000
```

> Los contadores son 64-bit (HC — High Capacity). No deberían hacer wrap en intervalos de ~24h pero el código aplica `max(0, delta)` como salvaguarda.

### 6.4 Correlación VLAN → Usuario

```mermaid
flowchart LR
    CONFIG["config/config.json
    vlan_mappings:
      JaMB:
        ext: 'VM Network EXT 14'
        int: 'VM Network INT 15'"]

    CONFIG --> BUILD["_build_vlan_user_map:
    regex r'(\\d+)\\s*$' sobre net_name
    → {14: {user:'JaMB', network_type:'ext'},
       15: {user:'JaMB', network_type:'int'}}"]

    BUILD --> IFACE["_is_vlan_iface(name):
    regex r'[Vv]l(?:an)?\\d+$'
    Vlan14 · vlan2 · Vl14 → True"]

    IFACE --> MAP["_parse_vlan_id:
    'Vlan14' → 14"]

    MAP --> RESULT["vlan_traffic entry:
    {vlan_id: 14, user: 'JaMB',
    network_type: 'ext',
    in_mbps: 2.34, out_mbps: 0.87}"]
```

### 6.5 OIDs por sección

| Sección | OID | Descripción |
|---------|-----|-------------|
| **Sistema** | `1.3.6.1.2.1.1.5.0` | sysName |
| | `1.3.6.1.2.1.1.1.0` | sysDescr (versión IOS) |
| | `1.3.6.1.2.1.1.3.0` | sysUpTime |
| | `1.3.6.1.2.1.1.6.0` | sysLocation |
| **CPU** | `1.3.6.1.4.1.9.9.109.1.1.1.1.6.1` | cpmCPUTotal5sec (%) |
| | `1.3.6.1.4.1.9.9.109.1.1.1.1.7.1` | cpmCPUTotal1min (%) |
| | `1.3.6.1.4.1.9.9.109.1.1.1.1.8.1` | cpmCPUTotal5min (%) |
| **Memoria** | `1.3.6.1.4.1.9.9.48.1.1.1.2` | ciscoMemoryPoolName (walk) |
| | `1.3.6.1.4.1.9.9.48.1.1.1.5` | ciscoMemoryPoolUsed (bytes) |
| | `1.3.6.1.4.1.9.9.48.1.1.1.6` | ciscoMemoryPoolFree (bytes) |
| | `1.3.6.1.4.1.9.9.48.1.1.1.7` | ciscoMemoryPoolUsedMax (bytes) |
| **Interfaces** | `1.3.6.1.2.1.31.1.1.1.1` | ifName (walk) |
| | `1.3.6.1.2.1.31.1.1.1.18` | ifAlias (descripción admin) |
| | `1.3.6.1.2.1.2.2.1.8` | ifOperStatus |
| | `1.3.6.1.2.1.31.1.1.1.15` | ifHighSpeed (Mbps) |
| | `1.3.6.1.2.1.31.1.1.1.6` | ifHCInOctets (64-bit) |
| | `1.3.6.1.2.1.31.1.1.1.10` | ifHCOutOctets (64-bit) |
| | `1.3.6.1.2.1.2.2.1.14` | ifInErrors |
| | `1.3.6.1.2.1.2.2.1.20` | ifOutErrors |
| | `1.3.6.1.2.1.2.2.1.13` | ifInDiscards |
| | `1.3.6.1.2.1.2.2.1.19` | ifOutDiscards |
| **VLANs** | `1.3.6.1.4.1.9.9.46.1.3.1.1.2.1` | vtpVlanState (walk) |
| | `1.3.6.1.4.1.9.9.46.1.3.1.1.4.1` | vtpVlanName (walk) |
| **Temperatura** | `1.3.6.1.4.1.9.9.13.1.3.1.2` | ciscoEnvMonTempStatusDescr |
| | `1.3.6.1.4.1.9.9.13.1.3.1.3` | ciscoEnvMonTempStatusValue (°C) |
| | `1.3.6.1.4.1.9.9.13.1.3.1.6` | ciscoEnvMonTempState |
| **Fans** | `1.3.6.1.4.1.9.9.13.1.4.1.2` | ciscoEnvMonFanStatusDescr |
| | `1.3.6.1.4.1.9.9.13.1.4.1.3` | ciscoEnvMonFanState |
| **Fuentes alim.** | `1.3.6.1.4.1.9.9.13.1.5.1.2` | ciscoEnvMonSupplyStatusDescr |
| | `1.3.6.1.4.1.9.9.13.1.5.1.3` | ciscoEnvMonSupplyState |

**Códigos de estado ENVMON:**

| Código | Estado |
|--------|--------|
| 1 | `normal` |
| 2 | `warning` |
| 3 | `critical` |
| 4 | `shutdown` |
| 5 | `notPresent` |
| 6 | `notFunctioning` |

### 6.6 Configuración

Sección `cisco_catalyst` en `config/config.json`:

```json
{
  "cisco_catalyst": {
    "enabled": true,
    "host": "192.168.X.X",
    "port": 161,
    "timeout_seconds": 10,
    "retries": 2,
    "snmp_community": "TTCF",
    "collect_hardware_health": true,
    "collect_vlan_traffic": true,
    "state_file": "data/cisco_catalyst_state.json",
    "metrics_file": "logs/cisco_catalyst_metrics.jsonl",
    "section_timeouts": {
      "system": 5,
      "cpu_memory": 8,
      "interfaces": 15,
      "vlans": 10,
      "hardware": 10
    }
  },
  "vlan_mappings": {
    "JaMB": {
      "ext": "VM Network EXT 14",
      "int": "VM Network INT 15"
    }
  }
}
```

---

## 7. Integración de colectores en el informe

```mermaid
flowchart TD
    subgraph "PerformanceReportAgent.run()"
        BLOCK_TN["try:
        collect_truenas_data(config)
        → truenas_data"]

        BLOCK_CC["try:
        collect_cisco_catalyst_data(config)
        → cisco_data"]

        NOTE["Ambos bloques son NO BLOQUEANTES:
        si fallan → truenas_data = None
        cisco_data = None
        El informe continúa sin ellos"]
    end

    subgraph "generate_pdf"
        TRUENAS_SEC["Sección TrueNAS:
        Solo si truenas_data.accessible == True
        CPU · Memoria · Pools ZFS · Interfaces · Temp"]

        CISCO_SEC["Sección Cisco Catalyst:
        Solo si cisco_data.accessible == True
        CPU · Memoria pools · Interfaces · VLANs · HW"]
    end

    subgraph "generate_llm_analyses"
        LLM_INFRA["Prompt P6 infra:
        TrueNAS + Cisco Catalyst
        análisis narrativo conjunto"]
    end

    BLOCK_TN --> TRUENAS_SEC
    BLOCK_CC --> CISCO_SEC
    BLOCK_TN --> LLM_INFRA
    BLOCK_CC --> LLM_INFRA
```

---

## 8. Tolerancia a fallos

```mermaid
flowchart TD
    subgraph "Nivel colector (TrueNAS / Cisco)"
        CF1["OID no disponible
        → añadir a errors[]
        continuar con siguiente sección"]
        CF2["Timeout de sección
        → error no fatal
        resultado parcial disponible"]
        CF3["Host inaccesible
        → accessible=False
        retornar dict vacío válido"]
    end

    subgraph "Nivel PerformanceReportAgent"
        PF1["collect_truenas_data falla
        → truenas_data = None
        PDF se genera sin sección TrueNAS"]
        PF2["collect_cisco_catalyst_data falla
        → cisco_data = None
        PDF se genera sin sección Cisco"]
        PF3["vCenter inaccesible
        → vcenter_accessible=False
        secciones vCenter vacías en PDF"]
        PF4["LLM timeout (30s)
        → párrafo = None
        PDF omite ese bloque de análisis"]
    end

    subgraph "Nivel Scheduler"
        SF1["Job falla con excepción
        → logger.error
        próxima ejecución: mañana 07:00"]
        SF2["Sistema apagado durante 07:00
        → misfire_grace_time=3600s
        ejecuta al arrancar si < 1h tarde"]
    end
```

**Principio**: ningún fallo individual interrumpe la generación del PDF. El informe puede ser parcial pero siempre se genera.

---

## 9. Observabilidad y logs

### Archivos de log

| Log | Ruta | Contenido |
|-----|------|-----------|
| Performance Report Agent | `logs/system.log` | Errores + info del agente |
| Report Scheduler | `logs/system.log` | Start/stop scheduler, job completado |
| Cisco métricas históricas | `logs/cisco_catalyst_metrics.jsonl` | Una línea JSONL por poll |
| Estado VLAN counters | `data/cisco_catalyst_state.json` | Contadores del último poll |

### Diagnóstico rápido

```powershell
# Ver último informe generado
ls reports/ | sort -Property LastWriteTime -Descending | head -1

# Verificar que el scheduler está activo
Get-Content logs/system.log -Tail 50 | Select-String "Scheduler de informes"

# Ver último poll Cisco Catalyst
Get-Content logs/cisco_catalyst_metrics.jsonl -Tail 1 | ConvertFrom-Json

# Forzar generación manual (consola Python)
# from background_agents.report_scheduler import generate_report_now
# generate_report_now()

# Ver errores de los colectores SNMP en el último informe
Get-Content logs/system.log -Tail 100 | Select-String "truenas|cisco_catalyst"
```

### Formato JSONL Cisco Catalyst

```json
{
  "accessible": true,
  "collected_at": "2026-03-12T07:00:05+00:00",
  "host": "192.168.X.X",
  "system": {"name": "SW-TTCF", "uptime_hours": 2160.5},
  "cpu": {"available": true, "pct_5sec": 4.0, "pct_1min": 3.0, "pct_5min": 3.0},
  "memory": [{"pool": "Processor", "total_mb": 870.6, "used_mb": 540.1, "usage_percent": 62.0}],
  "interfaces": [...],
  "vlan_traffic": [
    {"vlan_id": 14, "user": "JaMB", "network_type": "ext", "in_mbps": 2.34, "out_mbps": 0.87}
  ],
  "temperatures": [{"sensor": "Switch 1 - Inlet Temp Sensor", "temp_celsius": 28, "status": "normal"}],
  "fans": [{"fan": "Switch 1 - FAN 1", "status": "normal"}],
  "power": [{"psu": "Switch 1 - Power Supply A", "status": "normal"}],
  "errors": []
}
```

---

## 10. Parámetros de configuración

### Resumen global

| Parámetro | Archivo | Valor por defecto | Descripción |
|-----------|---------|-------------------|-------------|
| Scheduler hora | `report_scheduler.py` | `hour=7, minute=0` | Hora de ejecución diaria |
| `misfire_grace_time` | `report_scheduler.py` | `3600` s | Tolerancia por sistema apagado |
| `threshold_days` snapshots | `performance_report_agent.py` | `7` días | Snapshots considerados obsoletos |
| LLM timeout | `performance_report_agent.py` | `30` s por párrafo | Tiempo máximo por análisis LLM |
| Eventos vCenter max | `performance_report_agent.py` | `50` | Top N eventos en el informe |
| TrueNAS timeout global | `config.json → truenas` | `10` s | Fallback por sección |
| Cisco timeout global | `config.json → cisco_catalyst` | `10` s | Fallback por sección |
| Storage filter prefixes | `config.json → truenas` | `["/mnt", "/dev/zvol"]` | Filtrar pools ZFS relevantes |
| SNMP retries | ambos colectores | `2` | Reintentos antes de error |
| Cisco state file | `config.json → cisco_catalyst` | `data/cisco_catalyst_state.json` | Persistencia counters VLAN |
| Cisco metrics JSONL | `config.json → cisco_catalyst` | `logs/cisco_catalyst_metrics.jsonl` | Histórico de polls |

---

## 11. Referencia de archivos

```mermaid
graph LR
    subgraph "Scheduling"
        RS[report_scheduler.py<br/>APScheduler · CronTrigger 07:00]
    end

    subgraph "Orquestación"
        PRA[performance_report_agent.py<br/>PerformanceReportAgent<br/>Pipeline completo → PDF]
    end

    subgraph "Colectores"
        TN[truenas_snmp_collector.py<br/>TrueNASSnmpCollector<br/>SNMPv3]
        CC[cisco_catalyst_snmp_collector.py<br/>CiscoCatalystSnmpCollector<br/>SNMPv2c]
        AEC[advanced_esxi_collector.py<br/>AdvancedESXiCollector<br/>pyvmomi directo]
        HDC[historical_data_collector.py<br/>Thread cada 10min]
    end

    subgraph "Dependencias externas"
        SNAP[snmpget / snmpwalk<br/>binarios SO]
        RL[reportlab<br/>generación PDF]
        APS[apscheduler<br/>BackgroundScheduler]
        OLLAMA[ChatOllama gpt-oss:20b<br/>análisis LLM]
    end

    RS -->|invoca| PRA
    PRA -->|colecta| TN
    PRA -->|colecta| CC
    PRA -->|colecta| AEC
    PRA -->|lee| HDC
    TN --> SNAP
    CC --> SNAP
    PRA --> RL
    RS --> APS
    PRA --> OLLAMA
```

### Tabla de archivos

| Archivo | Clase / Función principal | Propósito |
|---------|--------------------------|-----------|
| `background_agents/report_scheduler.py` | `start_report_scheduler()` | Scheduler APScheduler diario 07:00 |
| `background_agents/performance_report_agent.py` | `PerformanceReportAgent` | Orquestación completa + PDF ReportLab |
| `background_agents/truenas_snmp_collector.py` | `TrueNASSnmpCollector` | Métricas TrueNAS via SNMPv3 |
| `background_agents/cisco_catalyst_snmp_collector.py` | `CiscoCatalystSnmpCollector` | Métricas Cisco Catalyst 3850 via SNMPv2c |
| `advanced_esxi_collector.py` | `AdvancedESXiCollector` | Métricas avanzadas ESXi via pyvmomi |
| `historical_data_collector.py` | `start_historical_collection()` | Serie temporal CPU/RAM ESXi (10 min) |
| `data/cisco_catalyst_state.json` | — | Persistencia de contadores VLAN entre polls |
| `logs/cisco_catalyst_metrics.jsonl` | — | Histórico de polls Cisco en formato JSONL |
| `reports/` | — | PDFs generados (`informe_rendimiento_YYYY-MM-DD.pdf`) |
| `historical_data/` | — | Series temporales `{host_id}_historical.json` |

---

*Documentación generada a partir del código fuente de `vcenter_agent_system/background_agents/`.*
*Para detalles de la arquitectura general del sistema ver `README.md` y `.github/copilot-instructions.md`.*
