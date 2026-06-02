---
tipo: referencia
versión: 1.2
tags: [config, configuracion, json, vcenter, rag, esxi, vcsim]
última_actualización: 2026-04-20
relacionado:
  - "[[Stack-Tecnologico]]"
  - "[[Connection-Pool]]"
  - "[[Sistema-RAG-v2]]"
  - "[[Agente-vCenter]]"
---

# Configuración del Sistema

Archivo central de configuración en formato JSON ubicado en `vcenter_agent_system/config/config.json`. Contiene credenciales vCenter, hosts ESXi, configuración RAG v2.4, fallback a vcsim, VLANs y parámetros de recolección. El catálogo de **plantillas** ya no se hardcodea aquí: se autodetecta desde vCenter.

## Ubicación

**Archivo:** `config/config.json`

## Estructura General

```json
{
  "deployment": { /* Hosts ESXi y templates */ },
  "vlan_mappings": { /* VLANs por usuario */ },
  "vcenter_host": "...",
  "vcenter_user": "...",
  "vcenter_pass": "...",
  "esxi_hosts": { /* Hosts ESXi con credenciales */ },
  "truenas": { /* SNMP colector TrueNAS */ },
  "cisco_catalyst": { /* SNMP colector Cisco */ },
  "vcenter_fallback": { /* Fallback a vcsim */ },
  "rag_v2": { /* Configuración RAG v2.4 */ },
  "reporting": { /* Retención de datos históricos */ },
  "vcenter_user_activity": { /* Colector actividad usuarios */ }
}
```

## Secciones

### 1. deployment

Configuración de infraestructura de despliegue: hosts ESXi destino por tipo de VM.

```json
{
  "deployment": {
    "hosts": {
      "mcu_sim": {
        "ip": "172.30.188.144",
        "datastore": "datastore_44"
      },
      "eqsim": {
        "ip": "172.30.188.145",
        "datastore": "datastore_45"
      }
    },
    "templates": {
      "_comment": "DEPRECATED: Las plantillas se autodetectan desde vCenter (patrón MCU/Eqsim/SIM + legacy). Este bloque se deja vacío para evitar hardcode.",
      "mcu": {},
      "sim": {},
      "eqsim": {}
    }
  }
}
```

**Propósito:** Define:
- **hosts**: ESXi destino según tipo de VM (mcu_sim, eqsim)
- **templates**: **Deprecated** (ya no se mantiene un catálogo en `config.json`)

**Autodescubrimiento de plantillas (estado actual):**
- Las herramientas de despliegue (`deploy_dev_env`, `deploy_dev_env_2mcu`, `clone_mcu_template`) obtienen las plantillas **directamente desde vCenter** y construyen un mapping dinámico.
- Convenciones soportadas:
  - Estándar: `MCU-P28`, `Eqsim-P28`, `SIM-P24` (case-insensitive)
  - Legacy MCU: `SLES...P27...` (se asume MCU)
  - Legacy Eqsim: `EqSim-24-plantilla`
- Cache TTL configurable: `template_cache_ttl_seconds`.
- Si no se indica versión, se usa la **más alta disponible** (alias `pNN`).

> Si una plantilla no se detecta, hay que renombrarla para que cumpla el patrón o extender la lógica en `server/mcp_tool_registry.py`.

### 2. vlan_mappings

Mapeo de abreviatura de usuario a VLANs del proyecto.

```json
{
  "vlan_mappings": {
    "JaMB": {"ext": "VM Network EXT 14", "sec": "VM Network SECMON 16"},
    "PBA":  {"ext": "VM Network EXT 35", "sec": "VM Network SEC 37"},
    "JJPR": {"ext": "VM Network EXT 11", "sec": "VM Network SEC 13"},
    "FGM":  {"ext": "VM Network EXT 5",  "sec": "VM Network SEC 6"},
    "PGF":  {"ext": "VM Network EXT 41", "sec": "VM Network SEC 43"},
    "ELG":  {"ext": "VM Network EXT 29", "sec": "VM Network SEC 31"},
    "VEG":  {"ext": "VM Network EXT 23", "sec": "VM Network SEC 25"},
    "ASF":  {"ext": "VM Network EXT 8",  "sec": "VM Network SEC 10"}
  }
}
```

**Campos:**
- `ext`: Red externa (comunicación inter-VM)
- `sec`: Red de monitoreo/seguridad

**Uso:** Asignación automática de redes durante despliegue de VMs basada en usuario propietario.

### 3. vCenter Credentials

```json
{
  "vcenter_host": "172.30.188.136",
  "vcenter_user": "agent@vcenter.local",
  "vcenter_pass": "gal1$LEO"
}
```

**Seguridad:** Credenciales en texto plano (⚠️ solo para desarrollo/lab). En producción usar:
- Variables de entorno (`$env:VCENTER_USER`, `$env:VCENTER_PASS`)
- Secrets management (HashiCorp Vault, AWS Secrets Manager)

### 4. esxi_hosts

Hosts ESXi gestionados con credenciales para conexión directa.

```json
{
  "esxi_hosts": {
    "esxi8-135": {
      "host": "172.30.188.135",
      "user": "root",
      "pass": "gal1$LEO",
      "enabled": true
    },
    "esxi8-144": {
      "host": "172.30.188.144",
      "user": "root",
      "pass": "gal1$LEO",
      "enabled": true,
      "note": "Pendiente de configuración"
    },
    "esxi8-145": {
      "host": "172.30.188.145",
      "user": "root",
      "pass": "gal1$LEO",
      "enabled": true,
      "note": "Pendiente de configuración"
    }
  }
}
```

**Propósito:** Monitorización directa ESXi (CPU, memoria, red, storage) independiente de vCenter.

**Ver:** [[Agente-vCenter#Monitorización ESXi]] para endpoints de monitorización.

### 5. truenas

Configuración del colector TrueNAS vía SNMP v3.

```json
{
  "truenas": {
    "enabled": true,
    "host": "172.30.188.138",
    "port": 161,
    "snmp_version": "v3",
    "snmp_user": "agent",
    "snmp_auth_protocol": "SHA",
    "snmp_auth_password": "gal1$LEO",
    "snmp_priv_protocol": "AES",
    "snmp_priv_password": "gal1$LEO",
    "timeout_seconds": 10,
    "retries": 2,
    "collect_temperatures": true,
    "collect_network": true,
    "storage_filter_prefixes": ["/mnt", "/dev/zvol"],
    "thresholds": {
      "storage_warn_pct": 70,
      "storage_crit_pct": 85,
      "temp_warn_celsius": 65,
      "temp_crit_celsius": 80
    },
    "section_timeouts": {
      "system": 5,
      "cpu_memory": 5,
      "storage": 15,
      "network": 10,
      "temperatures": 8
    }
  }
}
```

**Métricas recolectadas:**
- Sistema (uptime, hostname, descripción)
- CPU / Memoria (uso porcentual)
- Storage (pools, datasets, uso)
- Red (interfaces, tráfico)
- Temperaturas (sensores térmicos)

**Thresholds:** Alertas warning/critical configurables por tipo de métrica.

### 6. cisco_catalyst

Colector SNMP del switch Cisco Catalyst 3850 (WS-C3850-24T).

```json
{
  "cisco_catalyst": {
    "enabled": true,
    "host": "172.30.188.151",
    "port": 161,
    "snmp_version": "v2c",
    "snmp_community": "TTCF",
    "timeout_seconds": 10,
    "retries": 2,
    "collect_vlan_traffic": true,
    "collect_hardware_health": true,
    "collect_stp": false,
    "collect_cdp": false,
    "polling_interval_seconds": 180,
    "metrics_file": "logs/cisco_catalyst_metrics.jsonl",
    "state_file": "data/cisco_catalyst_state.json",
    "section_timeouts": {
      "system": 5,
      "cpu_memory": 5,
      "interfaces": 15,
      "vlans": 10,
      "hardware": 8
    }
  }
}
```

**Métricas recolectadas:**
- Tráfico por VLAN (octetos in/out, paquetes, errores)
- Salud de hardware (fans, power supplies, temperaturas)
- CPU / Memoria del switch
- Estado de interfaces

**Output:** Métricas en `logs/cisco_catalyst_metrics.jsonl` (JSON Lines), estado en `data/cisco_catalyst_state.json`.

### 7. vcenter_fallback

Sistema de fallback automático a vcsim cuando vCenter no está disponible.

```json
{
  "vcenter_fallback": {
    "enabled": true,
    "mode": "auto",
    "ping_timeout_s": 3,
    "ping_port": 443,
    "vcsim": {
      "docker_image": "vmware/vcsim:latest",
      "container_name": "vcenter_agent_vcsim",
      "host": "127.0.0.1",
      "port": 8989,
      "user": "user",
      "pass": "pass",
      "inventory_path": "data/vcsim_inventory",
      "auto_start": true,
      "wait_ready_timeout_s": 30
    },
    "govc": {
      "binary_path": "tools/govc.exe",
      "download_if_missing": true
    }
  }
}
```

#### Modos de Fallback

| Modo | `enabled` | `mode` | Comportamiento |
|------|-----------|--------|----------------|
| **Deshabilitado** | `false` | — | Solo vCenter real, falla si inaccesible |
| **Auto** | `true` | `"auto"` | Intenta vCenter; si TCP ping falla → vcsim |
| **Forzado** | `true` | `"force_vcsim"` | Siempre usa vcsim, ignora vCenter real |

**Ver:** [[Connection-Pool#Modos de Fallback]] para flujo detallado.

#### vcsim (vCenter Simulator)

- **Docker image:** `vmware/vcsim:latest`
- **Puerto:** 8989 (configurable)
- **Auto-start:** Levanta contenedor Docker automáticamente si `auto_start: true`
- **Inventario:** `data/vcsim_inventory` — estructura de VMs/Datastores simuladas
- **Uso:** Desarrollo, testing CI/CD, demos sin vCenter real

### 8. rag_v2

Configuración del sistema RAG v2.4 (Hybrid ChromaDB + BM25).

```json
{
  "rag_v2": {
    "enabled": true,
    "features": {
      "query_expansion_v2": true,
      "embedding_cache": true,
      "reranking": true,
      "folder_filtering": true,
      "hybrid_search": true
    },
    "vector_store": {
      "db_path": "data/chroma_db",
      "embedding_model": "nomic-embed-text",
      "chunk_size": 1200,
      "chunk_overlap": 250,
      "force_rebuild": false
    },
    "hybrid_retrieval": {
      "base_alpha": 0.5,
      "initial_k": 40,
      "bm25_k1": 1.5,
      "bm25_b": 0.75,
      "internal_docs_boost": 0.75
    },
    "parameters": {
      "cache_max_size": 1000,
      "rerank_candidates": 40,
      "rerank_top_k": 12,
      "internal_doc_boost": 0.75,
      "score_threshold": 0.0
    },
    "logging": {
      "metrics_enabled": true,
      "metrics_file": "logs/retrieval_metrics.jsonl"
    }
  }
}
```

#### Parámetros Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `chunk_size` | 1200 | Caracteres por chunk (MD-aware en doc_tools.py) |
| `chunk_overlap` | 250 | Caracteres de solapamiento entre chunks |
| `embedding_model` | `nomic-embed-text` | Modelo Ollama para embeddings (768 dims) |
| `base_alpha` | 0.5 | Mix ratio vector/BM25 (adaptativo según query) |
| `initial_k` | 40 | Candidatos iniciales híbridos |
| `rerank_top_k` | 12 | Documentos finales tras reranking |
| `bm25_k1` | 1.5 | Parámetro BM25 saturation |
| `bm25_b` | 0.75 | Parámetro BM25 length normalization |
| `internal_docs_boost` | 0.75 | +75% boost para .md files (docs internas) |
| `cache_max_size` | 1000 | Queries en LRU cache de embeddings |

**Ver:** [[Sistema-RAG-v2]] para arquitectura completa del pipeline de 8 etapas.

#### Features Flags

```json
{
  "query_expansion_v2": true,      // 62 term families expansion
  "embedding_cache": true,         // LRU cache 1000 queries
  "reranking": true,               // FastReranker 40→12
  "folder_filtering": true,        // Strict/boosting/global search
  "hybrid_search": true            // ChromaDB + BM25
}
```

### 9. reporting

Configuración de retención de datos históricos y ventana del panel de administración.

```json
{
  "reporting": {
    "historical_retention_hours": 168,
    "chart_window_hours": 24
  }
}
```

**Parámetros:**
- `historical_retention_hours`: 168h (7 días) — datos antiguos se eliminan
- `chart_window_hours`: 24h — ventana de gráficos en dashboard admin

### 10. vcenter_user_activity

Recolector de actividad de usuarios nativos vCenter (tareas y eventos).

```json
{
  "vcenter_user_activity": {
    "enabled": true,
    "period_hours": 24,
    "max_tasks_per_collection": 5000,
    "system_users_filter": [
      "vpxd-extension",
      "vpxuser",
      "vsphere-webclient",
      "vdcs",
      "nsx-t"
    ],
    "error_rate_warn_pct": 10,
    "error_rate_crit_pct": 20,
    "persist_days": 30
  }
}
```

**Propósito:** Monitoriza actividad de usuarios en vCenter (no confundir con autenticación del chat):
- Recolecta tareas vCenter (`TaskInfo`) en las últimas 24h
- Filtra usuarios del sistema (vpxd-extension, vpxuser, etc.)
- Calcula tasa de error por usuario
- Persiste datos 30 días

**Output:** Dashboard de administración (`/admin/stats`) muestra usuarios más activos y tasa de error.

## Variables de Entorno (Alternativa Segura)

Para evitar credenciales en texto plano:

```powershell
# Establecer variables de entorno
$env:VCENTER_HOST = "172.30.188.136"
$env:VCENTER_USER = "agent@vcenter.local"
$env:VCENTER_PASS = "gal1$LEO"

# El código puede leer:
vcenter_host = os.getenv("VCENTER_HOST", config.get("vcenter_host"))
vcenter_user = os.getenv("VCENTER_USER", config.get("vcenter_user"))
vcenter_pass = os.getenv("VCENTER_PASS", config.get("vcenter_pass"))
```

## Archivos de Configuración Relacionados

### agents.yaml

Configuración de routing del orquestador (keywords por agente).

**Ubicación:** `config/agents.yaml`

```yaml
vcenter:
  route_keywords:
    - vm
    - virtual machine
    - vcenter
    - esxi
    - mcu
    - eqsim
    - plantilla
    # ... 50+ términos

documentation:
  route_keywords:
    - documentacion
    - archivo
    - explicame
    - como se
    # ... 30+ términos
```

**Ver:** [[Orquestador#Layer 0]] para uso en clasificación de queries.

### logging_config.json

Configuración del [[Structured-Logging]].

**Ubicación:** `config/logging_config.json`

```json
{
  "version": 1,
  "formatters": {
    "structured": {
      "()": "src.utils.structured_logger.StructuredFormatter"
    }
  },
  "handlers": {
    "api_file": {
      "class": "logging.handlers.RotatingFileHandler",
      "filename": "logs/api/api.log",
      "maxBytes": 10485760,
      "backupCount": 5
    }
    // ...
  },
  "loggers": {
    "api": {"level": "INFO", "handlers": ["api_file"]},
    "audit": {"level": "INFO", "handlers": ["audit_file"]},
    "security": {"level": "INFO", "handlers": ["security_file"]}
    // ...
  }
}
```

## Carga de Configuración

```python
import json

def load_config():
    """Carga config.json con fallback a valores por defecto"""
    config_path = "config/config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        return {}
```

**Uso en código:**

```python
config = load_config()
vcenter_host = config.get("vcenter_host")
rag_enabled = config.get("rag_v2", {}).get("enabled", True)
```

## Validación de Configuración

Validar config.json al inicio de la aplicación:

```python
def validate_config(config: dict):
    """Valida campos obligatorios"""
    required_fields = ["vcenter_host", "vcenter_user", "vcenter_pass"]
    
    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"Missing required config field: {field}")
    
    # Validar RAG v2
    if config.get("rag_v2", {}).get("enabled"):
        if not config["rag_v2"].get("vector_store", {}).get("db_path"):
            raise ValueError("RAG v2 enabled but db_path not configured")
```

## Enlaces Relacionados

- [[Stack-Tecnologico]] — Dependencias y versiones del sistema
- [[Connection-Pool]] — Uso de vcenter_fallback
- [[Sistema-RAG-v2]] — Configuración detallada de rag_v2
- [[Agente-vCenter]] — Uso de deployment, vlan_mappings, esxi_hosts
- [[Structured-Logging]] — Configuración de logging

***

**Versión del documento:** 1.2  
**Fuente:** `vcenter_agent_system/config/config.json`
