# 📡 Pipeline - Git Monitor (Fase 1)

## Visión General

**Git Monitor** es la primera fase del pipeline CI/CD. Detecta nuevos tags en el repositorio Git remoto y los marca para procesamiento.

**Relacionado con**:
- [[Arquitectura del Pipeline]] - Contexto general del pipeline
- [[Pipeline - Compilación]] - Siguiente fase del pipeline
- [[Modelo de Datos#processed_tags]] - Tabla para evitar reprocesamiento
- [[Referencia - APIs Externas#Git]] - API Git utilizada

---

## Responsabilidades

1. **Polling periódico** - Consulta repositorio Git cada 5 minutos
2. **Filtrado de tags** - Aplica regex para validar formato de tags
3. **Detección de nuevos** - Compara con tabla `processed_tags`
4. **Checkout** - Clona/actualiza repo y hace checkout del tag
5. **Marcado** - Registra tag en DB para evitar reprocesar

---

## Ubicación y Ejecución

**Script**: `scripts/git_monitor.sh`

**Invocación**:
```bash
# Desde ci_cd.sh (daemon mode)
./scripts/git_monitor.sh detect

# Checkout manual de tag específico
./scripts/git_monitor.sh checkout MAC_1_V24_02_15_01
```

**Dependencias**:
- `git` CLI tool
- Credenciales en `config/.env` (GIT_USER, GIT_PASSWORD)
- Tabla `processed_tags` en SQLite
- [[Pipeline - Common Functions]] (`common.sh`)

---

## Arquitectura

```mermaid
sequenceDiagram
    participant Daemon as ci_cd.sh (daemon)
    participant GM as git_monitor.sh
    participant Git as Git Remote
    participant DB as SQLite
    participant FS as File System
    
    loop Every 5 minutes
        Daemon->>GM: detect_new_tags()
        GM->>Git: git ls-remote --tags
        Git-->>GM: List of all tags
        GM->>GM: Filter by regex pattern
        GM->>DB: SELECT FROM processed_tags
        DB-->>GM: Already processed tags
        GM->>GM: Compare and find new
        
        alt New tag found
            GM-->>Daemon: Return: MAC_1_V24_02_15_01
            Daemon->>GM: checkout_tag(MAC_1_V24_02_15_01)
            GM->>FS: mkdir -p /home/YOUR_USER/compile
            GM->>Git: git clone/fetch + checkout TAG
            Git-->>GM: Files ready
            GM->>DB: INSERT INTO processed_tags
            GM-->>Daemon: Checkout OK
        else No new tags
            GM-->>Daemon: Return: empty
        end
    end
```

---

## Funciones Principales

### 1. `detect_new_tags()`

**Propósito**: Detecta tags nuevos que no han sido procesados.

**Algoritmo**:
```bash
detect_new_tags() {
    local GIT_URL=$(config_get "git.url")
    local TAG_PATTERN=$(config_get "git.tag_pattern")
    
    # 1. Obtener todos los tags remotos
    log_info "Fetching tags from remote repository..."
    local remote_tags=$(git ls-remote --tags "$GIT_URL" | awk '{print $2}' | sed 's|refs/tags/||' | grep -E "$TAG_PATTERN")
    
    # 2. Obtener tags ya procesados de DB
    local processed_tags=$(db_query "SELECT tag_name FROM processed_tags")
    
    # 3. Comparar y encontrar nuevos
    local new_tags=""
    for tag in $remote_tags; do
        if ! echo "$processed_tags" | grep -q "^$tag$"; then
            new_tags="$new_tags $tag"
        fi
    done
    
    # 4. Retornar nuevos tags (stdout)
    echo "$new_tags" | xargs
}
```

**Output**: Lista de tags nuevos separados por espacio (o vacío si no hay nuevos)

**Ejemplo**:
```bash
NEW_TAGS=$(./scripts/git_monitor.sh detect)
# Output: "MAC_1_V24_02_15_01 MAC_1_V24_02_15_02"
```

### 2. `checkout_tag()`

**Propósito**: Clona repositorio y hace checkout de tag específico.

**Algoritmo**:
```bash
checkout_tag() {
    local TAG_NAME="$1"
    local GIT_URL=$(config_get "git.url")
    local BRANCH=$(config_get "git.branch")
    local COMPILE_DIR=$(config_get "compilation.compile_dir")
    
    log_info "Checking out tag: $TAG_NAME"
    
    # 1. Preparar directorio
    mkdir -p "$COMPILE_DIR"
    
    # 2. Si ya existe, actualizar; si no, clonar
    if [ -d "$COMPILE_DIR/.git" ]; then
        log_info "Updating existing repository..."
        cd "$COMPILE_DIR"
        git fetch --all --tags
        git checkout "$TAG_NAME"
    else
        log_info "Cloning repository..."
        git clone --branch "$BRANCH" "$GIT_URL" "$COMPILE_DIR"
        cd "$COMPILE_DIR"
        git checkout "$TAG_NAME"
    fi
    
    # 3. Verificar checkout exitoso
    local current_tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
    if [ "$current_tag" != "$TAG_NAME" ]; then
        log_error "Failed to checkout tag $TAG_NAME"
        return 1
    fi
    
    log_ok "Successfully checked out tag: $TAG_NAME"
    return 0
}
```

**Validaciones**:
- Directorio de compilación existe y es escribible
- Git puede conectar al repositorio (credenciales válidas)
- Tag existe en el remoto
- Checkout exitoso (verificado con `git describe`)

### 3. `mark_tag_processed()`

**Propósito**: Registra tag en DB para evitar reprocesamiento.

**Implementación**:
```bash
mark_tag_processed() {
    local TAG_NAME="$1"
    
    db_query "INSERT INTO processed_tags (tag_name, detected_at) VALUES ('$TAG_NAME', CURRENT_TIMESTAMP)"
    
    if [ $? -eq 0 ]; then
        log_ok "Tag $TAG_NAME marked as processed"
    else
        log_error "Failed to mark tag $TAG_NAME as processed"
        return 1
    fi
}
```

**Nota**: Se marca al INICIO del procesamiento para evitar reprocesar si el pipeline falla a mitad.

---

## Configuración

### YAML (`config/ci_cd_config.yaml`)

```yaml
git:
  url: "https://${GIT_USER}:${GIT_PASSWORD}@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO"
  branch: "YOUR_GIT_BRANCH"
  tag_pattern: "^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$"
  
compilation:
  compile_dir: "/home/YOUR_USER/compile"
```

**Ver detalles**: [[Referencia - Configuración#Git]]

### Variables de Entorno (`.env`)

```bash
GIT_USER=automation
GIT_PASSWORD=ghp_xxxxxxxxxxxxxxxxxxxx
```

**⚠️ Seguridad**: Nunca commitear `.env`. Usar `.env.example` como template.

---

## Validación de Tags

### Regex Pattern

**Pattern**: `^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$`

**Formatos válidos**:
- `MAC_1_V24_02_15_01` - Tag oficial con prefijo MAC
- `MAC_12_V24_03_20_02` - MAC con múltiples dígitos
- `V24_02_15_01` - Tag interno sin prefijo

**Formatos inválidos**:
- `v24_02_15_01` - Minúscula (regex es case-sensitive)
- `MAC_V24_02_15_01` - Falta número después de MAC_
- `V2024_02_15_01` - Año con 4 dígitos
- `MAC_1_V24_2_15_1` - Dígitos sin padding de ceros

**Componentes**:
- `MAC_[0-9]+_` - Prefijo opcional con número de MAC
- `V[0-9]{2}` - Año (2 dígitos)
- `_[0-9]{2}` - Mes
- `_[0-9]{2}` - Día
- `_[0-9]{2}` - Número de versión del día

### Validación en Script

```bash
validate_tag_format() {
    local TAG_NAME="$1"
    local TAG_PATTERN=$(config_get "git.tag_pattern")
    
    if echo "$TAG_NAME" | grep -E "$TAG_PATTERN" > /dev/null; then
        log_ok "Tag format valid: $TAG_NAME"
        return 0
    else
        log_error "Invalid tag format: $TAG_NAME (expected pattern: $TAG_PATTERN)"
        return 1
    fi
}
```

---

## Flujo de Datos

```mermaid
flowchart TD
    A[Daemon Loop] -->|Every 5 min| B[detect_new_tags]
    B --> C{git ls-remote}
    C -->|All tags| D[Filter by regex]
    D -->|Valid tags| E[Query processed_tags DB]
    E -->|Already processed| F{New tags?}
    F -->|No| G[Sleep 5 min]
    G --> A
    F -->|Yes| H[For each new tag]
    H --> I[checkout_tag]
    I --> J{Checkout OK?}
    J -->|No| K[Log error]
    K --> G
    J -->|Yes| L[mark_tag_processed]
    L --> M[Start Pipeline]
    M --> N[Compilation Phase]
    
    style B fill:#4ECDC4
    style I fill:#95E1D3
    style M fill:#FF6B6B,color:#fff
```

**Ver pipeline completo**: [[Diagrama - Flujo Completo]]

---

## Interacción con Base de Datos

### Tabla `processed_tags`

**Ver schema**: [[Modelo de Datos#processed_tags]]

**Queries ejecutadas**:

```sql
-- 1. Obtener tags ya procesados
SELECT tag_name FROM processed_tags;

-- 2. Marcar nuevo tag como detectado
INSERT INTO processed_tags (tag_name, detected_at)
VALUES ('MAC_1_V24_02_15_01', CURRENT_TIMESTAMP);

-- 3. Actualizar timestamp de procesamiento completo
UPDATE processed_tags
SET processed_at = CURRENT_TIMESTAMP
WHERE tag_name = 'MAC_1_V24_02_15_01';

-- 4. Verificar si tag ya fue procesado (before processing)
SELECT COUNT(*) FROM processed_tags
WHERE tag_name = 'MAC_1_V24_02_15_01';
```

---

## Logging

### Mensajes de Log

**Formato estándar** (via [[Pipeline - Common Functions]]):
```
[2026-03-20 10:05:33] [INFO] Git monitor: Checking for new tags...
[2026-03-20 10:05:34] [INFO] Found 2 remote tags matching pattern
[2026-03-20 10:05:34] [OK] Found 1 new tag: MAC_1_V24_02_15_01
[2026-03-20 10:05:35] [INFO] Checking out tag: MAC_1_V24_02_15_01
[2026-03-20 10:06:02] [OK] Successfully checked out tag: MAC_1_V24_02_15_01
[2026-03-20 10:06:03] [OK] Tag MAC_1_V24_02_15_01 marked as processed
```

**Ubicación**: `logs/pipeline_YYYYMMDD.log`

**Ver sistema de logs**: [[Referencia - Logs]]

---

## Gestión de Errores

### Errores Comunes

#### 1. **Fallo de Autenticación**

**Síntoma**:
```
[ERROR] Git monitor: Failed to fetch tags from remote
fatal: Authentication failed for 'https://YOUR_GIT_SERVER/...'
```

**Causa**: Credenciales inválidas o expiradas

**Solución**:
```bash
# Verificar credenciales en .env
cat config/.env | grep GIT_

# Probar conexión manualmente
git ls-remote --tags https://$GIT_USER:$GIT_PASSWORD@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO

# Regenerar token si es necesario (en GitLab/GitHub)
```

**Ver troubleshooting**: [[Operación - Troubleshooting#Git Monitor]]

#### 2. **Tag No Existe**

**Síntoma**:
```
[ERROR] Failed to checkout tag MAC_1_V24_02_15_99
fatal: reference is not a tree: MAC_1_V24_02_15_99
```

**Causa**: Tag fue eliminado del remoto o nombre incorrecto

**Solución**:
```bash
# Listar tags remotos
git ls-remote --tags https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO | grep V24_02

# Verificar formato
./scripts/git_monitor.sh detect
```

#### 3. **Directorio de Compilación No Escribible**

**Síntoma**:
```
[ERROR] Cannot create directory /home/YOUR_USER/compile
mkdir: cannot create directory '/home/YOUR_USER/compile': Permission denied
```

**Causa**: Permisos insuficientes

**Solución**:
```bash
# Verificar permisos
ls -ld /home/YOUR_USER/

# Crear directorio con permisos correctos
sudo mkdir -p /home/YOUR_USER/compile
sudo chown agent:agent /home/YOUR_USER/compile
chmod 755 /home/YOUR_USER/compile
```

#### 4. **Base de Datos Bloqueada**

**Síntoma**:
```
[ERROR] Failed to mark tag as processed
Error: database is locked
```

**Causa**: Escritura concurrente sin WAL mode

**Solución**:
```bash
# Activar WAL mode
sqlite3 db/pipeline.db "PRAGMA journal_mode = WAL;"

# Verificar
sqlite3 db/pipeline.db "PRAGMA journal_mode;"
```

**Ver DB troubleshooting**: [[Operación - Troubleshooting#Base de Datos]]

---

## Manejo de Edge Cases

### Tag Duplicado en Remoto

**Problema**: Desarrollador crea tag, lo borra y lo recrea con mismo nombre.

**Comportamiento**:
- Git monitor detecta el "nuevo" tag (mismo nombre)
- Lookup en `processed_tags` encuentra el tag
- **Skip** (no reprocesa)

**Override manual** (si es necesario):
```bash
# Eliminar de processed_tags para forzar reprocesamiento
sqlite3 db/pipeline.db "DELETE FROM processed_tags WHERE tag_name='MAC_1_V24_02_15_01'"

# Ejecutar manualmente
./ci_cd.sh --tag MAC_1_V24_02_15_01
```

### Múltiples Tags Nuevos Simultáneos

**Problema**: Desarrollador crea 5 tags de golpe.

**Comportamiento**:
- `detect_new_tags()` retorna: `"TAG1 TAG2 TAG3 TAG4 TAG5"`
- Daemon procesa **uno por uno** secuencialmente
- Si uno falla, los siguientes se procesan igual

**Código en `ci_cd.sh`**:
```bash
NEW_TAGS=$(./scripts/git_monitor.sh detect)
for tag in $NEW_TAGS; do
    log_info "Processing tag: $tag"
    ./scripts/git_monitor.sh checkout "$tag"
    run_pipeline "$tag"  # Bloquea hasta completar
done
```

### Rollback de Tag

**Problema**: Tag desplegado tiene bug crítico, necesita rollback.

**Solución manual**:
```bash
# 1. Identificar tag anterior exitoso
sqlite3 db/pipeline.db "SELECT tag_name FROM deployments WHERE status='success' ORDER BY started_at DESC LIMIT 5"

# 2. Desplegar tag anterior manualmente
./ci_cd.sh --tag MAC_1_V24_02_14_02

# 3. Opcional: marcar tag problemático como failed
sqlite3 db/pipeline.db "UPDATE deployments SET status='failed', error_message='Manual rollback' WHERE tag_name='MAC_1_V24_02_15_01'"
```

---

## Performance y Optimización

### Estrategias Implementadas

1. **Polling vs Webhooks**
   - **Actual**: Polling cada 5 minutos
   - **Ventaja**: Simple, no requiere endpoint público
   - **Desventaja**: Latencia de hasta 5 minutos

2. **Shallow Clone**
   - **No implementado**: Clone completo del repo
   - **Mejora potencial**:
     ```bash
     git clone --depth 1 --branch "$TAG_NAME" "$GIT_URL" "$COMPILE_DIR"
     ```
   - **Ganancia**: ~70% menos tiempo de clone (si repo es grande)

3. **Cache de Clones**
   - **Implementado**: Usa `git fetch` si repo ya existe
   - **Ganancia**: Ahorro de ~2-3 minutos en clones subsecuentes

### Métricas Típicas

| Operación | Tiempo Promedio | Notas |
|-----------|-----------------|-------|
| `git ls-remote` | 2-5 segundos | Depende de latencia red |
| `git clone` (primera vez) | 3-5 minutos | Repo ~500 MB |
| `git fetch` (subsecuente) | 30-60 segundos | Solo delta |
| `git checkout TAG` | 5-10 segundos | Cambio de archivos |
| Query `processed_tags` | <100ms | DB local, indexed |
| Insert `processed_tags` | <50ms | DB local |

**Total Fase 1**: ~5-7 minutos (primera vez), ~1-2 minutos (subsecuente)

---

## Testing Manual

### Test de Detección

```bash
cd /home/YOUR_USER/cicd

# Test: detectar tags nuevos
./scripts/git_monitor.sh detect

# Output esperado (si hay nuevos):
# MAC_1_V24_02_15_01 MAC_1_V24_02_15_02

# Output esperado (si no hay nuevos):
# (vacío)
```

### Test de Checkout

```bash
# Test: checkout de tag específico
./scripts/git_monitor.sh checkout MAC_1_V24_02_15_01

# Verificar resultado
cd /home/YOUR_USER/compile
git describe --tags --exact-match
# Debe retornar: MAC_1_V24_02_15_01
```

### Test de Validación de Formato

```bash
# Tags válidos
validate_tag_format "MAC_1_V24_02_15_01"  # ✓
validate_tag_format "V24_02_15_01"        # ✓

# Tags inválidos
validate_tag_format "v24_02_15_01"        # ✗ (minúscula)
validate_tag_format "MAC_V24_02_15_01"    # ✗ (falta número)
validate_tag_format "V2024_02_15_01"      # ✗ (año 4 dígitos)
```

### Test de DB Integration

```bash
# Verificar tabla processed_tags
sqlite3 db/pipeline.db "SELECT * FROM processed_tags ORDER BY detected_at DESC LIMIT 10"

# Forzar marcado manual
sqlite3 db/pipeline.db "INSERT INTO processed_tags (tag_name) VALUES ('TEST_TAG_V99_99_99_99')"

# Limpiar test tag
sqlite3 db/pipeline.db "DELETE FROM processed_tags WHERE tag_name LIKE 'TEST_%'"
```

---

## Integración con CI/CD Daemon

### Invocación desde `ci_cd.sh`

**Código simplificado**:
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

daemon_mode() {
    log_info "Starting CI/CD daemon (polling every 5 minutes)..."
    
    while true; do
        # Fase 1: Git Monitor
        log_info "Checking for new tags..."
        NEW_TAGS=$("$SCRIPT_DIR/scripts/git_monitor.sh" detect)
        
        if [ -n "$NEW_TAGS" ]; then
            log_ok "Found new tags: $NEW_TAGS"
            
            for tag in $NEW_TAGS; do
                log_info "Processing tag: $tag"
                
                # Checkout
                if ! "$SCRIPT_DIR/scripts/git_monitor.sh" checkout "$tag"; then
                    log_error "Failed to checkout tag: $tag"
                    continue
                fi
                
                # Marcar como procesado
                db_query "INSERT INTO processed_tags (tag_name) VALUES ('$tag')"
                
                # Ejecutar pipeline completo (fases 2-5)
                run_pipeline "$tag"
            done
        else
            log_info "No new tags detected"
        fi
        
        # Sleep 5 minutos
        log_info "Sleeping for 5 minutes..."
        sleep 300
    done
}
```

**Ver orquestador completo**: [[Arquitectura del Pipeline#Orquestador Principal]]

---

## Monitorización

### Verificar Funcionamiento

```bash
# Ver logs de Git Monitor en tiempo real
tail -f logs/pipeline_$(date +%Y%m%d).log | grep "Git monitor"

# Ver tags procesados recientemente
sqlite3 db/pipeline.db "
SELECT tag_name, detected_at, processed_at
FROM processed_tags
ORDER BY detected_at DESC
LIMIT 10
"

# Verificar conectividad Git
timeout 10 git ls-remote --tags https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO | head -5
```

### Métricas en Web UI

**Dashboard muestra**:
- Último tag procesado
- Número de tags procesados hoy
- Tasa de éxito de checkouts

**Ver**: http://YOUR_PIPELINE_HOST_IP:8080

**API endpoint**:
```bash
curl http://YOUR_PIPELINE_HOST_IP:8080/api/processed-tags | jq
```

---

## Extensiones Futuras

### 1. Webhook Integration

**Concepto**: Recibir notificaciones push de GitLab/GitHub en lugar de polling.

**Ventajas**:
- Latencia < 1 segundo (vs 5 minutos)
- Menos carga en servidor Git

**Implementación básica**:
```bash
# Endpoint webhook (nginx reverse proxy)
location /webhook/git {
    proxy_pass http://localhost:9000;
}

# Listener Python
from flask import Flask, request
app = Flask(__name__)

@app.route('/webhook/git', methods=['POST'])
def git_webhook():
    data = request.json
    tag_name = data.get('ref', '').replace('refs/tags/', '')
    
    # Trigger pipeline
    subprocess.run(['./ci_cd.sh', '--tag', tag_name])
    
    return 'OK', 200
```

### 2. Múltiples Repositorios

**Concepto**: Monitorizar varios repos simultáneamente.

**Config YAML**:
```yaml
git:
  repositories:
    - name: GALTTCMC
      url: "https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO"
      branch: YOUR_GIT_BRANCH
      tag_pattern: "^MAC_.*"
    - name: GALTTCMC_TOOLS
      url: "https://YOUR_GIT_SERVER/YOUR_ORG/Tools"
      branch: main
      tag_pattern: "^v[0-9]+\\.[0-9]+\\.[0-9]+$"
```

### 3. Tag Filtering Avanzado

**Concepto**: Filtros adicionales más allá de regex (ej: solo tags con annotations).

**Ejemplo**:
```bash
# Solo tags annotated (no lightweight)
git ls-remote --tags "$GIT_URL" | grep '\^{}' | sed 's/\^{}//'

# Solo tags de cierta fecha en adelante
git for-each-ref --sort=creatordate --format '%(refname:short) %(creatordate:short)' refs/tags | awk '$2 >= "2026-03-01"'
```

---

## Enlaces Relacionados

### Documentación del Pipeline
- [[Arquitectura del Pipeline]] - Contexto general
- [[Pipeline - Compilación]] - Siguiente fase (Fase 2)
- [[Pipeline - Common Functions]] - Funciones compartidas
- [[Diagrama - Flujo Completo]] - Visualización completa

### Configuración y Datos
- [[Referencia - Configuración#Git]] - Opciones de configuración
- [[Modelo de Datos#processed_tags]] - Tabla de tags procesados
- [[Referencia - APIs Externas#Git]] - Integración con Git

### Operación
- [[Operación - Troubleshooting#Git Monitor]] - Problemas comunes
- [[Operación - Monitorización]] - Supervisión del sistema
- [[01 - Quick Start#Testing de Fases Individuales]] - Comandos rápidos
