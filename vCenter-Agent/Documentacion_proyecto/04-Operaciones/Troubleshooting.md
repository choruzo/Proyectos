---
tipo: operacional
versión: 1.0
tags: [troubleshooting, diagnostico, errores, debugging, soluciones]
última_actualización: 2026-03-24
relacionado:
  - "[[Guia-Implementacion]]"
  - "[[Structured-Logging]]"
  - "[[Connection-Pool]]"
  - "[[Sistema-RAG-v2]]"
---

# Troubleshooting — Solución de Problemas

Guía de diagnóstico y solución de problemas comunes del vCenter Multi-Agent System.

## Autenticación

### Sesión Expirada

**Síntoma:** Error 401 al hacer requests, redirige a login.

**Causas:**
- Sesión inactiva >3600s (ACTIVE_SESSIONS timeout)
- Cookie de sesión eliminada
- Servidor reiniciado (sesiones in-memory)

**Solución:**
```bash
# Re-login desde UI
http://localhost:9100/ → Login de nuevo

# O forzar logout + login via API
curl -X POST http://localhost:9100/logout
curl -X POST http://localhost:9100/login -d "username=admin&password=pass"
```

### IP Bloqueada (429 Too Many Requests)

**Síntoma:** Error 429 tras múltiples intentos de login fallidos.

**Causa:** Protección anti-brute-force (5 intentos en 5 min).

**Solución:**
```bash
# Esperar 5 minutos o reiniciar servidor
# O borrar bloqueo manualmente en logs/security/security.log (no recomendado)

# Verificar bloqueos activos:
Get-Content logs/security/security.log | Where-Object { $_ -match '"event":"login_blocked"' }
```

### Usuario No Existe

**Síntoma:** Error 401 "Invalid credentials" con usuario correcto.

**Causa:** Usuario no creado en `data/auth.db`.

**Solución:**
```python
# Crear usuario manualmente
python -c "
from src.auth import create_user
create_user('newuser', 'password123', abbr='NU', role='user')
"
```

## vCenter

### Connection Pool Lleno

**Síntoma:** "Connection pool exhausted" en logs/system/system.log.

**Causa:** 5 conexiones simultáneas activas (max del pool).

**Solución:**
```bash
# 1. Esperar 30s (timeout de conexiones inactivas)
# 2. Reiniciar servidor para limpiar pool

# Ver conexiones activas en logs:
Get-Content logs/business/business.log | Where-Object { $_ -match 'connection_pool' }
```

**Prevención:** Aumentar `_max_connections` en `VCenterConnectionPool` (requiere cambio en código).

### Tool Execution Failed

**Síntoma:** "Tool execution failed: [Errno 13003]" o "Vim.fault.NotAuthenticated".

**Causas:**
- Sesión vCenter expirada (>30 min inactiva)
- Credenciales vCenter incorrectas en config.json
- vCenter inaccesible (red, firewall)

**Diagnóstico:**
```bash
# 1. Ping a vCenter
ping 172.30.188.136

# 2. Curl al UI (debe devolver HTML)
curl -k https://172.30.188.136/ui

# 3. Verificar credenciales en config.json
cat config/config.json | grep vcenter_

# 4. Ver errores en logs
Get-Content logs/system/system.log -Tail 50
```

**Solución:**
```json
// Habilitar fallback a vcsim temporalmente
{
  "vcenter_fallback": {
    "enabled": true,
    "mode": "auto"
  }
}
```

### vCenter Simulator (vcsim) No Arranca

**Síntoma:** "vcsim container failed to start" en logs.

**Causa:** Docker no instalado/corriendo, puerto 8989 ocupado.

**Solución:**
```powershell
# Verificar Docker
docker ps

# Verificar puerto
netstat -ano | findstr :8989

# Reiniciar vcsim manualmente
docker stop vcenter_agent_vcsim
docker rm vcenter_agent_vcsim
docker run -d --name vcenter_agent_vcsim -p 8989:8989 vmware/vcsim
```

## RAG v2.4 (Documentación)

### Sin Resultados en Búsqueda

**Síntoma:** Agente documentación responde "No encontré información relevante".

**Causas:**
- ChromaDB no inicializado (`data/chroma_db/` vacío)
- Documentos no indexados
- Query demasiado específica (no hay match)

**Diagnóstico:**
```bash
# Verificar índice ChromaDB
ls data/chroma_db/
# Debe contener: chroma.sqlite3, archivos de índice

# Ver métricas RAG en logs
Get-Content logs/retrieval_metrics.jsonl -Tail 10
```

**Solución:**
```json
// Forzar reindexación en config.json
{
  "rag_v2": {
    "vector_store": {
      "force_rebuild": true
    }
  }
}
// Reiniciar servidor → reindexará docs/
```

### Respuestas Genéricas (No Usa Documentos)

**Síntoma:** Respuestas sin referencias a fuentes, no menciona documentos.

**Causa:** Documentos recuperados con score muy bajo (< threshold implícito).

**Solución:**
```json
// Bajar threshold en config.json (default 0.0 ya es mínimo)
// O mejorar query expansion agregando términos relevantes
// Ver src/utils/query_expander.py → TERM_FAMILIES
```

### Embedding Cache Miss Rate Alto

**Síntoma:** Métricas RAG muestran cache_hit_rate < 0.3.

**Causa:** Queries siempre distintas (no hay repeticiones).

**Solución:** Normal si queries varían mucho. Cache optimiza solo repeticiones. Aumentar `cache_max_size` si hay muchas queries similares:

```json
{
  "rag_v2": {
    "parameters": {
      "cache_max_size": 2000  // Default: 1000
    }
  }
}
```

## Ollama / LLM

### Ollama No Responde

**Síntoma:** "Connection refused" al llamar Ollama API.

**Causa:** Servicio Ollama no corriendo.

**Solución:**
```powershell
# Windows: Verificar servicio
Get-Service -Name ollama
Start-Service ollama

# Linux: Verificar systemd
systemctl status ollama
systemctl start ollama

# Verificar modelos descargados
ollama list
# Debe mostrar: gpt-oss:20b, nomic-embed-text
```

### Modelo No Encontrado

**Síntoma:** "Model 'gpt-oss:20b' not found".

**Solución:**
```bash
# Descargar modelo
ollama pull gpt-oss:20b

# Verificar
ollama list
```

### Respuestas Muy Lentas (>30s)

**Síntoma:** Timeouts, UI se congela.

**Causas:**
- Modelo muy grande para hardware (gpt-oss:20b requiere 16GB+ RAM)
- num_ctx muy alto (16384 tokens)
- CPU/GPU insuficiente

**Solución:**
```powershell
# Usar modelo más pequeño
$env:ORCH_EXECUTOR_MODEL = "llama3.1:8b"
python run.py

# O reducir num_ctx en config LLM
# Ver src/api/main_agent.py → executor_llm initialization
```

## Flask / API

### Port Already in Use

**Síntoma:** "Address already in use: 0.0.0.0:9100".

**Solución:**
```powershell
# Windows: Matar proceso en puerto 9100
netstat -ano | findstr :9100
taskkill /PID <pid> /F

# Linux
lsof -i :9100
kill -9 <pid>

# O cambiar puerto
$env:FLASK_PORT = "8080"
python run.py
```

### CORS Errors (Cross-Origin)

**Síntoma:** "No 'Access-Control-Allow-Origin' header".

**Causa:** Frontend en origen distinto (ej. localhost:3000 → API localhost:9100).

**Solución:** Habilitar CORS en `src/api/main_agent.py`:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
```

### SSE Stream Corta Inesperadamente

**Síntoma:** Evento `done` nunca llega, stream se cierra a mitad.

**Causas:**
- Timeout de proxy/nginx intermedio
- Error no capturado en agente

**Diagnóstico:**
```bash
# Ver último error en logs
Get-Content logs/system/system.log -Tail 50
Get-Content logs/api/api.log -Tail 50
```

**Solución:** Aumentar timeout en servidor web si hay proxy:
```nginx
# nginx.conf
proxy_read_timeout 300s;
proxy_send_timeout 300s;
```

## Sistema

### Error "Module not found"

**Síntoma:** `ModuleNotFoundError: No module named 'langchain'`.

**Causa:** Dependencias no instaladas o venv no activado.

**Solución:**
```bash
# Activar venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux

# Reinstalar dependencias
pip install --force-reinstall -r requirements_oficial.txt
```

### ChromaDB/SQLite Corrupted

**Síntoma:** "database disk image is malformed".

**Solución:**
```bash
# Backup + eliminar bases corruptas
cp -r data/ data_backup/
rm -rf data/chroma_db/
rm data/users.db data/auth.db

# Reiniciar → reconstruye automáticamente
python run.py
```

### Disco Lleno (Logs Gigantes)

**Síntoma:** Aplicación lenta, "No space left on device".

**Causa:** Logs sin rotación (aunque RotatingFileHandler debería evitarlo).

**Solución:**
```bash
# Limpiar logs antiguos
rm logs/*/*.log.1
rm logs/*/*.log.2

# Verificar configuración de rotación en config/logging_config.json
# maxBytes: 10485760 (10MB)
# backupCount: 5
```

## Performance

### Alta Latencia en Queries

**Síntoma:** Respuestas tardan >10s constantemente.

**Diagnóstico:**
```bash
# Ver métricas de performance
Get-Content logs/performance/performance.log | ConvertFrom-Json | 
  Where-Object { $_.performance.duration_ms -gt 10000 }

# Ver métricas RAG
Get-Content logs/retrieval_metrics.jsonl -Tail 20 | ConvertFrom-Json |
  Select-Object query_text, retrieval_time_ms
```

**Soluciones:**
- **RAG lento:** Reducir `initial_k` (40 → 20) en config.json
- **LLM lento:** Usar modelo más pequeño o aumentar num_ctx gradualmente
- **Connection pool:** Ver logs si hay wait times altos

### Alto Uso de Memoria

**Síntoma:** >8GB RAM usados, sistema se ralentiza.

**Causas:**
- Modelo LLM muy grande (gpt-oss:20b usa 16GB+)
- ChromaDB + embedding cache grande

**Solución:**
```powershell
# Reducir cache size
// config.json
{"rag_v2": {"parameters": {"cache_max_size": 500}}}

# Usar modelo más pequeño
$env:ORCH_EXECUTOR_MODEL = "llama3.1:8b"
```

## Logs y Diagnóstico

### Ver Logs en Tiempo Real

```powershell
# API requests
Get-Content logs/api/api.log -Wait -Tail 20

# vCenter operations
Get-Content logs/business/business.log -Wait -Tail 20

# Errores del sistema
Get-Content logs/system/system.log -Wait -Tail 20

# Seguridad (login, bloqueos)
Get-Content logs/security/security.log -Wait -Tail 20

# RAG métricas
Get-Content logs/retrieval_metrics.jsonl -Wait -Tail 5
```

### Buscar Errores Específicos

```powershell
# Errores de conexión vCenter
Get-Content logs/system/system.log | Where-Object { $_ -match 'connection.*failed' }

# Queries lentas (>5s)
Get-Content logs/performance/performance.log | ConvertFrom-Json |
  Where-Object { $_.performance.duration_ms -gt 5000 }

# Usuarios bloqueados
Get-Content logs/security/security.log | Where-Object { $_ -match 'login_blocked' }
```

Ver [[Structured-Logging]] para formato de logs y categorías.

## Enlaces Relacionados

- [[Guia-Implementacion]] — Instalación y configuración inicial
- [[Structured-Logging]] — Sistema de logging estructurado
- [[Connection-Pool]] — Gestión de conexiones vCenter
- [[Sistema-RAG-v2]] — Pipeline RAG y configuración
- [[Configuracion]] — Referencia de config.json

***

**Versión del documento:** 1.0  
**Fuentes originales:**  
- `vcenter_agent_system/DOCS_proyect/Chat/GUIA_FUNCIONAMIENTO.md` (sección troubleshooting)  
- `vcenter_agent_system/DOCS_proyect/vCenter_Agent/TROUBLESHOOTING.md`

