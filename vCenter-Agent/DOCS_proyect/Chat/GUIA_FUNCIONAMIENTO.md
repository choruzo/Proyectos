# Guía de Funcionamiento - Chat del Orquestador

**Versión:** 1.0  
**Última actualización:** Enero 2026  
**Categoría:** Guía de Usuario y Operador

---

## 🎯 Objetivo

Esta guía describe **cómo funciona el chat** desde la perspectiva del usuario y del operador del sistema. Cubre casos de uso, limitaciones, troubleshooting y best practices.

---

## 📱 Interfaz de Usuario

### Vista General

```
┌─────────────────────────────────────────────────────┐
│  Orquestador de Agentes        [☀️]                 │
│  Canal unificado para vCenter y futuros agentes.    │
├─────────────────────────────────────────────────────┤
│  💡 Consejo: Incluye palabras clave ("VM",          │
│     "datastore", "snapshot") para enrutar           │
│     automáticamente al agente de vCenter.           │
├─────────────────────────────────────────────────────┤
│  Usuario: [jmartinb          ]                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐   │
│  │ 🤖 Orquestador listo. Pregunta algo sobre   │   │
│  │    vCenter o realiza una consulta general.  │   │
│  │                        [14:32]              │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 👤 ¿Cuántas VMs hay en producción?          │   │
│  │                        [14:33] · vcenter    │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 🤖 Hay 12 máquinas virtuales activas...     │   │
│  │                        [14:33] · vcenter    │   │
│  └─────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  [Escribe tu mensaje. Enter = enviar..]             │
│  [Shift+Enter = nueva línea]                        │
│                                        [Enviar ➤]   │
└─────────────────────────────────────────────────────┘
```

### Componentes de la UI

#### 1. **Campo de Mensaje** (textarea)
- **Funcionalidad**: Captura entrada del usuario
- **Atajo Enter**: Envía mensaje
- **Atajo Shift+Enter**: Nueva línea sin enviar
- **Placeholder**: "Escribe tu mensaje. Enter = enviar..."

#### 2. **Log de Chat** (div#log)
- **Visualización**: Historial conversacional
- **Auto-scroll**: Se desplaza al nuevo mensaje
- **Atributos ARIA**: Accesible para lectores de pantalla

#### 3. **Botón Toggle Tema**
- **Ubicación**: Esquina superior derecha (☀️/🌙)
- **Temas**: Dark (default) / Light
- **Persistencia**: Guarda en `localStorage.orch_public_theme`

#### 4. **Badge Agente** (last-agent)
- **Información**: Muestra el último agente que respondió
- **Valores**: vcenter / documentation / general
- **Actualización**: Se actualiza con cada respuesta

---

## 🤖 Comportamiento del Sistema

### Flujo de Entrada

```
Usuario escribe mensaje
    ↓
JavaScript valida (no vacío)
    ↓
POST /chat con {username, message}
    ↓
Flask valida sesión
    ↓
Extrae y clasifica mensaje
    ↓
Formatea (opcional)
    ↓
Enruta a agente apropiado
    ↓
Agente procesa
    ↓
Retorna respuesta
    ↓
JavaScript renderiza
    ↓
Usuario ve respuesta en chat
```

---

## 📊 Casos de Uso

### Caso 1: Consulta sobre VMs en vCenter

**Entrada del usuario**:
```
"¿Cuántas VMs hay en el cluster de producción?"
```

**Proceso**:
1. Sistema detecta keywords: "VMs", "cluster", "producción"
2. Clasifica como: **vcenter**
3. Formatea mensaje (si está habilitado)
4. Invoca Agent vCenter
5. Agent consulta pyvmomi
6. Retorna lista de VMs

**Respuesta esperada**:
```
Hay 12 máquinas virtuales activas en el clúster de producción:
- vm-prod-01 (4 vCPU, 16GB RAM)
- vm-prod-02 (8 vCPU, 32GB RAM)
...
```

**Badge mostrado**: `vcenter`

---

### Caso 2: Consulta sobre Documentación

**Entrada del usuario**:
```
"¿Cómo instalo el DNS según la documentación?"
```

**Proceso**:
1. Sistema detecta keywords: "instalar", "DNS", "documentación"
2. Clasifica como: **documentation**
3. Invoca Agent Documentación
4. Agent busca en archivos .docx con Whoosh
5. Retorna procedimiento

**Respuesta esperada**:
```
Según la documentación (Configuracion_templates.md):

1. Acceder a la consola del host ESXi
2. Editar /etc/resolv.conf
3. Agregar nameserver 10.0.0.53
4. Reiniciar el servicio de red
...
```

**Badge mostrado**: `documentation`

---

### Caso 3: Consulta General (Sin Clasificación)

**Entrada del usuario**:
```
"¿Qué es un snapshot?"
```

**Proceso**:
1. No coincide con keywords documentales o vCenter
2. Invoca LLM para clasificación fallback
3. LLM decide: **general**
4. Invoca Agent General (executor_llm)
5. Genera respuesta informativa

**Respuesta esperada**:
```
Un snapshot es una fotografía puntual del estado de una máquina virtual 
en un momento específico, incluyendo:
- Estado de la memoria
- Discos virtuales
- Configuración de red

Los snapshots pueden usarse para recuperación rápida ante fallos.
```

**Badge mostrado**: `general`

---

## 🔍 Palabras Clave de Enrutamiento

### Keywords de vCenter

```
vm, vcenter, host, datastore, template, snapshot,
clonar, desplegar, power, encender, apagar,
cpu, memoria, almacenamiento, performance,
metrics, monitoreo, cluster, esxi, vmware
```

### Keywords de Documentación

```
instalar, configurar, documentación, manual,
guía, procedimiento, paso a paso, cómo,
template, dns, network, usuario, permiso,
archivo, doc, readme, setup, installation
```

**Nota**: Si el mensaje contiene keywords de ambas categorías, **documentación tiene prioridad**.

---

## ⚙️ Operación del Sistema

### Inicio del Servidor

```bash
cd vcenter_agent_system
python run.py
```

**Salida esperada**:
```
Starting vCenter Agent System...
Running from: d:\Archivos\Javier\Scritp_python\Agente\vcenter_agent_system

📊 Iniciando recolección de datos históricos...
✓ Recolección de datos históricos iniciada correctamente

🚀 Iniciando servidor Flask en http://0.0.0.0:5000
```

### Acceso a la Interfaz

```
URL: http://localhost:5000/ui/chat
```

**Requisitos previos**:
1. Estar autenticado (sesión válida)
2. Variables de entorno configuradas
3. Modelos LLM (Qwen, Llama) disponibles localmente

### Variables de Entorno Críticas

```bash
# Modelos LLM
ORCH_FORMATTER_MODEL=gpt-oss:20b
ORCH_EXECUTOR_MODEL=gpt-oss:20b
ENABLE_QUERY_FORMATTING=true
FORMATTER_TIMEOUT=5

# Seguridad
ORCH_SECRET=tu_secreto_aleatorio_16_hex
```

---

## 📊 Monitorización en Tiempo Real

### Observar Logs Estructurados

```bash
# Terminal 1: Logs de API
tail -f logs/api/api.log

# Terminal 2: Logs de Auditoría
tail -f logs/audit/audit.log

# Terminal 3: Logs de Performance
tail -f logs/performance/performance.log
```

### Ejemplo de Log de Operación Exitosa

```json
{
  "timestamp": "2026-01-15T14:32:45.890Z",
  "level": "INFO",
  "category": "API",
  "message": "[TIMING] Petición /chat RECIBIDA en servidor",
  "user": "jmartinb"
}

{
  "timestamp": "2026-01-15T14:32:46.013Z",
  "level": "INFO",
  "category": "API",
  "message": "[TIMING] Sesión validada",
  "duration_ms": 123,
  "user": "jmartinb"
}

{
  "timestamp": "2026-01-15T14:32:46.018Z",
  "level": "INFO",
  "category": "AUDIT",
  "message": "message_routing",
  "user": "jmartinb",
  "target": "vcenter",
  "message": "¿Cuántas VMs hay?"
}

{
  "timestamp": "2026-01-15T14:32:46.567Z",
  "level": "INFO",
  "category": "PERFORMANCE",
  "message": "query_formatting_completed",
  "duration_ms": 234,
  "metadata": {
    "original_length": 18,
    "formatted_length": 24,
    "model": "gpt-oss:20b"
  }
}
```

---

## 🐛 Troubleshooting

### Problema: "Sesión expirada"

**Síntoma**: 
```json
{"error": "Sesión expirada"}
```

**Causa**: La sesión ha superado el timeout de 3600 segundos (1 hora) o no existe.

**Solución**:
1. Actualizar la página (F5)
2. Si persiste, hacer login nuevamente
3. Verificar que SESSION_TIMEOUT no sea muy bajo en main_agent.py

---

### Problema: "Mensaje vacío"

**Síntoma**:
```json
{"error": "Mensaje vacío"}
```

**Causa**: El usuario intentó enviar un mensaje solo con espacios en blanco.

**Solución**:
1. Escribir un mensaje válido (mínimo 1 carácter)
2. Asegurar que no sea solo espacios

---

### Problema: "Error de red"

**Síntoma**: El navegador muestra "Error de red: Failed to fetch"

**Causa** (posibles):
1. Servidor Flask no está corriendo
2. URL incorrecta o puerto diferente
3. CORS bloqueado (improbable en setup local)
4. Timeout de conexión

**Solución**:
1. Verificar que `python run.py` está activo
2. Comprobar que la URL sea `http://localhost:5000`
3. Revisar logs del servidor
4. Aumentar timeout en orchestrator_chat.js si es necesario

---

### Problema: Respuestas lentas (> 5 segundos)

**Síntoma**: El chat tarda mucho en responder

**Causa** (posibles):
1. Modelos LLM lentos o sobrecargados
2. Formateo habilitado con timeout corto
3. Agent vCenter haciendo consultas pesadas a pyvmomi
4. Sistema con recursos limitados

**Solución**:
```bash
# 1. Deshabilitar formateo si no es crítico
export ENABLE_QUERY_FORMATTING=false

# 2. Aumentar timeout del formateador
export FORMATTER_TIMEOUT=10

# 3. Revisar performance logs
tail -f logs/performance/performance.log | grep query_formatting

# 4. Usar modelo más ligero
export ORCH_FORMATTER_MODEL=phi2:2.7b
```

---

### Problema: Modelo LLM no disponible

**Síntoma**:
```
Error: Could not connect to Ollama service
```

**Causa**: Ollama no está corriendo o modelo no está descargado

**Solución**:
```bash
# 1. Iniciar Ollama (en otra terminal)
ollama serve

# 2. En otra terminal, descargar modelos
ollama pull gpt-oss:20b
ollama pull gpt-oss:20b

# 3. Verificar modelos disponibles
ollama list

# 4. Reintentar acceso al chat
```

---

### Problema: Clasificación incorrecta

**Síntoma**: El mensaje se envía al agente equivocado

**Ejemplo**:
```
"¿Cómo optimizar el rendimiento de las VMs?"
→ Se clasifica como 'documentation' cuando debería ser 'general'
```

**Causa**: Palabras clave ambiguas o clasificación LLM inconsistente

**Solución**:
1. **Reformular el mensaje**: Ser más específico
   - ❌ "¿Cómo optimizar?"
   - ✅ "Dame metrics de performance de vm-prod-01"

2. **Agregar keywords explícitas**: 
   - Para vCenter: incluir "VM", "datastore", "snapshot"
   - Para Documentación: incluir "documentación", "guía"

3. **Debug en logs**:
   ```bash
   grep "classify_task" logs/api/api.log | tail -10
   ```

---

## 🚀 Performance y Optimización

### Métricas de Latencia Esperada

| Fase | Rango Normal | Máximo Aceptable |
|------|-------------|-----------------|
| Validación de sesión | 10-50ms | 100ms |
| Parsing de mensaje | 5-20ms | 50ms |
| Clasificación keywords | 5-10ms | 50ms |
| Formateo de query | 100-500ms | 1000ms |
| Procesamiento agente | 1000-5000ms | 10000ms |
| **Total** | **1.1-5.5s** | **11.2s** |

### Optimización

#### 1. Deshabilitar Formateo

```bash
export ENABLE_QUERY_FORMATTING=false
```

**Impacto**: -500ms en latencia promedio

#### 2. Usar Modelo Formateador Más Ligero

```bash
# Cambiar de qwen3 (1.7B) a phi2 (2.7B)
export ORCH_FORMATTER_MODEL=phi2:2.7b
```

#### 3. Aumentar Resources de Ollama

```bash
# Configurar GPU acceleration
export OLLAMA_CUDA_VISIBLE_DEVICES=0
```

#### 4. Cache de Respuestas

Modificar `format_user_query()` para cachear respuestas frecuentes:

```python
# Pseudo-código
query_cache = {}
if message in query_cache:
    return query_cache[message]
# ... procesar ...
query_cache[message] = formatted_result
```

---

## 📝 Best Practices para Usuarios

### ✅ DO's

1. **Sé específico**
   ```
   ✓ "Listar todas las VMs del cluster producción"
   ✗ "dame vm"
   ```

2. **Incluye contexto**
   ```
   ✓ "¿Cuál es el CPU de vm-prod-02?"
   ✗ "¿CPU?"
   ```

3. **Usa palabras clave**
   ```
   ✓ "Crear snapshot de vm-test-01"
   ✗ "Hacer foto de la VM"
   ```

4. **Un tema por mensaje**
   ```
   ✓ "Listar VMs en producción"
   ✗ "Listar VMs y también datastores y hosts"
   ```

### ❌ DON'Ts

1. **No esperes conversación multi-turno compleja**
   - Cada mensaje es independiente
   - Proporciona contexto completo siempre

2. **No uses comandos CLI directamente**
   ```
   ✗ "Get-VM | Where-Object { ... }"
   ✓ "Listar VMs filtradas por CPU > 4"
   ```

3. **No preguntes sobre datos sensibles sin permiso**
   ```
   ✗ "Dame contraseñas de vCenter"
   ✓ "¿Cuál es el estado de autenticación?"
   ```

4. **No esperes operaciones sin confirmación**
   ```
   ✗ "Borra todas las VMs de desarrollo"
   ✓ "¿Cuáles son las VMs de desarrollo?"
   ```

---

## 🔒 Consideraciones de Seguridad

### Control de Acceso

- Toda operación requiere autenticación (sesión válida)
- Username se obtiene de la sesión, no del request
- Auditoría registra usuario, mensaje y target

### Logging Sensible

```python
# Mensaje completo se trunca a 120 caracteres en auditoría
audit_logger.audit(
    f"message_routing", 
    user=username, 
    target=target, 
    message=message[:120]  # ← Protección
)
```

### Timeout de Sesión

- Sesiones expiran después de 1 hora de inactividad
- Se pueden ajustar via `SESSION_TIMEOUT` en main_agent.py

---

## 📚 Referencia Rápida de Endpoints

### Endpoint: POST /chat

```http
POST /chat HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
  "username": "jmartinb",
  "message": "¿Cuántas VMs hay?"
}
```

**Respuesta exitosa** (200 OK):
```json
{
  "response": "Hay 12 máquinas virtuales activas",
  "agent": "vcenter"
}
```

**Respuesta error de sesión** (401 Unauthorized):
```json
{
  "error": "Sesión expirada"
}
```

**Respuesta error de mensaje vacío** (400 Bad Request):
```json
{
  "error": "Mensaje vacío"
}
```

---

## 🎓 Ejemplos de Consultas Efectivas

### vCenter Queries

```
1. "¿Cuántas VMs hay en el clúster de producción?"
2. "Listar hosts ESXi y su estado"
3. "¿Cuál es el almacenamiento disponible en datastore-prod?"
4. "Crear un snapshot de vm-backup-01"
5. "¿Cuáles son los recursos de vm-web-server?"
6. "Obtener métricas de CPU para vm-prod-02"
7. "Clonar vm-template-linux a vm-nueva"
```

### Documentation Queries

```
1. "¿Cómo instalar DNS en ESXi?"
2. "Cuéntame el procedimiento de backup"
3. "¿Cuál es la guía de configuración de redes?"
4. "¿Dónde está la documentación del TrueNAS?"
5. "Necesito el procedimiento de disaster recovery"
6. "¿Cuáles son los pasos para actualizar vCenter?"
```

### General Queries

```
1. "¿Qué es un snapshot?"
2. "Explícame los tipos de almacenamiento"
3. "¿Cuál es la diferencia entre template y snapshot?"
4. "Cuéntame sobre clustering"
```

---

## 📞 Soporte y Escalación

### Niveles de Soporte

| Problema | Responsable | Acción |
|----------|------------|--------|
| Sesión expirada | Usuario | Hacer login nuevamente |
| Mensaje vacío | Usuario | Escribir mensaje válido |
| Clasificación incorrecta | Administrador | Revisar keywords en config |
| Formateo lento | Administrador | Ajustar timeout o deshabilitar |
| Model no disponible | DevOps | Verificar Ollama y modelos |
| vCenter no responde | Infrastructure | Verificar conexión vCenter |

### Contactos

- **Problemas de Chat**: jmartinb@domain.com
- **Problemas vCenter**: vcenter-admin@domain.com
- **Problemas Infraestructura**: devops@domain.com

---

## 📈 Métricas Clave a Monitorear

```bash
# 1. Tasa de error por endpoint
grep '"error"' logs/api/api.log | wc -l

# 2. Latencia promedio
grep "\[TIMING\]" logs/api/api.log | tail -100

# 3. Operaciones por usuario
grep "message_routing" logs/audit/audit.log | grep -c "jmartinb"

# 4. Agente más usado
grep "message_routing" logs/audit/audit.log | cut -d'"' -f4 | sort | uniq -c
```

---

---

## ⚡ Referencia Rápida

### Estado del sistema

| Componente | OK cuando... | Síntoma de fallo |
|------------|-------------|------------------|
| Flask Server | Responde en < 200ms | Timeout o "connection refused" |
| Sesión | No devuelve 401 | "Sesión expirada" al enviar |
| vCenter API | Badge `vcenter` responde | "Error consultando vCenter" |
| Ollama | Responde en < 5s | "Could not connect to Ollama" |
| Documentos indexados | Badge `documentation` responde | "No encontré información" |
| Logs | Archivos en `logs/` crecen | Sin actividad en `logs/api/api.log` |

### Troubleshooting en una línea

| Problema | Acción inmediata |
|----------|-----------------|
| "Sesión expirada" | F5 → login nuevamente |
| "Mensaje vacío" | Escribir al menos 1 carácter |
| Respuesta lenta (> 5s) | `export ENABLE_QUERY_FORMATTING=false` |
| Ollama no disponible | `ollama serve` + `ollama pull gpt-oss:20b` |
| Chat no carga | Verificar `http://localhost:5000` activo |
| Logs no aparecen | Verificar que `logs/` existe en el directorio
