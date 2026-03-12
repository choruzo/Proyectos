# Referencia de API - Endpoint /chat

**Versión:** 1.0  
**Última actualización:** Enero 2026  
**Tipo:** API Reference

---

## 📍 Endpoint Overview

```
POST /chat
```

**Propósito**: Procesar consultas de chat enrutándolas al agente apropiado.

**URL Base**: `http://localhost:5000`

**Path Completo**: `http://localhost:5000/chat`

---

## 🔐 Autenticación

**Tipo**: Session-based

**Requerimiento**: Debe estar autenticado y tener una sesión válida

```
Cookie: session=abc123def456...
```

**Validación**:
- Sesión debe existir en `ACTIVE_SESSIONS`
- Sesión debe estar dentro del timeout (3600 segundos)
- Si falla, retorna 401 Unauthorized

---

## 📤 Request

### Headers Requeridos

```http
POST /chat HTTP/1.1
Host: localhost:5000
Content-Type: application/json
```

### Body JSON

```json
{
  "username": "string",
  "message": "string"
}
```

#### Parámetros

| Parámetro | Tipo | Requerido | Descripción | Ejemplo |
|-----------|------|----------|------------|---------|
| `username` | string | ✓ | Identificador del usuario | "jmartinb" |
| `message` | string | ✓ | Consulta a procesar | "¿Cuántas VMs hay?" |

### Validaciones

```
1. username
   - No puede estar vacío
   - Máximo 256 caracteres
   
2. message
   - No puede estar vacío o solo espacios en blanco
   - Máximo 5000 caracteres (recomendado)
```

---

## 📥 Response

### Success Response (200 OK)

```json
{
  "response": "string",
  "agent": "string"
}
```

#### Estructura

| Campo | Tipo | Descripción |
|-------|------|------------|
| `response` | string | Respuesta generada por el agente |
| `agent` | string | Agente que procesó: "vcenter", "documentation", "general" |

### Ejemplos de Response Exitoso

#### Ejemplo 1: vCenter Query

**Request**:
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jmartinb",
    "message": "¿Cuántas VMs hay en producción?"
  }'
```

**Response** (200):
```json
{
  "response": "Hay 12 máquinas virtuales activas en el clúster de producción:\n- vm-prod-01: 4 vCPU, 16GB RAM\n- vm-prod-02: 8 vCPU, 32GB RAM\n...",
  "agent": "vcenter"
}
```

---

#### Ejemplo 2: Documentation Query

**Request**:
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jmartinb",
    "message": "¿Cómo instalar DNS?"
  }'
```

**Response** (200):
```json
{
  "response": "Según la documentación:\n\n1. Acceder a la consola ESXi\n2. Editar /etc/resolv.conf\n3. Agregar: nameserver 10.0.0.53\n...",
  "agent": "documentation"
}
```

---

#### Ejemplo 3: General Query

**Request**:
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jmartinb",
    "message": "¿Qué es un snapshot?"
  }'
```

**Response** (200):
```json
{
  "response": "Un snapshot es una fotografía puntual del estado de una máquina virtual. Captura:\n- Estado de memoria\n- Discos virtuales\n- Configuración de red\n...",
  "agent": "general"
}
```

---

### Error Responses

#### 401 Unauthorized - Sesión Expirada

**Causa**: No hay sesión válida o ha expirado

**Response**:
```json
{
  "error": "Sesión expirada"
}
```

**HTTP Status**: `401`

**Acciones recomendadas**:
1. Actualizar la página (F5)
2. Hacer login nuevamente
3. Verificar que las cookies estén habilitadas

---

#### 400 Bad Request - Mensaje Vacío

**Causa**: El campo `message` está vacío o solo contiene espacios

**Response**:
```json
{
  "error": "Mensaje vacío"
}
```

**HTTP Status**: `400`

**Ejemplo**:
```bash
# ❌ Request inválido
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jmartinb",
    "message": "   "
  }'
```

---

#### 400 Bad Request - JSON Malformado

**Causa**: El body JSON no es válido

**Response**:
```json
{
  "error": "Invalid JSON"
}
```

**HTTP Status**: `400`

**Ejemplo**:
```bash
# ❌ JSON inválido (falta comilla)
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jmartinb
    "message": "test"
  }'
```

---

#### 500 Internal Server Error - Error en Agente

**Causa**: Fallo en el procesamiento del agente

**Response**:
```json
{
  "error": "Error procesando petición en agente vCenter: [error details]"
}
```

**HTTP Status**: `500`

**Debugging**:
- Revisar logs en `logs/api/api.log`
- Revisar estado de vCenter/Documentos
- Verificar que Ollama está corriendo

---

## 🔄 Ciclo de Vida de la Solicitud

### Timeline

```
0ms     [Client] POST /chat enviado
        └─ Contiene username + message

50ms    [Server] Recibido en main_agent.py
        ├─ TIMING 1: time_received = now()
        └─ Middleware: Validar sesión

100ms   [Server] Sesión validada
        ├─ TIMING 2: time_after_session (+50ms)
        ├─ Obtener username de sesión
        └─ Extraer datos JSON

105ms   [Server] Mensaje parseado
        ├─ TIMING 3: time_after_parse (+5ms)
        ├─ classify_task(message) → 'vcenter'
        └─ Validar que no esté vacío

110ms   [Server] Clasificación completada
        └─ Target: 'vcenter'

115ms   [Formatter] Format query (si ENABLE_FORMATTING=true)
        ├─ Llama a gpt-oss:20b
        └─ Timeout: 5000ms

350ms   [Formatter] Respuesta formateada
        ├─ TIMING 4: time_after_format (+240ms)
        └─ Input: "me list as las vms"
           Output: "Listar las máquinas virtuales"

360ms   [Agent] Procesa con executor_llm
        ├─ Invoca get_user_context(username)
        ├─ Ejecuta agent_executor.invoke()
        └─ Consulta pyvmomi → vCenter

2500ms  [vCenter] Respuesta recibida
        └─ Retorna lista de VMs

2600ms  [Server] Agente retorna respuesta
        └─ answer = "Hay 12 VMs en producción..."

2610ms  [Logging] Registrar operación
        ├─ api.log: timing completo
        ├─ audit.log: message_routing
        └─ performance.log: métricas

2620ms  [Server] JSON serializado
        └─ {'response': '...', 'agent': 'vcenter'}

2625ms  [Network] HTTP 200 + JSON enviado
        └─ Response headers + body

2650ms  [Client] JSON recibido
        ├─ response.ok = true
        ├─ data.agent = 'vcenter'
        └─ data.response = "Hay 12 VMs..."

2655ms  [Client] DOM actualizado
        ├─ appendMessage('agent', response_text)
        ├─ lastAgent.textContent = 'vcenter'
        └─ log.scrollTop = log.scrollHeight

2660ms  [UI] Mensaje visible
        └─ Usuario ve respuesta en chat
```

---

## 📊 Matriz de Enrutamiento

### Flujo de Decisión

```
Input Message
    ↓
¿Contiene keywords documentación?
    ├─ SÍ → Retorna 'documentation'
    └─ NO ↓
    
¿Contiene keywords vCenter?
    ├─ SÍ → Retorna 'vcenter'
    └─ NO ↓

¿Invoca LLM para clasificación?
    ├─ Respuesta contiene 'documentation'? → Retorna 'documentation'
    ├─ Respuesta contiene 'vcenter'? → Retorna 'vcenter'
    └─ Otro → Retorna 'general'
```

### Tabla de Ejemplos

| Mensaje | Keywords Encontradas | LLM Verdict | Target | Razón |
|---------|---------------------|------------|--------|-------|
| "¿Cuántas VMs?" | vm | - | vcenter | Keyword match |
| "Cómo instalar DNS" | instalar | - | documentation | Keyword match |
| "¿Qué es una snapshot?" | snapshot | - | vcenter | Keyword match |
| "Hola" | - | general | general | LLM fallback |
| "Información sobre vCenter" | - | vcenter | vcenter | LLM classify |
| "Me ayudas?" | - | general | general | LLM fallback |

---

## 🔍 Debugging y Troubleshooting

### Verificar Endpoint está Activo

```bash
# Test simple
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","message":"hola"}'

# Esperado: HTTP 401 (sesión inválida) o 200 (éxito)
# NO esperado: Error de conexión
```

### Monitorear Requests en Tiempo Real

```bash
# Terminal 1: Tail de logs API
tail -f vcenter_agent_system/logs/api/api.log | grep "\[TIMING\]"

# Terminal 2: Hacer request
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","message":"¿VMs?"}'
```

---

### Capturar Detalles de Request/Response

```bash
# Verbose mode con curl
curl -v -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"username":"jmartinb","message":"test"}' \
  2>&1 | grep -E "(Request|Response|<|>)"
```

---

## 🧪 Testeo

### Test con Python Requests

```python
import requests
import json

BASE_URL = "http://localhost:5000"
ENDPOINT = "/chat"

# Session cookie (si está requerida)
session = requests.Session()
session.cookies.set('session', 'your_session_id')

# Request
response = session.post(
    f"{BASE_URL}{ENDPOINT}",
    json={
        "username": "jmartinb",
        "message": "¿Cuántas VMs hay?"
    },
    timeout=10
)

# Response
print(f"Status: {response.status_code}")
print(f"Body: {json.dumps(response.json(), indent=2)}")

# Assertions
assert response.status_code == 200
assert 'response' in response.json()
assert 'agent' in response.json()
```

---

### Test con JavaScript/Fetch

```javascript
const response = await fetch('http://localhost:5000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    username: 'jmartinb',
    message: '¿Cuántas VMs hay?'
  })
});

const data = await response.json();
console.log(`Status: ${response.status}`);
console.log(`Response:`, data);
```

---

### Test con Postman

1. **Crear nuevo Request**
   - Método: POST
   - URL: `http://localhost:5000/chat`

2. **Headers**
   - Content-Type: application/json

3. **Body (raw)**
   ```json
   {
     "username": "jmartinb",
     "message": "¿Cuántas VMs hay?"
   }
   ```

4. **Pre-request Script** (si necesita auth)
   ```javascript
   // Obtener sessionId de login primero
   // Luego setear como cookie
   ```

5. **Send** y verificar Response

---

## 📈 Limits y Quotas

### Límites Actuales

```
┌─────────────────────────────┐
│ LÍMITES DE OPERACIÓN        │
├─────────────────────────────┤
│ Timeout total: 30 segundos  │
│ Timeout formatter: 5 seg    │
│ Tamaño máx message: 5000 ch │
│ Tamaño máx response: 10000+ │
│ Sesiones activas: Sin límite│
│ Timeout de sesión: 3600 seg │
└─────────────────────────────┘
```

### Rate Limiting

**Actualmente**: No implementado

**Futuro**: Considerar per-user rate limiting

```python
# Pseudo-código para futuro
@limiter.limit("100 per hour")
@app.route('/chat', methods=['POST'])
def chat_api():
    ...
```

---

## 📝 Especificación OpenAPI/Swagger

```yaml
openapi: 3.0.0
info:
  title: Chat Orchestrator API
  version: 1.0.0
  description: API de chat con enrutamiento inteligente

servers:
  - url: http://localhost:5000
    description: Local development

paths:
  /chat:
    post:
      summary: Procesar consulta de chat
      description: Enruta la consulta al agente apropiado
      
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                username:
                  type: string
                  example: jmartinb
                message:
                  type: string
                  example: "¿Cuántas VMs hay?"
      
      responses:
        '200':
          description: Éxito
          content:
            application/json:
              schema:
                type: object
                properties:
                  response:
                    type: string
                  agent:
                    type: string
                    enum: [vcenter, documentation, general]
        
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
        
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
        
        '500':
          description: Server error
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
```

---

## 🔗 Recursos Relacionados

- [Arquitectura del Chat](./ARQUITECTURA_CHAT.md)
- [Guía de Funcionamiento](./GUIA_FUNCIONAMIENTO.md)
- [Guía de Implementación](./GUIA_IMPLEMENTACION_TECNICA.md)

---

## 📞 Soporte

Para problemas con el endpoint:
1. Revisar sección "Troubleshooting" en [GUIA_FUNCIONAMIENTO.md](./GUIA_FUNCIONAMIENTO.md)
2. Revisar logs en `logs/api/api.log`
3. Verificar que Ollama y modelos están disponibles
4. Contactar al equipo de desarrollo
