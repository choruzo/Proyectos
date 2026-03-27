# 🔌 Referencia - APIs Externas

## Visión General

Documentación de integración con servicios externos: Git, SonarQube y vCenter.

**Relacionado con**:
- [[Pipeline - Git Monitor#APIs de Git]]
- [[Pipeline - SonarQube#APIs de SonarQube]]
- [[Pipeline - vCenter#APIs de vCenter]]

---

## Git API

### Endpoint: ls-remote

**Propósito**: Listar tags remotos sin clonar

**Comando**:
```bash
git ls-remote --tags https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
```

**Output**:
```
a1b2c3d4... refs/tags/MAC_1_V24_02_15_01
e5f6g7h8... refs/tags/MAC_1_V24_02_15_02
```

**Filtrado**:
```bash
git ls-remote --tags "$GIT_URL" | awk '{print $2}' | sed 's|refs/tags/||' | grep -E "$TAG_PATTERN"
```

---

### Autenticación

**Basic Auth** (HTTPS):
```bash
git clone https://$GIT_USER:$GIT_PASSWORD@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
```

**Variables en `.env`**:
```bash
GIT_USER=automation
GIT_PASSWORD=ghp_xxxxxxxxxxxxxxxxxxxx
```

---

## SonarQube API

**Base URL**: `https://YOUR_SONARQUBE_SERVER`

### Autenticación

**Token-based** (basic auth con token como username):
```bash
curl -u "$SONAR_TOKEN:" https://YOUR_SONARQUBE_SERVER/api/...
```

---

### 1. Component Analysis Task

**GET** `/api/ce/component?component={projectKey}`

**Response**:
```json
{
  "queue": [],
  "current": {
    "id": "AYxxxxx",
    "type": "REPORT",
    "status": "SUCCESS",
    "submittedAt": "2026-03-20T10:30:00+0000",
    "executionTimeMs": 45000
  }
}
```

---

### 2. Quality Gate Status

**GET** `/api/qualitygates/project_status?projectKey={projectKey}`

**Response**:
```json
{
  "projectStatus": {
    "status": "OK",
    "conditions": [
      {
        "status": "OK",
        "metricKey": "new_coverage",
        "comparator": "LT",
        "errorThreshold": "80",
        "actualValue": "85.3"
      }
    ]
  }
}
```

---

### 3. Project Measures

**GET** `/api/measures/component?component={projectKey}&metricKeys=coverage,bugs,vulnerabilities`

**Response**:
```json
{
  "component": {
    "key": "GALTTCMC",
    "measures": [
      {"metric": "coverage", "value": "85.3"},
      {"metric": "bugs", "value": "0"}
    ]
  }
}
```

---

### Error Handling

**401 Unauthorized**: Token expirado
```bash
# Regenerar token en SonarQube UI
# My Account → Security → Generate Tokens
```

**404 Not Found**: Proyecto no existe
```bash
# Crear proyecto via API
curl -u "$SONAR_TOKEN:" -X POST \
  "https://YOUR_SONARQUBE_SERVER/api/projects/create?name=GALTTCMC&project=GALTTCMC"
```

---

## vCenter REST API

**Base URL**: `https://vcenter.example.com/rest`

**⚠️ No usa pyvmomi** - Solo REST API nativo

---

### Autenticación

**POST** `/rest/com/vmware/cis/session`

**Request**:
```bash
curl -X POST -u "$VCENTER_USER:$VCENTER_PASSWORD" \
  "https://vcenter.example.com/rest/com/vmware/cis/session" \
  --insecure
```

**Response**:
```json
{
  "value": "vmware-api-session-id-here"
}
```

**Session duration**: ~30 minutos

---

### 1. Get VM by Name

**GET** `/rest/vcenter/vm?names={vmName}`

**Headers**:
```
vmware-api-session-id: {sessionId}
```

**Response**:
```json
{
  "value": [
    {
      "vm": "vm-123",
      "name": "Releases",
      "power_state": "POWERED_ON"
    }
  ]
}
```

---

### 2. Upload ISO to Datastore

**PUT** `/folder/{path}?dsName={datastoreName}`

**Headers**:
```
vmware-api-session-id: {sessionId}
Content-Type: application/octet-stream
```

**Body**: Binary ISO file

**Example**:
```python
upload_url = "{}/folder/P27/Versiones/{}.iso?dsName=YOUR_DATASTORE".format(
    base_url.replace('/rest', ''),
    tag_name
)

with open(iso_path, 'rb') as f:
    requests.put(upload_url, headers=headers, data=f, verify=False)
```

---

### 3. Configure CD-ROM

**GET** `/rest/vcenter/vm/{vmId}/hardware/cdrom`

**PATCH** `/rest/vcenter/vm/{vmId}/hardware/cdrom/{cdromId}`

**Request Body**:
```json
{
  "spec": {
    "backing": {
      "type": "ISO_FILE",
      "iso_file": "[YOUR_DATASTORE] P27/Versiones/TAG.iso"
    },
    "start_connected": true
  }
}
```

---

### 4. Power On VM

**POST** `/rest/vcenter/vm/{vmId}/power/start`

**Response**: 204 No Content (success)

---

### Session Timeout Handling

```python
def api_call_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            # Session expired, re-authenticate
            session = authenticate(...)
            kwargs['session'] = session
            return func(*args, **kwargs)
        raise
```

---

## Rate Limits

### Git

- **No rate limit** oficial, pero:
- Avoid polling < 1 minuto
- Usar `ls-remote` en lugar de `clone` cuando sea posible

---

### SonarQube

- **Analysis uploads**: ~10 por hora (recomendado)
- **API queries**: Sin límite estricto

---

### vCenter

- **Session limit**: 100 sesiones concurrentes por usuario
- **API calls**: Sin límite estricto, pero usar reasonable rate

---

## SSL/TLS

Todos los servicios externos usan HTTPS. El pipeline **desactiva verificación SSL**:

```python
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

requests.get(url, verify=False)
```

**⚠️ Solo aceptable en red interna confiable**

---

## Enlaces Relacionados

- [[Pipeline - Git Monitor]] - Uso de Git API
- [[Pipeline - SonarQube]] - Uso de SonarQube API
- [[Pipeline - vCenter]] - Uso de vCenter API
- [[Referencia - Configuración]] - Configuración de endpoints
