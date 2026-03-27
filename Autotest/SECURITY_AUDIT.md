# Informe de Auditoría de Seguridad — GALTTCMC CI/CD Pipeline

**Fecha:** 2026-03-16
**Archivos analizados:** 35
**Vulnerabilidades encontradas:** 23

---

## Resumen Ejecutivo

| Severidad | Cantidad |
|-----------|----------|
| Critical  | 3        |
| High      | 6        |
| Medium    | 9        |
| Low       | 4        |
| **Total** | **23**   |

---

## Hallazgos Detallados

### FINDING-01: Inyección de Comandos via `eval` en `config_get()`
- **Archivo:** `scripts/common.sh`, línea 93
- **Tipo:** A03:2021 - Injection (Command Injection)
- **Severidad:** CRITICAL

**Descripción:** La función `config_get()` usa `eval "echo \"$value\""` para expandir variables de entorno en valores leídos del YAML. Si un atacante consigue inyectar contenido malicioso en el fichero YAML (o si el valor YAML contiene caracteres de shell), se ejecutarán comandos arbitrarios con los privilegios del usuario `agent`.

**Evidencia:**
```bash
# common.sh línea 93
eval "echo \"$value\""
```

Un valor YAML como `$(rm -rf /)` o `` `id` `` sería ejecutado por `eval`. Cualquier fuente no confiable que alimente el YAML (tags Git, parámetros) podría desencadenar ejecución remota de código.

---

### FINDING-02: SQL Injection en Múltiples Queries Bash
- **Archivos:** `scripts/common.sh` líneas 150-185; `ci_cd.sh` líneas 203-244; `scripts/git_monitor.sh` líneas 324-353; `scripts/fix_inconsistencies.sh` líneas 63-109; `scripts/diagnose_tag.sh` líneas 48-121
- **Tipo:** A03:2021 - Injection (SQL Injection)
- **Severidad:** CRITICAL

**Descripción:** Todas las funciones `db_query()`, `db_tag_processed()`, `db_log_execution()` y las queries directas construyen sentencias SQL interpolando variables directamente en la cadena sin sanitizar ni usar parámetros preparados. Un tag Git con nombre malicioso como `V01'; DROP TABLE deployments; --` podría destruir la base de datos.

**Evidencia:**
```bash
# common.sh línea 169-170
db_query "INSERT INTO execution_log (deployment_id, phase, message, level)
          VALUES ($deployment_id, '$phase', '$message', '$level')"

# ci_cd.sh línea 203
existing_deployment=$(db_query "SELECT id FROM deployments WHERE tag_name='$tag'" ...)

# ci_cd.sh líneas 219-222
deployment_id=$(db_query \
    "INSERT INTO deployments (tag_name, status, started_at, triggered_by)
     VALUES ('$tag', 'pending', datetime('now'), '$triggered_by');
     SELECT last_insert_rowid();")

# fix_inconsistencies.sh línea 63
sqlite3 "$DB_PATH" "DELETE FROM deployments WHERE tag_name='$tag'"
```

---

### FINDING-03: Credenciales en URL Git (Exposición en Logs y Procesos)
- **Archivo:** `scripts/git_monitor.sh`, líneas 58-60 y 301-306
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** CRITICAL

**Descripción:** La contraseña Git se inyecta directamente en la URL para `git ls-remote` y `git clone`. Esto expone las credenciales en:
1. La tabla de procesos del sistema (`/proc/*/cmdline`, `ps aux`)
2. Los logs del pipeline (via `tee -a "$LOG_FILE"` en línea 306)
3. Histórico de shell si se ejecuta manualmente

**Evidencia:**
```bash
# git_monitor.sh línea 59
auth_url=$(echo "$repo_url" | sed "s|https://|https://${GIT_USERNAME:-agent}:${GIT_PASSWORD}@|")
all_tags=$(git ls-remote --tags "$auth_url" 2>/dev/null || echo "")

# git_monitor.sh línea 302
clone_url=$(echo "$repo_url" | sed "s|https://|https://${GIT_USERNAME:-agent}:${GIT_PASSWORD}@|")
git clone --branch "$branch" "$clone_url" "$repo_path" 2>&1 | tee -a "$LOG_FILE"
```

---

### FINDING-04: SECRET_KEY Hardcodeada en Flask
- **Archivo:** `web/config.py`, línea 15
- **Tipo:** A02:2021 - Cryptographic Failures
- **Severidad:** HIGH

**Descripción:** La clave secreta de Flask tiene un valor por defecto predecible `'galttcmc-cicd-web-ui-secret-key-change-me'`. Si la variable de entorno `SECRET_KEY` no está definida, las sesiones de usuario pueden ser falsificadas.

**Evidencia:**
```python
SECRET_KEY = os.getenv('SECRET_KEY', 'galttcmc-cicd-web-ui-secret-key-change-me')
```

---

### FINDING-05: SSL/TLS Verification Deshabilitada Globalmente
- **Archivos:** `python/vcenter_api.py` líneas 30 y 42; `python/sonar_check.py` líneas 53, 144, 199
- **Tipo:** A02:2021 - Cryptographic Failures
- **Severidad:** HIGH

**Descripción:** La verificación SSL está deshabilitada en ambos clientes API, permitiendo ataques Man-in-the-Middle contra vCenter y SonarQube.

**Evidencia:**
```python
# vcenter_api.py línea 30
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# vcenter_api.py línea 42
self.verify_ssl = False  # Cambiar a True si tienes certificados válidos

# sonar_check.py línea 53
r = requests.get(url, params=params, auth=auth, verify=False, timeout=30)
```

---

### FINDING-06: SSH con `StrictHostKeyChecking=no`
- **Archivos:** `scripts/common.sh` líneas 196-197; `scripts/deploy.sh` línea 35
- **Tipo:** A02:2021 - Cryptographic Failures
- **Severidad:** HIGH

**Descripción:** Todas las conexiones SSH aceptan cualquier clave de host sin verificación. Un atacante que suplante la IP YOUR_TARGET_VM_IP (ARP spoofing, DNS hijacking) puede interceptar la sesión SSH completa y capturar comandos ejecutados como `root`.

**Evidencia:**
```bash
# common.sh línea 196
local ssh_opts="-o StrictHostKeyChecking=no -o BatchMode=yes -i $ssh_key"

# deploy.sh línea 35
echo "-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 -i $ssh_key"
```

---

### FINDING-07: Ejecución Remota como Root sin Restricciones
- **Archivo:** `scripts/deploy.sh`, líneas 39-46 y 264-266
- **Tipo:** A04:2021 - Insecure Design
- **Severidad:** HIGH

**Descripción:** El pipeline ejecuta comandos arbitrarios como `root` en la VM destino. Los parámetros provienen de la configuración YAML (que pasa por `config_get` con `eval`) y se ejecutan sin sanitización.

**Evidencia:**
```bash
# deploy.sh líneas 264-266
ssh $ssh_opts "${TARGET_VM_USER}@${TARGET_VM_IP}" \
    "cd $install_path && ./$install_script $install_params" \
    <<< "" 2>&1 | tee -a "$DEPLOY_LOG_FILE"
```

---

### FINDING-08: Inyección de Comandos en `compile.sh` via Variables No Sanitizadas
- **Archivo:** `scripts/compile.sh`, líneas 157-160
- **Tipo:** A03:2021 - Injection (Command Injection)
- **Severidad:** HIGH

**Descripción:** El script ejecuta un script externo con `bash -c` interpolando directamente `$compile_dir` y `$build_script` procedentes del YAML. Las comillas simples en esas variables rompen el contexto del comando.

**Evidencia:**
```bash
# compile.sh líneas 157-160
timeout "$timeout_secs" bash -c "
    cd '$compile_dir'
    './$build_script' 2>&1
" 2>&1 | tee -a "$COMPILE_LOG_FILE"
```

---

### FINDING-09: Función `wait_for()` Usa `eval` con Input No Validado
- **Archivo:** `scripts/common.sh`, línea 238
- **Tipo:** A03:2021 - Injection (Command Injection)
- **Severidad:** HIGH

**Descripción:** La función `wait_for()` ejecuta `eval "$check_cmd"` donde `$check_cmd` es un parámetro genérico. Cualquier uso futuro con input no confiable sería explotable.

**Evidencia:**
```bash
# common.sh línea 238
while ! eval "$check_cmd"; do
```

---

### FINDING-10: Cookie de Sesión sin Flag `Secure`
- **Archivo:** `web/config.py`, líneas 19-20
- **Tipo:** A05:2021 - Security Misconfiguration
- **Severidad:** MEDIUM

**Descripción:** La configuración define `SESSION_COOKIE_HTTPONLY = True` y `SESSION_COOKIE_SAMESITE = 'Lax'`, pero falta `SESSION_COOKIE_SECURE = True`. La cookie de sesión se transmite en conexiones HTTP sin cifrar.

**Evidencia:**
```python
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
# Falta: SESSION_COOKIE_SECURE = True
```

---

### FINDING-11: Web UI Sin HTTPS (HTTP en Puerto 8080)
- **Archivos:** `web.service` línea 19; `web/config.py` línea 10
- **Tipo:** A02:2021 - Cryptographic Failures
- **Severidad:** MEDIUM

**Descripción:** La Web UI escucha en HTTP plano (puerto 8080) sin TLS. Las credenciales de login viajan en texto claro por la red.

**Evidencia:**
```
# web.service
ExecStart=/usr/bin/python3.6 -m gunicorn --bind 0.0.0.0:8080 ...
```

---

### FINDING-12: Escucha en Todas las Interfaces (0.0.0.0)
- **Archivos:** `web/config.py` línea 10; `web.service` línea 14
- **Tipo:** A05:2021 - Security Misconfiguration
- **Severidad:** MEDIUM

**Descripción:** Gunicorn escucha en `0.0.0.0`, exponiendo la aplicación en todas las interfaces de red.

**Evidencia:**
```python
HOST = os.getenv('WEB_HOST', '0.0.0.0')
```

---

### FINDING-13: Exposición de Excepciones en Respuestas API
- **Archivo:** `web/app.py`, múltiples endpoints (líneas 241, 270, 323, 386, 430, 453...)
- **Tipo:** A05:2021 - Security Misconfiguration
- **Severidad:** MEDIUM

**Descripción:** Todos los endpoints API devuelven `str(e)` directamente al cliente, exponiendo rutas internas, nombres de tablas y estado de la base de datos.

**Evidencia:**
```python
except Exception as e:
    return jsonify({'error': str(e)}), 500
```

---

### FINDING-14: Ausencia de Rate Limiting en Login
- **Archivo:** `web/app.py`, líneas 111-151
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** MEDIUM

**Descripción:** El endpoint `/login` no implementa rate limiting ni bloqueo de cuenta tras intentos fallidos. Un atacante puede realizar ataques de fuerza bruta sin restricción.

---

### FINDING-15: Política de Contraseñas Débil
- **Archivos:** `web/app.py` línea 848; `python/manage_web_users.py` línea 30
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** MEDIUM

**Descripción:** La única restricción es longitud mínima de 8 caracteres. Sin requisitos de complejidad (mayúsculas, números, caracteres especiales) ni verificación contra contraseñas comunes.

---

### FINDING-16: Falta Autorización Basada en Roles (RBAC)
- **Archivo:** `web/app.py`, líneas 66-76 y 822-914
- **Tipo:** A01:2021 - Broken Access Control
- **Severidad:** MEDIUM

**Descripción:** No existe sistema de roles. Cualquier usuario autenticado puede gestionar otros usuarios (crear, desactivar, cambiar contraseñas).

---

### FINDING-17: XSS Potencial en Visor de Logs via `x-html`
- **Archivo:** `web/templates/logs.html`, línea 169
- **Tipo:** A03:2021 - Injection (XSS)
- **Severidad:** MEDIUM

**Descripción:** El visor de logs usa `x-html="colorizeLog(logContent)"` para renderizar HTML. Aunque `escapeHtml()` mitiga el riesgo, el uso de `x-html` en lugar de `x-text` introduce un vector de ataque que depende de la correcta implementación del escape.

**Evidencia:**
```html
<pre x-ref="logContainer" ...><code x-html="colorizeLog(logContent)"></code></pre>
```

---

### FINDING-18: Credenciales de Administrador en Texto Plano
- **Archivo:** `config/.env.example`, líneas 32-33
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** MEDIUM

**Descripción:** El fichero `.env` almacena credenciales de vCenter, SonarQube y Git en texto plano sin protección programática de permisos (solo recomendación `chmod 600` en documentación).

**Evidencia:**
```
VCENTER_USER=administrator@vsphere.local
VCENTER_PASSWORD=your_vcenter_password_here
```

---

### FINDING-19: Información Sensible en Templates HTML
- **Archivos:** `web/templates/base.html` líneas 161-162; `web/templates/login.html` línea 114; `web/templates/dashboard.html` línea 169
- **Tipo:** A01:2021 - Broken Access Control (Information Disclosure)
- **Severidad:** LOW

**Descripción:** Las plantillas exponen la IP interna del servidor (`YOUR_PIPELINE_HOST_IP`) y la versión del SO (`SUSE Linux 15`) directamente en el HTML.

**Evidencia:**
```html
<p class="font-semibold">SUSE Linux 15</p>
<p>YOUR_PIPELINE_HOST_IP</p>
```

---

### FINDING-20: Contraseña de Servidor Expuesta en CLAUDE.md
- **Archivo:** `CLAUDE.md` (tabla Infrastructure)
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** LOW

**Descripción:** El fichero `CLAUDE.md` incluye la contraseña del servidor dev en texto claro. Si se commitea al repositorio, cualquier persona con acceso al repo obtiene acceso al servidor.

---

### FINDING-21: Dependencias con Versiones Antiguas
- **Archivos:** `web/requirements.txt`; `python/requirements.txt`
- **Tipo:** A06:2021 - Vulnerable and Outdated Components
- **Severidad:** LOW

**Descripción:** El proyecto usa Python 3.6 (EOL desde diciembre 2021) y versiones antiguas de Flask 2.0.3, Werkzeug 2.0.3, Jinja2 3.0.3, requests, urllib3. Estas versiones tienen CVEs conocidos sin parche.

---

### FINDING-22: Servicio `cicd.service` Sin Hardening Completo
- **Archivo:** `cicd.service`
- **Tipo:** A05:2021 - Security Misconfiguration
- **Severidad:** LOW

**Descripción:** El servicio aplica `NoNewPrivileges=true` y `PrivateTmp=true`, pero carece de `ProtectSystem=strict`, `ProtectHome=read-only`, `ReadWritePaths` restringido y otras directivas de sandboxing que sí tiene `web.service`.

---

### FINDING-23: Contraseña Visible en Argumentos de Línea de Comandos
- **Archivo:** `python/manage_web_users.py`, líneas 173-178
- **Tipo:** A07:2021 - Identification and Authentication Failures
- **Severidad:** LOW

**Descripción:** El script acepta contraseñas como argumentos de línea de comandos. Los argumentos de proceso son visibles en `/proc/*/cmdline` y en el histórico de shell.

**Evidencia:**
```python
cmd_add(sys.argv[2], sys.argv[3])
```

---

## Tabla Resumen de Riesgos

| ID   | Severidad | Categoría OWASP                  | Descripción Breve                              |
|------|-----------|----------------------------------|------------------------------------------------|
| F-01 | CRITICAL  | A03 Injection                    | `eval` en `config_get()`                       |
| F-02 | CRITICAL  | A03 Injection                    | SQL Injection en queries Bash                  |
| F-03 | CRITICAL  | A07 Auth Failures                | Credenciales Git en URL/logs                   |
| F-04 | HIGH      | A02 Crypto Failures              | SECRET_KEY hardcodeada                         |
| F-05 | HIGH      | A02 Crypto Failures              | SSL verification deshabilitada                 |
| F-06 | HIGH      | A02 Crypto Failures              | SSH StrictHostKeyChecking=no                   |
| F-07 | HIGH      | A04 Insecure Design              | Ejecución remota root sin restricción          |
| F-08 | HIGH      | A03 Injection                    | Command injection en compile.sh                |
| F-09 | HIGH      | A03 Injection                    | `eval` en `wait_for()`                         |
| F-10 | MEDIUM    | A05 Misconfiguration             | Cookie sin flag Secure                         |
| F-11 | MEDIUM    | A02 Crypto Failures              | Web UI sin HTTPS                               |
| F-12 | MEDIUM    | A05 Misconfiguration             | Bind en 0.0.0.0                                |
| F-13 | MEDIUM    | A05 Misconfiguration             | Excepciones expuestas en API                   |
| F-14 | MEDIUM    | A07 Auth Failures                | Sin rate limiting en login                     |
| F-15 | MEDIUM    | A07 Auth Failures                | Política contraseñas débil                     |
| F-16 | MEDIUM    | A01 Access Control               | Sin RBAC                                       |
| F-17 | MEDIUM    | A03 Injection                    | XSS potencial via x-html                       |
| F-18 | MEDIUM    | A07 Auth Failures                | Credenciales en .env sin protección            |
| F-19 | LOW       | A01 Access Control               | Info sensible en HTML                          |
| F-20 | LOW       | A07 Auth Failures                | Contraseña en CLAUDE.md                        |
| F-21 | LOW       | A06 Vulnerable Components        | Dependencias antiguas / Python 3.6 EOL         |
| F-22 | LOW       | A05 Misconfiguration             | cicd.service sin hardening completo            |
| F-23 | LOW       | A07 Auth Failures                | Contraseña en args de proceso                  |

---

## Aspectos Positivos Encontrados

Los siguientes controles de seguridad están correctamente implementados:

1. **CSRF Protection:** La Web UI implementa tokens CSRF con validación `hmac.compare_digest()` (resistente a timing attacks).
2. **SQL Parametrizado en Web UI:** Los queries SQLite en `app.py` usan correctamente parámetros `?` en vez de interpolación directa.
3. **Open Redirect Prevention:** El login valida que el parámetro `next` sea una ruta relativa.
4. **Path Traversal Prevention en Logs:** El endpoint de logs valida contra `..`, `/` y `\\` en nombres de fichero.
5. **Password Hashing:** Las contraseñas se almacenan con `werkzeug.security.generate_password_hash()` (bcrypt/pbkdf2).
6. **Session Security:** `SESSION_COOKIE_HTTPONLY=True` y `SESSION_COOKIE_SAMESITE='Lax'` están configurados.
7. **.gitignore Correcto:** `.env`, bases de datos y logs están excluidos del repositorio.
8. **`set -euo pipefail`:** Todos los scripts Bash usan modo estricto.
9. **XSS Prevention en Toast:** La función `showToast()` usa `textContent` en vez de `innerHTML`.
10. **Status Filter Whitelist:** El endpoint `/api/deployments` valida el filtro de status contra un set fijo `VALID_STATUSES`.

---

## Archivos Auditados

| Archivo |
|---------|
| `ci_cd.sh` |
| `scripts/common.sh` |
| `scripts/git_monitor.sh` |
| `scripts/compile.sh` |
| `scripts/deploy.sh` |
| `scripts/notify.sh` |
| `scripts/fix_inconsistencies.sh` |
| `scripts/diagnose_tag.sh` |
| `python/sonar_check.py` |
| `python/vcenter_api.py` |
| `python/manage_web_users.py` |
| `web/app.py` |
| `web/config.py` |
| `config/ci_cd_config.yaml` |
| `config/.env.example` |
| `db/init_db.sql` |
| `cicd.service` |
| `web.service` |
| `web/templates/base.html` |
| `web/templates/login.html` |
| `web/templates/logs.html` |
| `web/templates/dashboard.html` |
| `web/templates/users.html` |
| `web/static/js/app.js` |
| `.gitignore` |
| `install_service.sh` |
| `web/requirements.txt` |
| `python/requirements.txt` |
