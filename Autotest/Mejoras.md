# Mejoras Web UI — GALTTCMC CI/CD

## 1. Seguridad (crítico)

- ~~**SQL injection** en `/api/deployments` — el filtro `status` se concatena directamente en la query sin parametrizar~~ ✅ Resuelto
- ~~**XSS potencial** en el sistema de toasts — `innerHTML` sin sanitizar~~ ✅ Resuelto
- ~~**Sin autenticación** en ninguna ruta, ni protección CSRF~~ ✅ Resuelto

---

## 2. Funcionalidad incompleta

- ~~**Modal de Pipeline Runs** — El backend ya devuelve build logs y resultados SonarQube en `GET /api/deployment/<id>`, pero el modal no los renderiza. Solo muestra estado, duración e ISO path.~~ ✅ Resuelto
- ~~**Coloreado de logs** — El visor es texto verde plano. No distingue `ERROR`/`WARN`/`INFO`/`DEBUG` con colores (mejora de UX más impactante para diagnóstico).~~ ✅ Resuelto
- ~~**`new_coverage`** — La columna existe en la API de SonarQube pero no aparece en la tabla de resultados.~~ ✅ Resuelto
- ~~**Estado real del pipeline** — "Pipeline Status: Active" está hardcodeado en el HTML. Nunca refleja si `cicd.service` está corriendo o parado.~~ ✅ Resuelto
- ~~**Enlace "Ver todo"** en el dashboard — Las tarjetas de deployments recientes no tienen link a la vista completa.~~ ✅ Resuelto

---

## 3. Dependencias CDN ✅ Resuelto

~~Tailwind, Alpine.js y Chart.js se cargan desde internet (`cdn.tailwindcss.com`). En una red local aislada (`172.30.188.x`) esto es un punto de fallo.~~

Descargadas a `web/static/js/` y `base.html` actualizado con rutas locales.

---

## 4. Pipeline — Mejoras de despliegue ✅ Resuelto

### Revertir snapshot antes del despliegue (vCenter API)

Antes de ejecutar la **Phase 5 – SSH Deploy**, la **Phase 4 – vCenter** debe:

1. **Listar las snapshots** de la VM `YOUR_CICD_VM` vía API (`GET /rest/vcenter/vm/{vm_id}/snapshot`).
2. **Revertir al snapshot existente** (`POST /rest/vcenter/vm/{vm_id}/snapshot/{snapshot_id}?action=revert`) — garantiza un estado limpio y reproducible para cada despliegue.
3. **Esperar a que la VM esté en estado `POWERED_OFF`** tras el revert (polling con `wait_for()`).
4. **Arrancar la VM** (`POST /rest/vcenter/vm/{vm_id}/power/start`).
5. **Esperar a que la VM esté `POWERED_ON`** antes de ceder el control a la Phase 5.

**Motivación:** Evita que instalaciones previas o estados corruptos interfieran en el despliegue. Cada pipeline run parte de una imagen limpia y consistente.

**Archivos a modificar:**
- `python/vcenter_api.py` — añadir métodos `list_snapshots()`, `revert_to_snapshot()`, `wait_vm_power_state()`
- `scripts/compile.sh` o `ci_cd.sh` — orquestar la llamada al revert antes del deploy
- `config/ci_cd_config.yaml` — añadir clave `snapshot_name` (nombre del snapshot a usar, p.ej. `"clean_install"`)

---

## 5. Características de alto valor que faltan

| Feature | Impacto |
|---|---|
| ~~Live-tail de logs (auto-follow + polling)~~ ✅ Resuelto | Alto |
| Búsqueda por tag en Pipeline Runs | Alto |
| ~~Visualización de fases del pipeline (barras de progreso por fase)~~ ✅ Resuelto | Medio |
| Filtro por rango de fechas en Pipeline Runs | Medio |
| Link desde SonarQube result → deployment | Medio |
| Paginación numerada (en vez de solo Prev/Next) | Bajo |
| Tendencia de Security Hotspots en el gráfico Sonar | Bajo |
| `triggered_by` visible en el modal | Bajo |

---

## 7. Phase 6 – Verificación Post-Instalación

### Contexto y reto

Cuando `install.sh` termina, la VM hace `reboot`. Al reconectarse por SSH, el sistema **fuerza un cambio de contraseña** porque la instalación deja la cuenta `root` con la contraseña expirada (`root`/`root`). El flujo interactivo es:

```
1. SSH conecta con root / root   (autenticación OK, pero passwd caducada)
2. WARNING: Your password has expired.
3. (current) UNIX password: root       ← pide la contraseña actual de nuevo
4. Enter new UNIX password: <nueva>    ← nueva contraseña
5. Retype new UNIX password: <nueva>   ← confirmación
6. Shell disponible
```

Este proceso **bloquea `BatchMode=yes`** (usado actualmente en `deploy.sh`), por lo que no se puede verificar nada hasta haberlo resuelto.

### Solución propuesta: `pexpect` en Python

Usar `pexpect` (Python) en lugar de `expect` (Bash) para manejar el diálogo interactivo, ya que Python ya es una dependencia del proyecto.

**Flujo automatizado (nuevo script `python/post_install_check.py`):**

1. **Actualizar `known_hosts`** — tras el reinstall el host key cambia; ejecutar `ssh-keyscan` antes de conectar:
   ```bash
   ssh-keyscan -H YOUR_TARGET_VM_IP >> ~/.ssh/known_hosts
   ```
   O mejor: pasar `StrictHostKeyChecking=no` solo en este paso y después forzar `ssh-keyscan` para registrar la nueva clave.

2. **Cambiar la contraseña via `pexpect`:**
   ```python
   import pexpect

   child = pexpect.spawn(
       'ssh -o StrictHostKeyChecking=no root@{ip}'.format(ip=target_ip),
       timeout=30
   )
   child.expect('password:')
   child.sendline('root')                          # login
   child.expect(r'(current).*password:')
   child.sendline('root')                          # confirma contraseña actual
   child.expect(r'[Ee]nter new.*password:')
   child.sendline(new_password)
   child.expect(r'[Rr]etype new.*password:')
   child.sendline(new_password)
   child.expect(r'[$#]')                           # shell listo
   ```

3. **Instalar clave SSH** (para las comprobaciones siguientes sin contraseña):
   ```python
   child.sendline(
       'mkdir -p ~/.ssh && echo "{pubkey}" >> ~/.ssh/authorized_keys'
       ' && chmod 600 ~/.ssh/authorized_keys'.format(pubkey=ssh_pubkey)
   )
   child.expect(r'[$#]')
   child.sendline('exit')
   ```

4. **Ejecutar checks de verificación** via SSH normal (`BatchMode=yes` ya funciona):

   | Check | Comando remoto | Criterio de éxito |
   |-------|----------------|-------------------|
   | Servicios activos | `systemctl is-active ttc chronyd network firewalld` | Todos `active` |
   | Bond — estado interfaz | `ip link show bond0` | `state UP` |
   | Bond — slaves activos | `cat /proc/net/bonding/bond0` | `eth0` y `eth1` con `MII Status: up` |
   | Directorios instalación | `test -d /opt/fdf && test -d /opt/gcss && test -d /opt/mc && test -d /opt/mmi` | Exit code 0 |
   | Conectividad gateway | `ping -c1 -W2 <gateway>` | Exit code 0 |
   | Espacio en disco | `df -h /` | Uso < umbral configurable |
   | Arranque limpio | `uptime -s` | Fecha posterior al reboot del pipeline |

### Contraseña por defecto

La instalación deja `root`/`root`. La nueva contraseña debe cumplir los requisitos típicos de PAM en SUSE 15 (mínimo 8 caracteres, mayúsculas, minúsculas, dígito, símbolo). Se propone como valor por defecto:

```
YOUR_VM_PASSWORD
```

Se almacena en `config/.env` como `TARGET_VM_NEW_PASSWORD=YOUR_VM_PASSWORD` y **no se hardcodea** en ningún script. Si en el futuro se requiere rotación, se cambia solo en `.env`.

### Configuración necesaria en `ci_cd_config.yaml`

```yaml
post_install:
  enabled: true
  new_root_password: ""          # leer de TARGET_VM_NEW_PASSWORD en .env
  wait_reboot_timeout: 300       # segundos esperando que la VM vuelva tras el reboot
  wait_reboot_interval: 15
  checks:
    services:
      - ttc
      - chronyd
      - network
      - firewalld
    bond:
      interface: bond0
      slaves:
        - eth0
        - eth1
    opt_dirs:
      - /opt/fdf
      - /opt/gcss
      - /opt/mc
      - /opt/mmi
    disk_usage_threshold: 90     # % máximo uso de disco (/)
```

### Archivos a crear/modificar

| Archivo | Cambio |
|---------|--------|
| `python/post_install_check.py` | Nuevo script: cambio de passwd + checks |
| `ci_cd.sh` | Añadir llamada a Phase 6 tras `deploy.sh` |
| `config/ci_cd_config.yaml` | Añadir bloque `post_install` |
| `config/.env.example` | Añadir `TARGET_VM_NEW_PASSWORD=` |
| `db/init_db.sql` | Añadir tabla `post_install_results` o columnas en `deployments` |
| `requirements_post.txt` | `pexpect>=4.8` (solo para esta fase) |

### Prerequisitos en la máquina Linux (`YOUR_PIPELINE_HOST_IP`)

```bash
pip3.6 install pexpect
# pexpect está en repos SUSE: zypper install python3-pexpect
```

### Integración en el pipeline

```
Phase 5 – SSH Deploy (install.sh → reboot)
    ↓
[wait_for_reboot: polling SSH hasta que la VM responda]
    ↓
Phase 6 – Post-Install Check
    ├─ Cambio de contraseña root (pexpect)
    ├─ Instalar SSH key para futuros accesos
    └─ Verificar servicios / bond / rutas
    ↓
Pipeline: SUCCESS o FAILED (con detalle del check fallido)
```

**Motivación:** Sin esta fase, el pipeline reporta `SUCCESS` en cuanto `install.sh` devuelve código 0 (o la VM reinicia), sin garantizar que el sistema instalado arranca correctamente y está operativo.

---

## 6. Código muerto a limpiar

- `.badge-success` / `.badge-failed` CSS nunca usados (los templates usan clases inline)
- `.card-hover` y `.modal-backdrop` definidos pero no aplicados
- `formatBytes()` y `formatDuration()` en `app.js` nunca llamados
- `load_config()` en `app.py` definido pero sin usar
