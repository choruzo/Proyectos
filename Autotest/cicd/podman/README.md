# PoC Podman — CI/CD GALTTCMC (híbrido)

Contenedores para la **app** (orquestador, BD, Web UI) con la **compilación
y el análisis Sonar ejecutándose en el host SUSE**. Las imágenes se construyen
en local y se exportan como `.tar` para cargarlas en la máquina destino.

```
                 máquina local (Windows/Linux)                máquina SUSE
  ┌──────────────────────────────────────┐         ┌──────────────────────────────┐
  │ podman/docker build                  │  .tar   │ podman load                  │
  │  - cicd-orchestrator:local  ─────────┼────────►│  - orquestador (daemon)      │
  │  - cicd-web:local                    │         │  - web (gunicorn :8080)      │
  └──────────────────────────────────────┘         │                              │
                                                   │  compile.sh --ssh──┐         │
                                                   │        ▲           │         │
                                                   │        │  bind     ▼         │
                                                   │   /home/agent/V08... (host)  │
                                                   │   build_DVDs.sh + wrapper    │
                                                   └──────────────────────────────┘
```

## Qué se contenedoriza y qué no

| Componente | Dónde corre |
|---|---|
| `ci_cd.sh` (daemon), `git_monitor`, vCenter, `deploy.sh`, Quality Gate API | Contenedor orquestador |
| Web UI (Flask + Gunicorn) | Contenedor web |
| SQLite (`db/`), logs (`logs/`), `config/` | Bind-mount desde el host |
| **`build_DVDs.sh` + build-wrapper** | **Host SUSE** (por SSH a `127.0.0.1`) |
| **`sonar-scanner` (análisis)** | **Host SUSE** (por SSH a `127.0.0.1`) |
| `wall` | Deshabilitado en contenedor (`CICD_WALL_ENABLED=false`) |
| `/etc/profile.d/informacion.sh` | Bind-mount del fichero del host |

## Ficheros

```
cicd/podman/
├── Containerfile.orchestrator
├── Containerfile.web
├── build-and-export.ps1     # local Windows: build + save a dist/*.tar
├── build-and-export.sh      # local Linux/macOS
├── load-and-run.sh          # SUSE: load + host runner + Quadlet
├── run.sh                   # SUSE: fallback manual (start/stop/status/logs)
├── quadlet/
│   ├── cicd.container       # unidad Quadlet del orquestador
│   ├── cicd-web.container   # unidad Quadlet de la Web
│   └── install_quadlet.sh   # instala/arranca o desinstala las unidades
├── compose.yaml             # alternativa con podman-compose (opcional)
├── env.example              # variables de run.sh
└── host/
    ├── build_runner.sh          # ejecuta la compilación en el host
    ├── sonar_runner.sh          # ejecuta sonar-scanner en el host
    └── install_host_runner.sh   # instala runners + wrapper + SSH local
```

La compilación **y el análisis Sonar** se ejecutan en el host SUSE: el
contenedor solo prepara el workspace, orquesta y consulta el Quality Gate. Así
se usa el toolchain y el truststore corporativo del host, sin replicarlos en la
imagen.

## Prerrequisitos

### Local (build)
- Podman **o** Docker. Si no hay Podman, el script usa Docker y el `.tar`
  resultante se carga igual con `podman load`.
- Acceso a Docker Hub (`opensuse/leap:15.5`) y a GitHub (binario `yq`).

### SUSE (runtime)
- `podman` 4.4+ rootless (Quadlet) y `sshd` activo:
  `sudo systemctl enable --now sshd`. En el entorno verificado: Podman 4.9.5.
- `yq` en el host (lo usa `run.sh` para derivar rutas y `ci_cd.sh` para la
  configuración).
- El UID/GID de `agent` debe coincidir con `AGENT_UID`/`AGENT_GID` de la
  imagen. Usa `id -u agent` / `id -g agent`.
- Clave `~/.ssh/id_rsa` (se autoriza a sí misma para el SSH local).
- El clon del repo en `/home/agent/GALTTCMC` y el directorio de compilación
  (`git.compile_path`, p. ej. `/home/agent/V08.XX.XX.XX`).
- `config/.env` con las credenciales (no está en el repo; ver Paso 2).
- Misma conectividad que el host actual (git, SonarQube, vCenter y VM destino):
  el contenedor usa `--network host`.
- Si necesitas acceso externo a la Web:
  `sudo firewall-cmd --permanent --add-port=8080/tcp && sudo firewall-cmd --reload`.
- **Detener los servicios systemd antiguos** para no duplicar el pipeline:
  `sudo systemctl disable --now cicd cicd-web web.service`
  (`web.service` es el del RPM y también ocupa el 8080).

## Paso 1 — construir y exportar (local)

```powershell
# Windows (ajusta UID/GID a los del usuario agent del host)
pwsh ./cicd/podman/build-and-export.ps1 -AgentUid <UID> -AgentGid <GID>

# Linux/macOS
AGENT_UID=<UID> AGENT_GID=<GID> ./cicd/podman/build-and-export.sh
```

Usa `id -u agent` / `id -g agent`. Si no coinciden con
`id -u agent`/`id -g agent`, los bind-mounts fallarán por permisos.

Genera:

```
cicd/podman/dist/cicd-orchestrator.tar
cicd/podman/dist/cicd-web.tar
```

## Paso 2 — copiar a SUSE

### Máquina nueva (no existe `/home/agent/cicd`)

```bash
ssh agent@<DEV_HOST_IP> 'mkdir -p /home/agent/cicd/db /home/agent/cicd/logs'
scp -r cicd/config cicd/utils cicd/python cicd/scripts cicd/web cicd/podman \
    agent@<DEV_HOST_IP>:/home/agent/cicd/
scp cicd/db/init_db.sql agent@<DEV_HOST_IP>:/home/agent/cicd/db/

# Credenciales: no vienen en el repo
ssh agent@<DEV_HOST_IP> \
  'cp /home/agent/cicd/config/.env.example /home/agent/cicd/config/.env && chmod 600 /home/agent/cicd/config/.env'
# Editar /home/agent/cicd/config/.env: GIT_USERNAME, GIT_PASSWORD,
# SONAR_TOKEN, VCENTER_USER, VCENTER_PASSWORD, SECRET_KEY.
```

### Actualización (ya existe `/home/agent/cicd`)

```bash
# Solo scripts/runners e imágenes; NO copies db/pipeline.db ni logs/
scp -r cicd/podman/* agent@<DEV_HOST_IP>:/home/agent/cicd/podman/
```

> No copies `db/pipeline.db`, `logs/` ni `config/ci_cd_config.yaml` desde local
> sobre una instalación en marcha: sobrescribirías el estado y la configuración
> del host. La configuración viva es la del host (se monta en el contenedor).

## Paso 3 — cargar y arrancar (SUSE)

```bash
cd /home/agent/cicd/podman
./load-and-run.sh
```

> Antes, comprueba que `cicd.service`, `cicd-web.service` y `web.service` están
> parados; si no, el contenedor web no podrá usar el 8080. Para un arranque
> gradual: `systemctl --user start cicd.service` o
> `systemctl --user start cicd-web.service`.

Esto hace: `podman load` de ambas imágenes, instala los runners en
`~/cicd-host/`, prepara el SSH local e instala/arranca las unidades **Quadlet**
(runtime oficial). Si Quadlet no estuviera disponible, `load-and-run.sh` usa
`run.sh` como fallback.

## Runtime oficial: Quadlet

```bash
cd /home/agent/cicd/podman/quadlet
./install_quadlet.sh
```

- Unidades: `~/.config/containers/systemd/cicd.container` y `cicd-web.container`.
- Gestión: `systemctl --user status|start|stop|restart cicd.service cicd-web.service`.
- Logs: `journalctl --user -u cicd -f` (además de los ficheros de `logs/`).
- Autoarranque: `[Install] WantedBy=default.target` + `loginctl enable-linger agent`.
- Las unidades aparecen como `generated` en `systemctl --user is-enabled`; es
  normal: se arrancan con `start`, no con `enable`.
- El instalador deshabilita `podman-restart.service` (ya no hace falta).
- `run.sh` queda como fallback manual: para usarlo, antes
  `systemctl --user stop cicd.service cicd-web.service`.

## Verificación

```bash
systemctl --user status cicd.service cicd-web.service
journalctl --user -u cicd -f          # logs del daemon (journald)
journalctl --user -u cicd-web -f      # logs de gunicorn
tail -f /home/agent/cicd/logs/pipeline_$(date +%Y%m%d).log
tail -f /home/agent/cicd/logs/web_access.log

curl -s -o /dev/null -w 'login=%{http_code}\n' http://127.0.0.1:8080/login
podman exec -it cicd /home/agent/cicd/ci_cd.sh verify
podman exec -it cicd /home/agent/cicd/ci_cd.sh status
```

Prueba de compilación delegada (sin desplegar). **Borra y recrea el
`compile_path` real**, no la ejecutes con un pipeline en curso:

```bash
podman exec -it cicd /home/agent/cicd/scripts/compile.sh verify
podman exec -it cicd /home/agent/cicd/scripts/compile.sh prepare
podman exec -it cicd /home/agent/cicd/scripts/compile.sh build
podman exec -it cicd /home/agent/cicd/scripts/compile.sh validate
```

Señales de que la delegación funciona:

- Compilación: `Compilación delegada al host: ...` y
  `ps -ef | grep build_DVDs` en el host.
- Sonar: `Análisis SonarQube delegado al host: ...` y
  `[sonar_runner] host=... java=... branch=...`.

Pipeline completo (redespliega en la VM destino):

```bash
podman exec -d cicd /bin/bash -lc \
  '/home/agent/cicd/ci_cd.sh --tag <TAG> > /home/agent/cicd/logs/manual_<TAG>.log 2>&1'
```

## Arranque por partes

```bash
# Quadlet (runtime oficial)
systemctl --user start cicd.service
systemctl --user start cicd-web.service
systemctl --user stop cicd.service cicd-web.service

# Fallback run.sh (parar antes las unidades Quadlet)
systemctl --user stop cicd.service cicd-web.service
WEB_PORT=8081 ./run.sh start-web   # solo Web en 8081
./run.sh start-orchestrator        # solo el daemon
```

> El orquestador **no debe arrancarse** mientras el `cicd.service` antiguo
> (system-level) siga activo: habría dos daemons compitiendo por los mismos
> tags y la misma VM.

## Parar / desmontar

```bash
# Quadlet
systemctl --user stop cicd.service cicd-web.service
./quadlet/install_quadlet.sh uninstall

# Fallback run.sh
./run.sh stop
./run.sh rm        # elimina contenedores (la BD/logs del host se conservan)
```

## Cutover desde systemd

```bash
# 1. Parar los servicios antiguos (requiere sudo)
#    OJO: si existe ADEMAS `web.service` (RPM), que ocupa el 8080.
sudo systemctl disable --now cicd cicd-web web.service
# 2. Instalar/arrancar Quadlet (runtime oficial)
./quadlet/install_quadlet.sh
# 3. Comprobar
systemctl --user status cicd.service cicd-web.service
journalctl --user -u cicd -f
```

Rollback: `./quadlet/install_quadlet.sh uninstall && sudo systemctl enable --now cicd web.service`
(usa `cicd-web` si esa es la unidad web de tu instalación).

## Persistencia tras reinicio (Quadlet)

```bash
loginctl enable-linger agent
```

Quadlet genera las unidades y los enlaces de `default.target` en cada arranque;
con linger, el gestor de usuario arranca sin sesión. **No** hace falta
`podman-restart.service` (el instalador de Quadlet lo deshabilita). Las
unidades aparecen como `generated` en `systemctl --user is-enabled`; es normal.

## Variables (run.sh)

| Variable | Default | Debe coincidir con |
|---|---|---|
| `CICD_DIR` | `/home/agent/cicd` | `general.working_directory` |
| `REPO_DIR` | se deriva de `git.repo_local_path` | `git.repo_local_path` |
| `COMPILE_PATH` | se deriva de `git.compile_path` | `git.compile_path` |
| `SSH_DIR` | directorio de `target_vm.ssh_key_path` | `target_vm.ssh_key_path` |
| `PROFILE_SCRIPT` | se deriva de `notifications.profile_script` | `notifications.profile_script` |

`run.sh` lee esos valores de `config/ci_cd_config.yaml` con `yq`; las variables
de entorno solo hacen falta para sobrescribirlos.

## Credenciales y CA corporativa

- `config/.env` debe incluir `GIT_USERNAME` con el propietario del token. En
  Usa el usuario propietario del token; si falta, `git_monitor.sh` usa `agent` y Git
  devuelve 401.
- `run.sh` monta `/etc/ssl/ca-bundle.pem` del host y exporta `GIT_SSL_CAINFO`,
  `CURL_CA_BUNDLE`, `SSL_CERT_FILE` y `REQUESTS_CA_BUNDLE`. Sin esto, el
  contenedor no confía en la CA corporativa (`<CORPORATE_CA>`) y git
  falla con `unable to get local issuer certificate`.

## Limitaciones conocidas del PoC

- **`wall`** no alcanza las TTY del host desde el contenedor. Se desactiva por
  defecto; el resto de notificaciones (`profile.d`, estado en BD) sí funciona.
  Para `wall` real haría falta un pequeño hook en el host (systemd path/socket).
- **SQLite** debe vivir en disco local del host (no NFS/SMB): WAL y `flock`
  no son fiables en red.
- **UID**: las imágenes se construyen con `AGENT_UID`/`AGENT_GID` (1000 por
  defecto; usa el UID/GID de `agent`). Deben coincidir con `id -u agent` /
  `id -g agent` o los bind-mounts fallarán por permisos.
- **`podman --userns=keep-id`** requiere podman rootless (ejecutar como
  `agent`, no con `sudo`).
- La imagen lleva el código dentro: para actualizar scripts hay que
  reconstruir y reexportar (o montar `cicd/` en modo desarrollo).
- Si usas `podman-compose`, `userns_mode: keep-id` puede no estar soportado en
  versiones antiguas; la ruta recomendada es `run.sh`.
- El análisis Sonar se ejecuta en el host (`CICD_SONAR_MODE=ssh`) con su
  Java/truststore. En el contenedor solo se extrae `mmi.jar` con `unzip`
  (instalado) y se consulta la API del Quality Gate.

## Estado verificado (entorno de desarrollo)

- Podman 4.9.5 rootless; usuario `agent` con el UID/GID del host.
- Imágenes construidas con `-AgentUid <UID> -AgentGid <GID>`.
- `GIT_USERNAME=<GIT_USER>` en `config/.env` (el token pertenece a ese usuario)
  y credential helper en el clon para que funcione `git fetch`.
- CA corporativa montada desde `/etc/ssl/ca-bundle.pem` del host.
- Contenedores con `restart=always`; persistencia con
  `loginctl enable-linger agent` y
  `systemctl --user enable --now podman-restart.service`.
- `cicd.service`, `cicd-web.service` y `web.service` (RPM) deshabilitados.
- Verificado: detección de tags, parada limpia por `SIGTERM`, Web en `:8080`
  (interna y externa) y arranque automático de ambos contenedores tras
  simular un reinicio.
- Pipeline completo verificado con `<TAG>` (deployment <id>):
  compilación y Sonar en el host, checksums/vCenter/deploy en el contenedor,
  `success` en ~13m. Quality Gate `ERROR` pero `allow_override=true`.
- Runtime oficial con **Quadlet**: `systemctl --user start/stop/status` y
  autoarranque vía `default.target` (unidades `generated`) + linger.
  `podman-restart.service` deshabilitado; `run.sh` queda como fallback.
