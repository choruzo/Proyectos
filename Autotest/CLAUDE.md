# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GALTTCMC CI/CD Pipeline** – Daemon systemd en SUSE Linux 15 (`172.30.188.137`, user `agent`) que monitoriza tags Git cada 5 minutos, compila un instalador DVD/ISO, valida calidad con SonarQube, sube el ISO a vCenter y lo despliega por SSH a una VM destino.

Todo el código vive en `cicd/`. Los archivos en la raíz son solo documentación de referencia (`Aclaraciones.md`, `Pasos_manuales.md`, `aplicar_fix.ps1`).

## Commands

Todos los comandos se ejecutan desde `cicd/` en la máquina Linux destino:

```bash
# Orquestador principal
./ci_cd.sh daemon              # Polling continuo cada 5 minutos
./ci_cd.sh --tag V24_02_15_01  # Procesar tag específico manualmente
./ci_cd.sh status              # Estadísticas de despliegues
./ci_cd.sh verify              # Verificar prerequisites del entorno
./ci_cd.sh init                # Inicializar base de datos SQLite
./ci_cd.sh logs 100            # Ver últimas 100 líneas de log

# Fases individuales
./scripts/git_monitor.sh detect           # Detectar nuevos tags
./scripts/git_monitor.sh checkout <tag>   # Checkout de tag específico
./scripts/compile.sh                      # Pipeline de compilación completo
./scripts/deploy.sh                       # Despliegue SSH completo

# Python API clients
python3.6 python/sonar_check.py config/ci_cd_config.yaml V24_02_15_01
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status
python3.6 python/vcenter_api.py config/ci_cd_config.yaml upload_iso /path/to.iso

# Servicio systemd
sudo ./install_service.sh install
sudo systemctl start cicd
sudo journalctl -u cicd -f

# Empaquetar RPM
./build_rpm.sh    # Genera ~/rpmbuild/RPMS/noarch/cicd-galttcmc-*.rpm
```

## Architecture

### Pipeline de 5 Fases (ejecutadas secuencialmente por `ci_cd.sh`)

1. **Phase 1 – Git Monitor** (`scripts/git_monitor.sh`): `git ls-remote` cada 5 min, filtra por regex `^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$`, verifica tabla `processed_tags` en SQLite para evitar reprocesar.

2. **Phase 2 – Compilación** (`scripts/compile.sh`): Copia fuentes a `/home/agent/compile`, da permisos de ejecución con `find`, ejecuta `Development_TTCF/ttcf/utils/dvds/build_DVDs.sh`, valida salida `InstallationDVD.iso`. Timeout: 3600s.

3. **Phase 3 – SonarQube** (`python/sonar_check.py`): REST API client contra SonarQube. Ejecuta `build-wrapper` + `sonar-scanner` (prebundleados en `utils/`). Umbrales: coverage ≥80%, bugs=0, vulnerabilities=0, code_smells≤10, security_hotspots=0.

4. **Phase 4 – vCenter** (`python/vcenter_api.py`): REST API (sin pyvmomi). Sube ISO al datastore, configura CD-ROM en VM `MCU-CI_CD`, enciende la VM.

5. **Phase 5 – SSH Deploy** (`scripts/deploy.sh`): SSH con clave a `172.30.188.147`, monta ISO en `/mnt/cdrom`, copia a `/root/install`, ejecuta `install.sh "ope 1 - YES yes"`, limpia y notifica.

### Key Files

| Archivo | Propósito |
|---------|-----------|
| `ci_cd.sh` | Orquestador principal; entrada para todas las operaciones |
| `scripts/common.sh` | Librería compartida: `config_get()`, logging, `wait_for()`, `db_query()` |
| `config/ci_cd_config.yaml` | Toda la configuración (endpoints, rutas, umbrales) |
| `config/.env` | Credenciales en runtime (no commitear; usar `.env.example`) |
| `db/init_db.sql` | Schema SQLite: 5 tablas, 2 vistas |
| `utils/` | sonar-scanner + build-wrapper prebundleados (no modificar) |

### Sistema de Configuración

Todos los scripts hacen `source common.sh` y usan `config_get "yaml.path"` para leer `ci_cd_config.yaml`. Las credenciales (`${GIT_PASSWORD}`, `${SONAR_TOKEN}`, `${VCENTER_USER}`, `${VCENTER_PASSWORD}`) se expanden desde `.env`, cargado por el servicio systemd.

### Base de Datos SQLite (`db/pipeline.db`)

Tablas: `deployments` (pipeline runs, status: pending→compiling→analyzing→deploying→success/failed), `build_logs`, `sonar_results`, `execution_log`, `processed_tags`. Vistas: `v_recent_deployments`, `v_deployment_stats`.

```bash
sqlite3 db/pipeline.db "SELECT * FROM v_deployment_stats"
sqlite3 db/pipeline.db "SELECT * FROM deployments ORDER BY started_at DESC LIMIT 10"
```

## Conventions

### Bash
- `set -euo pipefail` obligatorio en todos los scripts
- Logging siempre via `log_info/log_warn/log_error/log_ok/log_debug` (van a stderr, nunca stdout, para no interferir con captura de valores)
- No hacer echo a stdout en funciones de `common.sh`

### Python
- **Solo Python 3.6+** — sin f-strings, usar `.format()` o `%`
- `requests` con control explícito de SSL; `urllib3.disable_warnings(...)` para certificados self-signed
- Reconectar en errores 401 de vCenter (session timeout)

### Credenciales y Configuración
- Cargar `.env` antes de parsear config (las credenciales deben estar en entorno primero)
- Tag format: `MAC_X_VXX_XX_XX_XX` (oficial) o `VXX_XX_XX_XX` (interno); usar regex de config, no hardcodear
- SSH key esperada en `/home/agent/.ssh/id_rsa`

## Setup

```bash
./setup_phase0.sh                              # Setup inicial
cp config/.env.example config/.env            # Rellenar credenciales
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
ssh-copy-id root@172.30.188.147
./ci_cd.sh init
sudo ./install_service.sh install
```

## Infrastructure

| Componente | Dirección | Auth |
|------------|-----------|------|
| Dev machine (SUSE 15) | 172.30.188.137 | agent / gal1$LEO |
| Git repo | https://git.indra.es/git/GALTTCMC/GALTTCMC branch `WORKING_G2G_DEVELOPMENT` | token en `.env` |
| SonarQube | https://sonarqube.indra.es | token en `.env` |
| Target VM "Releases" | 172.30.188.147 | root, SSH key |
| Web UI | http://172.30.188.137:8080 | sin auth (red local) |

## Web UI - Interfaz de Monitorización

### Overview

Aplicación web Flask que consume la base de datos SQLite del pipeline y expone métricas vía REST API + interfaz gráfica.

### Stack
- **Backend**: Flask 2.0.3 + Gunicorn 20.1.0 (WSGI server)
- **Frontend**: Alpine.js 3.x (reactividad), Tailwind CSS (estilos), Chart.js (gráficos)
- **Deployment**: Servicio systemd `cicd-web` en puerto 8080

### Instalación

```bash
cd /home/agent/cicd
sudo ./install_web.sh install      # Instala deps, configura firewall, inicia servicio
sudo systemctl status cicd-web     # Verificar estado
```

### Comandos Web UI

```bash
# Gestión del servicio
sudo systemctl start|stop|restart cicd-web
sudo journalctl -u cicd-web -f

# Logs de aplicación
tail -f logs/web_access.log
tail -f logs/web_error.log

# Desarrollo (modo debug, sin Gunicorn)
cd web && python3.6 app.py

# Desinstalar
sudo ./install_web.sh uninstall
```

### Estructura Web UI

```
web/
├── app.py              # Flask app: rutas, API endpoints, lógica
├── config.py           # Configuración Flask (puerto, debug, etc.)
├── requirements.txt    # Flask, Gunicorn, PyYAML
├── templates/          # Jinja2 templates
│   ├── base.html      # Layout base con navbar, sidebar
│   ├── dashboard.html # Dashboard principal con métricas
│   ├── pipeline_runs.html # Tabla de deployments con paginación
│   ├── logs.html      # Visor de logs con búsqueda
│   └── sonar_results.html # Resultados SonarQube + gráficos
└── static/
    ├── css/style.css  # Estilos custom + scrollbar, badges
    └── js/app.js      # Utilidades JS: toast, formatters, auto-refresh
```

### API Endpoints

Todos bajo `/api/`:

- **Dashboard**: `/api/dashboard/stats`, `/api/dashboard/recent-deployments`, `/api/dashboard/chart-data`
- **Deployments**: `/api/deployments?page=1&status=all`, `/api/deployment/<id>`
- **Logs**: `/api/logs/list`, `/api/logs/view/<filename>?lines=500&search=error`
- **SonarQube**: `/api/sonar/results`, `/api/sonar/trends`

Ejemplo:
```bash
curl http://localhost:8080/api/dashboard/stats | jq
curl http://localhost:8080/api/deployments?status=success&page=1
```

### Convenciones Web UI

- **Python 3.6 compatible** — mismo requisito que el resto del proyecto
- **No usar f-strings**, usar `.format()` o `%`
- **Queries SQLite** con `conn.row_factory = sqlite3.Row` para dict-like access
- **Logging** va a stderr (común con el resto del proyecto)
- **Templates** usan Alpine.js `x-data` para estado reactivo
- **Rutas absolutas** siempre para acceder a `db/`, `logs/`, `config/`

### Troubleshooting Web UI

```bash
# Servicio no inicia
sudo journalctl -u cicd-web -n 50 --no-pager
cd web && python3.6 app.py  # Probar manualmente

# Puerto 8080 en uso
sudo lsof -i :8080
# Cambiar puerto en web.service: Environment="WEB_PORT=9090"

# No se ven datos
ls -la db/pipeline.db  # Verificar que existe y tiene permisos
sqlite3 db/pipeline.db "SELECT COUNT(*) FROM deployments"

# Permisos de logs
chmod 755 logs && chmod 644 logs/*.log
```

Más info: Ver [web/README.md](cicd/web/README.md) para documentación completa de la Web UI.

