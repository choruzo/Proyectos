# CI/CD Pipeline - GALTTCMC

## Project Overview

Automated CI/CD pipeline for GALTTCMC that monitors Git tags, compiles DVDs/ISOs, runs SonarQube quality checks, and deploys to vCenter VMs via SSH. Target environment is SUSE Linux 15 running on 172.30.188.137 as a systemd daemon.

## Architecture

**Pipeline Flow**: Git Monitor → Compile → SonarQube Analysis → vCenter Deploy → SSH Installation → Notifications

- **Orchestrator**: [cicd/ci_cd.sh](cicd/ci_cd.sh) coordinates all phases
- **Modular scripts**: Each phase is a separate script in [cicd/scripts/](cicd/scripts/)
- **Python APIs**: [python/sonar_check.py](cicd/python/sonar_check.py) and [vcenter_api.py](cicd/python/vcenter_api.py) handle external service integration
- **Audit database**: SQLite at `db/pipeline.db` tracks all deployments, build logs, and SonarQube results
- **Systemd service**: [cicd.service](cicd/cicd.service) runs pipeline as daemon with auto-restart

## Critical Conventions

### Bash Scripts
- **Always** use `set -euo pipefail` at script start
- Source `common.sh` for shared functions: `source "$SCRIPT_DIR/common.sh"`
- Use logging functions: `log_info`, `log_error`, `log_ok`, `log_warn`, `log_debug`
- Log to **stderr** (not stdout) to avoid interfering with value capture - common.sh already does this
- Store config in YAML, access via `config_get "path.to.key"` function
- DB queries use `db_query "SELECT ..."` helper from common.sh

### Python Code
- **Python 3.6+ compatibility only** - no f-strings, use `.format()` or `%` formatting
- Use `from __future__ import print_function` for Python 2/3 compatibility habits
- Config loaded via `load_config()` with env var expansion for `${VAR_NAME}` patterns
- All API calls use `requests` library with explicit SSL verification control
- Disable SSL warnings: `urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)`

### Configuration Management
- Master config: [config/ci_cd_config.yaml](cicd/config/ci_cd_config.yaml)
- Credentials: `config/.env` (never commit, use `.env.example` as template)
- Environment variable substitution in YAML: `"${GIT_PASSWORD}"` → replaced by `load_config()` or `yq` in bash
- vCenter uses REST API directly (no pyvmomi dependency)

### Database Schema
- **deployments** table tracks pipeline runs with status: pending → compiling → analyzing → deploying → success/failed
- **build_logs** stores per-phase compilation metrics
- **sonar_results** caches SonarQube quality gate results
- **processed_tags** prevents reprocessing tags
- Use indexed queries (see [db/init_db.sql](cicd/db/init_db.sql) for schema)

## Key Developer Workflows

### Running Pipeline
```bash
# Daemon mode (continuous monitoring every 5 minutes)
./ci_cd.sh daemon

# Manual tag processing
./ci_cd.sh --tag MAC_1_V24_02_15_01

# Check status
./ci_cd.sh status

# Initialize database
./ci_cd.sh init
```

### Testing Individual Phases
```bash
# Test Git monitoring
./scripts/git_monitor.sh detect

# Test compilation only
./scripts/compile.sh

# Test SonarQube check
python3.6 python/sonar_check.py config/ci_cd_config.yaml V24_02_15_01

# Test vCenter operations
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status
```

### Debugging
- Logs organized by date: `logs/pipeline_YYYYMMDD.log`
- Phase-specific logs: `logs/compile_YYYYMMDD_HHMMSS.log`, `logs/deploy_YYYYMMDD_HHMMSS.log`
- Query audit trail: `sqlite3 db/pipeline.db "SELECT * FROM deployments ORDER BY started_at DESC LIMIT 10"`
- Service logs: `journalctl -u cicd.service -f`

### Setup New Environment
1. Run `./setup_phase0.sh` for initial setup
2. Copy `config/.env.example` to `config/.env` and fill credentials
3. Generate SSH key: `ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa`
4. Copy public key to target VM: `ssh-copy-id root@172.30.188.147`
5. Initialize database: `./ci_cd.sh init`
6. Install service: `sudo ./install_service.sh`

## Integration Points

### Git Repository
- URL: `https://git.indra.es/git/GALTTCMC/GALTTCMC`
- Branch: `WORKING_G2G_DEVELOPMENT`
- Tag pattern: `^(MAC_[0-9]+_)?V[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$` (e.g., MAC_1_V24_02_15_01 or V24_02_15_01)
- Uses basic auth via credentials in `.env`

### SonarQube API
- Base URL: `https://sonarqube.indra.es`
- Authentication: Token-based (basic auth with token as username, empty password)
- Quality thresholds enforced: coverage ≥80%, bugs=0, vulnerabilities=0, security_hotspots=0, code_smells≤10
- Override flag: `allow_override: false` in config blocks deployment on quality gate failure

### vCenter REST API
- Uses session-based authentication (not pyvmomi)
- Operations: upload ISO to datastore, configure VM CD-ROM, power on/off
- Datastore path format: `[NAS_LIBRERIA] P27/Versiones/TAG_NAME.iso`
- Target VM: `Releases` in datacenter `TTCF_desar`

### Target VM Deployment
- SSH to 172.30.188.147 as root
- Process: mount ISO → copy contents → execute `install.sh`
- ISO mounted at `/mnt/cdrom` on target VM
- Installation copies to `/opt/releases/TAG_NAME/`

## Common Pitfalls

- **Don't echo to stdout in common.sh functions** - logs go to stderr to preserve value returns
- **Always load .env before config parsing** - credentials need to be in environment first
- **Tag format validation** - Use regex pattern from config, don't hardcode
- **SSH key location** - Expected at `/home/agent/.ssh/id_rsa`, configurable via `TARGET_VM_KEY`
- **Python version** - Must work with Python 3.6 (SUSE 15 default), no modern syntax
- **SQLite concurrent writes** - Use transactions for multi-row inserts
- **vCenter session timeout** - Reconnect on 401 errors in vcenter_api.py

## File Organization

```
cicd/
├── ci_cd.sh                 # Main orchestrator, entry point
├── scripts/
│   ├── common.sh            # Shared: logging, config parsing, DB helpers
│   ├── git_monitor.sh       # Phase 1: Detect new tags, checkout
│   ├── compile.sh           # Phase 2: Build DVDs/ISOs
│   ├── deploy.sh            # Phase 4: SSH deployment
│   └── notify.sh            # Phase 5: Wall notifications, profile update
├── python/
│   ├── sonar_check.py       # Phase 3: SonarQube API, threshold validation
│   └── vcenter_api.py       # vCenter REST API wrapper
├── config/
│   ├── ci_cd_config.yaml    # Main configuration (commit)
│   ├── .env                 # Credentials (NEVER commit)
│   └── sonar-project.properties  # SonarQube project config
└── db/
    ├── init_db.sql          # Database schema
    └── pipeline.db          # SQLite audit database
```

## Extending the Pipeline

- **Add new phase**: Create script in `scripts/`, source `common.sh`, integrate in `ci_cd.sh` run_pipeline()
- **Modify quality gates**: Update `sonarqube.thresholds` in YAML, adjust validation in sonar_check.py
- **Add notifications**: Extend `notify.sh` (currently supports wall + /etc/profile.d/informacion.sh)
- **New DB tables**: Add to init_db.sql with indices, create views for common queries

## Web UI - Monitoring Interface

### Overview

Flask web application that provides real-time visualization of the CI/CD pipeline status. Built with Flask 2.0.3 + Gunicorn, Alpine.js for reactivity, Tailwind CSS for styling, and Chart.js for metrics visualization.

### Installation

```bash
cd /home/agent/cicd
sudo ./install_web.sh install
```

Access at: `http://172.30.188.137:8080`

### Architecture

- **Backend**: Flask app (`web/app.py`) with REST API endpoints
- **Frontend**: Jinja2 templates + Alpine.js for reactive components
- **Data Source**: Direct SQLite queries to `db/pipeline.db`
- **Deployment**: systemd service `cicd-web` with Gunicorn WSGI server
- **Port**: 8080 (configurable via `WEB_PORT` env var)

### Key Features

1. **Dashboard** - Real-time metrics, success rate, recent deployments, charts
2. **Pipeline Runs** - Full deployment history with filtering and pagination
3. **Logs Viewer** - Live log viewing with search and line-filtering
4. **SonarQube Results** - Quality metrics table and trend charts
5. **Dark Mode** - User preference persisted in localStorage
6. **Auto-refresh** - Optional automatic data refresh

### File Structure

```
web/
├── app.py              # Flask routes, API endpoints, database queries
├── config.py           # Flask configuration (host, port, debug)
├── requirements.txt    # Flask==2.0.3, Gunicorn==20.1.0, PyYAML==6.0.1
├── templates/
│   ├── base.html      # Base layout with navbar, sidebar, dark mode toggle
│   ├── dashboard.html # Dashboard with stats cards and charts
│   ├── pipeline_runs.html # Deployments table with modal details
│   ├── logs.html      # Log viewer with file selector and search
│   └── sonar_results.html # SonarQube results table and trend charts
└── static/
    ├── css/style.css  # Custom CSS (scrollbar, badges, transitions)
    └── js/app.js      # Utility functions (toast, formatters, clipboard)
```

### API Endpoints

All under `/api/`:

- `GET /api/dashboard/stats` - Total deployments, success rate, last 24h, avg duration
- `GET /api/dashboard/recent-deployments` - Last 10 deployments
- `GET /api/dashboard/chart-data` - Last 7 days deployment data for charts
- `GET /api/deployments?page=1&per_page=20&status=all` - Paginated deployment list
- `GET /api/deployment/<id>` - Full deployment details including build logs and SonarQube results
- `GET /api/logs/list` - Available log files with size and modification date
- `GET /api/logs/view/<filename>?lines=500&search=error` - Log content with optional filtering
- `GET /api/sonar/results` - Last 50 SonarQube analysis results
- `GET /api/sonar/trends` - Last 10 successful deployments SonarQube metrics for charting

### Service Management

```bash
# Status
sudo systemctl status cicd-web

# Logs
sudo journalctl -u cicd-web -f
tail -f /home/agent/cicd/logs/web_access.log
tail -f /home/agent/cicd/logs/web_error.log

# Control
sudo systemctl start|stop|restart cicd-web

# Uninstall
sudo ./install_web.sh uninstall
```

### Development Mode

For testing without Gunicorn:

```bash
cd /home/agent/cicd/web
python3.6 app.py
# Runs on http://0.0.0.0:8080 with Flask dev server
```

### Web UI Conventions

- **Python 3.6 compatibility** - No f-strings, use `.format()` or `%` formatting
- **Error handling** - All API endpoints return JSON with error messages on failure
- **Database access** - Uses `sqlite3.Row` factory for dict-like row access
- **Relative imports** - Flask app uses absolute paths to `db/`, `logs/`, `config/`
- **Frontend state** - Alpine.js `x-data` functions manage component state reactively
- **Logging** - Application logs go to stderr (same as rest of project)
- **Security** - Service runs with `NoNewPrivileges`, `ProtectSystem=strict`, limited file access

### Common Issues

- **Port 8080 in use**: Change `WEB_PORT` in `/etc/systemd/system/cicd-web.service`, then `systemctl daemon-reload && systemctl restart cicd-web`
- **No data displayed**: Verify `db/pipeline.db` exists and has deployments: `sqlite3 db/pipeline.db "SELECT COUNT(*) FROM deployments"`
- **Permission denied on logs**: `chmod 755 logs && chmod 644 logs/*.log`
- **Service won't start**: Check `journalctl -u cicd-web -n 50` for errors, verify Python deps installed

Full documentation: [web/README.md](cicd/web/README.md)
