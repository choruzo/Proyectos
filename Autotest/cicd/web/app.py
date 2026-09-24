#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web UI for GALTTCMC CI/CD Pipeline
Visualiza logs, deployments, metricas de SonarQube y estado del pipeline
"""

from __future__ import print_function
import os
import sys
import time
import sqlite3
import subprocess
from collections import deque
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, g
import secrets
import hmac
from werkzeug.security import check_password_hash, generate_password_hash
import yaml

# Añadir path del proyecto para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config.from_object('config')

# Rutas de configuracion
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'pipeline.db')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'ci_cd_config.yaml')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

GENERIC_ERROR = 'Error interno del servidor. Consulta el log de la Web UI.'

# Fases del pipeline en orden de ejecucion. Las claves son las que ci_cd.sh
# guarda en deployments.current_phase / failed_phase y en execution_log.phase.
PIPELINE_PHASES = [
    ('checkout',   'Git Checkout'),
    ('compile',    'Compilación'),
    ('sonarqube',  'SonarQube'),
    ('checksums',  'Checksums y Doxygen'),
    ('vcenter',    'vCenter'),
    ('ssh_deploy', 'Instalación SSH'),
]
PHASE_KEYS = [p[0] for p in PIPELINE_PHASES]

# Ejecuciones anteriores a current_phase/failed_phase: la fase se deduce del
# prefijo "[fase]" de error_message que escribe cleanup_on_error.
LEGACY_ERROR_PHASES = {
    'checkout': 'checkout',
    'compile': 'compile',
    'sonarqube_prepare': 'sonarqube',
    'sonarqube': 'sonarqube',
    'checksums': 'checksums',
    'deploy': 'vcenter',
    'deploy_upload': 'vcenter',
    'deploy_snapshot': 'vcenter',
    'deploy_snapshot_wait': 'vcenter',
    'deploy_cdrom': 'vcenter',
    'deploy_power': 'vcenter',
    'deploy_power_wait': 'vcenter',
    'deploy_ssh': 'ssh_deploy',
}

# Fase en curso deducida del estado global (ejecuciones sin current_phase)
LEGACY_STATUS_PHASES = {
    'compiling': 'compile',
    'analyzing': 'sonarqube',
    'deploying': 'vcenter',
}

RUNNING_STATUSES = ('pending', 'compiling', 'analyzing', 'deploying')
VALID_STATUSES = set(RUNNING_STATUSES) | {'success', 'failed'}


class BadRequest(Exception):
    """Parametro de peticion no valido (se responde con 400)."""
    pass


def load_config():
    """Carga configuracion desde ci_cd_config.yaml"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        app.logger.exception('Error cargando %s', CONFIG_PATH)
        return {}


def get_db_connection():
    """Obtiene conexion a SQLite"""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    """Columnas de una tabla (para convivir con BDs sin migrar)."""
    return set(row[1] for row in conn.execute('PRAGMA table_info({})'.format(table)).fetchall())


def api_error(context):
    """Registra la excepcion en curso en el servidor y devuelve un 500 generico."""
    app.logger.exception('Error en %s', context)
    return jsonify({'error': GENERIC_ERROR}), 500


def int_arg(name, default, minimum=None, maximum=None):
    """Lee un parametro entero de la query string y lo acota a [minimum, maximum]."""
    raw = request.args.get(name, '')
    if raw == '':
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise BadRequest('El parámetro "{}" debe ser un número entero.'.format(name))
    if minimum is not None and value < minimum:
        raise BadRequest('El parámetro "{}" debe ser mayor o igual que {}.'.format(name, minimum))
    if maximum is not None and value > maximum:
        value = maximum
    return value


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    return jsonify({'error': str(error)}), 400


# ==================== SCHEMA ====================

def _add_column(conn, table, column, ddl):
    """ALTER TABLE ADD COLUMN tolerante a que otro worker lo haya hecho antes."""
    if column in table_columns(conn, table):
        return False
    try:
        conn.execute('ALTER TABLE {} ADD COLUMN {} {}'.format(table, column, ddl))
        return True
    except sqlite3.OperationalError as e:
        if 'duplicate column' in str(e).lower():
            return False
        raise


def ensure_schema():
    """Migraciones ligeras e idempotentes que necesita la Web UI."""
    if not os.path.exists(DB_PATH):
        return
    conn = get_db_connection()
    try:
        tables = set(r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall())

        if 'web_users' in tables:
            _add_column(conn, 'web_users', 'password_changed_at', 'TEXT')
            if _add_column(conn, 'web_users', 'is_admin', 'INTEGER NOT NULL DEFAULT 0'):
                # Primer arranque con roles: 'admin' (o el usuario activo mas
                # antiguo) pasa a ser administrador para no perder la gestion.
                conn.execute("UPDATE web_users SET is_admin = 1 WHERE username = 'admin'")
                if not conn.execute('SELECT 1 FROM web_users WHERE is_admin = 1').fetchone():
                    conn.execute(
                        'UPDATE web_users SET is_admin = 1 WHERE id = '
                        '(SELECT MIN(id) FROM web_users WHERE is_active = 1)')

        conn.execute(
            """CREATE TABLE IF NOT EXISTS web_login_attempts (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   username TEXT,
                   ip TEXT,
                   attempted_at TEXT DEFAULT (datetime('now'))
               )""")
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_web_login_attempts_time ON web_login_attempts(attempted_at)')

        if 'deployments' in tables:
            _add_column(conn, 'deployments', 'current_phase', 'TEXT')
            _add_column(conn, 'deployments', 'failed_phase', 'TEXT')
        conn.commit()
    finally:
        conn.close()


try:
    ensure_schema()
except Exception:
    app.logger.exception('No se pudo aplicar la migracion del schema de la Web UI')


# ==================== AUTH ====================

def generate_csrf_token():
    """Genera o recupera el CSRF token de la sesion actual."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def password_stamp():
    """Marca que cambia con cada cambio de contraseña e invalida las sesiones previas."""
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.context_processor
def inject_globals():
    """Inyecta csrf_token y el usuario actual en todos los templates."""
    user = getattr(g, 'user', None)
    return {
        'csrf_token': generate_csrf_token(),
        'current_user': {
            'username': user['username'] if user else '',
            'is_admin': bool(user and user['is_admin']),
        },
    }


def _end_session(reason):
    session.clear()
    if request.path.startswith('/api/'):
        return jsonify({'error': reason}), 401
    return redirect(url_for('login', next=request.path))


@app.before_request
def require_login():
    """Verifica autenticacion en todas las rutas excepto /login y archivos estaticos.

    El usuario se relee de la BD en cada peticion: desactivarlo, cambiarle la
    contraseña o que caduque la sesion surte efecto inmediato.
    """
    g.user = None
    public_endpoints = {'login', 'logout', 'static'}
    if request.endpoint in public_endpoints:
        return None
    if not session.get('authenticated'):
        return _end_session('Authentication required')

    max_age = app.config['PERMANENT_SESSION_LIFETIME'].total_seconds()
    if time.time() - session.get('login_at', 0) > max_age:
        return _end_session('Sesión caducada')

    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT username, is_active, is_admin, password_changed_at FROM web_users WHERE username = ?',
            (session.get('username', ''),)
        ).fetchone()
    finally:
        conn.close()
    if not user or not user['is_active']:
        return _end_session('Usuario no válido')
    if (user['password_changed_at'] or '') != session.get('pwd_stamp', ''):
        return _end_session('La contraseña ha cambiado, inicia sesión de nuevo')
    g.user = user
    return None


def admin_required(view):
    """Restringe una ruta a usuarios administradores."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not g.user or not g.user['is_admin']:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Se requieren permisos de administrador.'}), 403
            return render_template('403.html'), 403
        return view(*args, **kwargs)
    return wrapper


def format_datetime(dt_str):
    """Formatea datetime string para display"""
    if not dt_str:
        return 'N/A'
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%d %H:%M')
    except (TypeError, ValueError):
        return dt_str


def format_seconds(total):
    """Segundos -> 'Xm Ys'"""
    minutes, seconds = divmod(int(total), 60)
    if minutes > 0:
        return "{}m {}s".format(minutes, seconds)
    return "{}s".format(seconds)


def parse_db_datetime(value):
    """Timestamp de SQLite (datetime('now')) -> datetime, o None."""
    if not value:
        return None
    try:
        return datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return None


def calculate_duration(start, end):
    """Calcula duracion entre dos timestamps"""
    dt_start = parse_db_datetime(start)
    dt_end = parse_db_datetime(end)
    if not dt_start or not dt_end:
        return 'N/A'
    return format_seconds((dt_end - dt_start).total_seconds())


# ==================== LOGIN RATE LIMIT ====================

def client_ip():
    return request.remote_addr or 'unknown'


def login_blocked(conn, username, ip):
    """Devuelve un mensaje si el usuario o la IP han superado los fallos permitidos."""
    window = '-{} minutes'.format(int(app.config['LOGIN_WINDOW_MINUTES']))
    user_failures = conn.execute(
        "SELECT COUNT(*) FROM web_login_attempts WHERE username = ? AND attempted_at >= datetime('now', ?)",
        (username, window)
    ).fetchone()[0]
    ip_failures = conn.execute(
        "SELECT COUNT(*) FROM web_login_attempts WHERE ip = ? AND attempted_at >= datetime('now', ?)",
        (ip, window)
    ).fetchone()[0]
    if (user_failures >= app.config['LOGIN_MAX_FAILURES_PER_USER'] or
            ip_failures >= app.config['LOGIN_MAX_FAILURES_PER_IP']):
        return ('Demasiados intentos fallidos. Espera {} minutos antes de volver a intentarlo.'
                .format(app.config['LOGIN_WINDOW_MINUTES']))
    return None


def register_login_failure(conn, username, ip):
    conn.execute('INSERT INTO web_login_attempts (username, ip) VALUES (?, ?)', (username, ip))
    conn.execute("DELETE FROM web_login_attempts WHERE attempted_at < datetime('now', '-1 day')")
    conn.commit()


# ==================== ROUTES: AUTH ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Pagina de login con proteccion CSRF y limitacion de intentos."""
    error = None
    if request.method == 'POST':
        form_token = request.form.get('csrf_token', '')
        session_token = session.get('csrf_token', '')
        if not session_token or not hmac.compare_digest(session_token, form_token):
            error = 'Token de seguridad invalido. Recarga la pagina e intentalo de nuevo.'
        else:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            if not username or not password:
                error = 'Introduce usuario y contrasena.'
            else:
                conn = None
                try:
                    conn = get_db_connection()
                    ip = client_ip()
                    error = login_blocked(conn, username, ip)
                    if not error:
                        user = conn.execute(
                            'SELECT password_hash, password_changed_at FROM web_users '
                            'WHERE username = ? AND is_active = 1',
                            (username,)
                        ).fetchone()
                        if user and check_password_hash(user['password_hash'], password):
                            conn.execute(
                                "UPDATE web_users SET last_login = datetime('now') WHERE username = ?",
                                (username,)
                            )
                            conn.execute('DELETE FROM web_login_attempts WHERE username = ?', (username,))
                            conn.commit()
                            session.clear()
                            session.permanent = True
                            session['authenticated'] = True
                            session['username'] = username
                            session['login_at'] = time.time()
                            session['pwd_stamp'] = user['password_changed_at'] or ''
                            next_url = request.args.get('next', '/')
                            # Evitar open redirect: solo permitir rutas relativas del mismo origen
                            if not next_url.startswith('/') or next_url.startswith('//'):
                                next_url = '/'
                            return redirect(next_url)
                        register_login_failure(conn, username, ip)
                        app.logger.warning('Login fallido: usuario=%s ip=%s', username, ip)
                        error = 'Usuario o contrasena incorrectos.'
                except Exception:
                    app.logger.exception('Error en login')
                    error = 'Error de autenticacion. Contacte al administrador.'
                finally:
                    if conn is not None:
                        conn.close()
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Cierra la sesion activa."""
    session.clear()
    return redirect(url_for('login'))


# ==================== ROUTES: PAGES ====================

@app.route('/')
def index():
    """Dashboard principal"""
    return render_template('dashboard.html')


@app.route('/pipeline-runs')
def pipeline_runs():
    """Vista de pipeline runs"""
    return render_template('pipeline_runs.html')


@app.route('/logs')
def logs():
    """Vista de logs"""
    return render_template('logs.html')


@app.route('/sonar-results')
def sonar_results():
    """Vista de resultados SonarQube"""
    return render_template('sonar_results.html')


# ==================== FASES ====================

def resolve_phases(dep):
    """Devuelve (fase_en_curso, fase_fallida) de un deployment.

    Usa deployments.current_phase / failed_phase (los escribe ci_cd.sh) y, para
    ejecuciones anteriores a esas columnas, los deduce del estado y del mensaje
    de error. Una fase None significa que no se puede determinar.
    """
    keys = dep.keys()
    status = dep['status']
    current = dep['current_phase'] if 'current_phase' in keys else None
    failed = dep['failed_phase'] if 'failed_phase' in keys else None
    if current not in PHASE_KEYS:
        current = None
    if failed not in PHASE_KEYS:
        failed = None

    if status == 'failed' and failed is None:
        failed = current
        message = dep['error_message'] or ''
        if failed is None and message.startswith('['):
            prefix = message[1:message.find(']')] if ']' in message else ''
            failed = LEGACY_ERROR_PHASES.get(prefix)
    if status in RUNNING_STATUSES and current is None:
        current = LEGACY_STATUS_PHASES.get(status)
    return current, failed


def phase_states(status, current, failed):
    """Estado de cada fase: completed / active / failed / pending / unknown."""
    states = []
    for key in PHASE_KEYS:
        if status == 'success':
            states.append('completed')
        elif status == 'failed':
            if failed is None:
                states.append('unknown')
            elif PHASE_KEYS.index(key) < PHASE_KEYS.index(failed):
                states.append('completed')
            elif key == failed:
                states.append('failed')
            else:
                states.append('pending')
        elif current is None:
            states.append('pending')
        elif PHASE_KEYS.index(key) < PHASE_KEYS.index(current):
            states.append('completed')
        elif key == current:
            states.append('active')
        else:
            states.append('pending')
    return states


def deployment_attempt(d):
    """Nº de ejecución del tag (1 si la BD aún no tiene la columna attempt)."""
    return d['attempt'] if 'attempt' in d.keys() and d['attempt'] else 1


def deployment_summary(d):
    """Fila de deployment para las listas de la API."""
    current, failed = resolve_phases(d)
    return {
        'id': d['id'],
        'tag': d['tag_name'],
        'attempt': deployment_attempt(d),
        'status': d['status'],
        'started_at': format_datetime(d['started_at']),
        'finished_at': format_datetime(d['completed_at']),
        'duration': calculate_duration(d['started_at'], d['completed_at']),
        'error_message': d['error_message'],
        'current_phase': current,
        'failed_phase': failed,
        'phase_states': phase_states(d['status'], current, failed),
    }


def sonar_rows_for_deployment(conn, deployment):
    """Resultados Sonar de un deployment: por deployment_id, o por tag en filas antiguas."""
    return conn.execute(
        """SELECT * FROM sonar_results
           WHERE deployment_id = ? OR (deployment_id IS NULL AND tag = ?)
           ORDER BY created_at DESC, id DESC""",
        (deployment['id'], deployment['tag_name'])
    ).fetchall()


# ==================== API ENDPOINTS ====================

@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """Estadisticas para dashboard"""
    try:
        conn = get_db_connection()

        # Total deployments
        total = conn.execute('SELECT COUNT(*) as count FROM deployments').fetchone()['count']

        # Success rate
        success = conn.execute(
            "SELECT COUNT(*) as count FROM deployments WHERE status = 'success'"
        ).fetchone()['count']

        success_rate = round((success / float(total) * 100), 1) if total > 0 else 0

        # Last 24h deployments
        last_24h = conn.execute(
            """SELECT COUNT(*) as count FROM deployments
               WHERE started_at >= datetime('now', '-1 day')"""
        ).fetchone()['count']

        # Currently running
        running = conn.execute(
            """SELECT COUNT(*) as count FROM deployments
               WHERE status IN ('pending', 'compiling', 'analyzing', 'deploying')"""
        ).fetchone()['count']

        # Average duration (last 10 successful): el LIMIT va en la subconsulta,
        # si no AVG agrega todo el historial antes de aplicarlo.
        avg_duration = conn.execute(
            """SELECT AVG(secs) as avg_seconds FROM (
                   SELECT COALESCE(duration_seconds,
                                   (julianday(completed_at) - julianday(started_at)) * 86400) as secs
                   FROM deployments
                   WHERE status = 'success' AND completed_at IS NOT NULL
                   ORDER BY started_at DESC LIMIT 10
               )"""
        ).fetchone()['avg_seconds']

        avg_duration_str = format_seconds(avg_duration) if avg_duration else 'N/A'

        conn.close()

        return jsonify({
            'total_deployments': total,
            'success_rate': success_rate,
            'last_24h': last_24h,
            'currently_running': running,
            'avg_duration': avg_duration_str
        })
    except Exception:
        return api_error('api_dashboard_stats')


@app.route('/api/dashboard/recent-deployments')
def api_recent_deployments():
    """Ultimos 10 deployments"""
    try:
        conn = get_db_connection()
        deployments = conn.execute(
            """SELECT * FROM deployments
               ORDER BY started_at DESC LIMIT 10"""
        ).fetchall()
        conn.close()

        return jsonify([deployment_summary(d) for d in deployments])
    except Exception:
        return api_error('api_recent_deployments')


@app.route('/api/dashboard/chart-data')
def api_chart_data():
    """Datos para graficos (ultimos 7 dias)"""
    try:
        conn = get_db_connection()

        # Deployments por dia (ultimos 7 dias)
        daily_stats = conn.execute(
            """SELECT
                date(started_at) as date,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
               FROM deployments
               WHERE started_at >= date('now', '-7 days')
               GROUP BY date(started_at)
               ORDER BY date"""
        ).fetchall()

        conn.close()

        labels = []
        success_data = []
        failed_data = []

        for row in daily_stats:
            labels.append(row['date'])
            success_data.append(row['success'])
            failed_data.append(row['failed'])

        return jsonify({
            'labels': labels,
            'datasets': [
                {
                    'label': 'Success',
                    'data': success_data,
                    'backgroundColor': 'rgba(34, 197, 94, 0.5)',
                    'borderColor': 'rgb(34, 197, 94)',
                    'borderWidth': 2
                },
                {
                    'label': 'Failed',
                    'data': failed_data,
                    'backgroundColor': 'rgba(239, 68, 68, 0.5)',
                    'borderColor': 'rgb(239, 68, 68)',
                    'borderWidth': 2
                }
            ]
        })
    except Exception:
        return api_error('api_chart_data')


@app.route('/api/deployments')
def api_deployments():
    """Lista todos los deployments con paginacion"""
    page = int_arg('page', 1, minimum=1)
    per_page = int_arg('per_page', app.config['DEFAULT_PAGE_SIZE'],
                       minimum=1, maximum=app.config['MAX_PAGE_SIZE'])
    status_filter = request.args.get('status', '')
    if status_filter and status_filter != 'all' and status_filter not in VALID_STATUSES:
        return jsonify({'error': 'Invalid status filter'}), 400

    try:
        offset = (page - 1) * per_page

        conn = get_db_connection()

        # Query base
        params = []
        where_clause = ""
        if status_filter and status_filter != 'all':
            where_clause = "WHERE status = ?"
            params.append(status_filter)

        # Total count
        total = conn.execute(
            "SELECT COUNT(*) as count FROM deployments {}".format(where_clause),
            params
        ).fetchone()['count']

        # Deployments
        deployments = conn.execute(
            """SELECT * FROM deployments {}
               ORDER BY started_at DESC LIMIT ? OFFSET ?""".format(where_clause),
            params + [per_page, offset]
        ).fetchall()

        conn.close()

        return jsonify({
            'deployments': [deployment_summary(d) for d in deployments],
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception:
        return api_error('api_deployments')


@app.route('/api/deployment/<int:deployment_id>')
def api_deployment_detail(deployment_id):
    """Detalle de un deployment especifico"""
    try:
        conn = get_db_connection()

        deployment = conn.execute(
            'SELECT * FROM deployments WHERE id = ?', (deployment_id,)
        ).fetchone()

        if not deployment:
            conn.close()
            return jsonify({'error': 'Deployment not found'}), 404

        # Build logs
        build_logs = conn.execute(
            'SELECT * FROM build_logs WHERE deployment_id = ?', (deployment_id,)
        ).fetchall()

        sonar = sonar_rows_for_deployment(conn, deployment)

        conn.close()

        current, failed = resolve_phases(deployment)
        result = {
            'id': deployment['id'],
            'tag': deployment['tag_name'],
            'attempt': deployment_attempt(deployment),
            'status': deployment['status'],
            'started_at': deployment['started_at'],
            'finished_at': deployment['completed_at'],
            'duration': calculate_duration(deployment['started_at'], deployment['completed_at']),
            'error_message': deployment['error_message'],
            'triggered_by': deployment['triggered_by'],
            'current_phase': current,
            'failed_phase': failed,
            'build_logs': [dict(log) for log in build_logs],
            'sonar_results': [dict(row) for row in sonar]
        }

        return jsonify(result)
    except Exception:
        return api_error('api_deployment_detail')


# ==================== LOGS ====================

def _count_lines(f, size):
    """Numero de lineas de los primeros `size` bytes, leyendo por bloques."""
    f.seek(0)
    count = 0
    remaining = size
    last = b''
    while remaining > 0:
        chunk = f.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        count += chunk.count(b'\n')
        remaining -= len(chunk)
        last = chunk[-1:]
    if size > 0 and last != b'\n':
        count += 1
    return count


def _tail_lines(f, end, n):
    """Ultimas n lineas antes del byte `end`, leyendo desde el final por bloques."""
    block = 64 * 1024
    pos = end
    data = b''
    while pos > 0 and data.count(b'\n') <= n:
        size = min(block, pos)
        pos -= size
        f.seek(pos)
        data = f.read(size) + data
    lines = data.splitlines(True)
    if pos > 0 and lines:
        lines = lines[1:]   # la primera puede estar cortada por el bloque
    return lines[-n:] if n > 0 else []


def _complete_end(f, size):
    """Offset tras el ultimo salto de linea: la linea que se esta escribiendo
    se devuelve entera en la siguiente consulta del live-tail."""
    if size == 0:
        return 0
    pos = size
    block = 64 * 1024
    while pos > 0:
        start = max(0, pos - block)
        f.seek(start)
        chunk = f.read(pos - start)
        idx = chunk.rfind(b'\n')
        if idx >= 0:
            return start + idx + 1
        pos = start
    return 0


def _decode(lines):
    return b''.join(lines).decode('utf-8', 'replace')


@app.route('/api/logs/list')
def api_logs_list():
    """Lista archivos de logs disponibles"""
    try:
        log_files = []
        for filename in sorted(os.listdir(LOGS_DIR), reverse=True):
            if filename.endswith('.log'):
                filepath = os.path.join(LOGS_DIR, filename)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)

                log_files.append({
                    'name': filename,
                    'size': size,
                    'size_mb': round(size / 1024.0 / 1024.0, 2),
                    'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

        return jsonify(log_files)
    except Exception:
        return api_error('api_logs_list')


@app.route('/api/logs/view/<filename>')
def api_logs_view(filename):
    """Lee un log sin cargarlo entero en memoria.

    - Sin `offset`: ultimas `lines` lineas (opcionalmente filtradas por `search`)
      y el offset en bytes desde el que seguir con el live-tail.
    - Con `offset`: solo las lineas completas añadidas desde ese byte. Si el
      fichero es mas pequeño que el offset (truncado o rotado) devuelve reset.
    """
    # Validar filename (seguridad)
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({'error': 'Log file not found'}), 404

    max_lines = app.config['MAX_LOG_LINES']
    lines = int_arg('lines', app.config['DEFAULT_LOG_LINES'], minimum=1, maximum=max_lines)
    offset = int_arg('offset', -1, minimum=0)
    search = request.args.get('search', '').lower()

    try:
        with open(filepath, 'rb') as f:
            size = os.fstat(f.fileno()).st_size

            if offset >= 0:
                if offset > size:
                    return jsonify({'filename': filename, 'reset': True, 'offset': 0})
                f.seek(offset)
                data = f.read(min(size - offset, app.config['LIVE_TAIL_MAX_BYTES']))
                cut = data.rfind(b'\n')
                data = data[:cut + 1] if cut >= 0 else b''
                new_offset = offset + len(data)
                return jsonify({
                    'filename': filename,
                    'reset': False,
                    'offset': new_offset,
                    'more': new_offset < size,
                    'new_lines': data.count(b'\n'),
                    'content': data.decode('utf-8', 'replace')
                })

            end = _complete_end(f, size)
            if search:
                # Recorre el fichero linea a linea guardando solo las ultimas coincidencias
                matches = deque(maxlen=lines)
                matched = 0
                file_total = 0
                consumed = 0
                f.seek(0)
                for raw in f:
                    # No pasar de `end`: lo posterior lo entrega el live-tail
                    consumed += len(raw)
                    if consumed > end:
                        break
                    file_total += 1
                    if search in raw.decode('utf-8', 'replace').lower():
                        matched += 1
                        matches.append(raw)
                content_lines = list(matches)
                total_lines = matched
            else:
                content_lines = _tail_lines(f, end, lines)
                file_total = _count_lines(f, end)
                total_lines = file_total

        return jsonify({
            'filename': filename,
            'total_lines': total_lines,
            'file_total_lines': file_total,
            'displayed_lines': len(content_lines),
            'max_lines': max_lines,
            'offset': end,
            'content': _decode(content_lines)
        })
    except Exception:
        return api_error('api_logs_view')


# ==================== SONARQUBE ====================

SONAR_DEPLOYMENT_JOIN = """
    LEFT JOIN deployments d ON d.id = COALESCE(
        sr.deployment_id,
        (SELECT d2.id FROM deployments d2 WHERE d2.tag_name = sr.tag ORDER BY d2.id DESC LIMIT 1))
"""


@app.route('/api/sonar/results')
def api_sonar_results():
    """Resultados SonarQube de todos los analisis, con el estado de su deployment"""
    try:
        conn = get_db_connection()

        # Detectar si las columnas new_* ya existen (migracion gradual)
        has_new_cols = 'new_bugs' in table_columns(conn, 'sonar_results')

        if has_new_cols:
            select_new = 'sr.new_coverage, sr.new_bugs, sr.new_vulnerabilities, sr.new_code_smells, sr.new_security_hotspots,'
        else:
            select_new = 'NULL as new_coverage, NULL as new_bugs, NULL as new_vulnerabilities, NULL as new_code_smells, NULL as new_security_hotspots,'

        results = conn.execute(
            """SELECT sr.id as sonar_id, sr.tag, sr.created_at,
                      sr.coverage, sr.bugs, sr.vulnerabilities, sr.code_smells, sr.security_hotspots,
                      {new_cols}
                      sr.passed, sr.quality_gate_status,
                      d.id as dep_id, d.status as dep_status
               FROM sonar_results sr
               {join}
               ORDER BY sr.created_at DESC, sr.id DESC LIMIT 50""".format(
                new_cols=select_new, join=SONAR_DEPLOYMENT_JOIN)
        ).fetchall()

        conn.close()

        data = []
        for r in results:
            data.append({
                'sonar_id': r['sonar_id'],
                'deployment_id': r['dep_id'],
                'deployment_status': r['dep_status'],
                'tag': r['tag'],
                'date': format_datetime(r['created_at']),
                'quality_gate': r['quality_gate_status'],
                'passed': r['passed'],
                # Overall Code metrics (informativo)
                'coverage': r['coverage'],
                'bugs': r['bugs'],
                'vulnerabilities': r['vulnerabilities'],
                'code_smells': r['code_smells'],
                'security_hotspots': r['security_hotspots'],
                # New Code metrics (los que evalua el Quality Gate)
                'new_coverage': r['new_coverage'],
                'new_bugs': r['new_bugs'],
                'new_vulnerabilities': r['new_vulnerabilities'],
                'new_code_smells': r['new_code_smells'],
                'new_security_hotspots': r['new_security_hotspots'],
            })

        return jsonify(data)
    except Exception:
        return api_error('api_sonar_results')


@app.route('/api/sonar/trends')
def api_sonar_trends():
    """Tendencias de metricas SonarQube (ultimos 10 analisis)"""
    try:
        conn = get_db_connection()

        # Detectar si las columnas new_* existen
        has_new_cols = 'new_bugs' in table_columns(conn, 'sonar_results')

        if has_new_cols:
            select_new = 'sr.new_coverage, sr.new_bugs, sr.new_vulnerabilities, sr.new_code_smells,'
        else:
            select_new = 'NULL as new_coverage, NULL as new_bugs, NULL as new_vulnerabilities, NULL as new_code_smells,'

        results = conn.execute(
            """SELECT sr.coverage, sr.bugs, sr.vulnerabilities, sr.code_smells,
                      {new_cols}
                      sr.tag
               FROM sonar_results sr
               ORDER BY sr.created_at DESC, sr.id DESC LIMIT 10""".format(new_cols=select_new)
        ).fetchall()

        conn.close()

        labels = []
        new_bugs_data = []
        new_vulnerabilities_data = []
        new_code_smells_data = []
        overall_coverage_data = []
        new_coverage_data = []

        for r in reversed(list(results)):
            labels.append(r['tag'])
            new_bugs_data.append(r['new_bugs'] if r['new_bugs'] is not None else 0)
            new_vulnerabilities_data.append(r['new_vulnerabilities'] if r['new_vulnerabilities'] is not None else 0)
            new_code_smells_data.append(r['new_code_smells'] if r['new_code_smells'] is not None else 0)
            overall_coverage_data.append(r['coverage'] if r['coverage'] is not None else 0)
            new_coverage_data.append(r['new_coverage'] if r['new_coverage'] is not None else None)

        return jsonify({
            'labels': labels,
            'coverage': overall_coverage_data,
            'new_coverage': new_coverage_data,
            'bugs': new_bugs_data,
            'vulnerabilities': new_vulnerabilities_data,
            'code_smells': new_code_smells_data
        })
    except Exception:
        return api_error('api_sonar_trends')


@app.route('/api/pipeline/status')
def api_pipeline_status():
    """Estado real del servicio cicd.service via systemctl"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'cicd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        state = result.stdout.decode('utf-8', errors='replace').strip()
        if not state:
            state = 'unknown'
        return jsonify({
            'status': state,
            'running': state == 'active'
        })
    except Exception:
        app.logger.exception('Error consultando systemctl is-active cicd')
        return jsonify({'status': 'unknown', 'running': False})


@app.route('/api/deployment/<int:deployment_id>/phases')
def api_deployment_phases(deployment_id):
    """Datos de fases del pipeline para visualizacion de progreso por fase.

    El estado sale de deployments.current_phase / failed_phase y los tiempos del
    primer registro de cada fase en execution_log (los escribe ci_cd.sh al
    empezar cada fase). Una fase termina cuando empieza la siguiente o, la
    ultima, en completed_at (o ahora, si sigue en curso).
    """
    try:
        conn = get_db_connection()
        deployment = conn.execute(
            'SELECT * FROM deployments WHERE id = ?', (deployment_id,)
        ).fetchone()

        if not deployment:
            conn.close()
            return jsonify({'error': 'Deployment not found'}), 404

        dep_status = deployment['status']

        phase_start = {}
        for row in conn.execute(
                """SELECT phase, MIN(timestamp) as first_ts
                   FROM execution_log
                   WHERE deployment_id = ?
                   GROUP BY phase""",
                (deployment_id,)).fetchall():
            if row['phase'] in PHASE_KEYS:
                phase_start[row['phase']] = parse_db_datetime(row['first_ts'])

        # Build logs for compile sub-phases detail
        build_logs = conn.execute(
            'SELECT phase, start_time, duration, exit_code FROM build_logs WHERE deployment_id = ? ORDER BY start_time',
            (deployment_id,)
        ).fetchall()

        # SonarQube result for the analyze phase
        sonar = sonar_rows_for_deployment(conn, deployment)
        sonar = sonar[0] if sonar else None

        conn.close()

        current, failed = resolve_phases(deployment)
        states = phase_states(dep_status, current, failed)

        finished_at = parse_db_datetime(deployment['completed_at'])
        if finished_at is None and dep_status in RUNNING_STATUSES:
            finished_at = datetime.utcnow()

        phases = []
        for i, (pkey, plabel) in enumerate(PIPELINE_PHASES):
            ph_status = states[i]

            # Duracion: desde su inicio hasta el inicio de la siguiente fase registrada
            duration_secs = None
            start = phase_start.get(pkey)
            if start and ph_status in ('completed', 'failed', 'active'):
                end = None
                for next_key in PHASE_KEYS[i + 1:]:
                    if next_key in phase_start:
                        end = phase_start[next_key]
                        break
                if end is None:
                    end = finished_at
                if end is not None:
                    duration_secs = max(1, int((end - start).total_seconds()))

            # Compile phase: derive total duration from build_logs if exec_log has no timing
            if pkey == 'compile' and build_logs and duration_secs is None:
                total_secs = sum(bl['duration'] or 0 for bl in build_logs)
                if total_secs > 0:
                    duration_secs = total_secs

            duration_str = format_seconds(duration_secs) if duration_secs else None

            # Sub-phase details
            details = []
            if pkey == 'compile' and build_logs:
                for bl in build_logs:
                    bl_secs = bl['duration']
                    exit_ok = (bl['exit_code'] == 0) if bl['exit_code'] is not None else None
                    details.append({
                        'name': bl['phase'],
                        'duration': format_seconds(bl_secs) if bl_secs else None,
                        'duration_seconds': bl_secs,
                        'exit_code': bl['exit_code'],
                        'ok': exit_ok
                    })
            if pkey == 'sonarqube' and sonar:
                details.append({
                    'name': 'Quality Gate',
                    'value': sonar['quality_gate_status'],
                    'ok': bool(sonar['passed'])
                })

            phases.append({
                'key': pkey,
                'label': plabel,
                'status': ph_status,
                'duration': duration_str,
                'duration_seconds': duration_secs,
                'details': details
            })

        note = None
        if dep_status == 'failed' and failed is None:
            note = 'No consta en qué fase falló esta ejecución (es anterior al registro de fases).'
        elif not phase_start and dep_status != 'pending':
            note = 'Ejecución anterior al registro de fases: no hay tiempos por fase.'

        return jsonify({
            'deployment_id': deployment_id,
            'tag': deployment['tag_name'],
            'status': dep_status,
            'current_phase': current,
            'failed_phase': failed,
            'note': note,
            'phases': phases
        })
    except Exception:
        return api_error('api_deployment_phases')


# ==================== ROUTES & API: USERS ====================

@app.route('/users')
@admin_required
def users():
    """Pagina de gestion de usuarios."""
    return render_template('users.html')


def _validate_csrf_api():
    """Valida el CSRF token para llamadas AJAX (header X-CSRF-Token o campo de formulario)."""
    session_token = session.get('csrf_token', '')
    request_token = (
        request.headers.get('X-CSRF-Token', '') or
        request.form.get('csrf_token', '')
    )
    if not session_token or not request_token:
        return False
    return hmac.compare_digest(session_token, request_token)


def _password_error(password):
    if len(password) < app.config['MIN_PASSWORD_LENGTH']:
        return 'La contrasena debe tener al menos {} caracteres.'.format(app.config['MIN_PASSWORD_LENGTH'])
    return None


@app.route('/api/users')
@admin_required
def api_users_list():
    """Lista todos los usuarios (sin hashes)."""
    try:
        conn = get_db_connection()
        rows = conn.execute(
            'SELECT id, username, is_active, is_admin, created_at, last_login FROM web_users ORDER BY username'
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return api_error('api_users_list')


@app.route('/api/users', methods=['POST'])
@admin_required
def api_users_create():
    """Crea un nuevo usuario."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''
        is_admin = 1 if data.get('is_admin') else 0
        if not username or len(username) > app.config['MAX_USERNAME_LENGTH']:
            return jsonify({'error': 'El nombre de usuario debe tener entre 1 y {} caracteres.'.format(
                app.config['MAX_USERNAME_LENGTH'])}), 400
        error = _password_error(password)
        if error:
            return jsonify({'error': error}), 400
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO web_users (username, password_hash, is_admin, password_changed_at) VALUES (?, ?, ?, ?)',
                (username, password_hash, is_admin, password_stamp())
            )
            conn.commit()
            return jsonify({'ok': True, 'username': username, 'is_admin': is_admin})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'El usuario "{}" ya existe.'.format(username)}), 409
        finally:
            conn.close()
    except Exception:
        return api_error('api_users_create')


@app.route('/api/users/<username>/toggle', methods=['POST'])
@admin_required
def api_users_toggle(username):
    """Activa o desactiva un usuario."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    if username == g.user['username']:
        return jsonify({'error': 'No puedes desactivar tu propio usuario.'}), 400
    try:
        conn = get_db_connection()
        user = conn.execute(
            'SELECT is_active FROM web_users WHERE username = ?', (username,)
        ).fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'Usuario no encontrado.'}), 404
        new_state = 0 if user['is_active'] else 1
        conn.execute(
            'UPDATE web_users SET is_active = ? WHERE username = ?', (new_state, username)
        )
        conn.commit()
        conn.close()
        return jsonify({'ok': True, 'username': username, 'is_active': new_state})
    except Exception:
        return api_error('api_users_toggle')


@app.route('/api/users/<username>/admin', methods=['POST'])
@admin_required
def api_users_toggle_admin(username):
    """Concede o retira el rol de administrador."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    if username == g.user['username']:
        return jsonify({'error': 'No puedes cambiar tu propio rol.'}), 400
    try:
        conn = get_db_connection()
        user = conn.execute(
            'SELECT is_admin FROM web_users WHERE username = ?', (username,)
        ).fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'Usuario no encontrado.'}), 404
        new_state = 0 if user['is_admin'] else 1
        conn.execute('UPDATE web_users SET is_admin = ? WHERE username = ?', (new_state, username))
        conn.commit()
        conn.close()
        return jsonify({'ok': True, 'username': username, 'is_admin': new_state})
    except Exception:
        return api_error('api_users_toggle_admin')


@app.route('/api/users/<username>/change-password', methods=['POST'])
@admin_required
def api_users_change_password(username):
    """Restablece la contrasena de otro usuario (cierra sus sesiones abiertas)."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    if username == g.user['username']:
        return jsonify({'error': 'Para cambiar tu contraseña usa "Mi contraseña" (pide la actual).'}), 400
    try:
        data = request.get_json(silent=True) or {}
        password = data.get('password') or ''
        error = _password_error(password)
        if error:
            return jsonify({'error': error}), 400
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.execute(
            'UPDATE web_users SET password_hash = ?, password_changed_at = ? WHERE username = ?',
            (password_hash, password_stamp(), username)
        )
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Usuario no encontrado.'}), 404
        return jsonify({'ok': True})
    except Exception:
        return api_error('api_users_change_password')


@app.route('/api/account/password', methods=['POST'])
def api_account_password():
    """Cambia la contrasena del usuario actual; exige la contrasena actual."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    try:
        data = request.get_json(silent=True) or {}
        current = data.get('current_password') or ''
        password = data.get('password') or ''
        username = g.user['username']

        conn = get_db_connection()
        try:
            row = conn.execute(
                'SELECT password_hash FROM web_users WHERE username = ?', (username,)
            ).fetchone()
            if not row or not check_password_hash(row['password_hash'], current):
                return jsonify({'error': 'La contraseña actual no es correcta.'}), 400
            error = _password_error(password)
            if error:
                return jsonify({'error': error}), 400
            stamp = password_stamp()
            conn.execute(
                'UPDATE web_users SET password_hash = ?, password_changed_at = ? WHERE username = ?',
                (generate_password_hash(password), stamp, username)
            )
            conn.commit()
        finally:
            conn.close()
        # Mantener esta sesion; las demas sesiones del usuario quedan invalidadas
        session['pwd_stamp'] = stamp
        return jsonify({'ok': True})
    except Exception:
        return api_error('api_account_password')


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': GENERIC_ERROR}), 500
    return render_template('500.html'), 500


# ==================== FILTERS ====================

app.jinja_env.filters['datetime'] = format_datetime
app.jinja_env.filters['duration'] = calculate_duration


if __name__ == '__main__':
    # Desarrollo: Flask server directo
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 8080),
        debug=app.config.get('DEBUG', False)
    )
