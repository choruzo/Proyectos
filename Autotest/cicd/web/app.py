#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web UI for GALTTCMC CI/CD Pipeline
Visualiza logs, deployments, metricas de SonarQube y estado del pipeline
"""

from __future__ import print_function
import os
import sys
import sqlite3
import json
import subprocess
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory, session, redirect, url_for
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


def load_config():
    """Carga configuracion desde ci_cd_config.yaml"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print("Error loading config: {}".format(str(e)))
        return {}


def get_db_connection():
    """Obtiene conexion a SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ==================== AUTH ====================

def generate_csrf_token():
    """Genera o recupera el CSRF token de la sesion actual."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


@app.context_processor
def inject_csrf_token():
    """Inyecta csrf_token en todos los templates."""
    return {'csrf_token': generate_csrf_token()}


@app.before_request
def require_login():
    """Verifica autenticacion en todas las rutas excepto /login y archivos estaticos."""
    public_endpoints = {'login', 'logout', 'static'}
    if request.endpoint in public_endpoints:
        return None
    if not session.get('authenticated'):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Authentication required'}), 401
        return redirect(url_for('login', next=request.path))
    return None


def format_datetime(dt_str):
    """Formatea datetime string para display"""
    if not dt_str:
        return 'N/A'
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return dt_str


def calculate_duration(start, end):
    """Calcula duracion entre dos timestamps"""
    if not start or not end:
        return 'N/A'
    try:
        dt_start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        dt_end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        duration = dt_end - dt_start
        
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        
        if minutes > 0:
            return "{}m {}s".format(minutes, seconds)
        return "{}s".format(seconds)
    except:
        return 'N/A'


# ==================== ROUTES: AUTH ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Pagina de login con proteccion CSRF."""
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
                try:
                    conn = get_db_connection()
                    user = conn.execute(
                        'SELECT password_hash FROM web_users WHERE username = ? AND is_active = 1',
                        (username,)
                    ).fetchone()
                    if user and check_password_hash(user['password_hash'], password):
                        conn.execute(
                            "UPDATE web_users SET last_login = datetime('now') WHERE username = ?",
                            (username,)
                        )
                        conn.commit()
                        conn.close()
                        session.clear()
                        session['authenticated'] = True
                        session['username'] = username
                        next_url = request.args.get('next', '/')
                        # Evitar open redirect: solo permitir rutas relativas del mismo origen
                        if not next_url.startswith('/') or next_url.startswith('//'):
                            next_url = '/'
                        return redirect(next_url)
                    conn.close()
                    error = 'Usuario o contrasena incorrectos.'
                except Exception:
                    error = 'Error de autenticacion. Contacte al administrador.'
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
        
        # Average duration (last 10 successful)
        avg_duration = conn.execute(
            """SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400) as avg_seconds
               FROM deployments 
               WHERE status = 'success' AND completed_at IS NOT NULL
               ORDER BY started_at DESC LIMIT 10"""
        ).fetchone()['avg_seconds']
        
        avg_duration_str = 'N/A'
        if avg_duration:
            minutes = int(avg_duration / 60)
            seconds = int(avg_duration % 60)
            avg_duration_str = "{}m {}s".format(minutes, seconds) if minutes > 0 else "{}s".format(seconds)
        
        conn.close()
        
        return jsonify({
            'total_deployments': total,
            'success_rate': success_rate,
            'last_24h': last_24h,
            'currently_running': running,
            'avg_duration': avg_duration_str
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/recent-deployments')
def api_recent_deployments():
    """Ultimos 10 deployments"""
    try:
        conn = get_db_connection()
        deployments = conn.execute(
            """SELECT id, tag_name, status, started_at, completed_at, error_message
               FROM deployments 
               ORDER BY started_at DESC LIMIT 10"""
        ).fetchall()
        conn.close()
        
        result = []
        for d in deployments:
            result.append({
                'id': d['id'],
                'tag': d['tag_name'],
                'status': d['status'],
                'started_at': format_datetime(d['started_at']),
                'finished_at': format_datetime(d['completed_at']),
                'duration': calculate_duration(d['started_at'], d['completed_at']),
                'error_message': d['error_message']
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/deployments')
def api_deployments():
    """Lista todos los deployments con paginacion"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        status_filter = request.args.get('status', '')
        
        offset = (page - 1) * per_page
        
        conn = get_db_connection()

        VALID_STATUSES = {'pending', 'compiling', 'analyzing', 'deploying', 'success', 'failed'}

        # Query base
        params = []
        where_clause = ""
        if status_filter and status_filter != 'all':
            if status_filter not in VALID_STATUSES:
                conn.close()
                return jsonify({'error': 'Invalid status filter'}), 400
            where_clause = "WHERE status = ?"
            params.append(status_filter)

        # Total count
        total = conn.execute(
            "SELECT COUNT(*) as count FROM deployments {}".format(where_clause),
            params
        ).fetchone()['count']

        # Deployments
        deployments = conn.execute(
            """SELECT id, tag_name, status, started_at, completed_at, error_message
               FROM deployments {}
               ORDER BY started_at DESC LIMIT ? OFFSET ?""".format(where_clause),
            params + [per_page, offset]
        ).fetchall()
        
        conn.close()
        
        result = []
        for d in deployments:
            result.append({
                'id': d['id'],
                'tag': d['tag_name'],
                'status': d['status'],
                'started_at': format_datetime(d['started_at']),
                'finished_at': format_datetime(d['completed_at']),
                'duration': calculate_duration(d['started_at'], d['completed_at']),
                'error_message': d['error_message']
            })
        
        return jsonify({
            'deployments': result,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        
        # Sonar results - buscar por tag en lugar de deployment_id
        sonar_results = conn.execute(
            'SELECT * FROM sonar_results WHERE tag = ?', (deployment['tag_name'],)
        ).fetchall()
        
        conn.close()
        
        result = {
            'id': deployment['id'],
            'tag': deployment['tag_name'],
            'status': deployment['status'],
            'started_at': deployment['started_at'],
            'finished_at': deployment['completed_at'],
            'duration': calculate_duration(deployment['started_at'], deployment['completed_at']),
            'error_message': deployment['error_message'],
            'triggered_by': deployment['triggered_by'],
            'build_logs': [dict(log) for log in build_logs],
            'sonar_results': [dict(result) for result in sonar_results]
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/view/<filename>')
def api_logs_view(filename):
    """Lee contenido de un log file"""
    try:
        # Validar filename (seguridad)
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(LOGS_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Log file not found'}), 404
        
        lines = int(request.args.get('lines', 500))
        search = request.args.get('search', '').lower()
        start_line = int(request.args.get('start_line', -1))

        with open(filepath, 'r') as f:
            all_lines = f.readlines()

        file_total_lines = len(all_lines)

        if start_line >= 0:
            # Incremental fetch for live-tail: only return new lines since start_line
            content_lines = all_lines[start_line:]
            matched_lines = file_total_lines
        else:
            # Normal fetch with optional search filter and line truncation
            if search:
                all_lines = [line for line in all_lines if search in line.lower()]
            matched_lines = len(all_lines)
            content_lines = all_lines[-lines:] if matched_lines > lines else all_lines

        return jsonify({
            'filename': filename,
            'total_lines': matched_lines,
            'file_total_lines': file_total_lines,
            'displayed_lines': len(content_lines),
            'content': ''.join(content_lines)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sonar/results')
def api_sonar_results():
    """Resultados SonarQube de todos los deployments"""
    try:
        conn = get_db_connection()

        # Detectar si las columnas new_* ya existen (migracion gradual)
        cols = set(row[1] for row in conn.execute('PRAGMA table_info(sonar_results)').fetchall())
        has_new_cols = 'new_bugs' in cols

        if has_new_cols:
            select_new = 'sr.new_coverage, sr.new_bugs, sr.new_vulnerabilities, sr.new_code_smells, sr.new_security_hotspots,'
        else:
            select_new = 'NULL as new_coverage, NULL as new_bugs, NULL as new_vulnerabilities, NULL as new_code_smells, NULL as new_security_hotspots,'

        results = conn.execute(
            """SELECT sr.id as sonar_id, sr.tag, sr.created_at,
                      sr.coverage, sr.bugs, sr.vulnerabilities, sr.code_smells, sr.security_hotspots,
                      {new_cols}
                      sr.passed, sr.quality_gate_status,
                      d.id as dep_id, d.tag_name, d.started_at
               FROM sonar_results sr
               JOIN deployments d ON sr.tag = d.tag_name
               WHERE d.status = 'success'
               ORDER BY sr.created_at DESC LIMIT 50""".format(new_cols=select_new)
        ).fetchall()

        conn.close()

        data = []
        for r in results:
            data.append({
                'sonar_id': r['sonar_id'],
                'deployment_id': r['dep_id'],
                'tag': r['tag_name'],
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
    except Exception as e:
        print('Error in api_sonar_results: {}'.format(str(e)), file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sonar/trends')
def api_sonar_trends():
    """Tendencias de metricas SonarQube (ultimos 10 deployments)"""
    try:
        conn = get_db_connection()

        # Detectar si las columnas new_* existen
        cols = set(row[1] for row in conn.execute('PRAGMA table_info(sonar_results)').fetchall())
        has_new_cols = 'new_bugs' in cols

        if has_new_cols:
            select_new = 'sr.new_coverage, sr.new_bugs, sr.new_vulnerabilities, sr.new_code_smells,'
        else:
            select_new = 'NULL as new_coverage, NULL as new_bugs, NULL as new_vulnerabilities, NULL as new_code_smells,'

        results = conn.execute(
            """SELECT sr.coverage, sr.bugs, sr.vulnerabilities, sr.code_smells,
                      {new_cols}
                      d.tag_name
               FROM sonar_results sr
               JOIN deployments d ON sr.tag = d.tag_name
               WHERE d.status = 'success'
               ORDER BY sr.created_at DESC LIMIT 10""".format(new_cols=select_new)
        ).fetchall()

        conn.close()

        labels = []
        new_bugs_data = []
        new_vulnerabilities_data = []
        new_code_smells_data = []
        overall_coverage_data = []
        new_coverage_data = []

        for r in reversed(list(results)):
            labels.append(r['tag_name'])
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    except Exception as e:
        return jsonify({'status': 'unknown', 'running': False, 'error': str(e)})


# ==================== ERROR HANDLERS ====================

@app.route('/api/deployment/<int:deployment_id>/phases')
def api_deployment_phases(deployment_id):
    """Datos de fases del pipeline para visualizacion de progreso por fase"""
    try:
        conn = get_db_connection()
        deployment = conn.execute(
            'SELECT * FROM deployments WHERE id = ?', (deployment_id,)
        ).fetchone()

        if not deployment:
            conn.close()
            return jsonify({'error': 'Deployment not found'}), 404

        dep_status = deployment['status']

        # Phase timings from execution_log (may not always have data)
        exec_phase_map = {}
        try:
            exec_logs = conn.execute(
                """SELECT phase,
                          MIN(timestamp) as first_ts,
                          MAX(timestamp) as last_ts,
                          SUM(CASE WHEN level='ERROR' THEN 1 ELSE 0 END) as errors
                   FROM execution_log
                   WHERE deployment_id = ?
                   GROUP BY phase""",
                (deployment_id,)
            ).fetchall()
            for row in exec_logs:
                exec_phase_map[row['phase']] = {
                    'start': row['first_ts'],
                    'end': row['last_ts'],
                    'errors': row['errors']
                }
        except Exception:
            pass

        # Build logs for compile sub-phases detail
        build_logs = conn.execute(
            'SELECT phase, start_time, duration, exit_code FROM build_logs WHERE deployment_id = ? ORDER BY start_time',
            (deployment_id,)
        ).fetchall()

        # SonarQube result for the analyze phase
        sonar = conn.execute(
            'SELECT quality_gate_status, passed FROM sonar_results WHERE tag = ? ORDER BY created_at DESC LIMIT 1',
            (deployment['tag_name'],)
        ).fetchone()

        conn.close()

        # Pipeline phases in execution order: (key, label, execution_log_phase_key)
        PHASES = [
            ('checkout', 'Git Checkout',   'checkout'),
            ('compile',  'Compilacion',    'compile'),
            ('analyze',  'SonarQube',      'sonar'),
            ('deploy',   'vCenter Deploy', 'deploy'),
            ('notify',   'SSH Install',    'notify'),
        ]

        # Number of phases completed given overall deployment status
        STATUS_COMPLETED = {
            'pending':   0,
            'compiling': 1,
            'analyzing': 2,
            'deploying': 3,
            'success':   5,
            'failed':    0,  # determined below via execution_log
        }

        num_completed = STATUS_COMPLETED.get(dep_status, 0)
        failed_phase_idx = None

        if dep_status == 'failed':
            # Walk phases in order; count how many ran successfully in exec_log
            for i, (pkey, plabel, exec_key) in enumerate(PHASES):
                if exec_key in exec_phase_map:
                    if exec_phase_map[exec_key]['errors'] > 0:
                        failed_phase_idx = i
                        break
                    num_completed = i + 1
            if failed_phase_idx is None:
                # No explicit error found: mark the phase after last seen as failed
                failed_phase_idx = num_completed if num_completed < len(PHASES) else len(PHASES) - 1

        phases = []
        for i, (pkey, plabel, exec_key) in enumerate(PHASES):
            # Determine per-phase status
            if dep_status == 'success':
                ph_status = 'completed'
            elif dep_status == 'failed':
                if i < num_completed:
                    ph_status = 'completed'
                elif i == failed_phase_idx:
                    ph_status = 'failed'
                else:
                    ph_status = 'pending'
            elif i < num_completed:
                ph_status = 'completed'
            elif i == num_completed:
                ph_status = 'active'
            else:
                ph_status = 'pending'

            # Timing from execution_log
            duration_secs = None
            duration_str = None
            if exec_key in exec_phase_map:
                try:
                    ts_s = datetime.strptime(exec_phase_map[exec_key]['start'], '%Y-%m-%d %H:%M:%S')
                    ts_e = datetime.strptime(exec_phase_map[exec_key]['end'], '%Y-%m-%d %H:%M:%S')
                    duration_secs = max(1, int((ts_e - ts_s).total_seconds()))
                    m, s = divmod(duration_secs, 60)
                    duration_str = '{}m {}s'.format(m, s) if m > 0 else '{}s'.format(s)
                except Exception:
                    pass

            # Compile phase: derive total duration from build_logs if exec_log has no timing
            if pkey == 'compile' and build_logs and duration_secs is None:
                total_secs = sum(bl['duration'] or 0 for bl in build_logs)
                if total_secs > 0:
                    duration_secs = total_secs
                    m, s = divmod(total_secs, 60)
                    duration_str = '{}m {}s'.format(m, s) if m > 0 else '{}s'.format(s)

            # Sub-phase details
            details = []
            if pkey == 'compile' and build_logs:
                for bl in build_logs:
                    bl_secs = bl['duration']
                    bl_dur_str = None
                    if bl_secs:
                        m2, s2 = divmod(bl_secs, 60)
                        bl_dur_str = '{}m {}s'.format(m2, s2) if m2 > 0 else '{}s'.format(s2)
                    exit_ok = (bl['exit_code'] == 0) if bl['exit_code'] is not None else None
                    details.append({
                        'name': bl['phase'],
                        'duration': bl_dur_str,
                        'duration_seconds': bl_secs,
                        'exit_code': bl['exit_code'],
                        'ok': exit_ok
                    })
            if pkey == 'analyze' and sonar:
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

        return jsonify({
            'deployment_id': deployment_id,
            'tag': deployment['tag_name'],
            'status': dep_status,
            'phases': phases
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== ROUTES & API: USERS ====================

@app.route('/users')
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


@app.route('/api/users')
def api_users_list():
    """Lista todos los usuarios (sin hashes)."""
    try:
        conn = get_db_connection()
        rows = conn.execute(
            'SELECT id, username, is_active, created_at, last_login FROM web_users ORDER BY username'
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['POST'])
def api_users_create():
    """Crea un nuevo usuario."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''
        if not username or len(username) > 64:
            return jsonify({'error': 'El nombre de usuario debe tener entre 1 y 64 caracteres.'}), 400
        if len(password) < 8:
            return jsonify({'error': 'La contrasena debe tener al menos 8 caracteres.'}), 400
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO web_users (username, password_hash) VALUES (?, ?)',
                (username, password_hash)
            )
            conn.commit()
            conn.close()
            return jsonify({'ok': True, 'username': username})
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'error': 'El usuario "{}" ya existe.'.format(username)}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<username>/toggle', methods=['POST'])
def api_users_toggle(username):
    """Activa o desactiva un usuario."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    if username == session.get('username'):
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<username>/change-password', methods=['POST'])
def api_users_change_password(username):
    """Cambia la contrasena de un usuario."""
    if not _validate_csrf_api():
        return jsonify({'error': 'Token CSRF invalido.'}), 400
    try:
        data = request.get_json() or {}
        password = data.get('password') or ''
        if len(password) < 8:
            return jsonify({'error': 'La contrasena debe tener al menos 8 caracteres.'}), 400
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.execute(
            'UPDATE web_users SET password_hash = ? WHERE username = ?',
            (password_hash, username)
        )
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Usuario no encontrado.'}), 404
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
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
