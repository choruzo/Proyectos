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
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory
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
        
        # Query base
        where_clause = ""
        if status_filter and status_filter != 'all':
            where_clause = "WHERE status = '{}'".format(status_filter)
        
        # Total count
        total = conn.execute(
            "SELECT COUNT(*) as count FROM deployments {}".format(where_clause)
        ).fetchone()['count']
        
        # Deployments
        deployments = conn.execute(
            """SELECT id, tag_name, status, started_at, completed_at, error_message
               FROM deployments {}
               ORDER BY started_at DESC LIMIT {} OFFSET {}""".format(
                where_clause, per_page, offset
            )
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
        
        with open(filepath, 'r') as f:
            all_lines = f.readlines()
        
        # Filtrar por busqueda si existe
        if search:
            all_lines = [line for line in all_lines if search in line.lower()]
        
        # Ultimas N lineas
        content_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return jsonify({
            'filename': filename,
            'total_lines': len(all_lines),
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
        
        # JOIN por tag en lugar de deployment_id (sonar_results.deployment_id puede ser NULL)
        # Filtrar solo deployments exitosos para consistencia
        results = conn.execute(
            """SELECT sr.*, d.tag_name, d.started_at, d.id as deployment_id
               FROM sonar_results sr
               JOIN deployments d ON sr.tag = d.tag_name
               WHERE d.status = 'success'
               ORDER BY sr.created_at DESC LIMIT 50"""
        ).fetchall()
        
        conn.close()
        
        data = []
        for r in results:
            data.append({
                'deployment_id': r['deployment_id'],
                'tag': r['tag_name'],
                'date': format_datetime(r['created_at']),
                'quality_gate': r['quality_gate_status'],
                'coverage': r['coverage'],
                'bugs': r['bugs'],
                'vulnerabilities': r['vulnerabilities'],
                'code_smells': r['code_smells'],
                'security_hotspots': r['security_hotspots'],
                'passed': r['passed']
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
        
        # JOIN por tag en lugar de deployment_id (sonar_results.deployment_id puede ser NULL)
        results = conn.execute(
            """SELECT sr.*, d.tag_name, d.started_at
               FROM sonar_results sr
               JOIN deployments d ON sr.tag = d.tag_name
               WHERE d.status = 'success'
               ORDER BY sr.created_at DESC LIMIT 10"""
        ).fetchall()
        
        conn.close()
        
        labels = []
        coverage_data = []
        bugs_data = []
        vulnerabilities_data = []
        code_smells_data = []
        
        for r in reversed(list(results)):
            labels.append(r['tag_name'])
            coverage_data.append(r['coverage'] if r['coverage'] else 0)
            bugs_data.append(r['bugs'] if r['bugs'] else 0)
            vulnerabilities_data.append(r['vulnerabilities'] if r['vulnerabilities'] else 0)
            code_smells_data.append(r['code_smells'] if r['code_smells'] else 0)
        
        return jsonify({
            'labels': labels,
            'coverage': coverage_data,
            'bugs': bugs_data,
            'vulnerabilities': vulnerabilities_data,
            'code_smells': code_smells_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

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
