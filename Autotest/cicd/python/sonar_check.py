#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sonar_check.py - Consulta API SonarQube y valida umbrales

Compatible con Python 3.6+

Uso:
    python3.6 sonar_check.py <config_path> <tag>
"""

from __future__ import print_function
import sys
import os
import time
import yaml
import requests
import sqlite3
from datetime import datetime


def load_config(config_path):
    """Cargar configuración YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expandir variables de entorno
    def expand_env(obj):
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.environ.get(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(i) for i in obj]
        return obj
    
    return expand_env(config)


def _sonar_request(config, url, params):
    """Helper: hace GET a SonarQube con reintentos SSL"""
    sonar_config = config.get('sonarqube', {})
    token = os.environ.get('SONAR_TOKEN', sonar_config.get('token', ''))
    auth = (token, '') if token else None
    try:
        r = requests.get(url, params=params, auth=auth, verify=True, timeout=30)
        r.raise_for_status()
    except requests.exceptions.SSLError:
        print('[WARN] Error SSL, reintentando sin verificación...')
        r = requests.get(url, params=params, auth=auth, verify=False, timeout=30)
        r.raise_for_status()
    return r


def get_sonar_metrics(config):
    """Obtiene métricas Overall Code y New Code desde SonarQube API"""
    sonar_config = config.get('sonarqube', {})
    base_url = sonar_config.get('url', '').rstrip('/')
    project_key = sonar_config.get('project_key', 'GALTTCMC_interno')

    url = '{}/api/measures/component'.format(base_url)

    print('[*] Consultando SonarQube: {}'.format(url))
    print('[*] Proyecto: {}'.format(project_key))

    # Overall Code metrics
    params_overall = {
        'component': project_key,
        'metricKeys': 'coverage,bugs,vulnerabilities,code_smells,security_hotspots'
    }
    data_overall = _sonar_request(config, url, params_overall).json()
    measures_overall = data_overall.get('component', {}).get('measures', [])

    metrics = {}
    for m in measures_overall:
        try:
            metrics[m['metric']] = float(m.get('value', '0'))
        except (ValueError, TypeError):
            metrics[m['metric']] = 0.0

    # New Code metrics (devueltos en el array `periods`)
    new_metric_keys = 'new_coverage,new_bugs,new_vulnerabilities,new_code_smells,new_security_hotspots'
    params_new = {
        'component': project_key,
        'metricKeys': new_metric_keys,
        'additionalFields': 'periods'
    }
    data_new = _sonar_request(config, url, params_new).json()
    measures_new = data_new.get('component', {}).get('measures', [])

    for m in measures_new:
        metric_name = m.get('metric')  # e.g. 'new_bugs'
        # New code values come in 'periods' array, not 'value'
        periods = m.get('periods', [])
        val_str = periods[0].get('value', '0') if periods else m.get('value', '0')
        try:
            metrics[metric_name] = float(val_str)
        except (ValueError, TypeError):
            metrics[metric_name] = 0.0

    return metrics


def parse_report_task(report_task_file):
    """Leer report-task.txt generado por sonar-scanner y extraer ceTaskUrl y ceTaskId"""
    result = {}
    try:
        with open(report_task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip()
    except Exception as e:
        print('[WARN] No se pudo leer report-task.txt: {}'.format(str(e)))
    return result


def wait_for_ce_task(ce_task_url, config, max_wait=300, poll_interval=5):
    """Esperar a que SonarQube termine de procesar el análisis (CE task).

    SonarQube procesa los análisis de forma asíncrona. Si se consulta el
    quality gate antes de que termine el procesamiento se obtiene el estado
    del análisis ANTERIOR. Esta función sondea el endpoint del CE task hasta
    que el estado sea SUCCESS, FAILED o CANCELLED.

    Devuelve (task_status, analysis_id).
    """
    sonar_config = config.get('sonarqube', {})
    token = os.environ.get('SONAR_TOKEN', sonar_config.get('token', ''))
    auth = (token, '') if token else None

    print('[*] Esperando procesamiento SonarQube (max {}s)...'.format(max_wait))

    elapsed = 0
    while elapsed < max_wait:
        try:
            response = requests.get(
                ce_task_url,
                auth=auth,
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            task = data.get('task', {})
            status = task.get('status', 'UNKNOWN')
            analysis_id = task.get('analysisId', '')

            print('[*] CE Task status: {} ({}s)'.format(status, elapsed))

            if status in ('SUCCESS', 'FAILED', 'CANCELLED'):
                symbol = '[OK]' if status == 'SUCCESS' else '[X]'
                print('{} Procesamiento completado: {}'.format(symbol, status))
                return status, analysis_id

            time.sleep(poll_interval)
            elapsed += poll_interval

        except Exception as e:
            print('[WARN] Error consultando CE task: {}'.format(str(e)))
            time.sleep(poll_interval)
            elapsed += poll_interval

    print('[WARN] Timeout esperando CE task ({}s). '
          'Se usará el estado actual del proyecto.'.format(max_wait))
    return 'UNKNOWN', ''


def get_quality_gate_status(config, analysis_id=None):
    """Obtener estado del Quality Gate.

    Si se proporciona analysis_id se consulta el resultado exacto de ese
    análisis (evita leer el estado de un análisis anterior).
    """
    sonar_config = config.get('sonarqube', {})
    base_url = sonar_config.get('url', '').rstrip('/')
    project_key = sonar_config.get('project_key', 'GALTTCMC_interno')
    token = os.environ.get('SONAR_TOKEN', sonar_config.get('token', ''))

    url = '{}/api/qualitygates/project_status'.format(base_url)
    if analysis_id:
        params = {'analysisId': analysis_id}
        print('[*] Consultando Quality Gate para analysis ID: {}'.format(analysis_id))
    else:
        params = {'projectKey': project_key}
        print('[*] Consultando Quality Gate para proyecto: {}'.format(project_key))

    auth = (token, '') if token else None

    try:
        response = requests.get(
            url,
            params=params,
            auth=auth,
            verify=False,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('projectStatus', {}).get('status', 'UNKNOWN')
    except Exception as e:
        print('[WARN] No se pudo obtener Quality Gate: {}'.format(str(e)))
        return 'UNKNOWN'


def check_thresholds(metrics, thresholds):
    """Valida métricas contra umbrales configurados"""
    results = {}
    all_passed = True
    
    # Definir checks: (métrica, función de validación)
    checks = {
        'coverage': lambda v, t: v >= t,      # >= umbral
        'bugs': lambda v, t: v <= t,           # <= umbral
        'vulnerabilities': lambda v, t: v <= t,
        'code_smells': lambda v, t: v <= t,
        'security_hotspots': lambda v, t: v <= t
    }
    
    for metric, check_func in checks.items():
        value = metrics.get(metric, 0.0)
        threshold = thresholds.get(metric, 0)
        
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            threshold = 0.0
        
        passed = check_func(value, threshold)
        results[metric] = {
            'value': value,
            'threshold': threshold,
            'passed': passed
        }
        
        if not passed:
            all_passed = False
    
    return all_passed, results


def _ensure_new_columns(conn):
    """Migración: añade columnas new_* a sonar_results si no existen."""
    existing = set(row[1] for row in conn.execute('PRAGMA table_info(sonar_results)').fetchall())
    new_cols = [
        ('new_coverage', 'REAL'),
        ('new_bugs', 'INTEGER'),
        ('new_vulnerabilities', 'INTEGER'),
        ('new_code_smells', 'INTEGER'),
        ('new_security_hotspots', 'INTEGER'),
    ]
    for col, ctype in new_cols:
        if col not in existing:
            conn.execute('ALTER TABLE sonar_results ADD COLUMN {} {}'.format(col, ctype))
            print('[*] Columna migrada: {}'.format(col))


def save_to_db(db_path, tag, metrics, passed, quality_gate_status='UNKNOWN'):
    """Guarda resultados en SQLite (overall + new code metrics)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Migración automática si la BD es anterior
        _ensure_new_columns(conn)

        cursor.execute('''
            INSERT INTO sonar_results
            (tag, timestamp,
             coverage, bugs, vulnerabilities, code_smells, security_hotspots,
             new_coverage, new_bugs, new_vulnerabilities, new_code_smells, new_security_hotspots,
             passed, quality_gate_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tag,
            datetime.now().isoformat(),
            metrics.get('coverage', 0),
            int(metrics.get('bugs', 0)),
            int(metrics.get('vulnerabilities', 0)),
            int(metrics.get('code_smells', 0)),
            int(metrics.get('security_hotspots', 0)),
            metrics.get('new_coverage', None),
            int(metrics.get('new_bugs', 0)) if 'new_bugs' in metrics else None,
            int(metrics.get('new_vulnerabilities', 0)) if 'new_vulnerabilities' in metrics else None,
            int(metrics.get('new_code_smells', 0)) if 'new_code_smells' in metrics else None,
            int(metrics.get('new_security_hotspots', 0)) if 'new_security_hotspots' in metrics else None,
            1 if passed else 0,
            quality_gate_status
        ))

        conn.commit()
        conn.close()
        print('[OK] Resultados guardados en base de datos')

    except Exception as e:
        print('[WARN] No se pudieron guardar resultados en BD: {}'.format(str(e)))


def print_results(results, all_passed):
    """Imprimir resultados formateados"""
    print('')
    print('=' * 60)
    print('                 RESULTADOS SONARQUBE')
    print('=' * 60)
    
    for metric, data in results.items():
        status = 'PASS' if data['passed'] else 'FAIL'
        symbol = '[OK]' if data['passed'] else '[X]'
        
        # Formatear valor
        value = data['value']
        if metric == 'coverage':
            value_str = '{:.1f}%'.format(value)
            threshold_str = '>= {:.1f}%'.format(data['threshold'])
        else:
            value_str = '{:.0f}'.format(value)
            threshold_str = '<= {:.0f}'.format(data['threshold'])
        
        print('  {} {:20s} {:>10s}  ({})'.format(
            symbol, metric, value_str, threshold_str
        ))
    
    print('=' * 60)
    
    if all_passed:
        print('[OK] Analisis SonarQube: APROBADO')
    else:
        print('[X] Analisis SonarQube: RECHAZADO')
    
    print('=' * 60)
    print('')


def main():
    if len(sys.argv) < 3:
        print('Uso: sonar_check.py <config_path> <tag> [report_task_file]')
        print('')
        print('Ejemplo:')
        print('  python3.6 sonar_check.py /home/agent/cicd/config/ci_cd_config.yaml V01_02_03_04')
        print('  python3.6 sonar_check.py /home/agent/cicd/config/ci_cd_config.yaml V01_02_03_04 /home/agent/compile/.scannerwork/report-task.txt')
        sys.exit(1)

    config_path = sys.argv[1]
    tag = sys.argv[2]
    report_task_file = sys.argv[3] if len(sys.argv) > 3 else None

    print('[*] Cargando configuracion: {}'.format(config_path))
    config = load_config(config_path)

    # ------------------------------------------------------------------ #
    # PASO 1: Esperar a que SonarQube termine de procesar el análisis     #
    # ------------------------------------------------------------------ #
    analysis_id = None
    if report_task_file and os.path.isfile(report_task_file):
        print('[*] Leyendo report-task.txt: {}'.format(report_task_file))
        task_info = parse_report_task(report_task_file)
        ce_task_url = task_info.get('ceTaskUrl', '')
        if ce_task_url:
            ce_status, analysis_id = wait_for_ce_task(ce_task_url, config)
            if ce_status == 'FAILED':
                print('[X] El CE task de SonarQube falló. El análisis no se procesó.')
            elif ce_status == 'CANCELLED':
                print('[X] El CE task de SonarQube fue cancelado.')
            # analysis_id puede ser vacío si el CE task no terminó bien
        else:
            print('[WARN] ceTaskUrl no encontrada en report-task.txt')
    else:
        if report_task_file:
            print('[WARN] report-task.txt no encontrado: {}'.format(report_task_file))
        print('[*] Sin report-task.txt: consultando estado actual del proyecto '
              '(puede reflejar un análisis anterior)')

    # ------------------------------------------------------------------ #
    # PASO 2: Obtener métricas del proyecto (overall code, informativo)   #
    # ------------------------------------------------------------------ #
    print('[*] Consultando metricas SonarQube para proyecto: {}'.format(
        config.get('sonarqube', {}).get('project_key')
    ))

    try:
        metrics = get_sonar_metrics(config)
    except requests.exceptions.RequestException as e:
        print('[ERROR] No se pudo conectar a SonarQube: {}'.format(str(e)))
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # PASO 3: Obtener estado del Quality Gate (resultado oficial SonarQube) #
    # ------------------------------------------------------------------ #
    quality_gate_status = get_quality_gate_status(config, analysis_id=analysis_id)
    print('[*] Quality Gate status: {}'.format(quality_gate_status))

    # ------------------------------------------------------------------ #
    # PASO 4: Evaluar umbrales personalizados (informativo)               #
    # ------------------------------------------------------------------ #
    thresholds = config.get('sonarqube', {}).get('thresholds', {})
    custom_passed, results = check_thresholds(metrics, thresholds)

    # Mostrar resultados de métricas / umbrales personalizados
    print_results(results, custom_passed)

    # ------------------------------------------------------------------ #
    # PASO 5: Decisión final basada en Quality Gate de SonarQube          #
    # ------------------------------------------------------------------ #
    # El Quality Gate de SonarQube es la fuente de verdad.
    # Los umbrales personalizados se muestran como información adicional
    # pero no bloquean el pipeline si SonarQube dice OK.
    sonar_qg_passed = (quality_gate_status == 'OK')

    if sonar_qg_passed:
        print('[OK] Quality Gate SonarQube: APROBADO ({})'
              .format(quality_gate_status))
    else:
        print('[X] Quality Gate SonarQube: NO APROBADO ({})'
              .format(quality_gate_status))
        if not custom_passed:
            print('[INFO] Umbrales personalizados también fallaron '
                  '(puede ser que se estén evaluando métricas globales '
                  'en lugar de código nuevo).')

    # ------------------------------------------------------------------ #
    # PASO 6: Guardar en base de datos                                    #
    # ------------------------------------------------------------------ #
    db_path = config.get('general', {}).get('db_path')
    if db_path:
        save_to_db(db_path, tag, metrics, sonar_qg_passed, quality_gate_status)

    # Código de salida basado en Quality Gate de SonarQube
    if sonar_qg_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
