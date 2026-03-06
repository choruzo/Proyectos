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


def get_sonar_metrics(config):
    """Obtiene métricas del proyecto desde SonarQube API"""
    sonar_config = config.get('sonarqube', {})
    base_url = sonar_config.get('url', '').rstrip('/')
    project_key = sonar_config.get('project_key', 'GALTTCMC_interno')
    token = os.environ.get('SONAR_TOKEN', sonar_config.get('token', ''))
    
    url = '{}/api/measures/component'.format(base_url)
    params = {
        'component': project_key,
        'metricKeys': 'coverage,bugs,vulnerabilities,code_smells,security_hotspots'
    }
    
    # SonarQube usa token como usuario con password vacío
    auth = (token, '') if token else None
    headers = {}
    
    # Alternativa: usar header Authorization
    if token and not auth:
        headers['Authorization'] = 'Bearer {}'.format(token)
    
    print('[*] Consultando SonarQube: {}'.format(url))
    print('[*] Proyecto: {}'.format(project_key))
    
    try:
        response = requests.get(
            url,
            params=params,
            auth=auth,
            headers=headers,
            verify=True,  # Cambiar a False si hay problemas con certificados
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.SSLError:
        # Reintentar sin verificación SSL
        print('[WARN] Error SSL, reintentando sin verificación...')
        response = requests.get(
            url,
            params=params,
            auth=auth,
            headers=headers,
            verify=False,
            timeout=30
        )
        response.raise_for_status()
    
    data = response.json()
    
    # Extraer métricas
    measures = data.get('component', {}).get('measures', [])
    metrics = {}
    
    for m in measures:
        metric_name = m.get('metric')
        value = m.get('value', '0')
        try:
            metrics[metric_name] = float(value)
        except (ValueError, TypeError):
            metrics[metric_name] = 0.0
    
    return metrics


def get_quality_gate_status(config):
    """Obtener estado del Quality Gate"""
    sonar_config = config.get('sonarqube', {})
    base_url = sonar_config.get('url', '').rstrip('/')
    project_key = sonar_config.get('project_key', 'GALTTCMC_interno')
    token = os.environ.get('SONAR_TOKEN', sonar_config.get('token', ''))
    
    url = '{}/api/qualitygates/project_status'.format(base_url)
    params = {'projectKey': project_key}
    auth = (token, '') if token else None
    
    try:
        response = requests.get(
            url,
            params=params,
            auth=auth,
            verify=True,
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


def save_to_db(db_path, tag, metrics, passed, quality_gate_status='UNKNOWN'):
    """Guarda resultados en SQLite"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sonar_results 
            (tag, timestamp, coverage, bugs, vulnerabilities, code_smells, 
             security_hotspots, passed, quality_gate_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tag,
            datetime.now().isoformat(),
            metrics.get('coverage', 0),
            int(metrics.get('bugs', 0)),
            int(metrics.get('vulnerabilities', 0)),
            int(metrics.get('code_smells', 0)),
            int(metrics.get('security_hotspots', 0)),
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
        print('Uso: sonar_check.py <config_path> <tag>')
        print('')
        print('Ejemplo:')
        print('  python3.6 sonar_check.py /home/agent/cicd/config/ci_cd_config.yaml V01_02_03_04')
        sys.exit(1)
    
    config_path = sys.argv[1]
    tag = sys.argv[2]
    
    print('[*] Cargando configuracion: {}'.format(config_path))
    config = load_config(config_path)
    
    # Obtener métricas
    print('[*] Consultando metricas SonarQube para proyecto: {}'.format(
        config.get('sonarqube', {}).get('project_key')
    ))
    
    try:
        metrics = get_sonar_metrics(config)
    except requests.exceptions.RequestException as e:
        print('[ERROR] No se pudo conectar a SonarQube: {}'.format(str(e)))
        sys.exit(1)
    
    # Obtener estado del Quality Gate (informativo)
    quality_gate_status = get_quality_gate_status(config)
    print('[*] Quality Gate status: {}'.format(quality_gate_status))
    
    # Validar umbrales
    thresholds = config.get('sonarqube', {}).get('thresholds', {})
    all_passed, results = check_thresholds(metrics, thresholds)
    
    # Mostrar resultados
    print_results(results, all_passed)
    
    # Guardar en base de datos
    db_path = config.get('general', {}).get('db_path')
    if db_path:
        save_to_db(db_path, tag, metrics, all_passed, quality_gate_status)
    
    # Código de salida
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
