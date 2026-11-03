#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Configuration for GALTTCMC CI/CD Web UI
"""

import os
from datetime import timedelta

# Server configuration
HOST = os.getenv('WEB_HOST', '0.0.0.0')
PORT = int(os.getenv('WEB_PORT', '8080'))
DEBUG = os.getenv('WEB_DEBUG', 'False').lower() == 'true'

# Flask configuration
# W1: la SECRET_KEY es obligatoria. Con un valor por defecto conocido se puede
# falsificar la cookie de sesión y saltarse el login, así que la app no arranca
# sin una clave propia.
_DEFAULT_SECRET_KEY = 'galttcmc-cicd-web-ui-secret-key-change-me'
_WEAK_SECRET_KEYS = {
    _DEFAULT_SECRET_KEY,
    'change-me',
    'changeme',
    'secret',
    'your-secret-key-here',
    'your_random_secret_key_here',
}


def _load_secret_key():
    key = os.getenv('SECRET_KEY', '').strip()
    if not key or key.lower() in _WEAK_SECRET_KEYS:
        raise RuntimeError(
            'SECRET_KEY no está definida o usa un valor por defecto/conocido. '
            'Añade una clave aleatoria a config/.env antes de arrancar la Web, '
            'p. ej.: python3 -c "import binascii,os; '
            'print(binascii.hexlify(os.urandom(32)).decode())"'
        )
    if len(key) < 16:
        raise RuntimeError(
            'SECRET_KEY demasiado corta (mínimo 16 caracteres). '
            'Genera una aleatoria de 32 bytes para config/.env.'
        )
    return key


SECRET_KEY = _load_secret_key()
JSON_SORT_KEYS = False

# Session security
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
# Solo 'true' cuando la Web se sirva por HTTPS (proxy inverso con TLS);
# con HTTP en claro el navegador no enviaría la cookie y no se podría iniciar sesión.
SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
# Duración máxima de una sesión desde el login. No se renueva con cada petición:
# el auto-refresh de la Web la mantendría abierta indefinidamente.
PERMANENT_SESSION_LIFETIME = timedelta(hours=int(os.getenv('SESSION_LIFETIME_HOURS', '12')))
SESSION_REFRESH_EACH_REQUEST = False

# Limitación de intentos de login (fallos dentro de la ventana)
LOGIN_WINDOW_MINUTES = int(os.getenv('LOGIN_WINDOW_MINUTES', '15'))
LOGIN_MAX_FAILURES_PER_USER = int(os.getenv('LOGIN_MAX_FAILURES_PER_USER', '5'))
LOGIN_MAX_FAILURES_PER_IP = int(os.getenv('LOGIN_MAX_FAILURES_PER_IP', '20'))

# Contraseñas
MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64

# Application settings
APP_NAME = 'GALTTCMC CI/CD Pipeline'
APP_VERSION = '1.0.0'

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Logs
MAX_LOG_LINES = 5000
DEFAULT_LOG_LINES = 500
# Máximo de bytes nuevos devueltos por cada consulta del live-tail
LIVE_TAIL_MAX_BYTES = 1024 * 1024
