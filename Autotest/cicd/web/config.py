#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Configuration for GALTTCMC CI/CD Web UI
"""

import os

# Server configuration
HOST = os.getenv('WEB_HOST', '0.0.0.0')
PORT = int(os.getenv('WEB_PORT', '8080'))
DEBUG = os.getenv('WEB_DEBUG', 'False').lower() == 'true'

# Flask configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'galttcmc-cicd-web-ui-secret-key-change-me')
JSON_SORT_KEYS = False

# Session security
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# Application settings
APP_NAME = 'GALTTCMC CI/CD Pipeline'
APP_VERSION = '1.0.0'

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Logs
MAX_LOG_LINES = 1000
DEFAULT_LOG_LINES = 500
