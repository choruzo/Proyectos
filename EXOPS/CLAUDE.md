# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

**EXOPS** es una aplicación web interna para administrar entornos VMware vSphere 8.x.
Backend Python/FastAPI con HTTPS, frontend HTML/Bootstrap 5.3, sin frameworks JS pesados.
Diseñada para equipos pequeños (2-5 admins) con autenticación propia + credenciales vCenter por usuario.

## Comandos

### Instalación
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Editar con valores reales
```

### Generar certificado SSL autofirmado
```bash
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem \
  -out certs/cert.pem -days 365 -nodes \
  -subj "/C=ES/ST=State/L=City/O=MyOrg/CN=vcenter-admin.local"
```

### Ejecutar en desarrollo
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8443 \
  --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

### Ejecutar en producción
```bash
uvicorn main:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem \
  --workers 2
```

### Linting y formato
```bash
ruff check app/
ruff format app/
```

### Tests
```bash
pytest
pytest tests/test_auth.py          # Módulo específico
pytest -k "test_login"             # Test específico por nombre
pytest --asyncio-mode=auto         # Tests async
```

## Arquitectura

### Capas

```
Browser (HTML/JS/Bootstrap) → HTTPS → FastAPI → Service Layer → SQLite / pyVmomi → vCenter 8.x
```

- **`main.py`**: Punto de entrada. Crea la app FastAPI, registra routers, middleware de seguridad y arranca Uvicorn con TLS.
- **`app/config.py`**: `Settings` con Pydantic-Settings. Lee todo desde `.env`.
- **`app/database.py`**: Engine SQLAlchemy async + función `get_db()` como dependencia FastAPI. Inicializa tablas y crea el primer `admin` si no existe.

### Módulos principales

- **`app/auth/`**: Login, generación/validación de JWT, dependencia `get_current_user`, protección de rutas, brute-force guard.
- **`app/vcenter/connection.py`**: Pool de sesiones pyVmomi en memoria (`dict[user_id → ServiceInstance]`). Cada usuario tiene su propia conexión vCenter. Las credenciales vCenter **nunca se persisten**.
- **`app/vcenter/vms.py|hosts.py|datastores.py|snapshots.py`**: Servicios que usan pyVmomi para operar sobre vSphere. Reciben la `ServiceInstance` del pool.
- **`app/audit/service.py`**: Inserta entradas en `audit_logs`. Debe llamarse desde cada endpoint que realice una acción sobre vCenter.
- **`app/api/v1/`**: Routers FastAPI que unen endpoints → servicios → respuesta JSON.

### Frontend

- Plantillas Jinja2 en `templates/` (SSR para carga inicial).
- `static/js/api.js` es el cliente centralizado para todas las llamadas `fetch()` a `/api/v1/*`.
- Cada página tiene su propio JS en `static/js/<página>.js`.
- Bootstrap 5.3 y Chart.js se cargan desde CDN (no hay build step).
- **Modo oscuro/claro**: controlado por `data-bs-theme` en `<html>`. Preferencia persistida en `localStorage('exops-theme')`; valor inicial desde `prefers-color-scheme` del SO. Script anti-flash insertado en `<head>` **antes** de los `<link>` CSS para evitar parpadeo al cargar. No usar clases `bg-light` hardcodeadas en el área de contenido principal — Bootstrap adapta el fondo automáticamente con el tema activo.

### Autenticación (flujo crítico)

1. `POST /api/v1/auth/login` recibe credenciales app + credenciales vCenter.
2. Se valida bcrypt contra SQLite; luego se prueba `SmartConnect` al vCenter.
3. Si ambas OK: se guarda la `ServiceInstance` en el pool en memoria y se devuelve JWT como cookie `httpOnly`.
4. En cada request posterior, `get_current_user` (dependencia FastAPI) valida el JWT y recupera la sesión vCenter del pool.
5. Si la sesión vCenter expiró, se reconecta transparentemente usando las credenciales que el usuario proporcionó en el login (guardadas en la sesión en memoria, no en DB).

## Decisiones de diseño importantes

- **Credenciales vCenter en memoria, nunca en DB**: Al reiniciar el servidor, todos los usuarios deben volver a hacer login.
- **pyVmomi sobre la REST API**: Se usa la API SOAP clásica (`SmartConnect`). En entornos con cert autofirmado usar `SmartConnectNoSSL`. Siempre llamar `Disconnect()` al hacer logout.
- **Sin build step en frontend**: Todo el JS es ES6+ vanilla con `fetch`. No hay `package.json`, `webpack` ni transpilación.
- **Un solo archivo SQLite**: La DB vive en `data/exops.db`. Las migraciones se gestionan con `Base.metadata.create_all()` en el arranque (adecuado para la escala actual).
- **Roles solo en la app**: La autorización a nivel de vCenter la gestiona el propio vCenter según las credenciales del usuario. La app no replica esos permisos.

## Variables de entorno requeridas (`.env`)

```
JWT_SECRET_KEY=    # Secreto largo y aleatorio
JWT_EXPIRE_MINUTES=60
APP_ENV=production
DATABASE_URL=sqlite+aiosqlite:///./data/exops.db
SSL_CERT_PATH=certs/cert.pem
SSL_KEY_PATH=certs/key.pem
```
