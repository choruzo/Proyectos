# GALTTCMC CI/CD Web UI

Interfaz web moderna para monitorear y visualizar el pipeline CI/CD de GALTTCMC.

## 🚀 Características

- **Dashboard en tiempo real**: Métricas del pipeline, tasa de éxito, despliegues recientes
- **Historial de Pipeline Runs**: Vista completa de todas las ejecuciones con filtros
- **Visor de Logs**: Visualización de logs con búsqueda y filtrado en tiempo real
- **Resultados SonarQube**: Análisis de calidad de código con gráficos de tendencias
- **Modo oscuro**: Interfaz adaptable con soporte dark/light mode
- **Auto-refresh**: Actualización automática opcional de datos
- **Responsive**: Diseño adaptativo para escritorio, tablet y móvil

## 📋 Requisitos

- Python 3.6+
- SUSE Linux 15
- SQLite database (automático con el pipeline principal)
- Acceso al directorio `cicd/` y sus logs

## 🔧 Instalación

### Instalación Automática

```bash
cd /home/YOUR_USER/cicd
sudo ./install_web.sh install
```

Este script:
1. Verifica prerequisites (Python 3.6, pip)
2. Instala dependencias Python (Flask, Gunicorn, etc.)
3. Configura el firewall (puerto 8080)
4. Instala el servicio systemd
5. Inicia la aplicación web

### Instalación Manual

```bash
cd /home/YOUR_USER/cicd/web

# Instalar dependencias
python3.6 -m pip install -r requirements.txt

# Desarrollo (Flask dev server)
python3.6 app.py

# Producción (Gunicorn)
gunicorn --bind 0.0.0.0:8080 --workers 2 app:app
```

## 🌐 Acceso

Después de la instalación, accede a la Web UI en:

```
http://YOUR_PIPELINE_HOST_IP:8080
```

O desde cualquier navegador en la red:

```
http://<IP_DEL_SERVIDOR>:8080
```

## 🎮 Uso

### Gestión del Servicio

```bash
# Ver estado
sudo systemctl status cicd-web

# Iniciar servicio
sudo systemctl start cicd-web

# Detener servicio
sudo systemctl stop cicd-web

# Reiniciar servicio
sudo systemctl restart cicd-web

# Ver logs
sudo journalctl -u cicd-web -f

# Ver logs de aplicación
tail -f /home/YOUR_USER/cicd/logs/web_access.log
tail -f /home/YOUR_USER/cicd/logs/web_error.log
```

### Navegación

1. **Dashboard** (`/`): Vista general con métricas y gráficos
2. **Pipeline Runs** (`/pipeline-runs`): Historial completo con filtros por estado
3. **Logs** (`/logs`): Visor de archivos de log con búsqueda
4. **SonarQube Results** (`/sonar-results`): Análisis de calidad y tendencias

### Auto-refresh

Activa el toggle "Auto-refresh" en la barra superior para actualizar datos automáticamente cada 30 segundos.

## 🏗️ Arquitectura

### Stack Tecnológico

**Backend:**
- Flask 2.0.3 (framework web)
- Gunicorn 20.1.0 (WSGI server)
- SQLite (base de datos)
- PyYAML (configuración)

**Frontend:**
- Alpine.js 3.x (reactividad)
- Tailwind CSS 3.x (estilos)
- Chart.js 3.9.1 (gráficos)
- Vanilla JavaScript (utilidades)

### Estructura de Archivos

```
web/
├── app.py                 # Aplicación Flask principal
├── config.py              # Configuración de Flask
├── requirements.txt       # Dependencias Python
├── static/
│   ├── css/
│   │   └── style.css     # Estilos personalizados
│   └── js/
│       └── app.js        # Utilidades JavaScript
└── templates/
    ├── base.html         # Template base
    ├── dashboard.html    # Dashboard principal
    ├── pipeline_runs.html # Historial de runs
    ├── logs.html         # Visor de logs
    ├── sonar_results.html # Resultados SonarQube
    ├── 404.html          # Error 404
    └── 500.html          # Error 500
```

## 🔌 API Endpoints

La Web UI expone los siguientes endpoints REST:

### Dashboard
- `GET /api/dashboard/stats` - Estadísticas generales
- `GET /api/dashboard/recent-deployments` - Últimos 10 despliegues
- `GET /api/dashboard/chart-data` - Datos para gráficos (últimos 7 días)

### Deployments
- `GET /api/deployments?page=1&per_page=20&status=all` - Lista de deployments
- `GET /api/deployment/<id>` - Detalle de un deployment específico

### Logs
- `GET /api/logs/list` - Lista de archivos de log disponibles
- `GET /api/logs/view/<filename>?lines=500&search=error` - Contenido de un log

### SonarQube
- `GET /api/sonar/results` - Resultados de análisis SonarQube
- `GET /api/sonar/trends` - Tendencias de métricas (últimos 10 deployments)

## ⚙️ Configuración

### Variables de Entorno

Configura en `/home/YOUR_USER/cicd/config/.env` o variables de entorno:

```bash
# Web server
WEB_HOST=0.0.0.0
WEB_PORT=8080
WEB_DEBUG=false

# Flask
SECRET_KEY=your-secret-key-here
```

### Cambiar Puerto

Edita `/etc/systemd/system/cicd-web.service`:

```ini
Environment="WEB_PORT=9090"
```

Luego recarga:

```bash
sudo systemctl daemon-reload
sudo systemctl restart cicd-web
```

### Seguridad

El servicio incluye configuraciones de seguridad:
- `NoNewPrivileges=true`
- `PrivateTmp=true`
- `ProtectSystem=strict`
- Acceso read-only a `/home/agent` excepto `logs/` y `db/`

## 🐛 Troubleshooting

### El servicio no inicia

```bash
# Ver logs detallados
sudo journalctl -u cicd-web -n 100 --no-pager

# Verificar permisos
ls -la /home/YOUR_USER/cicd/web
ls -la /home/YOUR_USER/cicd/db/pipeline.db

# Probar manualmente
cd /home/YOUR_USER/cicd/web
python3.6 app.py
```

### No se ven datos

1. Verifica que la base de datos existe: `ls -la /home/YOUR_USER/cicd/db/pipeline.db`
2. Ejecuta el pipeline al menos una vez: `cd /home/YOUR_USER/cicd && ./ci_cd.sh --tag TEST_TAG`
3. Verifica permisos de lectura en la base de datos

### Error 500 al cargar logs

Verifica permisos en el directorio de logs:

```bash
chmod 755 /home/YOUR_USER/cicd/logs
chmod 644 /home/YOUR_USER/cicd/logs/*.log
```

### Puerto 8080 en uso

Cambia el puerto en el servicio systemd o mata el proceso:

```bash
# Ver qué usa el puerto
sudo lsof -i :8080

# Cambiar puerto (ver sección Configuración)
```

## 🔄 Actualización

Para actualizar la Web UI después de cambios en el código:

```bash
# Opción 1: Reiniciar servicio
sudo systemctl restart cicd-web

# Opción 2: Reinstalar completamente
sudo ./install_web.sh reinstall
```

## 📊 Monitoreo

### Logs de Aplicación

```bash
# Logs de acceso HTTP
tail -f /home/YOUR_USER/cicd/logs/web_access.log

# Logs de errores
tail -f /home/YOUR_USER/cicd/logs/web_error.log

# Logs de systemd
sudo journalctl -u cicd-web -f
```

### Métricas de Performance

```bash
# Ver uso de recursos del servicio
systemctl status cicd-web

# Detalles de memoria/CPU
sudo systemctl show cicd-web | grep -E "(Memory|CPU)"
```

## 🎨 Personalización

### Tema Dark/Light

El modo oscuro se activa con el botón en la barra superior. La preferencia se guarda en `localStorage`.

### Logo y Branding

Edita [templates/base.html](templates/base.html):

```html
<!-- Cambiar emoji y título -->
<span class="text-2xl mr-2">🚀</span>
<h1 class="text-white text-lg font-bold">TU TÍTULO</h1>
```

### Colores

Edita [static/css/style.css](static/css/style.css) o usa clases de Tailwind en los templates.

## 📝 Desinstalación

```bash
sudo ./install_web.sh uninstall
```

Esto:
1. Detiene el servicio
2. Deshabilita el servicio
3. Elimina el archivo de servicio systemd
4. NO elimina dependenciasPython ni archivos de la aplicación

## 📖 Documentación Adicional

- [README principal del proyecto](../README.md)
- [Guía de desarrollo](../CLAUDE.md)
- [Instrucciones de Copilot](.github/copilot-instructions.md)

## 🤝 Contribuir

Para añadir nuevas funcionalidades a la Web UI:

1. Crea un nuevo endpoint en `app.py`
2. Añade la vista en `templates/`
3. Actualiza Alpine.js data en el template si es necesario
4. Prueba manualmente con `python3.6 app.py`
5. Actualiza esta documentación

## 📄 Licencia

© 2026 Indra Sistemas - GALTTCMC Project

---

**Versión:** 1.0.0  
**Última actualización:** Febrero 2026
