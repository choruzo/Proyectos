# EXOPS - vCenter Web Administration Platform

Aplicación web para la administración de entornos VMware vSphere 8.x.
Interfaz moderna, segura y accesible desde cualquier navegador dentro de la red interna.

## Documentación del Proyecto

| Documento | Descripción |
|-----------|-------------|
| [docs/objetivos.md](docs/objetivos.md) | Objetivos, alcance y casos de uso |
| [docs/arquitectura.md](docs/arquitectura.md) | Arquitectura técnica y stack tecnológico |
| [docs/requisitos.md](docs/requisitos.md) | Requisitos funcionales y no funcionales |
| [docs/librerias.md](docs/librerias.md) | Librerías y dependencias del proyecto |
| [docs/seguridad.md](docs/seguridad.md) | Modelo de seguridad, autenticación y HTTPS |
| [docs/bbdd.md](docs/bbdd.md) | Modelo de base de datos |

## Stack Principal

- **Backend**: Python 3.12+ / FastAPI
- **Frontend**: HTML5 / CSS3 / JavaScript / Bootstrap 5.3
- **vSphere API**: pyVmomi (VMware vSphere 8.x)
- **Base de datos**: SQLite
- **Servidor**: Uvicorn con TLS (HTTPS self-signed)
- **Despliegue**: VM Linux interna

## Funcionalidades implementadas

### Dashboard principal
- **4 tarjetas KPI en tiempo real**: VMs encendidas, hosts conectados, datastore más lleno y espacio libre total de datastores.
- **Auto-refresh cada 30 s** sin parpadeo (modo silencioso). Botón Refresh manual con spinner de carga.
- Datos obtenidos en paralelo con `Promise.all` desde los endpoints `/api/v1/vms/`, `/api/v1/hosts/` y `/api/v1/datastores/`.
- Colores adaptativos (success/warning/danger) según nivel de uso del datastore.

### Colector de métricas en background (opcional)
- **`asyncio.Task` de larga duración** que se conecta al vCenter con una cuenta de servicio propia y recopila 10 KPIs cada N minutos (por defecto 15).
- Independiente de las sesiones de usuario: persiste snapshots aunque nadie esté navegando el dashboard.
- Completamente opcional: si las variables `VCENTER_SERVICE_*` no están en `.env`, el comportamiento actual no cambia.
- Reconexión automática ante errores de red. Primera colecta inmediata al arrancar.
- Historial accesible vía `GET /api/v1/metrics/history?hours=N` (máximo 7 días).

### Gestión de VMs
- Listado con estado de encendido, recursos asignados y SO.
- Control de energía: encender, apagar, suspender, reiniciar.

### Gestión de Hosts ESXi
- Listado con uso de CPU/RAM en tiempo real y estado de conexión.
- Modo mantenimiento: entrar/salir con confirmación modal.

### Gestión de Datastores
- Listado con capacidad, espacio libre y porcentaje de uso.
- Ordenados por uso descendente.

### UI general
- **Modo oscuro/claro**: Toggle en la navbar. Persiste en `localStorage`; valor inicial desde `prefers-color-scheme` del SO. Sin parpadeo al cargar (script anti-flash en `<head>`).
- Toast de feedback para acciones y errores (patrón uniforme en todas las páginas).

## Variables de entorno

Copia `.env.example` a `.env` y edita los valores. Las variables marcadas con `#` son opcionales.

| Variable | Requerida | Descripción |
|---|---|---|
| `JWT_SECRET_KEY` | Sí | Secreto largo y aleatorio para firmar tokens |
| `JWT_EXPIRE_MINUTES` | No (60) | Duración de la sesión en minutos |
| `APP_ENV` | No (development) | `development` o `production` |
| `DEV_MODE` | No (false) | `true` omite la conexión real al vCenter |
| `DATABASE_URL` | No | Ruta SQLite (`sqlite+aiosqlite:///./data/exops.db`) |
| `SSL_CERT_PATH` | No | Ruta al certificado TLS |
| `SSL_KEY_PATH` | No | Ruta a la clave privada TLS |
| `VCENTER_SERVICE_HOST` | No | Host vCenter para el colector en background |
| `VCENTER_SERVICE_USER` | No | Usuario de la cuenta de servicio vCenter |
| `VCENTER_SERVICE_PASS` | No | Contraseña de la cuenta de servicio |
| `VCENTER_METRICS_INTERVAL_MINUTES` | No (15) | Frecuencia de colecta en minutos |
