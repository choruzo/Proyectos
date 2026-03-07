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
