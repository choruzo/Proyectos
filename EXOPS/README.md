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

## Funcionalidades UI

- **Modo oscuro/claro**: Toggle en la navbar. Persiste en `localStorage`; valor inicial desde `prefers-color-scheme` del SO. Sin parpadeo al cargar (script anti-flash en `<head>`).
