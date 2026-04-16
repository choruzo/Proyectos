# Task Platform (MVP)

MVP auto-alojado (LAN) para gestión de tareas por **Proyecto + Versión/Release**.

## Requisitos
- Docker + Docker Compose

## Arranque rápido
```powershell
cd task-platform
Copy-Item .env.example .env
# (opcional) edita .env

docker compose up --build -d
```

- Web: http://localhost:8080
- Admin UI (solo admin): http://localhost:8080/admin/users
- API (docs): http://localhost:8000/docs

## Qué incluye el MVP
- Login local (multiusuario) con distinción **Admin** vs **Usuario**.
- **Panel de administración** (solo admin):
  - Usuarios (crear/activar/desactivar, reset password, promover a admin).
  - Proyectos (crear/editar/borrar) + asignación de accesos ("quién ve qué proyecto").
  - Versiones/Releases (crear/renombrar/borrar) por proyecto.
  - Estadísticas (overview global y por proyecto) con gráficos.
- **Acceso por proyecto**: un usuario no-admin solo ve los proyectos que le asigne un admin.
- Permisos:
  - Admin: gestiona proyectos y releases.
  - Usuario: puede trabajar con tareas dentro de proyectos asignados (kanban/lista, filtros, adjuntos, presets) pero **no crea** proyectos ni releases.
- Campos de tareas: prioridad, estado, vencimiento, estimación (horas), tags.
- Adjuntos: subir/descargar/borrar (100MB por archivo).
- Presets (vistas guardadas) por proyecto.
- Export JSON: **solo admin** (UI y endpoint `/export`).

## Credenciales iniciales
Se crea un usuario admin al arrancar si no existe (variables en `.env`):
- `ADMIN_EMAIL` (ej: `admin@example.com`)
- `ADMIN_PASSWORD`

Después, entra en el panel admin (`/admin/users`) para **crear usuarios** y **asignarles proyectos**.

## Persistencia (tareas no deberían “perderse”)
Las tareas/proyectos y adjuntos se guardan en **volúmenes Docker** (`db_data` y `uploads`).

- `docker compose up --build` / rebuild de contenedores **NO borra** los datos.
- Se perderán si ejecutas **`docker compose down -v`** o si eliminas volúmenes (`docker volume rm ...`).
- También puede parecer que “faltan tareas” si estás viendo **Backlog** pero esas tareas están asignadas a una **Versión/Release** (usa el selector **Versión**).

Para que los volúmenes sean estables aunque cambie el nombre de la carpeta/proyecto, los nombres están fijados en `docker-compose.yml` (`task-platform_db_data` y `task-platform_uploads`). Si prefieres, también puedes usar `COMPOSE_PROJECT_NAME=task-platform` en `.env` (ya viene en `.env.example`).

## Backups / Restore
### Export lógico (JSON)
- En la UI (solo admin): botón **Exportar JSON**.
- Endpoint (solo admin): `GET http://localhost:8000/export` (requiere auth).

### Backup de Postgres (SQL)
```powershell
# genera un dump en el host
cd task-platform
docker compose exec -T db pg_dump -U $env:POSTGRES_USER -d $env:POSTGRES_DB > backup.sql
```

Restore:
```powershell
cd task-platform
Get-Content -Raw .\backup.sql | docker compose exec -T db psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
```

### Backup de uploads (adjuntos)
```powershell
cd task-platform
# copia el volumen de uploads desde el contenedor api
docker compose cp api:/data/uploads .\uploads-backup
```

Restore (sobrescribe):
```powershell
cd task-platform
docker compose cp .\uploads-backup\. api:/data/uploads
```
