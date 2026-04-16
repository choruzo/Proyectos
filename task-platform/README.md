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
- API (docs): http://localhost:8000/docs

## Qué incluye el MVP
- Login local (1 usuario admin).
- Proyectos + versiones (crear/editar/borrar).
- Tareas: Kanban (drag & drop) + Lista, filtros y búsqueda.
- Campos: prioridad, estado, vencimiento, estimación (horas), tags.
- Adjuntos: subir/descargar/borrar (100MB por archivo).
- Presets (vistas guardadas) por proyecto.
- Export JSON (desde la UI o endpoint `/export`).

## Credenciales iniciales
Se crea un usuario admin al arrancar si no existe (variables en `.env`):
- `ADMIN_EMAIL` (ej: `admin@example.com`)
- `ADMIN_PASSWORD`

## Backups / Restore
### Export lógico (JSON)
- En la UI: botón **Exportar JSON**.
- Endpoint: `GET http://localhost:8000/export` (requiere auth).

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
