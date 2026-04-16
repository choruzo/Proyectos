# Copilot instructions (task-platform)

## Build / run / dev commands

### Full stack (recommended)
From repo root:

```powershell
Copy-Item .env.example .env
# (optional) edit .env

docker compose up --build -d
```

URLs:
- Web: http://localhost:8080
- API (OpenAPI docs): http://localhost:8000/docs

Useful ops:
```powershell
# follow logs
docker compose logs -f api

# stop
docker compose down

# wipe DB + uploads volumes (destructive)
docker compose down -v
```

### Frontend (web/)
```powershell
cd web
npm install
npm run dev

npm run build
npm run preview
```

Notes:
- `VITE_API_URL` is used by `web/src/api/client.ts` and is **baked into the Docker image at build time** via `docker-compose.yml` build args → rebuild the `web` image after changing it.

### Backend (api/)
Docker runs:
- `uvicorn app.main:app --host 0.0.0.0 --port 8000` (see `api/Dockerfile`)

For running outside Docker, install deps and run uvicorn similarly. Config is **environment-variable driven** (see `app/core/config.py`); `.env` is not auto-loaded by the app.

### Tests / lint
No test runner or lint script is currently configured (no `pytest`, `ruff`, `eslint`, etc.).

## High-level architecture

### Containers and boundaries
- `db` (Postgres 16): persistent volume `db_data`.
- `api` (FastAPI + SQLAlchemy): talks to Postgres via `DATABASE_URL`; stores attachments under `/data/uploads` (volume `uploads`).
- `web` (React + Vite build): static bundle served by nginx; SPA fallback configured in `web/nginx.conf`.

### Backend structure (api/app)
- Entry point: `app/main.py`.
- Routers live in `app/api/routes/` and are mounted in `main.py`:
  - `/auth/*` (login/refresh/logout)
  - `/projects/*`, `/projects/{project_id}/releases/*`, `/tasks/*`, attachments, tags, saved views, audit, `/export`
- DB init: `app/db/init_db.py` runs on FastAPI startup:
  - `Base.metadata.create_all(...)` (no Alembic migrations yet; retries for Postgres readiness)
  - seeds/updates the admin user from `ADMIN_EMAIL`/`ADMIN_PASSWORD`

### Auth model (API + web coupling)
- `POST /auth/login` returns an **access token** in JSON and sets a **refresh token** as an `httpOnly` cookie.
- `POST /auth/refresh` rotates the refresh cookie and returns a new access token.
- Frontend keeps the access token **in memory** (`web/src/api/client.ts`) and does `credentials: 'include'` on requests so refresh works.

### Attachments
- API stores files on disk (uploads volume). Storage key format: `{task_id}/{uuid}`.
- Hard limit enforced server-side: **100MB** per file (`MAX_BYTES`).

### Export / backup
- `GET /export` returns a JSON dump of the main tables (projects, releases, tasks, tags, attachments, views, audit logs).
- README.md documents Docker-based Postgres dump + uploads copy procedures.

## Key conventions (project-specific)

- **Single-user MVP but multi-user-shaped data:** the DB has a `users` table; most routes depend on `get_current_user()`.
- **Task planning rule:** backlog is represented by `Task.release_id == NULL` (`list_tasks(..., backlog=true)`), otherwise tasks are filtered by `release_id`.
- **Allowed enums are centralized:** `TASK_STATUSES` and `TASK_PRIORITIES` in `app/core/constants.py` and validated in routes.
- **Audit logging is part of write paths:** create/update/delete endpoints call `app/services/audit.py::log_event(...)` with before/after/meta snapshots.
- **Saved views store JSON configs as strings:** `SavedView.config_json` persists `config` via `json.dumps(...)` and is parsed on output in `views` routes.
- **Conflict behavior:** duplicate names for projects and views return HTTP **409** (see `projects.py`, `views.py`).
