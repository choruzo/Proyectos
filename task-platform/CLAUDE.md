# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- **Backend**: FastAPI + SQLAlchemy (sync) + Alembic + Postgres 16 — lives in `api/`
- **Frontend**: React 18 + TypeScript + Vite (no UI framework; custom CSS in `duna-serena.css`) — lives in `web/`
- **Runtime**: Docker Compose (`db` + `api` + `web`/nginx)

## Commands

### Full stack
```bash
cp .env.example .env   # first time only
docker compose up --build -d

docker compose logs -f api     # tail API logs
docker compose down            # stop (data preserved)
docker compose down -v         # stop + wipe volumes (destructive)
```

### Frontend only (hot reload)
```bash
cd web
npm install
npm run dev      # Vite dev server
npm run build    # tsc + vite build
```

### Backend DB migrations (Alembic)
```bash
# new DB
docker compose exec api alembic upgrade head

# existing DB created before Alembic was added
docker compose exec api alembic stamp 0001_initial

# check state
docker compose exec api alembic current

# create new migration
docker compose exec api alembic revision --autogenerate -m "..."
docker compose exec api alembic upgrade head
```

### No tests or linters are configured.

## Architecture

### Services
| Container | Port | Notes |
|-----------|------|-------|
| `db` | 5432 | Postgres 16, volume `task-platform_db_data` |
| `api` | 8000 | FastAPI; uploads stored at `/data/uploads` (volume `task-platform_uploads`) |
| `web` | 8080 | React SPA served by nginx with SPA fallback |

### Backend (`api/app/`)
- `main.py` — FastAPI app; all routers mounted here; `init_db()` runs at startup (creates tables + seeds admin user).
- `core/config.py` — `Settings` via `pydantic-settings`; reads from **env vars only** (`.env` is NOT auto-loaded by the app, only by Docker Compose).
- `core/constants.py` — canonical `TASK_STATUSES` and `TASK_PRIORITIES` enums; validate against these in routes.
- `api/routes/` — one file per resource: `auth`, `admin`, `projects`, `releases`, `tasks`, `attachments`, `tags`, `views`, `audit`, `export`.
- `services/audit.py` — `log_event(...)` must be called on every create/update/delete path with before/after snapshots.
- `db/init_db.py` — on startup: retries for Postgres readiness, runs `Base.metadata.create_all`, stamps Alembic baseline if needed, seeds/updates admin.

### Frontend (`web/src/`)
- `api/client.ts` — Axios-like fetch wrapper; keeps access token **in memory** (never localStorage); all requests use `credentials: 'include'` for the refresh-token httpOnly cookie.
- `VITE_API_URL` is **baked at image build time** via Docker build arg — rebuild the `web` image after changing it.
- No state management library; state lives in component/context.

### Auth flow
1. `POST /auth/login` → access token in JSON body + refresh token as `httpOnly` cookie.
2. `POST /auth/refresh` → rotates cookie, returns new access token.
3. Frontend stores access token in memory; refresh is transparent.

## Key conventions

- **Backlog**: `Task.release_id IS NULL`. Use `list_tasks(..., backlog=True)` — never filter by a special status value.
- **Duplicate names** for projects and saved views return HTTP **409**.
- **Saved views**: `SavedView.config_json` stores JSON as a string (`json.dumps`/`json.loads` at the route boundary).
- **Attachments**: server-side hard limit of **100 MB** per file. Storage key: `{task_id}/{uuid}`.
- **Admin-only** endpoints: `GET /export`, all `/admin/*` routes — guarded by `get_current_user()` + is_admin check.
- **Volume names are fixed** in `docker-compose.yml` (`task-platform_db_data`, `task-platform_uploads`) so data survives folder renames.
