#!/bin/sh
set -e

cd /app

if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
  echo "[entrypoint] Running Alembic migrations..."

  FLAGS=$(python - <<'PY'
import os
from sqlalchemy import create_engine, text

url = os.environ.get("DATABASE_URL")
if not url:
    print("0 0")
    raise SystemExit(0)

engine = create_engine(url, pool_pre_ping=True)
with engine.connect() as conn:
    has_version = conn.scalar(text("select to_regclass('public.alembic_version')")) is not None
    has_users = conn.scalar(text("select to_regclass('public.users')")) is not None

print(f"{1 if has_version else 0} {1 if has_users else 0}")
PY
)

  set -- $FLAGS
  HAS_VERSION="$1"
  HAS_USERS="$2"

  # If DB already exists (created previously by create_all) but isn't stamped, stamp baseline then upgrade.
  if [ "$HAS_VERSION" = "0" ] && [ "$HAS_USERS" = "1" ]; then
    echo "[entrypoint] Existing DB detected without alembic_version; stamping baseline (0001_initial)"
    alembic stamp 0001_initial
  fi

  alembic upgrade head
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
