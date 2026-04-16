from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.models.base import Base
from app.models.user import User

# Import all models so they are registered in Base.metadata
import app.models  # noqa: F401


def init_db(db: Session) -> None:
    # Create tables (MVP). Later we can switch to Alembic migrations.
    # When running in Docker, Postgres might not be ready the instant the API starts.
    last_err: Exception | None = None
    for _attempt in range(30):
        try:
            Base.metadata.create_all(bind=db.get_bind())
            last_err = None
            break
        except OperationalError as e:
            last_err = e
            time.sleep(1)

    if last_err is not None:
        raise last_err

    admin = db.scalar(select(User).where(User.email == settings.admin_email))
    if admin is None:
        admin = User(
            email=settings.admin_email,
            password_hash=hash_password(settings.admin_password),
            is_admin=True,
        )
        db.add(admin)
        db.commit()
    else:
        # Single-user MVP convenience: allow rotating admin password via .env
        if not verify_password(settings.admin_password, admin.password_hash):
            admin.password_hash = hash_password(settings.admin_password)
            db.commit()
