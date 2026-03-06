import secrets
import string
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    # Importar modelos para que SQLAlchemy los registre antes de create_all
    from app.auth.models import User  # noqa: F401
    from app.audit.models import AuditLog  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await create_first_admin()


async def create_first_admin() -> None:
    from sqlalchemy import select

    import bcrypt

    from app.auth.models import User

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.role == "admin").limit(1))
        existing_admin = result.scalar_one_or_none()

        if existing_admin is None:
            alphabet = string.ascii_letters + string.digits
            plain_password = "".join(secrets.choice(alphabet) for _ in range(16))
            hashed = bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()

            admin_user = User(
                username="admin",
                hashed_password=hashed,
                role="admin",
                is_active=True,
            )
            db.add(admin_user)
            await db.commit()

            logger.info("=" * 60)
            logger.info("PRIMER ARRANQUE: usuario admin creado")
            logger.info(f"  Usuario:     admin")
            logger.info(f"  Contraseña:  {plain_password}")
            logger.info("Guarda esta contraseña — no se mostrará de nuevo.")
            logger.info("=" * 60)
