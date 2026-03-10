from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.database import init_db
from app.metrics.collector import collector as metrics_collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando EXOPS...")
    await init_db()
    logger.info("Base de datos lista.")

    if metrics_collector is not None:
        metrics_collector.start()

    yield

    if metrics_collector is not None:
        await metrics_collector.stop()

    logger.info("EXOPS detenido.")


app = FastAPI(
    title="EXOPS",
    version="1.0.0",
    description="Administración de entornos VMware vSphere 8.x",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# --- Archivos estáticos ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- CORS restringido ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Sin orígenes externos permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- Headers de seguridad ---
@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


# --- Routers API ---
from app.auth.router import router as auth_router
from app.audit.router import router as audit_router
from app.api.v1.vms import router as vms_router
from app.api.v1.hosts import router as hosts_router
from app.api.v1.datastores import router as datastores_router
from app.api.v1.snapshots import router as snapshots_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.users import router as users_router
from app.pages.router import router as pages_router

app.include_router(auth_router)
app.include_router(audit_router)
app.include_router(vms_router)
app.include_router(hosts_router)
app.include_router(datastores_router)
app.include_router(snapshots_router)
app.include_router(metrics_router)
app.include_router(users_router)
app.include_router(pages_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8443,
        ssl_keyfile=settings.SSL_KEY_PATH,
        ssl_certfile=settings.SSL_CERT_PATH,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
