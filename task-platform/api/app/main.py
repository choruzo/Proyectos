from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, attachments, audit, auth, export, projects, releases, tags, tasks, views
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.init_db import init_db

app = FastAPI(title="Task Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    with SessionLocal() as db:
        init_db(db)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(projects.router)
app.include_router(releases.router)
app.include_router(tasks.router)
app.include_router(attachments.router)
app.include_router(tags.router)
app.include_router(views.router)
app.include_router(audit.router)
app.include_router(export.router)
