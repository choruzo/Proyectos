from fastapi import APIRouter, Cookie, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional

from app.auth.service import decode_token

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="templates")


def _get_user_from_cookie(access_token: Optional[str] = Cookie(default=None)):
    """Devuelve TokenData si el cookie es válido, None si no."""
    if not access_token:
        return None
    try:
        return decode_token(access_token)
    except Exception:
        return None


@router.get("/", response_class=HTMLResponse)
async def root(access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if token_data:
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if token_data:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": token_data},
    )


@router.get("/vms", response_class=HTMLResponse)
async def vms_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("vms.html", {"request": request, "user": token_data})


@router.get("/hosts", response_class=HTMLResponse)
async def hosts_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("hosts.html", {"request": request, "user": token_data})


@router.get("/datastores", response_class=HTMLResponse)
async def datastores_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(
        "datastores.html", {"request": request, "user": token_data}
    )


@router.get("/snapshots", response_class=HTMLResponse)
async def snapshots_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("snapshots.html", {"request": request, "user": token_data})


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("audit.html", {"request": request, "user": token_data})


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request, access_token: Optional[str] = Cookie(default=None)):
    token_data = _get_user_from_cookie(access_token)
    if not token_data:
        return RedirectResponse(url="/login")
    if token_data.role != "admin":
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("users.html", {"request": request, "user": token_data})
