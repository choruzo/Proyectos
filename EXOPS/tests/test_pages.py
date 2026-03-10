"""Tests de integración para las páginas HTML."""
import pytest

from app.auth.service import create_access_token


def _set_token(client, role="admin"):
    token = create_access_token({"sub": "1", "username": "testuser", "role": role})
    client.cookies.set("access_token", token)
    return client


# --- Rutas sin autenticación → redirigen a /login ---

async def test_root_unauthenticated(client):
    resp = await client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/login"


async def test_dashboard_unauthenticated(client):
    resp = await client.get("/dashboard", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_vms_page_unauthenticated(client):
    resp = await client.get("/vms", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_hosts_page_unauthenticated(client):
    resp = await client.get("/hosts", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_datastores_page_unauthenticated(client):
    resp = await client.get("/datastores", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_snapshots_page_unauthenticated(client):
    resp = await client.get("/snapshots", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_audit_page_unauthenticated(client):
    resp = await client.get("/audit", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


async def test_users_page_unauthenticated(client):
    resp = await client.get("/users", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/login" in resp.headers["location"]


# --- Rutas con autenticación → devuelven HTML ---

async def test_dashboard_authenticated(client):
    _set_token(client)
    resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert b"html" in resp.content.lower()


async def test_vms_page_authenticated(client):
    _set_token(client)
    resp = await client.get("/vms")
    assert resp.status_code == 200


async def test_hosts_page_authenticated(client):
    _set_token(client)
    resp = await client.get("/hosts")
    assert resp.status_code == 200


async def test_datastores_page_authenticated(client):
    _set_token(client)
    resp = await client.get("/datastores")
    assert resp.status_code == 200


async def test_snapshots_page_authenticated(client):
    _set_token(client)
    resp = await client.get("/snapshots")
    assert resp.status_code == 200


async def test_audit_page_authenticated(client):
    _set_token(client)
    resp = await client.get("/audit")
    assert resp.status_code == 200


async def test_users_page_admin(client):
    _set_token(client, role="admin")
    resp = await client.get("/users")
    assert resp.status_code == 200


async def test_users_page_operator_redirected(client):
    """Un operator es redirigido a /dashboard al intentar acceder a /users."""
    _set_token(client, role="operator")
    resp = await client.get("/users", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/dashboard" in resp.headers["location"]


async def test_login_page_unauthenticated(client):
    resp = await client.get("/login")
    assert resp.status_code == 200


async def test_login_page_authenticated_redirects(client):
    """Si ya está autenticado, /login redirige a /dashboard."""
    _set_token(client)
    resp = await client.get("/login", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/dashboard" in resp.headers["location"]


async def test_root_authenticated_redirects(client):
    """Si ya está autenticado, / redirige a /dashboard."""
    _set_token(client)
    resp = await client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/dashboard" in resp.headers["location"]
