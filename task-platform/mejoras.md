# Mejoras propuestas (Task Platform)

Este documento recoge ideas de mejora basadas en el estado actual del repo (FastAPI + SQLAlchemy + Postgres + React/Vite, auth JWT con refresh cookie, tabla `users` con `is_admin`).

## 1) Multiusuario real + distinción Admin vs Usuario

### Roles y flujo
- **Mantener `users.is_admin`** (ya existe) como flag de rol.
- Añadir endpoint **`GET /auth/me`** que devuelva `{ id, email, is_admin, is_active }` para que el frontend pueda decidir navegación.
- Ajustar frontend: tras `login()` y tras `tryRestoreSession()` → llamar a `/auth/me`.
- **Redirección automática**:
  - Si `is_admin=true` → ir a `/admin` (panel de administración).
  - Si `is_admin=false` → ir a la app normal (proyectos/tablero).

### Gestión de usuarios (solo admin)
- Endpoints API (propuestos):
  - `GET /admin/users` (lista, filtros por activo/admin)
  - `POST /admin/users` (crear: email + password + is_admin)
  - `PATCH /admin/users/{id}` (activar/desactivar, promover a admin, etc.)
  - `POST /admin/users/{id}/reset-password` (reset a password temporal o establecer nueva)
  - `DELETE /admin/users/{id}` (o soft-delete: `is_active=false`)
- Dependencia backend: `require_admin(current_user)` para proteger rutas.

### Cambios de contraseña (self-service)
- Endpoint: `POST /auth/change-password` (requiere access token; valida contraseña actual).

## 2) Panel de administración (UI)

### Secciones
- **Usuarios**
  - Tabla con email, rol (admin/normal), estado (activo/inactivo), acciones.
  - Crear usuario (modal), desactivar, reset contraseña, cambiar rol.
- **Estadísticas (por proyecto)**
  - Resumen por proyecto: #tareas por estado/prioridad, backlog vs release, vencidas, sin vencimiento.
  - Actividad reciente: conteo de eventos audit por proyecto (usa `audit_logs`).
  - Adjuntos: #adjuntos y total `size_bytes` por proyecto.

### Enrutado
- SPA routes (ejemplo): `/admin`, `/admin/users`, `/admin/stats`.
- Guardas:
  - si no autenticado → login.
  - si autenticado pero no admin → 403 UI/volver a proyectos.

## 3) API de estadísticas

- `GET /admin/stats` (agregado global) y/o `GET /projects/{project_id}/stats`.
- Consultas típicas:
  - `COUNT(*) GROUP BY status, priority`.
  - `COUNT(*) WHERE release_id IS NULL` (backlog) vs por release.
  - vencidas: `due_date < today AND status != 'Done'`.
  - adjuntos: join Task→Attachment y sumar `size_bytes`.
  - actividad: `COUNT(*) FROM audit_logs WHERE project_id = ...` por ventana temporal.

## 4) Modelo de datos y ownership (para multiusuario)

Si se quiere pasar de “single-user MVP” a multiusuario real:
- Añadir `created_by`, `updated_by` y/o `owner_user_id` en Project/Task/Attachment.
- En `audit_logs.actor_user_id` ya se registra actor; aprovechar para trazabilidad.

## 5) Autenticación / seguridad (ajustes concretos)

- Cookie refresh actualmente `secure=False` (válido en LAN/local). Considerar:
  - `secure=True` cuando se despliegue detrás de HTTPS.
  - configurar `CORS_ORIGINS` para incluir `http://127.0.0.1:8080` además de `http://localhost:8080` (evita problemas de CORS al abrir con IP vs hostname).
- (Opcional) Incluir `is_admin` como claim del JWT access **o** preferir `/auth/me` para evitar duplicar lógica.

## 6) Migraciones y evolución del esquema

- Hoy se usa `Base.metadata.create_all(...)` en startup (sin Alembic). Para cambios como nuevos campos/índices:
  - incorporar Alembic y migraciones (especialmente si se añade ownership, roles o stats materializadas).

## 7) UX/UI (alineado con “Duna Serena”)

- Admin UI con misma estética (tokens ya en `web/src/duna-serena.css`).
- En la app normal: mostrar un acceso discreto a Admin (solo admin): “Administración”.
- Añadir estados vacíos con mensajes y acciones (p.ej. sin proyectos / sin releases / sin tareas).
