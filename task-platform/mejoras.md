# Mejoras propuestas (Task Platform)

Este documento recoge ideas de mejora basadas en el estado actual del repo (FastAPI + SQLAlchemy + Postgres + React/Vite, auth JWT con refresh cookie, tabla `users` con `is_admin`).

## 1) Multiusuario real + distinción Admin vs Usuario

**Estado:** ✅ Implementado (2026-04-16)

Incluye: `/auth/me`, `/auth/change-password`, endpoints `/admin/users*`, asignación `user_projects`, y enforcement de permisos (proyectos/releases solo admin; usuarios ven solo proyectos asignados; `/export` solo admin).

### Roles, acceso y flujo
- **Mantener `users.is_admin`** (ya existe) como flag de rol.
- Añadir endpoint **`GET /auth/me`** que devuelva `{ id, email, is_admin, is_active }` para que el frontend pueda decidir navegación.
- Ajustar frontend: tras `login()` y tras `tryRestoreSession()` → llamar a `/auth/me`.
- **Redirección automática**:
  - Si `is_admin=true` → ir a `/admin` (panel de administración).
  - Si `is_admin=false` → ir a la app normal (proyectos/tablero).

### Acceso a proyectos por usuario (asignación desde Admin)
- Cada usuario **no admin** solo puede ver/consultar los **proyectos que tenga asignados** desde el panel de administración.
- Si un usuario no tiene proyectos asignados → UI con estado vacío (“No tienes proyectos asignados”) y acción sugerida (“Contacta con un admin”).
- Backend:
  - Filtrar `GET /projects` por proyectos asignados al usuario.
  - En rutas con `project_id` (proyectos, releases, tareas, adjuntos, export por proyecto) validar que el usuario tiene acceso al proyecto.

### Permisos por rol (propuesta MVP)
- **Admin**:
  - Gestión de usuarios (roles/estado), **asignación/reasignación de proyectos**, creación/borrado/edición de proyectos y **versiones (releases)**.
  - Acceso al panel de métricas/overview (global y por proyecto).
- **Usuario**:
  - Solo puede ver proyectos/versions asignados.
  - Puede **seleccionar versiones** para ver tareas e **interactuar con tareas** (crear/editar, cambiar estado/prioridad, adjuntar evidencias) dentro de proyectos asignados.
  - **No puede crear** proyectos ni **versiones (releases)**.

### Gestión de usuarios y asignaciones (solo admin)
- Endpoints API (propuestos):
  - `GET /admin/users` (lista, filtros por activo/admin)
  - `POST /admin/users` (crear: email + password + is_admin)
  - `PATCH /admin/users/{id}` (activar/desactivar, promover a admin, etc.)
  - `POST /admin/users/{id}/reset-password` (reset a password temporal o establecer nueva)
  - `DELETE /admin/users/{id}` (o soft-delete: `is_active=false`)
  - `GET /admin/users/{id}/projects` (ver proyectos asignados)
  - `PUT /admin/users/{id}/projects` (reemplazar asignación completa) o `POST`/`DELETE` para asignar/desasignar
- Dependencia backend: `require_admin(current_user)` para proteger rutas + helper `require_project_access(current_user, project_id)`.

### Cambios de contraseña (self-service)
- Endpoint: `POST /auth/change-password` (requiere access token; valida contraseña actual).

## 2) Panel de administración (UI)

**Estado:** ✅ Implementado (2026-04-16)

Incluye rutas SPA `/admin/users`, `/admin/projects`, `/admin/stats` (con guardas) y soporte API `/admin/stats` + memberships por proyecto (`/admin/projects/{project_id}/users`).

### Objetivo
Panel dedicado a **administración, gestión y overview de métricas** por proyecto.

### Secciones
- **Usuarios**
  - Tabla con email, rol (admin/normal), estado (activo/inactivo), acciones.
  - Crear usuario (modal), desactivar, reset contraseña, cambiar rol.
- **Proyectos (gestión + asignaciones)**
  - Crear/borrar/editar proyectos (solo admin).
  - Asignar y desasignar proyectos a usuarios (y reasignar en bloque).
  - Vista “Quién tiene acceso a este proyecto” + búsqueda por usuario.
- **Versiones / Releases (solo admin)**
  - Crear/renombrar/borrar versiones de un proyecto.
  - Los usuarios no admin **no crean** versiones, pero sí pueden **seleccionarlas** para consumir el tablero/listado de tareas.
- **Estadísticas (overview global y por proyecto)**
  - Resumen por proyecto: #tareas por estado/prioridad, backlog vs release, vencidas, sin vencimiento.
  - Actividad reciente: conteo de eventos audit por proyecto (usa `audit_logs`).
  - Adjuntos: #adjuntos y total `size_bytes` por proyecto.

### Enrutado
- SPA routes (ejemplo): `/admin`, `/admin/users`, `/admin/projects`, `/admin/stats`.
- Guardas:
  - si no autenticado → login.
  - si autenticado pero no admin → 403 UI/volver a proyectos.

## 3) API de estadísticas

**Estado:** ✅ Implementado (2026-04-16)

- `GET /admin/stats` (admin-only, con `audit_days` + métricas: overdue/sin due_date, breakdown por release, adjuntos, actividad).
- `GET /projects/{project_id}/stats` (con `require_project_access`).

- `GET /admin/stats` (agregado global) y/o `GET /projects/{project_id}/stats`.
- Consultas típicas:
  - `COUNT(*) GROUP BY status, priority`.
  - `COUNT(*) WHERE release_id IS NULL` (backlog) vs por release.
  - vencidas: `due_date < today AND status != 'Done'`.
  - adjuntos: join Task→Attachment y sumar `size_bytes`.
  - actividad: `COUNT(*) FROM audit_logs WHERE project_id = ...` por ventana temporal.

## 4) Modelo de datos y ownership (para multiusuario)

**Estado:** ✅ Implementado (2026-04-16)

Incluye:
- Tabla `user_projects` (asignación de proyectos a usuarios).
- Ownership en entidades (`owner_user_id`, `created_by`, `updated_by`) vía Alembic + write paths que lo rellenan.

Si se quiere pasar de “single-user MVP” a multiusuario real:
- Añadir tabla de relación **`user_projects`** (o `project_memberships`):
  - Campos típicos: `user_id`, `project_id` (y opcional `role` dentro del proyecto).
  - `UNIQUE(user_id, project_id)` + índices por `user_id` y `project_id`.
  - Esta tabla es la base para: “cada usuario solo ve los proyectos asignados desde Admin”.
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

## 8) Asignación / “reclamar” tarea (Assignee)

**Estado:** 📝 Propuesto

Objetivo: que una tarea pueda estar **asignada a un usuario** (p.ej. “responsable”) y que un usuario pueda **reclamarla para sí** (auto-asignación) cuando esté libre.

### Modelo de datos
- Añadir en `tasks`:
  - `assigned_user_id` (nullable, FK a `users.id`).
  - (Opcional) `assigned_at` timestamp para trazabilidad.
- En respuestas de API incluir `assigned_user_id` y/o un objeto `assignee` mínimo (`{ id, email }`).

### Reglas y permisos (propuesta MVP)
- **Admin**: puede asignar/desasignar cualquier tarea de proyectos a los que tenga acceso.
- **Usuario normal**:
  - Puede **reclamar** una tarea **no asignada** de un proyecto al que tenga acceso.
  - Puede **liberar** una tarea si está asignada a sí mismo.
  - (Opcional) No puede asignar a otros usuarios.
- Conflicto: si la tarea ya está asignada a otro → devolver **409**.

### Endpoints sugeridos
- `PATCH /tasks/{task_id}`:
  - permitir `assigned_user_id` (admin-only) para asignar/desasignar.
- `POST /tasks/{task_id}/claim`:
  - asigna la tarea al `current_user` si `assigned_user_id IS NULL`.
- `POST /tasks/{task_id}/unclaim` (o `POST /tasks/{task_id}/release`):
  - desasigna si `assigned_user_id == current_user.id`.

### UI/UX
- En tarjeta/listado de tarea mostrar “Asignada a: …” (o icono/initials) y un filtro **“Mis tareas”**.
- Botón contextual:
  - si no asignada → “Reclamar”.
  - si asignada a mí → “Liberar”.
  - si asignada a otro → solo lectura.

### Auditoría
- Registrar eventos (ej.: `task_assigned`, `task_unassigned`, `task_claimed`) con before/after y `actor_user_id`.
