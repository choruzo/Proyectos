# Plataforma de gestión de tareas (Web + Docker) — Requisitos y diseño

**Fecha:** 2026-04-13  
**Estado:** Borrador v0.1  

## 1) Contexto / problema
- Actualmente uso Obsidian para tareas, pero se vuelve monolítico y rígido.
- Quiero una plataforma propia, desplegable en **Docker**, con **interfaz web**, que permita **modelar mejor** las tareas y adaptar flujos/vistas.

## 2) Objetivo
Construir una aplicación web auto-alojada (self-hosted) que permita gestionar tareas con un modelo flexible (campos configurables, flujos, vistas guardadas) y que pueda convivir con/absorber el flujo actual basado en notas.

## 3) Criterios de éxito (medibles)
- Puedo crear/editar/buscar tareas desde web.
- Puedo definir estados, prioridades y campos (al menos un conjunto configurable).
- Puedo visualizar por **Kanban** y por **lista** con filtros.
- Se despliega con `docker compose up` en una máquina/servidor propio.
- Exportación/backup de datos sencilla.

## 4) Alcance
### 4.1 MVP (primera versión usable)
- CRUD de tareas.
- Proyectos para agrupar.
- **Versiones/Releases por proyecto** (tablero por versión).
- Estados (workflow) y prioridades.
- Etiquetas/tags.
- Vistas: Lista + Kanban.
- Filtros y búsqueda.
- Persistencia en BD (Postgres).
- **Campos fijos extra:** vencimiento + estimación (horas).
- **Login local** (1 usuario en MVP, preparado para multiusuario).
- **Adjuntos/evidencias:** subir archivos a una tarea.

### 4.2 Futuro (post-MVP)
- Campos personalizados por proyecto (custom fields).
- Automatizaciones (reglas: si X entonces Y).
- Subtareas, checklists.
- Recurrencias.
- Calendario / timeline.
- Comentarios, historial de cambios (audit log).
- Integraciones (GitHub Issues, Slack/Teams, correo, webhooks).
- Multi-tenant / equipos.

## 5) Usuarios y permisos
- **Decisión:** empezar con **1 usuario en el MVP**, con diseño preparado para **multiusuario** más adelante (tabla `users`, ownership por recurso).
- (Futuro) Roles: admin / miembro / lector.
- (Futuro) Permisos por proyecto/equipo.

## 6) Modelo de datos (borrador)
**Task**
- id, título, descripción (Markdown)
- estado (Backlog/Todo/Doing/Done), prioridad (Low/Medium/High)
- proyecto
- **release/version (opcional, para backlog)**
- etiquetas
- fechas: creada, actualizada; vencimiento_fecha (YYYY-MM-DD)
- estimación_horas (opcional, decimal; p.ej. 1.5)
- asignado a (opcional)
- links (opcional): URLs, referencias

**Project**
- id, nombre, descripción
- configuración de workflow
- (futuro) campos personalizados

**Release/Version**
- id, project_id
- nombre (p.ej. `v1.2.0`)
- objetivo/nota (opcional)
- fechas (opcional): inicio, fin
- estado (planificada/en curso/cerrada)

**Attachment (Adjunto)**
- id, task_id
- nombre_original, mime_type, tamaño_bytes
- ruta_storage (o key)
- subido_por (user_id)
- creado_en

**Tag / Etiqueta**
- id, nombre
- (sugerencias) project_id + lista sugerida (o derivada por uso)
- relación: Task <-> Tag (many-to-many)

## 7) Funcionalidades clave
- Editor Markdown para descripción.
- Filtros combinables (estado, prioridad, etiqueta, texto, proyecto, versión).
- Vistas guardadas.
- Tablero Kanban por **versión/release**.
- Búsqueda simple por texto (título + descripción).
- Adjuntar archivos a tareas (subida/descarga).
- Atajos rápidos (crear tarea, mover estado).

## 8) Requisitos no funcionales
- **Despliegue:** Docker Compose en **PC/local**, accesible en **red local (LAN)**.
- **BD:** **Postgres** desde el inicio.
- **Adjuntos:** límite **100 MB por archivo**, **permitir cualquier tipo**.
- **Backups:** export **JSON** + backup de volúmenes (**Postgres + uploads**).
- **Seguridad:** no expuesto a internet; autenticación local; opcional HTTPS si hay proxy. JWT con **refresh en cookie httpOnly**.
- **Rendimiento:** búsqueda rápida con índices.

## 9) Stack elegido (MVP)
- **Backend:** Python **FastAPI** (API REST) + SQLAlchemy
- **Frontend:** **React + Vite**
- **DB:** **Postgres**
- **Auth:** **JWT** (access token en memoria + refresh token en cookie httpOnly)
- **Infra local:** Docker Compose (proxy opcional)

### Alternativas (descartadas por ahora)
- Backend Node (NestJS) o Go.
- Auth SSO (Keycloak) si hay equipo/empresa.

## 10) Arquitectura Docker (borrador)
- `web` (frontend)
- `api` (backend)
- `db` (postgres)
- volumen `uploads` (adjuntos) montado en `api`
- (opcional) `proxy` (caddy/traefik)

## 11) Migración / integración con Obsidian
- **Decisión:** MVP **independiente**, sin import/export con Obsidian.
- (Futuro) Importar/exportar Markdown/JSON si se necesita.

## 12) Decisiones pendientes / preguntas abiertas
(Se irán cerrando en orden.)
- Alcance: ✅ **MVP 1 usuario**, con ruta de evolución a multiusuario.
- Despliegue: ✅ **PC/local, solo LAN**.
- BD: ✅ **Postgres**.
- Modelo: ✅ **Proyecto + Versión/Release**.
- Regla de planificación: ✅ **0/1 versión**.
- Workflow: ✅ **fijo en MVP** (Backlog/Todo/Doing/Done).
- Auth: ✅ **login local**, preparado para multiusuario.
- Adjuntos: ✅ **sí en MVP** → límite **100 MB/archivo** ✅; **permitir cualquier tipo** ✅; ✅ borrar adjuntos al borrar tarea.
- Custom fields: ✅ **no configurables en MVP**; campos fijos extra: ✅ **vencimiento + estimación**.
- Integración con Obsidian: ✅ **ninguna** (app independiente).

---

## 13) Registro de decisiones (ADR-lite)
- **2026-04-13:** El **MVP** será para **1 usuario**, pero la arquitectura/datos se diseñan para habilitar **multiusuario** posteriormente sin reescritura grande.
- **2026-04-13:** Despliegue objetivo del MVP: **PC/local, acceso solo por LAN**.
- **2026-04-13:** Base de datos del MVP: **Postgres**.
- **2026-04-13:** Organización principal: **Proyecto + Versión/Release**.
- **2026-04-13:** Regla de planificación: **tarea pertenece a 0/1 versión** (backlog o una versión).
- **2026-04-13:** Workflow del MVP: **fijo** (Backlog/Todo/Doing/Done).
- **2026-04-13:** Autenticación MVP: **login local**, preparado para multiusuario.
- **2026-04-13:** Adjuntos: **incluidos en MVP** (subida desde web), **100 MB/archivo**, **cualquier tipo**.
- **2026-04-13:** Integración con Obsidian: **ninguna en MVP** (independiente).
- **2026-04-13:** Campos: sin custom fields en MVP; campos fijos extra: **vencimiento + estimación**.
- **2026-04-13:** Stack MVP: **FastAPI + React (Vite) + Postgres**.
- **2026-04-13:** Auth MVP: **JWT** (access en memoria + refresh cookie httpOnly).
- **2026-04-13:** Estimación: en **horas (decimal)**.
- **2026-04-13:** Vencimiento: **solo fecha** (sin hora).
- **2026-04-13:** Prioridad: **Low/Medium/High**.
- **2026-04-13:** Tags: **libres** con **sugerencias por proyecto**.
- **2026-04-13:** Búsqueda MVP: **simple** (contiene en título + descripción).
- **2026-04-13:** Kanban: **drag & drop** entre columnas.
- **2026-04-13:** Orden Kanban: **automático** (prioridad + vencimiento).
- **2026-04-13:** Vencimientos: sin notificaciones; **solo indicadores visuales**.
- **2026-04-13:** UI MVP: **escritorio** (responsive móvil post-MVP).
- **2026-04-13:** Subtareas/checklists: **post-MVP**.
- **2026-04-13:** Backups MVP: **volúmenes + export JSON** (sin adjuntos en export).
- **2026-04-13:** Adjuntos: al borrar tarea → **borrado en cascada**.
