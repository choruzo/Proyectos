#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestion de usuarios de la Web UI — GALTTCMC CI/CD

Uso:
    python3.6 manage_web_users.py list
    python3.6 manage_web_users.py add <usuario> <contrasena> [--admin]
    python3.6 manage_web_users.py remove <usuario>
    python3.6 manage_web_users.py change-password <usuario> <nueva_contrasena>
    python3.6 manage_web_users.py set-admin <usuario>
    python3.6 manage_web_users.py unset-admin <usuario>

Solo los administradores pueden gestionar usuarios desde la Web UI. Si no hay
ningun administrador activo, el usuario que se cree pasa a serlo automaticamente.

Requisitos:
    - Ejecutar desde el directorio cicd/ o cicd/python/
    - La base de datos db/pipeline.db debe existir (./ci_cd.sh init)
    - werkzeug instalado (dependencia de Flask)
"""

from __future__ import print_function
import os
import sys
import sqlite3
from datetime import datetime

# werkzeug.security esta disponible como dependencia de Flask
from werkzeug.security import generate_password_hash, check_password_hash

# Ruta a la base de datos relativa a este script (cicd/python/ -> ../db/pipeline.db)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'db', 'pipeline.db')

MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64


def get_db():
    """Abre conexion a la base de datos con comprobacion de existencia."""
    if not os.path.exists(DB_PATH):
        print("ERROR: Base de datos no encontrada en {}.".format(DB_PATH), file=sys.stderr)
        print("Ejecuta './ci_cd.sh init' primero para inicializarla.", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def password_stamp():
    """Marca de cambio de contrasena: al cambiar, la Web cierra las sesiones previas."""
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


def ensure_table(conn):
    """Crea la tabla web_users si no existe (migracion en caliente).

    Misma migracion que web/app.py: columnas is_admin y password_changed_at; al
    añadir is_admin, 'admin' (o el usuario activo mas antiguo) pasa a administrador.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS web_users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               username TEXT NOT NULL UNIQUE,
               password_hash TEXT NOT NULL,
               is_active INTEGER NOT NULL DEFAULT 1,
               created_at TEXT DEFAULT (datetime('now')),
               last_login TEXT
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_web_users_username ON web_users(username)"
    )
    cols = set(row[1] for row in conn.execute('PRAGMA table_info(web_users)').fetchall())
    if 'password_changed_at' not in cols:
        conn.execute('ALTER TABLE web_users ADD COLUMN password_changed_at TEXT')
    if 'is_admin' not in cols:
        conn.execute('ALTER TABLE web_users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0')
        conn.execute("UPDATE web_users SET is_admin = 1 WHERE username = 'admin'")
        if not conn.execute('SELECT 1 FROM web_users WHERE is_admin = 1').fetchone():
            conn.execute('UPDATE web_users SET is_admin = 1 WHERE id = '
                         '(SELECT MIN(id) FROM web_users WHERE is_active = 1)')
    conn.commit()


def cmd_add(username, password, is_admin=False):
    """Crea un nuevo usuario activo con la contrasena hasheada."""
    if not username or len(username) > MAX_USERNAME_LENGTH:
        print("ERROR: El nombre de usuario debe tener entre 1 y {} caracteres.".format(
            MAX_USERNAME_LENGTH), file=sys.stderr)
        sys.exit(1)
    if len(password) < MIN_PASSWORD_LENGTH:
        print("ERROR: La contrasena debe tener al menos {} caracteres.".format(
            MIN_PASSWORD_LENGTH), file=sys.stderr)
        sys.exit(1)

    password_hash = generate_password_hash(password)
    conn = get_db()
    ensure_table(conn)
    if not conn.execute('SELECT 1 FROM web_users WHERE is_admin = 1 AND is_active = 1').fetchone():
        is_admin = True
    try:
        conn.execute(
            'INSERT INTO web_users (username, password_hash, is_admin, password_changed_at) VALUES (?, ?, ?, ?)',
            (username, password_hash, 1 if is_admin else 0, password_stamp())
        )
        conn.commit()
        print("Usuario '{}' creado correctamente{}.".format(
            username, ' (administrador)' if is_admin else ''))
    except sqlite3.IntegrityError:
        print("ERROR: El usuario '{}' ya existe.".format(username), file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def cmd_remove(username):
    """Desactiva un usuario (no lo elimina fisicamente para conservar el historial)."""
    conn = get_db()
    ensure_table(conn)
    cursor = conn.execute(
        'UPDATE web_users SET is_active = 0 WHERE username = ? AND is_active = 1',
        (username,)
    )
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        print("ERROR: Usuario '{}' no encontrado o ya estaba desactivado.".format(
            username), file=sys.stderr)
        sys.exit(1)
    print("Usuario '{}' desactivado correctamente.".format(username))


def cmd_list():
    """Lista todos los usuarios con su estado, rol y fechas."""
    conn = get_db()
    ensure_table(conn)
    users = conn.execute(
        'SELECT username, is_active, is_admin, created_at, last_login FROM web_users ORDER BY username'
    ).fetchall()
    conn.close()

    if not users:
        print("No hay usuarios registrados.")
        return

    header = "{:<24} {:<8} {:<7} {:<20} {:<20}".format('USUARIO', 'ACTIVO', 'ADMIN', 'CREADO', 'ULTIMO LOGIN')
    print(header)
    print('-' * len(header))
    for u in users:
        last = u['last_login'] if u['last_login'] else 'Nunca'
        activo = 'Si' if u['is_active'] else 'No'
        admin = 'Si' if u['is_admin'] else 'No'
        print("{:<24} {:<8} {:<7} {:<20} {:<20}".format(
            u['username'], activo, admin, u['created_at'] or '', last))


def cmd_change_password(username, new_password):
    """Actualiza la contrasena de un usuario activo (cierra sus sesiones en la Web)."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        print("ERROR: La contrasena debe tener al menos {} caracteres.".format(
            MIN_PASSWORD_LENGTH), file=sys.stderr)
        sys.exit(1)

    password_hash = generate_password_hash(new_password)
    conn = get_db()
    ensure_table(conn)
    cursor = conn.execute(
        'UPDATE web_users SET password_hash = ?, password_changed_at = ? WHERE username = ? AND is_active = 1',
        (password_hash, password_stamp(), username)
    )
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        print("ERROR: Usuario '{}' no encontrado o desactivado.".format(
            username), file=sys.stderr)
        sys.exit(1)
    print("Contrasena de '{}' actualizada correctamente.".format(username))


def cmd_set_admin(username, is_admin):
    """Concede o retira el rol de administrador."""
    conn = get_db()
    ensure_table(conn)
    if not is_admin:
        others = conn.execute(
            'SELECT COUNT(*) FROM web_users WHERE is_admin = 1 AND is_active = 1 AND username != ?',
            (username,)
        ).fetchone()[0]
        if others == 0:
            conn.close()
            print("ERROR: '{}' es el unico administrador activo.".format(username), file=sys.stderr)
            sys.exit(1)
    cursor = conn.execute(
        'UPDATE web_users SET is_admin = ? WHERE username = ?', (1 if is_admin else 0, username)
    )
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        print("ERROR: Usuario '{}' no encontrado.".format(username), file=sys.stderr)
        sys.exit(1)
    print("Usuario '{}': administrador = {}.".format(username, 'Si' if is_admin else 'No'))


def print_usage():
    prog = os.path.basename(sys.argv[0])
    print("Uso:")
    print("  python3.6 {} list".format(prog))
    print("  python3.6 {} add <usuario> <contrasena> [--admin]".format(prog))
    print("  python3.6 {} remove <usuario>".format(prog))
    print("  python3.6 {} change-password <usuario> <nueva_contrasena>".format(prog))
    print("  python3.6 {} set-admin <usuario>".format(prog))
    print("  python3.6 {} unset-admin <usuario>".format(prog))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        cmd_list()

    elif cmd == 'add':
        if len(sys.argv) not in (4, 5) or (len(sys.argv) == 5 and sys.argv[4] != '--admin'):
            print("Uso: python3.6 {} add <usuario> <contrasena> [--admin]".format(
                os.path.basename(sys.argv[0])))
            sys.exit(1)
        cmd_add(sys.argv[2], sys.argv[3], is_admin=len(sys.argv) == 5)

    elif cmd == 'remove':
        if len(sys.argv) != 3:
            print("Uso: python3.6 {} remove <usuario>".format(
                os.path.basename(sys.argv[0])))
            sys.exit(1)
        cmd_remove(sys.argv[2])

    elif cmd == 'change-password':
        if len(sys.argv) != 4:
            print("Uso: python3.6 {} change-password <usuario> <nueva_contrasena>".format(
                os.path.basename(sys.argv[0])))
            sys.exit(1)
        cmd_change_password(sys.argv[2], sys.argv[3])

    elif cmd in ('set-admin', 'unset-admin'):
        if len(sys.argv) != 3:
            print("Uso: python3.6 {} {} <usuario>".format(
                os.path.basename(sys.argv[0]), cmd))
            sys.exit(1)
        cmd_set_admin(sys.argv[2], cmd == 'set-admin')

    else:
        print("ERROR: Comando desconocido '{}'.".format(cmd), file=sys.stderr)
        print_usage()
        sys.exit(1)
