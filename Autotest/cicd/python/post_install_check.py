#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_install_check.py - Fase 6: verificacion posterior a la instalacion

Tras el reboot que hace install.sh hay que entrar en la VM con el usuario de
verificacion (`post_install.ssh_user`) para comprobar que la instalacion ha
quedado operativa. Este script:

  1. Espera a que la VM vuelva a aceptar conexiones SSH.
  2. Entra con `post_install.ssh_user` y `post_install.current_password`
     (pexpect, porque el primer login puede pedir cambio de contrasena y eso
     bloquea BatchMode). Si el sistema fuerza el cambio, usa
     TARGET_VM_NEW_PASSWORD.
  3. Si `post_install.install_ssh_key` es true, instala la clave publica; si
     no (p. ej. el home del usuario no es escribible), ejecuta los checks en
     la misma sesion con contrasena.
  4. Ejecuta comprobaciones (servicios, bond, directorios, gateway, disco y
     arranque reciente). `ip` y `ping` pueden estar restringidos a root, asi
     que el gateway se valida con /proc/net/route.
  5. Guarda el resultado en post_install_results.

Compatible con Python 3.6+ (sin f-strings).

Uso:
    python3 post_install_check.py <config_path> <tag>

Variables de entorno:
    DEPLOYMENT_ID               fila de deployments asociada (la pasa ci_cd.sh)
    TARGET_VM_NEW_PASSWORD      contrasena nueva (solo si el login obliga a cambiarla)
    TARGET_VM_CURRENT_PASSWORD  contrasena actual (por defecto post_install.current_password)
    CICD_KNOWN_HOSTS_FILE       known_hosts a usar (por defecto logs/ssh_known_hosts)

Codigos de salida:
    0  todas las comprobaciones pasan (o la fase esta desactivada)
    1  fallo de conexion, cambio de contrasena o alguna comprobacion
    2  configuracion invalida / dependencias que faltan
"""

from __future__ import print_function

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

import yaml

try:
    import pexpect
except ImportError:  # se valida al arrancar, no al importar
    pexpect = None

PLACEHOLDER_PASSWORD = 'your_target_vm_new_password_here'

# Prompt de los shells del sistema (bash/csh/sh): termina en #, $, %, > o ~
# Anclado al final del buffer para no confundirlo con el "~ " intermedio de
# prompts tipo "usuario@host:~ > ".
PROMPT_RE = r'(?m)(?:[#\$%>~]) \r?\n?$'


def _log(message):
    print(message)
    sys.stdout.flush()


def _warn(message):
    print('[WARN] ' + message)
    sys.stdout.flush()


def _error(message):
    print('[ERROR] ' + message, file=sys.stderr)
    sys.stderr.flush()


#===============================================================================
# Configuracion
#===============================================================================

def load_config(config_path):
    """Cargar el YAML y expandir las variables de entorno ${VAR}."""
    with open(config_path, 'r') as fh:
        config = yaml.safe_load(fh)

    def expand_env(obj):
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                return os.environ.get(obj[2:-1], '')
            return obj
        if isinstance(obj, dict):
            return dict((k, expand_env(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [expand_env(i) for i in obj]
        return obj

    return expand_env(config)


def _get(config, dotted, default=None):
    node = config
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on', 'si', 'sí')


#===============================================================================
# Utilidades de proceso / red
#===============================================================================

def _text(data):
    if data is None:
        return ''
    if isinstance(data, bytes):
        return data.decode('utf-8', 'replace')
    return data


def _run(cmd, timeout=60):
    """Ejecutar un comando sin shell y devolver (rc, stdout, stderr)."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return 127, '', str(exc)

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, _text(out), _text(err)
    return proc.returncode, _text(out), _text(err)


def _wait_port(ip, port, timeout, interval):
    deadline = time.time() + timeout
    attempt = 1
    while True:
        try:
            sock = socket.create_connection((ip, port), timeout=5)
            sock.close()
            _log('[*] SSH disponible en {}:{} (intento {})'.format(ip, port, attempt))
            return True
        except (socket.error, OSError):
            pass
        if time.time() >= deadline:
            _error('Timeout esperando {}:{} tras {}s'.format(ip, port, timeout))
            return False
        _log('[*] Esperando SSH en {}:{} (intento {}, intervalo {}s)'.format(
            ip, port, attempt, interval))
        time.sleep(interval)
        attempt += 1


def _ssh_opts(known_hosts, key_path=None, batch=True):
    opts = [
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile={}'.format(known_hosts),
        '-o', 'ConnectTimeout=15',
    ]
    if batch:
        opts += ['-o', 'BatchMode=yes']
    if key_path:
        opts += ['-i', key_path]
    return opts


def _run_ssh(ip, user, known_hosts, key_path, remote_cmd, timeout=60):
    # PATH explícito: una sesión SSH no interactiva puede no incluir /usr/sbin
    # (ip, ping) en el PATH, y las comprobaciones se ejecutan sin login.
    remote = 'export PATH=/usr/sbin:/sbin:/usr/bin:/bin:$PATH; ' + remote_cmd
    cmd = ['ssh'] + _ssh_opts(known_hosts, key_path) + \
          ['{}@{}'.format(user, ip), remote]
    return _run(cmd, timeout=timeout)


def _remove_known_host(ip, known_hosts):
    """El reinstalado cambia la clave de host: se elimina la entrada anterior."""
    if not os.path.isfile(known_hosts):
        return
    _run(['ssh-keygen', '-R', ip, '-f', known_hosts], timeout=15)


def _public_key(key_path):
    pub_path = key_path + '.pub'
    if os.path.isfile(pub_path):
        with open(pub_path, 'r') as fh:
            return fh.read().strip()
    rc, out, _err = _run(['ssh-keygen', '-y', '-f', key_path], timeout=15)
    if rc == 0 and out.strip():
        return out.strip()
    return None


#===============================================================================
# Acceso SSH interactivo (cambio de contrasena + instalacion de clave)
#===============================================================================

def _spawn_ssh(ip, user, known_hosts):
    args = [
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile={}'.format(known_hosts),
        '-o', 'PreferredAuthentications=password,keyboard-interactive',
        '-o', 'PubkeyAuthentication=no',
        '-o', 'NumberOfPasswordPrompts=1',
        '-o', 'ConnectTimeout=15',
        '{}@{}'.format(user, ip),
    ]
    return pexpect.spawn('ssh', args, timeout=60, encoding='utf-8')


def _login(child, password):
    """Enviar la contrasena y clasificar la respuesta.

    Devuelve (estado, detalle) con estado en {'change', 'shell', 'denied', 'error'}.
    """
    index = child.expect([r'(?i)password:', pexpect.TIMEOUT, pexpect.EOF], timeout=45)
    if index != 0:
        tail = _text(getattr(child, 'before', '')).strip()
        if tail:
            tail = '; salida: ' + tail[-200:]
        return 'error', 'sin prompt de contrasena' + tail

    child.sendline(password)
    index = child.expect(
        [r'(?i)\(current\)\s*UNIX password:',
         r'(?i)(current|old).*password:',
         PROMPT_RE,
         r'(?i)permission denied',
         r'(?i)authentication failed',
         pexpect.TIMEOUT,
         pexpect.EOF],
        timeout=45)
    if index in (0, 1):
        return 'change', None
    if index == 2:
        return 'shell', None
    if index in (3, 4):
        return 'denied', None
    return 'error', 'timeout/EOF tras el login'


def _change_password(child, current_password, new_password):
    child.sendline(current_password)
    index = child.expect([r'(?i)new.*password:', pexpect.TIMEOUT, pexpect.EOF], timeout=45)
    if index != 0:
        return False, 'no se pidio la contrasena nueva'

    child.sendline(new_password)
    index = child.expect([r'(?i)retype.*password:', pexpect.TIMEOUT, pexpect.EOF], timeout=45)
    if index != 0:
        return False, 'no se pidio confirmacion de la contrasena'

    child.sendline(new_password)
    index = child.expect(
        [PROMPT_RE,
         r'(?i)password updated',
         r'(?i)successfully',
         r'(?i)authentication token manipulation',
         r'(?i)BAD PASSWORD',
         pexpect.TIMEOUT,
         pexpect.EOF],
        timeout=45)
    if index in (0, 1, 2, 6):
        return True, None
    if index == 4:
        return False, 'contrasena rechazada por PAM'
    if index == 3:
        return False, 'error al actualizar la contrasena'
    return False, 'sin confirmacion de cambio de contrasena'


def _install_key(child, pubkey):
    escaped = pubkey.replace('\\', '\\\\').replace('"', '\\"')
    remote = (
        'mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
        'touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && '
        '{{ grep -qxF "{k}" ~/.ssh/authorized_keys 2>/dev/null || '
        'echo "{k}" >> ~/.ssh/authorized_keys; }}'
    ).format(k=escaped)

    # No se espera el prompt: el eco del propio comando puede parecerse a uno.
    # Se encola el 'exit' y se espera a que la sesión termine; la clave se
    # valida después ejecutando las comprobaciones con autenticación por clave.
    child.sendline(remote)
    child.sendline('exit')
    try:
        index = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=60)
    except Exception:
        index = 1
    child.close(force=True)
    return index == 0


def _open_session(ip, user, known_hosts, current_password, new_password):
    """Abrir una sesion shell en la VM (login y cambio de contrasena si aplica).

    Devuelve (child, None) si hay shell, o (None, detalle) si falla. El
    llamador es responsable de cerrar el child.
    """
    child = _spawn_ssh(ip, user, known_hosts)
    state, detail = _login(child, current_password)

    if state == 'change':
        if not new_password:
            child.close(force=True)
            return None, ('el login exige cambiar la contrasena; '
                          'define TARGET_VM_NEW_PASSWORD')
        changed, detail = _change_password(child, current_password, new_password)
        child.close(force=True)
        if not changed:
            return None, detail
        # El cambio forzado cierra la sesion: reconectar con la nueva
        return _open_shell(ip, user, known_hosts, new_password)

    if state == 'shell':
        return child, None

    child.close(force=True)
    if state == 'denied' and new_password:
        # Ya se cambio en una ejecucion anterior: probar con la nueva
        return _open_shell(ip, user, known_hosts, new_password)
    if state == 'denied':
        return None, ('autenticacion fallida para {} con la contrasena '
                      'actual'.format(user))
    return None, detail


def _open_shell(ip, user, known_hosts, password):
    child = _spawn_ssh(ip, user, known_hosts)
    state, detail = _login(child, password)
    if state == 'shell':
        return child, None
    child.close(force=True)
    return None, detail or 'login no interactivo: {}'.format(state)


def _session_start(child):
    """Desactivar el eco del terminal para parsear limpio la salida."""
    child.sendline('stty -echo')
    index = child.expect([PROMPT_RE, pexpect.TIMEOUT, pexpect.EOF], timeout=15)
    return index == 0


def _session_run(child, remote_cmd, timeout=60):
    """Ejecutar un comando en la sesion ya abierta y devolver (rc, stdout)."""
    marker = '__CICD_MARK_{}__'.format(uuid.uuid4().hex)
    # Comando y marcador en la MISMA linea: si van en lineas distintas, el
    # shell imprime un prompt entre la salida y el marcador y contamina el parseo.
    child.sendline('{}; printf \'\\n{}:%s\\n\' "$?"'.format(remote_cmd, marker))
    index = child.expect(
        [r'(?m)^' + re.escape(marker) + r':(\d+)\r?$',
         pexpect.TIMEOUT, pexpect.EOF],
        timeout=timeout)
    if index != 0:
        return 124, _text(getattr(child, 'before', '')), 'timeout/EOF'
    rc = int(child.match.group(1))
    out = _text(child.before)
    # Consumir el prompt que queda tras el marcador
    child.expect([PROMPT_RE, pexpect.TIMEOUT, pexpect.EOF], timeout=15)
    return rc, out, ''


#===============================================================================
# Comprobaciones
#===============================================================================

def _bond_up_slaves(text):
    """Slaves con MII Status: up en /proc/net/bonding/<iface>."""
    up = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith('slave interface:'):
            current = line.split(':', 1)[1].strip()
        elif current and lower.startswith('mii status:'):
            if line.split(':', 1)[1].strip().lower() == 'up':
                up.append(current)
            current = None
    return up


def _check_services(run, config):
    results = []
    services = _get(config, 'post_install.checks.services', []) or []
    for service in services:
        rc, out, err = run("systemctl is-active '{}'".format(service))
        state = out.strip()
        results.append({
            'name': 'servicio ' + str(service),
            'ok': rc == 0 and state == 'active',
            'detail': state or err.strip() or 'sin respuesta',
        })
    return results


def _check_bond(run, config):
    results = []
    iface = _get(config, 'post_install.checks.bond.interface', 'bond0') or 'bond0'

    rc, out, err = run("cat /sys/class/net/{}/operstate 2>/dev/null".format(iface))
    state = out.strip()
    results.append({
        'name': 'bond ' + str(iface),
        'ok': state == 'up',
        'detail': state or 'interfaz no disponible',
    })

    slaves = _get(config, 'post_install.checks.bond.slaves', []) or []
    if slaves:
        rc, out, err = run("cat /proc/net/bonding/{} 2>/dev/null".format(iface))
        if rc != 0 or not out.strip():
            results.append({
                'name': 'bond esclavos',
                'ok': False,
                'detail': 'no se pudo leer /proc/net/bonding/' + str(iface),
            })
        else:
            up = _bond_up_slaves(out)
            missing = [s for s in slaves if s not in up]
            detail = 'up: ' + (', '.join(up) if up else 'ninguno')
            if missing:
                detail += '; faltan: ' + ', '.join(str(s) for s in missing)
            results.append({
                'name': 'bond esclavos',
                'ok': not missing,
                'detail': detail,
            })
    return results


def _check_dirs(run, config):
    results = []
    dirs = _get(config, 'post_install.checks.opt_dirs', []) or []
    for directory in dirs:
        safe = str(directory).replace("'", "'\\''")
        rc, _out, _err = run("test -d '{}'".format(safe))
        results.append({
            'name': 'directorio ' + str(directory),
            'ok': rc == 0,
            'detail': 'existe' if rc == 0 else 'no existe',
        })
    return results


def _hex_to_ip(value):
    """Convertir el gateway little-endian de /proc/net/route a a.b.c.d."""
    try:
        return '.'.join(str(int(value[i:i + 2], 16)) for i in (6, 4, 2, 0))
    except (TypeError, ValueError):
        return ''


def _check_gateway(run, config):
    """Comprobar la ruta por defecto desde /proc/net/route.

    `ip` y `ping` estan restringidos a root en la imagen instalada, asi que la
    comprobacion se limita a que exista una ruta por defecto valida.
    """
    rc, out, err = run("cat /proc/net/route")
    if rc != 0 or not out.strip():
        return [{
            'name': 'gateway',
            'ok': False,
            'detail': 'no se pudo leer /proc/net/route',
        }]

    gateway = None
    iface = None
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            flags = int(parts[3], 16)
        except ValueError:
            continue
        if parts[1] == '00000000' and (flags & 0x2):
            gateway = _hex_to_ip(parts[2])
            iface = parts[0]
            break

    configured = str(_get(config, 'post_install.checks.gateway', '') or '').strip()
    if not gateway:
        return [{
            'name': 'gateway',
            'ok': False,
            'detail': 'sin ruta por defecto',
        }]
    if configured and configured != gateway:
        return [{
            'name': 'gateway',
            'ok': False,
            'detail': 'ruta por defecto {} (esperado {})'.format(gateway, configured),
        }]

    arp_state = ''
    rc, arp, _err = run("cat /proc/net/arp")
    if rc == 0:
        for line in arp.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == gateway:
                arp_state = '; ARP flags={}'.format(parts[2])
                break

    return [{
        'name': 'gateway ' + gateway,
        'ok': True,
        'detail': 'via {}{}'.format(iface, arp_state),
    }]


def _check_disk(run, config):
    try:
        threshold = int(_get(config, 'post_install.checks.disk_usage_threshold', 90))
    except (TypeError, ValueError):
        threshold = 90

    rc, out, _err = run(
        "df -P / 2>/dev/null | awk 'NR==2 {gsub(/%/,\"\",$5); print $5}'")
    used = out.strip()
    ok = rc == 0 and used.isdigit() and int(used) <= threshold
    return [{
        'name': 'uso de disco /',
        'ok': ok,
        'detail': '{}% (maximo {}%)'.format(used if used else '?', threshold),
    }]


def _check_uptime(run, config):
    try:
        max_uptime = float(_get(config, 'post_install.checks.max_uptime_seconds', 1800))
    except (TypeError, ValueError):
        max_uptime = 1800.0

    rc, out, _err = run("cut -d' ' -f1 /proc/uptime")
    try:
        uptime = float(out.strip())
    except (TypeError, ValueError):
        uptime = None

    ok = uptime is not None and uptime <= max_uptime
    detail = '{:.0f}s (maximo {:.0f}s)'.format(uptime, max_uptime) \
        if uptime is not None else 'no disponible'
    return [{
        'name': 'arranque reciente',
        'ok': ok,
        'detail': detail,
    }]


def _run_checks(run, config):
    results = []
    results += _check_services(run, config)
    results += _check_bond(run, config)
    results += _check_dirs(run, config)
    results += _check_gateway(run, config)
    results += _check_disk(run, config)
    results += _check_uptime(run, config)
    return results


#===============================================================================
# Persistencia
#===============================================================================

def _record_results(db_path, deployment_id, tag, passed, results, error_message):
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS post_install_results (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   deployment_id INTEGER REFERENCES deployments(id),
                   tag TEXT NOT NULL,
                   passed INTEGER CHECK(passed IN (0, 1)),
                   checks_json TEXT,
                   error_message TEXT,
                   created_at TEXT DEFAULT (datetime('now'))
               )""")
        conn.execute(
            """INSERT INTO post_install_results
                   (deployment_id, tag, passed, checks_json, error_message)
               VALUES (?, ?, ?, ?, ?)""",
            (deployment_id, tag, 1 if passed else 0,
             json.dumps(results, ensure_ascii=False), error_message))
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        _warn('No se pudo guardar el resultado en la BD: {}'.format(exc))


#===============================================================================
# Main
#===============================================================================

def main(argv):
    if len(argv) < 3:
        _error('Uso: post_install_check.py <config_path> <tag>')
        return 2

    config_path = argv[1]
    tag = argv[2]

    if not os.path.isfile(config_path):
        _error('Configuracion no encontrada: {}'.format(config_path))
        return 2

    config = load_config(config_path)

    if not _as_bool(_get(config, 'post_install.enabled', False)):
        _log('[*] post_install deshabilitado (post_install.enabled=false); se omite')
        return 0

    if pexpect is None:
        _error('pexpect no esta instalado: reconstruye la imagen del orquestador '
               '(python3-pexpect en podman/Containerfile.orchestrator)')
        return 2

    # La contrasena nueva solo hace falta si el primer login obliga a cambiarla.
    new_password = (os.environ.get('TARGET_VM_NEW_PASSWORD') or
                    str(_get(config, 'post_install.new_password', '') or ''))
    if new_password == PLACEHOLDER_PASSWORD:
        new_password = ''
    if new_password and len(new_password) < 8:
        _error('TARGET_VM_NEW_PASSWORD debe tener al menos 8 caracteres')
        return 2

    current_password = (os.environ.get('TARGET_VM_CURRENT_PASSWORD') or
                        str(_get(config, 'post_install.current_password', '') or ''))

    ip = str(_get(config, 'target_vm.ip', '') or '')
    user = str(_get(config, 'post_install.ssh_user', '') or
               _get(config, 'target_vm.ssh_user', 'root') or 'root')

    if not current_password:
        _error('post_install.current_password no configurada')
        return 2
    key_path = str(_get(config, 'target_vm.ssh_key_path', '/home/agent/.ssh/id_rsa') or
                   '/home/agent/.ssh/id_rsa')

    if not ip:
        _error('target_vm.ip no configurada')
        return 2

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    known_hosts = (os.environ.get('CICD_KNOWN_HOSTS_FILE') or
                   os.path.join(base_dir, 'logs', 'ssh_known_hosts'))
    db_path = str(_get(config, 'general.db_path', '') or
                  os.path.join(base_dir, 'db', 'pipeline.db'))

    try:
        wait_timeout = int(_get(config, 'post_install.wait_reboot_timeout', 300))
    except (TypeError, ValueError):
        wait_timeout = 300
    try:
        wait_interval = int(_get(config, 'post_install.wait_reboot_interval', 15))
    except (TypeError, ValueError):
        wait_interval = 15
    try:
        settle = int(_get(config, 'post_install.settle_seconds', 10))
    except (TypeError, ValueError):
        settle = 10

    dep_env = os.environ.get('DEPLOYMENT_ID', '')
    deployment_id = int(dep_env) if dep_env.isdigit() else None

    _log('=== Verificacion post-instalacion: {} ==='.format(tag))
    _log('[*] VM: {}@{}'.format(user, ip))

    known_hosts_dir = os.path.dirname(known_hosts)
    if known_hosts_dir and not os.path.isdir(known_hosts_dir):
        try:
            os.makedirs(known_hosts_dir)
        except OSError:
            pass

    # install.sh termina con 'reboot': se deja unos segundos para que el
    # apagado empiece antes de esperar a que vuelva a aceptar SSH.
    if settle > 0:
        _log('[*] Esperando {}s a que la VM empiece el reboot'.format(settle))
        time.sleep(settle)

    _remove_known_host(ip, known_hosts)
    if not _wait_port(ip, 22, wait_timeout, wait_interval):
        _record_results(db_path, deployment_id, tag, False, [],
                        'SSH no disponible tras el reboot')
        return 1
    if settle > 0:
        time.sleep(settle)

    # El puerto 22 puede estar abierto un instante antes de que el reboot
    # termine (sshd antiguo apagándose) y cerrarse justo después. Se reintenta
    # el login completo hasta agotar wait_reboot_timeout.
    install_ssh_key = _as_bool(_get(config, 'post_install.install_ssh_key', True))

    deadline = time.time() + wait_timeout
    child = None
    detail = None
    while True:
        child, detail = _open_session(
            ip, user, known_hosts, current_password, new_password)
        if child is not None:
            break
        if time.time() >= deadline:
            _error('No se pudo preparar el acceso SSH: {}'.format(detail or ''))
            _record_results(db_path, deployment_id, tag, False, [],
                            'acceso SSH: {}'.format(detail or ''))
            return 1
        _warn('Acceso SSH no disponible todavía ({}); reintento en {}s'.format(
            detail or 'sin detalle', wait_interval))
        time.sleep(wait_interval)

    if install_ssh_key:
        pubkey = _public_key(key_path)
        if not pubkey:
            child.close(force=True)
            _error('No se pudo obtener la clave publica de {}'.format(key_path))
            _record_results(db_path, deployment_id, tag, False, [],
                            'sin clave publica para instalar')
            return 1
        if not _install_key(child, pubkey):
            child.close(force=True)
            _error('No se pudo instalar la clave SSH en {}'.format(user))
            _record_results(db_path, deployment_id, tag, False, [],
                            'instalacion de clave SSH fallida')
            return 1
        child.close(force=True)
        _log('[*] Acceso SSH preparado: clave instalada para {}'.format(user))
        runner = lambda cmd, timeout=60: _run_ssh(
            ip, user, known_hosts, key_path, cmd, timeout)
    else:
        if not _session_start(child):
            child.close(force=True)
            _error('No se pudo inicializar la sesion SSH con {}'.format(user))
            _record_results(db_path, deployment_id, tag, False, [],
                            'no se pudo inicializar la sesion SSH')
            return 1
        _log('[*] Acceso SSH preparado: sesion con {} (sin instalar clave)'.format(user))
        runner = lambda cmd, timeout=60: _session_run(child, cmd, timeout)

    try:
        results = _run_checks(runner, config)
    finally:
        if not install_ssh_key:
            try:
                child.close(force=True)
            except Exception:
                pass
    passed = all(r['ok'] for r in results)

    _log('')
    _log('Resultado de las comprobaciones:')
    for result in results:
        extra = (' - ' + result['detail']) if result.get('detail') else ''
        _log('  [{}] {}{}'.format('OK' if result['ok'] else 'FAIL',
                                  result['name'], extra))

    failed = [r['name'] for r in results if not r['ok']]
    error_message = ('comprobaciones fallidas: ' + ', '.join(failed)) if failed else None
    _record_results(db_path, deployment_id, tag, passed, results, error_message)

    if passed:
        _log('[*] Verificacion post-instalacion superada')
        return 0

    _error('Verificacion post-instalacion fallida: {}'.format(', '.join(failed)))
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
