#!/bin/bash
# =============================================================================
# install_quadlet.sh - Instala las unidades Quadlet del pipeline
# =============================================================================
# Ejecutar como el usuario 'agent' (rootless, sin sudo):
#   ./install_quadlet.sh            # instalar y arrancar
#   ./install_quadlet.sh uninstall  # parar y desinstalar
#
# Requiere: podman 4.4+, systemd de usuario activo y linger habilitado.
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
ACTION="${1:-install}"

check_linger() {
    local linger
    linger="$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true)"
    if [[ "$linger" != "yes" ]]; then
        echo "AVISO: linger deshabilitado. Ejecuta: loginctl enable-linger $(id -un)" >&2
    fi
}

uninstall_units() {
    echo "==> Parando y desinstalando unidades Quadlet"
    systemctl --user stop cicd-web.service cicd.service 2>/dev/null || true
    rm -f "$DEST/cicd-web.container" "$DEST/cicd.container"
    systemctl --user daemon-reload
    echo "==> Hecho"
}

install_units() {
    check_linger
    mkdir -p "$DEST"

    # Evita arrancar con un pipeline manual en curso.
    if podman exec cicd pgrep -f 'ci_cd.sh --tag' >/dev/null 2>&1; then
        echo "ERROR: hay un pipeline manual en curso; espera a que termine." >&2
        exit 1
    fi

    # Los contenedores de run.sh usan los mismos nombres: hay que retirarlos.
    echo "==> Retirando contenedores previos de run.sh (si existen)"
    podman stop cicd cicd-web 2>/dev/null || true
    podman rm cicd cicd-web 2>/dev/null || true

    echo "==> Instalando unidades en $DEST"
    install -m 0644 "$HERE/cicd.container" "$DEST/cicd.container"
    install -m 0644 "$HERE/cicd-web.container" "$DEST/cicd-web.container"

    systemctl --user daemon-reload
    echo "==> Arrancando servicios"
    # Las unidades de Quadlet son generadas: no se pueden 'enable'. El
    # autoarranque lo aporta [Install] WantedBy=default.target (el generador
    # crea los symlinks en cada arranque) junto con 'enable-linger'.
    systemctl --user start cicd.service cicd-web.service

    # Quadlet ya gestiona el arranque/restart; podman-restart deja de hacer falta.
    systemctl --user disable --now podman-restart.service 2>/dev/null || true

    echo ""
    systemctl --user --no-pager status cicd.service cicd-web.service || true
}

case "$ACTION" in
    install) install_units ;;
    uninstall) uninstall_units ;;
    *)
        echo "Uso: $0 {install|uninstall}" >&2
        exit 2
        ;;
esac
