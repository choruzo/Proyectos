#!/bin/bash
# =============================================================================
# load-and-run.sh - Carga las imagenes exportadas y arranca el PoC en SUSE
# =============================================================================
# Ejecutar en la maquina SUSE como 'agent' (rootless podman), desde el
# directorio cicd/podman/ copiado junto con dist/.
#
#   ./load-and-run.sh
#
# IMPORTANTE: antes de arrancar, detener los servicios systemd antiguos:
#   sudo systemctl disable --now cicd cicd-web web.service
# (web.service es el del RPM y tambien ocupa el puerto 8080)
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${DIST_DIR:-$HERE/dist}"
PODMAN="${PODMAN:-podman}"

if ! command -v "$PODMAN" &>/dev/null; then
    echo "ERROR: podman no esta instalado" >&2
    exit 1
fi

for img in cicd-orchestrator cicd-web; do
    tarfile="$DIST_DIR/$img.tar"
    if [[ ! -f "$tarfile" ]]; then
        echo "ERROR: falta $tarfile" >&2
        echo "       Ejecuta antes build-and-export.ps1/.sh en la maquina local." >&2
        exit 1
    fi
    echo "==> Cargando $tarfile"
    "$PODMAN" load -i "$tarfile"
done

echo "==> Preparando lado host (runners + build-wrapper + SSH local)"
"$HERE/host/install_host_runner.sh"

if [[ -x "$HERE/quadlet/install_quadlet.sh" ]]; then
    echo "==> Instalando runtime oficial (Quadlet)"
    "$HERE/quadlet/install_quadlet.sh"
else
    echo "==> Quadlet no disponible; usando run.sh como fallback"
    "$HERE/run.sh" start
fi

echo ""
"$HERE/run.sh" status
