#!/bin/bash
# =============================================================================
# run.sh - Arranca/para los contenedores del pipeline en la maquina SUSE
# =============================================================================
# Requiere haber cargado las imagenes antes (load-and-run.sh o podman load).
#
# Uso:
#   ./run.sh start|start-web|start-orchestrator|stop|rm|status|logs
#
# Si Quadlet esta activo (unidades en ~/.config/containers/systemd/), para
# usar este script como fallback primero:
#   systemctl --user stop cicd.service cicd-web.service
#
# Variables de entorno opcionales:
#   CICD_DIR        (default /home/agent/cicd)
#   REPO_DIR        (default /home/agent/GALTTCMC)
#   COMPILE_PATH    (default /home/agent/V08.XX.XX.XX)
#   SSH_DIR         (default /home/agent/.ssh)
#   PROFILE_SCRIPT  (default /etc/profile.d/informacion.sh)
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PODMAN="${PODMAN:-podman}"
IMAGE_ORCH="${IMAGE_ORCH:-localhost/cicd-orchestrator:local}"
IMAGE_WEB="${IMAGE_WEB:-localhost/cicd-web:local}"

CICD_DIR="${CICD_DIR:-/home/agent/cicd}"
CONFIG="${CICD_DIR}/config/ci_cd_config.yaml"
CICD_HOST_DIR="${CICD_HOST_DIR:-/home/agent/cicd-host}"
HOST_USER="${HOST_USER:-$(id -un)}"

# Deriva las rutas de ci_cd_config.yaml salvo que se pasen por entorno.
cfg_or() {
    local key=$1 fallback=$2 value=""
    if command -v yq &>/dev/null && [[ -f "$CONFIG" ]]; then
        value=$(yq ".$key" "$CONFIG" 2>/dev/null || true)
    fi
    if [[ -z "$value" || "$value" == "null" ]]; then
        printf '%s' "$fallback"
    else
        printf '%s' "$value"
    fi
}

REPO_DIR="${REPO_DIR:-$(cfg_or git.repo_local_path /home/agent/GALTTCMC)}"
COMPILE_PATH="${COMPILE_PATH:-$(cfg_or git.compile_path /home/agent/V08.XX.XX.XX)}"
SSH_KEY="${SSH_KEY:-$(cfg_or target_vm.ssh_key_path /home/agent/.ssh/id_rsa)}"
SSH_DIR="${SSH_DIR:-$(dirname "$SSH_KEY")}"
PROFILE_SCRIPT="${PROFILE_SCRIPT:-$(cfg_or notifications.profile_script /etc/profile.d/informacion.sh)}"

COMMON_ARGS=(
    --network host
    --userns=keep-id
    -e "HOME=/home/agent"
    -e "CICD_HOME=/home/agent/cicd"
    -e "CICD_BUILD_MODE=ssh"
    -e "CICD_BUILD_SSH=${HOST_USER}@127.0.0.1"
    -e "CICD_BUILD_RUNNER=${CICD_HOST_DIR}/build_runner.sh"
    -e "CICD_BUILD_WRAPPER=${CICD_HOST_DIR}/utils/build-wrapper-linux-x86/build-wrapper-linux-x86-64"
    -e "CICD_SONAR_MODE=ssh"
    -e "CICD_SONAR_SSH=${HOST_USER}@127.0.0.1"
    -e "CICD_SONAR_RUNNER=${CICD_HOST_DIR}/sonar_runner.sh"
    -e "CICD_WALL_ENABLED=false"
)

VOL_ARGS=(
    -v "${CICD_DIR}/config:/home/agent/cicd/config"
    -v "${CICD_DIR}/db:/home/agent/cicd/db"
    -v "${CICD_DIR}/logs:/home/agent/cicd/logs"
    -v "${REPO_DIR}:${REPO_DIR}"
    -v "${COMPILE_PATH}:${COMPILE_PATH}"
    -v "${SSH_DIR}:/home/agent/.ssh:ro"
)

# CA corporativa del host (git/curl/Sonar/vCenter). Sin esto, el contenedor
# no confia en la CA interna y git falla con "unable to get local issuer".
CA_BUNDLE="${CA_BUNDLE:-/etc/ssl/ca-bundle.pem}"
if [[ -f "$CA_BUNDLE" ]]; then
    VOL_ARGS+=(-v "${CA_BUNDLE}:/etc/ssl/certs/host-ca-bundle.pem:ro")
    COMMON_ARGS+=(
        -e "GIT_SSL_CAINFO=/etc/ssl/certs/host-ca-bundle.pem"
        -e "CURL_CA_BUNDLE=/etc/ssl/certs/host-ca-bundle.pem"
        -e "SSL_CERT_FILE=/etc/ssl/certs/host-ca-bundle.pem"
        -e "REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca-bundle.pem"
    )
fi

if [[ -f "$PROFILE_SCRIPT" ]]; then
    VOL_ARGS+=(-v "${PROFILE_SCRIPT}:${PROFILE_SCRIPT}")
fi

WEB_PORT="${WEB_PORT:-8080}"

WEB_ENV_ARGS=()
if [[ -f "${CICD_DIR}/config/.env" ]]; then
    WEB_ENV_ARGS+=(--env-file "${CICD_DIR}/config/.env")
fi

ensure_dirs() {
    mkdir -p "$CICD_DIR/db" "$CICD_DIR/logs"
    [[ -d "$REPO_DIR" ]] || echo "AVISO: no existe $REPO_DIR (git.repo_local_path)" >&2
    [[ -d "$COMPILE_PATH" ]] || mkdir -p "$COMPILE_PATH"
}

start_orchestrator() {
    ensure_dirs
    "$PODMAN" rm -f cicd >/dev/null 2>&1 || true

    echo "==> Arrancando orquestador"
    # stop-timeout alto: el daemon cierra la ejecucion en curso y la BD al
    # recibir SIGTERM (equivalente a TimeoutStopSec=180 del servicio systemd).
    "$PODMAN" run -d --name cicd --restart always --stop-timeout 180 \
        "${COMMON_ARGS[@]}" "${VOL_ARGS[@]}" \
        "$IMAGE_ORCH" daemon
}

start_web() {
    ensure_dirs
    "$PODMAN" rm -f cicd-web >/dev/null 2>&1 || true

    echo "==> Arrancando Web UI en :${WEB_PORT}"
    "$PODMAN" run -d --name cicd-web --restart always --stop-timeout 30 \
        "${COMMON_ARGS[@]}" "${VOL_ARGS[@]}" "${WEB_ENV_ARGS[@]}" \
        -e "WEB_PORT=${WEB_PORT}" \
        "$IMAGE_WEB" \
        --bind "0.0.0.0:${WEB_PORT}" --workers 2 --threads 2 --timeout 120 \
        --access-logfile /home/agent/cicd/logs/web_access.log \
        --error-logfile /home/agent/cicd/logs/web_error.log \
        --log-level info app:app
}

start_containers() {
    start_orchestrator
    start_web
}

case "${1:-start}" in
    start)
        start_containers
        ;;
    start-web)
        start_web
        ;;
    start-orchestrator)
        start_orchestrator
        ;;
    stop)
        "$PODMAN" stop cicd cicd-web 2>/dev/null || true
        ;;
    rm)
        "$PODMAN" stop cicd cicd-web 2>/dev/null || true
        "$PODMAN" rm cicd cicd-web 2>/dev/null || true
        ;;
    status)
        "$PODMAN" ps --filter 'name=^cicd' --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
        echo ""
        "$PODMAN" logs --tail 20 cicd 2>/dev/null || true
        ;;
    logs)
        "$PODMAN" logs -f --tail 50 cicd
        ;;
    *)
        echo "Uso: $0 {start|start-web|start-orchestrator|stop|rm|status|logs}" >&2
        exit 2
        ;;
esac
