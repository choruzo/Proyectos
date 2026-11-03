#!/bin/bash
# =============================================================================
# install_host_runner.sh - Prepara el lado host de la compilacion remota
# =============================================================================
# Ejecutar en la maquina SUSE como el usuario 'agent' (sin sudo).
#
# - Instala build_runner.sh y build-wrapper en ~/cicd-host/
# - Autoriza la clave del host para SSH local (contenedor -> host)
# - Comprueba que 'ssh agent@127.0.0.1' funciona
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CICD_HOST_DIR:-$HOME/cicd-host}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_UTILS="$REPO_ROOT/utils"

echo "==> Instalando runner en: $DEST"
mkdir -p "$DEST/utils"
install -m 0755 "$HERE/build_runner.sh" "$DEST/build_runner.sh"

# Runner de Sonar (puede venir junto al script o dentro de la imagen).
if [[ -f "$HERE/sonar_runner.sh" ]]; then
    install -m 0755 "$HERE/sonar_runner.sh" "$DEST/sonar_runner.sh"
elif command -v podman &>/dev/null && \
     podman image exists localhost/cicd-orchestrator:local 2>/dev/null; then
    sonar_cid="$(podman create localhost/cicd-orchestrator:local 2>/dev/null || true)"
    if [[ -n "$sonar_cid" ]]; then
        podman cp "$sonar_cid:/home/agent/cicd/podman/host/sonar_runner.sh" \
            "$DEST/sonar_runner.sh"
        podman rm "$sonar_cid" >/dev/null 2>&1 || true
        chmod 0755 "$DEST/sonar_runner.sh" 2>/dev/null || true
        echo "==> sonar_runner.sh extraido de la imagen"
    fi
fi

WRAPPER_SRC="$REPO_UTILS/build-wrapper-linux-x86"
if [[ -d "$WRAPPER_SRC" ]]; then
    rm -rf "$DEST/utils/build-wrapper-linux-x86"
    cp -a "$WRAPPER_SRC" "$DEST/utils/"
    chmod +x "$DEST/utils/build-wrapper-linux-x86/"* 2>/dev/null || true
    echo "==> build-wrapper copiado desde $WRAPPER_SRC"
elif command -v podman &>/dev/null && \
     podman image exists localhost/cicd-orchestrator:local 2>/dev/null; then
    # Fallback: extraer el wrapper de la imagen ya cargada.
    cid="$(podman create localhost/cicd-orchestrator:local 2>/dev/null || true)"
    if [[ -n "$cid" ]]; then
        podman cp "$cid:/home/agent/cicd/utils/build-wrapper-linux-x86" "$DEST/utils/"
        podman rm "$cid" >/dev/null 2>&1 || true
        chmod +x "$DEST/utils/build-wrapper-linux-x86/"* 2>/dev/null || true
        echo "==> build-wrapper extraido de la imagen"
    fi
else
    echo "AVISO: no existe $WRAPPER_SRC y no se pudo extraer de la imagen." >&2
    echo "       Copia cicd/utils en el host o ejecuta este script desde el repo completo." >&2
fi

# SSH local: la clave del host se autoriza a si misma. El contenedor monta
# ~/.ssh en solo lectura y usa la misma clave, asi que podra entrar.
KEY="$HOME/.ssh/id_rsa"
PUB="$KEY.pub"
if [[ -f "$KEY" && -f "$PUB" ]]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    if ! grep -qxF "$(cat "$PUB")" "$HOME/.ssh/authorized_keys"; then
        cat "$PUB" >> "$HOME/.ssh/authorized_keys"
        echo "==> Clave publica anadida a authorized_keys"
    fi

    echo "==> Probando SSH local"
    if ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 \
           -i "$KEY" "$(id -un)@127.0.0.1" "echo SSH local OK"; then
        :
    else
        echo "AVISO: SSH local no funciona. Comprueba que sshd esta activo:" >&2
        echo "       sudo systemctl enable --now sshd" >&2
    fi
else
    echo "AVISO: no existe $KEY (o $PUB); genera la clave y repite." >&2
fi

echo "==> Listo. Runner: $DEST/build_runner.sh"
