#!/usr/bin/env bash
# =============================================================================
# build-and-export.sh - Construye las imagenes en local y las exporta a .tar
# =============================================================================
# Equivalente Linux/macOS de build-and-export.ps1.
#
# Uso:
#   ./cicd/podman/build-and-export.sh
#   ENGINE=podman ./cicd/podman/build-and-export.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT="$(cd "$HERE/.." && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/dist}"

BASE_IMAGE="${BASE_IMAGE:-opensuse/leap:15.5}"
AGENT_UID="${AGENT_UID:-1000}"
AGENT_GID="${AGENT_GID:-1000}"

if [[ -z "${ENGINE:-}" ]]; then
    if command -v podman &>/dev/null; then
        ENGINE=podman
    elif command -v docker &>/dev/null; then
        ENGINE=docker
    else
        echo "ERROR: no se encontro podman ni docker" >&2
        exit 1
    fi
fi

ORCH_TAG="localhost/cicd-orchestrator:local"
WEB_TAG="localhost/cicd-web:local"
BUILD_ARGS=(
    --platform linux/amd64
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    --build-arg "AGENT_UID=$AGENT_UID"
    --build-arg "AGENT_GID=$AGENT_GID"
)

mkdir -p "$OUT_DIR"

echo "==> Construyendo $ORCH_TAG con $ENGINE"
"$ENGINE" build "${BUILD_ARGS[@]}" \
    -f "$HERE/Containerfile.orchestrator" -t "$ORCH_TAG" "$CONTEXT"

echo "==> Construyendo $WEB_TAG con $ENGINE"
"$ENGINE" build "${BUILD_ARGS[@]}" \
    -f "$HERE/Containerfile.web" -t "$WEB_TAG" "$CONTEXT"

echo "==> Exportando imagenes a $OUT_DIR"
if [[ "$ENGINE" == "podman" ]]; then
    podman save --format docker-archive -o "$OUT_DIR/cicd-orchestrator.tar" "$ORCH_TAG"
    podman save --format docker-archive -o "$OUT_DIR/cicd-web.tar" "$WEB_TAG"
else
    docker save -o "$OUT_DIR/cicd-orchestrator.tar" "$ORCH_TAG"
    docker save -o "$OUT_DIR/cicd-web.tar" "$WEB_TAG"
fi

echo ""
echo "Listo. Copia a la maquina SUSE:"
echo "  $OUT_DIR/cicd-orchestrator.tar"
echo "  $OUT_DIR/cicd-web.tar"
echo "  cicd/  (codigo, config y utils)"
