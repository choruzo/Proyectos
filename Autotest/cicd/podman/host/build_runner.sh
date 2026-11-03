#!/bin/bash
# =============================================================================
# build_runner.sh - Ejecuta la compilacion REAL en el host SUSE
# =============================================================================
# Lo invoca el contenedor orquestador por SSH a 127.0.0.1. El arbol de
# compilacion es un bind-mount compartido, asi que el contenedor y el host
# ven exactamente las mismas rutas.
#
# Uso:
#   build_runner.sh <compile_path> <build_script> <build_wrapper> <timeout_secs>
# =============================================================================

set -euo pipefail

# Entorno mínimo equivalente al del servicio systemd original. Evita que un
# PATH distinto en la sesión SSH no interactiva rompa build_DVDs.sh.
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export LANG="${LANG:-es_ES.UTF-8}"

if [[ $# -lt 4 ]]; then
    echo "Uso: $0 <compile_path> <build_script> <build_wrapper> <timeout_secs>" >&2
    exit 2
fi

COMPILE_PATH=$1
BUILD_SCRIPT=$2
BUILD_WRAPPER=$3
TIMEOUT_SECS=$4

if [[ ! -d "$COMPILE_PATH" ]]; then
    echo "ERROR: no existe el directorio de compilacion: $COMPILE_PATH" >&2
    exit 3
fi
if [[ ! -x "$BUILD_WRAPPER" ]]; then
    echo "ERROR: build-wrapper no ejecutable: $BUILD_WRAPPER" >&2
    exit 4
fi

mkdir -p "$COMPILE_PATH/bw-output"
cd "$COMPILE_PATH"

echo "[build_runner] host=$(hostname) compile_path=$COMPILE_PATH"
echo "[build_runner] script=$BUILD_SCRIPT timeout=${TIMEOUT_SECS}s"

exec timeout --kill-after=60 "$TIMEOUT_SECS" \
    "$BUILD_WRAPPER" --out-dir "$COMPILE_PATH/bw-output" \
    bash -c "./$BUILD_SCRIPT"
