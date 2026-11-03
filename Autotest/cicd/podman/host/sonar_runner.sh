#!/bin/bash
# =============================================================================
# sonar_runner.sh - Ejecuta sonar-scanner en el host SUSE
# =============================================================================
# Lo invoca el contenedor orquestador por SSH a 127.0.0.1. Usa el Java del
# host (java-25) y su truststore corporativo, y el árbol de compilación
# compartido (bw-output + mmi.jar ya extraído por el contenedor).
#
# Uso:
#   sonar_runner.sh <compile_path> <sonar_branch>
# =============================================================================

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Uso: $0 <compile_path> <sonar_branch>" >&2
    exit 2
fi

COMPILE_PATH=$1
SONAR_BRANCH=$2

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export LANG="${LANG:-es_ES.UTF-8}"

# Credenciales de Sonar (SONAR_TOKEN) desde el .env del host.
CICD_HOME="${CICD_HOME:-/home/agent/cicd}"
if [[ -f "$CICD_HOME/config/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$CICD_HOME/config/.env"
    set +a
fi

# Java del host: el mismo que usa el pipeline systemd y que confía en la CA.
if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    :
elif [[ -x /usr/lib64/jvm/java-25-openjdk-25/bin/java ]]; then
    JAVA_HOME=/usr/lib64/jvm/java-25-openjdk-25
elif command -v java >/dev/null 2>&1; then
    JAVA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")
else
    echo "ERROR: no se encontró Java en el host" >&2
    exit 3
fi
export JAVA_HOME

SCANNER_JAR="$COMPILE_PATH/utils/sonar-scanner-7.2.0.5079-linux-x64/lib/sonar-scanner-cli-7.2.0.5079.jar"
if [[ ! -f "$SCANNER_JAR" ]]; then
    echo "ERROR: no existe $SCANNER_JAR" >&2
    exit 4
fi
if [[ ! -f "$COMPILE_PATH/sonar-project.properties" ]]; then
    echo "ERROR: falta sonar-project.properties en $COMPILE_PATH" >&2
    exit 5
fi

cd "$COMPILE_PATH"

echo "[sonar_runner] host=$(hostname) java=$JAVA_HOME branch=$SONAR_BRANCH"
echo "[sonar_runner] SONAR_TOKEN=${SONAR_TOKEN:+set}"

exec "$JAVA_HOME/bin/java" -jar "$SCANNER_JAR" \
    -Dproject.settings=sonar-project.properties \
    -Dsonar.projectKey=GALTTCMC_interno \
    -Dsonar.projectName=GALTTCMC_interno \
    -Dsonar.branch.name="$SONAR_BRANCH" \
    -Dsonar.projectVersion="$SONAR_BRANCH"
