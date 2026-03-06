# ===== CONFIGURACIÓN =====
BRANCH_TEST="MASTER_TTC_Development_Pruebas"
SRC_DIR="Development_TTCF"
EXCLUDE_SUBDIR="Development_TTCF/ttcf/utils/makefile"   # <== Subcarpeta que NO quieres que se actualice
SCRIPT_COMPILE="Development_TTCF/ttcf/utils/makefile/compile_all.sh"
BUILD_WRAPPER="build-wrapper-linux-x86-64"
BW_OUTPUT_DIR="bw-output"
SONAR_CMD="sonar-scanner"
BASE_DIR="$(pwd)"

# ===== FUNCIONES =====
log() {
    echo -e "\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

error_exit() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
    exit 1
}

# ===== PASO 0: Compilar con build-wrapper =====
log "Iniciando compilación con build-wrapper..."
mkdir -p "${BW_OUTPUT_DIR}"

utils/build-wrapper-linux-x86/build-wrapper-linux-x86-64 --out-dir $BASE_DIR/bw-output Development_TTCF/ttcf/utils/makefile/compile_all.sh || error_exit "Error durante la compilación"

# ===== PASO 1 Extraer el contenido de mmi.jar en la carpeta target/ =====
mkdir -p target
cd target
jar xf ../mmi.jar
cd ..

# ===== PASO 5 Ejecutar análisis con SonarQube =====
log "Ejecutando análisis con SonarQube..."
utils/sonar-scanner-7.2.0.5079-linux-x64/bin/sonar-scanner   -Dproject.settings=sonar-project.properties   -Dsonar.projectKey=GALTTCMC_interno   -Dsonar.projectName=GALTTCMC_interno
