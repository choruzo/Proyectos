#!/bin/bash
#===============================================================================
# db_viewer.sh - Visor interactivo de la base de datos SQLite del pipeline
#===============================================================================
# Muestra las consultas del README mediante menús dialog
#
# Uso:
#   ./db_viewer.sh             # Lanzar visor interactivo
#   ./db_viewer.sh --db /ruta  # Especificar ruta a pipeline.db
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CICD_DIR="$(dirname "$SCRIPT_DIR")"

# Ruta por defecto a la base de datos
DB_FILE="${1:---}"
if [[ "$DB_FILE" == "--db" && -n "${2:-}" ]]; then
    DB_FILE="$2"
elif [[ "$DB_FILE" == "--" || "$DB_FILE" == "---" ]]; then
    DB_FILE="$CICD_DIR/db/pipeline.db"
fi

# Colores para cabeceras en textbox
TITLE="Pipeline CI/CD - Visor de Base de Datos"

#===============================================================================
# Utilidades
#===============================================================================

check_deps() {
    local missing=()
    for cmd in dialog sqlite3; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Faltan dependencias: ${missing[*]}" >&2
        echo "Instalar con: sudo zypper install ${missing[*]}" >&2
        exit 1
    fi
}

check_db() {
    if [[ ! -f "$DB_FILE" ]]; then
        dialog --title "Error" --msgbox \
            "Base de datos no encontrada:\n$DB_FILE\n\nEjecuta primero: ./ci_cd.sh init" \
            8 60
        exit 1
    fi
}

# Ejecuta una query y guarda resultado formateado en un fichero temporal
run_query() {
    local query="$1"
    local tmpfile
    tmpfile=$(mktemp /tmp/db_viewer_XXXXXX.txt)

    sqlite3 -column -header "$DB_FILE" "$query" > "$tmpfile" 2>&1 || \
        echo "(Error ejecutando consulta)" >> "$tmpfile"

    # Si está vacío, indicarlo
    if [[ ! -s "$tmpfile" ]]; then
        echo "(Sin resultados)" > "$tmpfile"
    fi

    echo "$tmpfile"
}

# Muestra un fichero en un textbox scrollable
show_result() {
    local title="$1"
    local tmpfile="$2"
    local height="${3:-30}"
    local width="${4:-100}"

    dialog --title "$title" \
           --textbox "$tmpfile" "$height" "$width"

    rm -f "$tmpfile"
}

# Muestra resultado de una query directamente
show_query() {
    local title="$1"
    local query="$2"
    local height="${3:-30}"
    local width="${4:-100}"

    local tmpfile
    tmpfile=$(run_query "$query")
    show_result "$title" "$tmpfile" "$height" "$width"
}

# Pide un valor al usuario y devuelve la selección
ask_input() {
    local title="$1"
    local prompt="$2"
    local default="${3:-}"
    local result

    result=$(dialog --title "$title" \
                    --inputbox "$prompt" 8 60 "$default" \
                    3>&1 1>&2 2>&3) || return 1
    echo "$result"
}

# Pide un número (limite)
ask_limit() {
    local default="${1:-10}"
    ask_input "Límite de resultados" "¿Cuántos registros mostrar?" "$default"
}

#===============================================================================
# Pantalla de estadísticas (dashboard)
#===============================================================================

show_dashboard() {
    local tmpfile
    tmpfile=$(mktemp /tmp/db_viewer_XXXXXX.txt)

    {
        echo "============================================================"
        echo "  DASHBOARD - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  DB: $DB_FILE"
        echo "============================================================"
        echo ""
        echo "--- ESTADÍSTICAS GLOBALES ---"
        sqlite3 -column -header "$DB_FILE" \
            "SELECT * FROM v_deployment_stats;" 2>/dev/null || \
            echo "(Vista no disponible)"
        echo ""
        echo "--- ÚLTIMOS 5 DESPLIEGUES ---"
        sqlite3 -column -header "$DB_FILE" \
            "SELECT tag_name, status, started_at, duration_seconds || 's' AS duracion
             FROM deployments ORDER BY id DESC LIMIT 5;" 2>/dev/null || \
            echo "(Sin datos)"
        echo ""
        echo "--- ÚLTIMO RESULTADO SONARQUBE ---"
        sqlite3 -column -header "$DB_FILE" \
            "SELECT tag, coverage || '%' AS coverage, bugs, vulnerabilities,
                    code_smells, CASE passed WHEN 1 THEN 'PASS' ELSE 'FAIL' END AS resultado
             FROM sonar_results ORDER BY id DESC LIMIT 1;" 2>/dev/null || \
            echo "(Sin datos)"
        echo ""
        echo "--- TAGS PENDIENTES ---"
        sqlite3 -column -header "$DB_FILE" \
            "SELECT tag_name, status, first_seen_at
             FROM processed_tags WHERE status IN ('pending','processing')
             ORDER BY id DESC LIMIT 5;" 2>/dev/null || \
            echo "(Sin datos)"
    } > "$tmpfile"

    show_result "Dashboard" "$tmpfile" 40 100
}

#===============================================================================
# Menú: Despliegues
#===============================================================================

menu_deployments() {
    while true; do
        local choice
        choice=$(dialog --title "Despliegues" \
                        --menu "Selecciona una consulta:" 18 60 10 \
                        1 "Últimos N despliegues" \
                        2 "Despliegues fallidos" \
                        3 "Despliegues exitosos" \
                        4 "Por estado (filtrar)" \
                        5 "Buscar por tag" \
                        6 "Vista completa (v_recent_deployments)" \
                        7 "Detalle de un despliegue" \
                        B "← Volver" \
                        3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            1)
                local n; n=$(ask_limit 10) || continue
                show_query "Últimos $n despliegues" \
                    "SELECT id, tag_name, status, started_at,
                            COALESCE(duration_seconds || 's', '-') AS duracion,
                            triggered_by
                     FROM deployments ORDER BY id DESC LIMIT $n;"
                ;;
            2)
                local n; n=$(ask_limit 10) || continue
                show_query "Despliegues fallidos (últimos $n)" \
                    "SELECT id, tag_name, status, started_at,
                            COALESCE(error_message,'(sin mensaje)') AS error
                     FROM deployments WHERE status='failed'
                     ORDER BY id DESC LIMIT $n;"
                ;;
            3)
                local n; n=$(ask_limit 10) || continue
                show_query "Despliegues exitosos (últimos $n)" \
                    "SELECT id, tag_name, status, started_at,
                            duration_seconds || 's' AS duracion
                     FROM deployments WHERE status='success'
                     ORDER BY id DESC LIMIT $n;"
                ;;
            4)
                local estado
                estado=$(dialog --title "Filtrar por estado" \
                                --menu "Estado:" 14 50 6 \
                                pending   "Pendiente" \
                                compiling "Compilando" \
                                analyzing "Analizando" \
                                deploying "Desplegando" \
                                success   "Exitoso" \
                                failed    "Fallido" \
                                3>&1 1>&2 2>&3) || continue
                show_query "Despliegues con estado: $estado" \
                    "SELECT id, tag_name, started_at,
                            COALESCE(duration_seconds || 's','-') AS duracion,
                            triggered_by
                     FROM deployments WHERE status='$estado'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            5)
                local tag; tag=$(ask_input "Buscar por tag" "Introduce el tag (o parte de él):") || continue
                show_query "Búsqueda: $tag" \
                    "SELECT id, tag_name, status, started_at,
                            COALESCE(duration_seconds || 's','-') AS duracion,
                            COALESCE(error_message,'-') AS error
                     FROM deployments WHERE tag_name LIKE '%$tag%'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            6)
                show_query "Vista: v_recent_deployments" \
                    "SELECT * FROM v_recent_deployments;" 30 120
                ;;
            7)
                local dep_id; dep_id=$(ask_input "Detalle de despliegue" "ID del despliegue:") || continue
                local tmpfile
                tmpfile=$(mktemp /tmp/db_viewer_XXXXXX.txt)
                {
                    echo "=== DESPLIEGUE #$dep_id ==="
                    sqlite3 -column -header "$DB_FILE" \
                        "SELECT * FROM deployments WHERE id=$dep_id;" 2>/dev/null
                    echo ""
                    echo "=== BUILD LOGS ==="
                    sqlite3 -column -header "$DB_FILE" \
                        "SELECT phase, start_time, duration || 's' AS dur, exit_code, log_file
                         FROM build_logs WHERE deployment_id=$dep_id;" 2>/dev/null
                    echo ""
                    echo "=== SONARQUBE ==="
                    sqlite3 -column -header "$DB_FILE" \
                        "SELECT coverage, bugs, vulnerabilities, code_smells,
                                security_hotspots, quality_gate_status,
                                CASE passed WHEN 1 THEN 'PASS' ELSE 'FAIL' END AS resultado
                         FROM sonar_results WHERE deployment_id=$dep_id;" 2>/dev/null
                    echo ""
                    echo "=== ÚLTIMOS 20 MENSAJES DE EJECUCIÓN ==="
                    sqlite3 -column -header "$DB_FILE" \
                        "SELECT timestamp, level, phase, message
                         FROM execution_log WHERE deployment_id=$dep_id
                         ORDER BY id DESC LIMIT 20;" 2>/dev/null
                } > "$tmpfile"
                show_result "Detalle despliegue #$dep_id" "$tmpfile" 40 120
                ;;
            B) return 0 ;;
        esac
    done
}

#===============================================================================
# Menú: SonarQube
#===============================================================================

menu_sonar() {
    while true; do
        local choice
        choice=$(dialog --title "SonarQube" \
                        --menu "Selecciona una consulta:" 14 60 6 \
                        1 "Últimos N resultados" \
                        2 "Resultados aprobados" \
                        3 "Resultados reprobados" \
                        4 "Buscar por tag" \
                        5 "Evolución de cobertura" \
                        B "← Volver" \
                        3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            1)
                local n; n=$(ask_limit 10) || continue
                show_query "Últimos $n resultados SonarQube" \
                    "SELECT tag, coverage || '%' AS coverage, bugs,
                            vulnerabilities, code_smells, security_hotspots,
                            quality_gate_status,
                            CASE passed WHEN 1 THEN 'PASS' ELSE 'FAIL' END AS resultado
                     FROM sonar_results ORDER BY id DESC LIMIT $n;" 20 110
                ;;
            2)
                local n; n=$(ask_limit 10) || continue
                show_query "Resultados SonarQube APROBADOS (últimos $n)" \
                    "SELECT tag, coverage || '%' AS coverage, bugs,
                            vulnerabilities, code_smells, timestamp
                     FROM sonar_results WHERE passed=1
                     ORDER BY id DESC LIMIT $n;"
                ;;
            3)
                local n; n=$(ask_limit 10) || continue
                show_query "Resultados SonarQube REPROBADOS (últimos $n)" \
                    "SELECT tag, coverage || '%' AS coverage, bugs,
                            vulnerabilities, code_smells, quality_gate_status, timestamp
                     FROM sonar_results WHERE passed=0
                     ORDER BY id DESC LIMIT $n;"
                ;;
            4)
                local tag; tag=$(ask_input "Buscar por tag" "Tag (o parte):") || continue
                show_query "SonarQube para: $tag" \
                    "SELECT tag, coverage || '%' AS coverage, bugs, vulnerabilities,
                            code_smells, security_hotspots, quality_gate_status,
                            CASE passed WHEN 1 THEN 'PASS' ELSE 'FAIL' END AS resultado,
                            timestamp
                     FROM sonar_results WHERE tag LIKE '%$tag%'
                     ORDER BY id DESC LIMIT 20;" 20 120
                ;;
            5)
                local n; n=$(ask_limit 15) || continue
                show_query "Evolución de cobertura (últimos $n)" \
                    "SELECT tag, coverage || '%' AS coverage,
                            CASE passed WHEN 1 THEN 'PASS' ELSE 'FAIL' END AS resultado,
                            timestamp
                     FROM sonar_results ORDER BY id DESC LIMIT $n;"
                ;;
            B) return 0 ;;
        esac
    done
}

#===============================================================================
# Menú: Compilación
#===============================================================================

menu_build() {
    while true; do
        local choice
        choice=$(dialog --title "Compilación (build_logs)" \
                        --menu "Selecciona una consulta:" 14 60 5 \
                        1 "Últimos N logs de compilación" \
                        2 "Logs por fase" \
                        3 "Compilaciones fallidas (exit_code != 0)" \
                        4 "Buscar por tag" \
                        B "← Volver" \
                        3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            1)
                local n; n=$(ask_limit 10) || continue
                show_query "Últimos $n logs de compilación" \
                    "SELECT b.id, b.tag, b.phase,
                            b.duration || 's' AS duracion, b.exit_code, b.log_file
                     FROM build_logs b ORDER BY b.id DESC LIMIT $n;" 20 100
                ;;
            2)
                local fase
                fase=$(dialog --title "Filtrar por fase" \
                              --menu "Fase:" 12 50 4 \
                              checkout "Checkout del código" \
                              prepare  "Preparación del workspace" \
                              compile  "Compilación" \
                              package  "Empaquetado/ISO" \
                              3>&1 1>&2 2>&3) || continue
                show_query "Build logs - fase: $fase" \
                    "SELECT id, tag, duration || 's' AS duracion,
                            exit_code, log_file, created_at
                     FROM build_logs WHERE phase='$fase'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            3)
                local n; n=$(ask_limit 10) || continue
                show_query "Compilaciones fallidas (últimas $n)" \
                    "SELECT id, tag, phase, duration || 's' AS duracion,
                            exit_code, log_file
                     FROM build_logs WHERE exit_code != 0
                     ORDER BY id DESC LIMIT $n;"
                ;;
            4)
                local tag; tag=$(ask_input "Buscar por tag" "Tag (o parte):") || continue
                show_query "Build logs para: $tag" \
                    "SELECT id, tag, phase, duration || 's' AS duracion,
                            exit_code, log_file
                     FROM build_logs WHERE tag LIKE '%$tag%'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            B) return 0 ;;
        esac
    done
}

#===============================================================================
# Menú: Tags procesados
#===============================================================================

menu_tags() {
    while true; do
        local choice
        choice=$(dialog --title "Tags procesados" \
                        --menu "Selecciona una consulta:" 14 60 5 \
                        1 "Últimos N tags procesados" \
                        2 "Tags por estado" \
                        3 "Buscar tag" \
                        4 "Resumen por estado" \
                        B "← Volver" \
                        3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            1)
                local n; n=$(ask_limit 10) || continue
                show_query "Últimos $n tags procesados" \
                    "SELECT id, tag_name, status, first_seen_at, processed_at
                     FROM processed_tags ORDER BY id DESC LIMIT $n;"
                ;;
            2)
                local estado
                estado=$(dialog --title "Filtrar por estado" \
                                --menu "Estado:" 12 50 4 \
                                pending    "Pendiente" \
                                processing "Procesando" \
                                completed  "Completado" \
                                skipped    "Omitido" \
                                3>&1 1>&2 2>&3) || continue
                show_query "Tags con estado: $estado" \
                    "SELECT id, tag_name, first_seen_at, processed_at
                     FROM processed_tags WHERE status='$estado'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            3)
                local tag; tag=$(ask_input "Buscar tag" "Tag (o parte):") || continue
                show_query "Búsqueda de tag: $tag" \
                    "SELECT id, tag_name, status, first_seen_at, processed_at
                     FROM processed_tags WHERE tag_name LIKE '%$tag%'
                     ORDER BY id DESC LIMIT 20;"
                ;;
            4)
                show_query "Resumen de tags por estado" \
                    "SELECT status,
                            COUNT(*) AS total,
                            MIN(first_seen_at) AS primer_tag,
                            MAX(first_seen_at) AS ultimo_tag
                     FROM processed_tags GROUP BY status ORDER BY total DESC;"
                ;;
            B) return 0 ;;
        esac
    done
}

#===============================================================================
# Menú: Log de ejecución
#===============================================================================

menu_execlog() {
    while true; do
        local choice
        choice=$(dialog --title "Log de ejecución" \
                        --menu "Selecciona una consulta:" 14 60 5 \
                        1 "Últimos N mensajes" \
                        2 "Solo errores (ERROR)" \
                        3 "Solo advertencias (WARN)" \
                        4 "Por despliegue ID" \
                        5 "Por fase" \
                        B "← Volver" \
                        3>&1 1>&2 2>&3) || return 0

        case "$choice" in
            1)
                local n; n=$(ask_limit 20) || continue
                show_query "Últimos $n mensajes de ejecución" \
                    "SELECT timestamp, level, phase,
                            SUBSTR(message,1,80) AS mensaje
                     FROM execution_log ORDER BY id DESC LIMIT $n;" 30 110
                ;;
            2)
                local n; n=$(ask_limit 20) || continue
                show_query "Últimos $n errores de ejecución" \
                    "SELECT timestamp, phase, deployment_id,
                            SUBSTR(message,1,90) AS mensaje
                     FROM execution_log WHERE level='ERROR'
                     ORDER BY id DESC LIMIT $n;" 30 120
                ;;
            3)
                local n; n=$(ask_limit 20) || continue
                show_query "Últimas $n advertencias de ejecución" \
                    "SELECT timestamp, phase, deployment_id,
                            SUBSTR(message,1,90) AS mensaje
                     FROM execution_log WHERE level='WARN'
                     ORDER BY id DESC LIMIT $n;" 30 120
                ;;
            4)
                local dep_id; dep_id=$(ask_input "Log por despliegue" "ID del despliegue:") || continue
                show_query "Ejecución del despliegue #$dep_id" \
                    "SELECT timestamp, level, phase,
                            SUBSTR(message,1,80) AS mensaje
                     FROM execution_log WHERE deployment_id=$dep_id
                     ORDER BY id;" 35 120
                ;;
            5)
                local fase; fase=$(ask_input "Log por fase" "Nombre de fase (ej: compile, deploy, sonar):") || continue
                show_query "Log de fase: $fase" \
                    "SELECT timestamp, level, deployment_id,
                            SUBSTR(message,1,80) AS mensaje
                     FROM execution_log WHERE phase LIKE '%$fase%'
                     ORDER BY id DESC LIMIT 30;" 35 120
                ;;
            B) return 0 ;;
        esac
    done
}

#===============================================================================
# Consulta SQL personalizada
#===============================================================================

show_custom_query() {
    local query
    query=$(dialog --title "Consulta SQL personalizada" \
                   --inputbox "Introduce la consulta SQL:\n(solo SELECT)" \
                   10 70 "SELECT * FROM deployments ORDER BY id DESC LIMIT 5;" \
                   3>&1 1>&2 2>&3) || return 0

    # Bloquear comandos destructivos
    local upper_query
    upper_query=$(echo "$query" | tr '[:lower:]' '[:upper:]')
    if echo "$upper_query" | grep -qE '^\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)'; then
        dialog --title "Error" --msgbox \
            "Solo se permiten consultas SELECT.\nOperación rechazada." 8 50
        return 0
    fi

    show_query "Resultado SQL personalizado" "$query" 35 120
}

#===============================================================================
# Información de la base de datos
#===============================================================================

show_db_info() {
    local tmpfile
    tmpfile=$(mktemp /tmp/db_viewer_XXXXXX.txt)

    {
        echo "=== INFORMACIÓN DE LA BASE DE DATOS ==="
        echo "Fichero: $DB_FILE"
        echo "Tamaño:  $(du -sh "$DB_FILE" 2>/dev/null | cut -f1)"
        echo ""
        echo "=== TABLAS ==="
        sqlite3 "$DB_FILE" ".tables" 2>/dev/null
        echo ""
        echo "=== CONTEO POR TABLA ==="
        for tabla in deployments build_logs sonar_results execution_log processed_tags; do
            local count
            count=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM $tabla;" 2>/dev/null || echo "?")
            printf "  %-20s %s registros\n" "$tabla" "$count"
        done
        echo ""
        echo "=== VISTAS ==="
        sqlite3 "$DB_FILE" \
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;" 2>/dev/null
        echo ""
        echo "=== PRAGMA ==="
        sqlite3 -column -header "$DB_FILE" "PRAGMA integrity_check;" 2>/dev/null
    } > "$tmpfile"

    show_result "Información de la base de datos" "$tmpfile" 30 80
}

#===============================================================================
# Menú principal
#===============================================================================

main_menu() {
    while true; do
        local choice
        choice=$(dialog --title "$TITLE" \
                        --menu "Base de datos: $(basename "$DB_FILE")\n\nSelecciona una sección:" \
                        20 65 10 \
                        1 "Dashboard (resumen general)" \
                        2 "Despliegues" \
                        3 "SonarQube" \
                        4 "Compilación (build logs)" \
                        5 "Tags procesados" \
                        6 "Log de ejecución" \
                        7 "Consulta SQL personalizada" \
                        8 "Información de la base de datos" \
                        9 "Salir" \
                        3>&1 1>&2 2>&3) || break

        case "$choice" in
            1) show_dashboard ;;
            2) menu_deployments ;;
            3) menu_sonar ;;
            4) menu_build ;;
            5) menu_tags ;;
            6) menu_execlog ;;
            7) show_custom_query ;;
            8) show_db_info ;;
            9) break ;;
        esac
    done
}

#===============================================================================
# Entrada principal
#===============================================================================

main() {
    check_deps
    check_db
    main_menu
    clear
    echo "Visor cerrado. Base de datos: $DB_FILE"
}

main "$@"
