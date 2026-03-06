#!/bin/bash
#===============================================================================
# diagnose_tag.sh - Diagnóstico de Tags en Base de Datos
#===============================================================================
# Verifica el estado de un tag específico en todas las tablas relevantes
#
# Uso:
#   ./diagnose_tag.sh V08_00_00_00_IntermediateVersion
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

# Cargar configuración
load_config

TAG="${1:-}"

if [[ -z "$TAG" ]]; then
    echo "Uso: $(basename "$0") <TAG_NAME>"
    echo ""
    echo "Ejemplo: $(basename "$0") V08_00_00_00_IntermediateVersion"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  DIAGNÓSTICO DE TAG: $TAG"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

DB_PATH="$SCRIPT_DIR/db/pipeline.db"

if [[ ! -f "$DB_PATH" ]]; then
    echo "❌ ERROR: Base de datos no encontrada: $DB_PATH"
    exit 1
fi

echo "📂 Base de datos: $DB_PATH"
echo ""

# 1. Verificar en deployments
echo "─────────────────────────────────────────────────────────────────"
echo "📋 Tabla: deployments"
echo "─────────────────────────────────────────────────────────────────"

deployment_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM deployments WHERE tag_name='$TAG'" 2>/dev/null || echo "0")

if [[ "$deployment_count" == "0" ]]; then
    echo "   (No se encontraron registros)"
else
    sqlite3 -line "$DB_PATH" \
        "SELECT id, tag_name, status, started_at, completed_at, 
                duration_seconds, triggered_by, error_message
         FROM deployments WHERE tag_name='$TAG'"
fi
echo ""

# 2. Verificar en processed_tags
echo "─────────────────────────────────────────────────────────────────"
echo "📋 Tabla: processed_tags"
echo "─────────────────────────────────────────────────────────────────"

processed_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM processed_tags WHERE tag_name='$TAG'" 2>/dev/null || echo "0")

if [[ "$processed_count" == "0" ]]; then
    echo "   (No se encontraron registros)"
else
    sqlite3 -line "$DB_PATH" \
        "SELECT id, tag_name, status, first_seen_at, processed_at
         FROM processed_tags WHERE tag_name='$TAG'"
fi
echo ""

# 3. Verificar en sonar_results
echo "─────────────────────────────────────────────────────────────────"
echo "📋 Tabla: sonar_results"
echo "─────────────────────────────────────────────────────────────────"

sonar_count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sonar_results WHERE tag='$TAG'" 2>/dev/null || echo "0")

if [[ "$sonar_count" == "0" ]]; then
    echo "   (No se encontraron registros)"
else
    sqlite3 -line "$DB_PATH" \
        "SELECT id, tag, 
                CASE WHEN passed=1 THEN 'PASS' ELSE 'FAIL' END as quality_gate,
                coverage, bugs, vulnerabilities, code_smells, security_hotspots,
                quality_gate_status, timestamp, created_at
         FROM sonar_results WHERE tag='$TAG'"
fi
echo ""

# 4. Verificar en build_logs
echo "─────────────────────────────────────────────────────────────────"
echo "📋 Tabla: build_logs"
echo "─────────────────────────────────────────────────────────────────"
build_logs=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM build_logs WHERE tag='$TAG'" 2>&1)
if [[ "$build_logs" == "0" ]]; then
    echo "   (No se encontraron registros)"
else
    echo "   Registros de compilación: $build_logs"
    sqlite3 -header -column "$DB_PATH" \
        "SELECT phase, status, duration_seconds, errors 
         FROM build_logs WHERE tag='$TAG' 
         ORDER BY started_at"
fi
echo ""

# 5. Resumen y recomendaciones
echo "═══════════════════════════════════════════════════════════════════"
echo "  RESUMEN Y RECOMENDACIONES"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Verificar estado
in_deployments=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM deployments WHERE tag_name='$TAG'" 2>/dev/null || echo "0")
in_processed=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM processed_tags WHERE tag_name='$TAG'" 2>/dev/null || echo "0")
deployment_status=$(sqlite3 "$DB_PATH" "SELECT status FROM deployments WHERE tag_name='$TAG'" 2>/dev/null || echo "")
processed_status=$(sqlite3 "$DB_PATH" "SELECT status FROM processed_tags WHERE tag_name='$TAG'" 2>/dev/null || echo "")

echo "✓ En tabla deployments: $in_deployments registro(s) - Estado: ${deployment_status:-N/A}"
echo "✓ En tabla processed_tags: $in_processed registro(s) - Estado: ${processed_status:-N/A}"
echo ""

# Detectar inconsistencias específicas
if [[ "$deployment_status" == "failed" && "$processed_status" == "completed" ]]; then
    echo "⚠️  ALERTA: INCONSISTENCIA DETECTADA"
    echo "   deployments dice 'failed' pero processed_tags dice 'completed'"
    echo ""
    echo "Esto ocurre cuando un procesamiento falló pero posteriormente se completó"
    echo "con otra ejecución. Para limpiar la inconsistencia:"
    echo ""
    echo "Opción 1 - Reprocesar (recomendado):"
    echo "  ./ci_cd.sh --tag $TAG"
    echo ""
    echo "Opción 2 - Limpiar manualmente:"
    echo "  sqlite3 db/pipeline.db \"DELETE FROM deployments WHERE tag_name='$TAG'\""
    echo "  sqlite3 db/pipeline.db \"DELETE FROM processed_tags WHERE tag_name='$TAG'\""
    echo "  ./ci_cd.sh --tag $TAG"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    exit 0
fi

if [[ "$in_deployments" == "0" && "$in_processed" == "0" ]]; then
    echo "🔍 CONCLUSIÓN: El tag NO ha sido procesado nunca"
    echo ""
    echo "Acciones sugeridas:"
    echo "  1. Verificar que el tag existe en el repositorio remoto:"
    echo "     ./scripts/git_monitor.sh list | grep $TAG"
    echo ""
    echo "  2. Procesar manualmente:"
    echo "     ./ci_cd.sh --tag $TAG"
    
elif [[ "$in_deployments" == "1" && "$in_processed" == "0" ]]; then
    echo "⚠️  CONCLUSIÓN: Registro inconsistente"
    echo "   El tag está en 'deployments' pero NO en 'processed_tags'"
    echo ""
    echo "Esto indica un fallo durante el procesamiento. Estado: $deployment_status"
    echo ""
    echo "Acciones sugeridas:"
    echo "  1. Si quieres reprocesarlo, el script ahora lo hará automáticamente:"
    echo "     ./ci_cd.sh --tag $TAG"
    echo ""
    echo "  2. O limpiar manualmente los registros:"
    echo "     sqlite3 db/pipeline.db \"DELETE FROM deployments WHERE tag_name='$TAG'\""
    
elif [[ "$in_deployments" == "1" && "$in_processed" == "1" ]]; then
    if [[ "$deployment_status" == "success" ]]; then
        echo "✅ CONCLUSIÓN: Tag procesado correctamente"
        echo ""
        echo "El pipeline se completó exitosamente para este tag."
    elif [[ "$deployment_status" == "failed" ]]; then
        echo "❌ CONCLUSIÓN: Tag procesado pero FALLÓ"
        echo ""
        echo "Acciones sugeridas:"
        echo "  1. Revisar logs del último procesamiento:"
        echo "     ./ci_cd.sh logs 200 | grep -A 50 '$TAG'"
        echo ""
        echo "  2. Reprocesar (el script limpiará registros antiguos automáticamente):"
        echo "     ./ci_cd.sh --tag $TAG"
    else
        echo "🔄 CONCLUSIÓN: Tag en progreso o estado desconocido"
        echo ""
        echo "Estado actual: $deployment_status"
        echo ""
        echo "Si el pipeline no está corriendo, el registro puede estar obsoleto."
        echo "Puedes forzar reprocesamiento con: ./ci_cd.sh --tag $TAG"
    fi
else
    echo "⚠️  CONCLUSIÓN: Registros duplicados detectados"
    echo ""
    echo "Hay múltiples registros para este tag (no debería ocurrir)."
    echo "Considera limpiar la base de datos manualmente."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
