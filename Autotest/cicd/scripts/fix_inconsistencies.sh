#!/bin/bash
#===============================================================================
# fix_inconsistencies.sh - Reparar Inconsistencias en Base de Datos
#===============================================================================
# Busca y repara inconsistencias comunes entre tablas deployments y processed_tags
#
# Uso:
#   ./fix_inconsistencies.sh           # Ver inconsistencias
#   ./fix_inconsistencies.sh --fix     # Reparar automáticamente
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

# Cargar configuración
load_config

DB_PATH="$SCRIPT_DIR/db/pipeline.db"
FIX_MODE=${1:-}

if [[ ! -f "$DB_PATH" ]]; then
    echo "❌ ERROR: Base de datos no encontrada: $DB_PATH"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  VERIFICACIÓN DE INCONSISTENCIAS EN BASE DE DATOS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Base de datos: $DB_PATH"
echo ""

#===============================================================================
# 1. Tags con deployment 'failed' pero processed_tags 'completed'
#===============================================================================

echo "─────────────────────────────────────────────────────────────────"
echo "🔍 Buscando: deployment='failed' + processed_tags='completed'"
echo "─────────────────────────────────────────────────────────────────"

inconsistency1=$(sqlite3 "$DB_PATH" "
    SELECT d.tag_name
    FROM deployments d
    INNER JOIN processed_tags p ON d.tag_name = p.tag_name
    WHERE d.status = 'failed' AND p.status = 'completed'
" 2>/dev/null)

if [[ -z "$inconsistency1" ]]; then
    echo "✅ No se encontraron inconsistencias de este tipo"
else
    echo "⚠️  Tags inconsistentes encontrados:"
    echo "$inconsistency1" | while read -r tag; do
        echo "   - $tag"
    done
    
    if [[ "$FIX_MODE" == "--fix" ]]; then
        echo ""
        echo "🔧 Reparando inconsistencias..."
        echo "$inconsistency1" | while read -r tag; do
            echo "   Limpiando registros de: $tag"
            sqlite3 "$DB_PATH" "DELETE FROM deployments WHERE tag_name='$tag'" 2>/dev/null || true
            sqlite3 "$DB_PATH" "DELETE FROM build_logs WHERE tag='$tag'" 2>/dev/null || true
            sqlite3 "$DB_PATH" "DELETE FROM sonar_results WHERE tag='$tag'" 2>/dev/null || true
            sqlite3 "$DB_PATH" "DELETE FROM processed_tags WHERE tag_name='$tag'" 2>/dev/null || true
            echo "   ✓ $tag limpiado"
        done
        echo ""
        echo "✅ Inconsistencias reparadas. Ahora puedes reprocesar los tags manualmente:"
        echo "$inconsistency1" | while read -r tag; do
            echo "   ./ci_cd.sh --tag $tag"
        done
    fi
fi
echo ""

#===============================================================================
# 2. Tags en deployments pero NO en processed_tags
#===============================================================================

echo "─────────────────────────────────────────────────────────────────"
echo "🔍 Buscando: deployments sin registro en processed_tags"
echo "─────────────────────────────────────────────────────────────────"

inconsistency2=$(sqlite3 "$DB_PATH" "
    SELECT d.tag_name, d.status
    FROM deployments d
    LEFT JOIN processed_tags p ON d.tag_name = p.tag_name
    WHERE p.tag_name IS NULL
" 2>/dev/null)

if [[ -z "$inconsistency2" ]]; then
    echo "✅ No se encontraron inconsistencias de este tipo"
else
    echo "⚠️  Tags sin registro en processed_tags:"
    echo "$inconsistency2" | while read -r line; do
        echo "   - $line"
    done
    
    if [[ "$FIX_MODE" == "--fix" ]]; then
        echo ""
        echo "🔧 Creando registros faltantes en processed_tags..."
        echo "$inconsistency2" | while read -r tag status; do
            echo "   Añadiendo: $tag con status=$status"
            sqlite3 "$DB_PATH" "
                INSERT OR IGNORE INTO processed_tags (tag_name, status, processed_at)
                VALUES ('$tag', '$status', datetime('now'))
            " 2>/dev/null || true
            echo "   ✓ $tag añadido"
        done
        echo ""
        echo "✅ Registros creados en processed_tags"
    fi
fi
echo ""

#===============================================================================
# 3. Tags en processed_tags pero NO en deployments
#===============================================================================

echo "─────────────────────────────────────────────────────────────────"
echo "🔍 Buscando: processed_tags sin registro en deployments"
echo "─────────────────────────────────────────────────────────────────"

inconsistency3=$(sqlite3 "$DB_PATH" "
    SELECT p.tag_name, p.status
    FROM processed_tags p
    LEFT JOIN deployments d ON p.tag_name = d.tag_name
    WHERE d.tag_name IS NULL
" 2>/dev/null)

if [[ -z "$inconsistency3" ]]; then
    echo "✅ No se encontraron inconsistencias de este tipo"
else
    echo "⚠️  Tags huérfanos en processed_tags (sin deployment asociado):"
    echo "$inconsistency3" | while read -r line; do
        echo "   - $line"
    done
    
    if [[ "$FIX_MODE" == "--fix" ]]; then
        echo ""
        echo "🔧 Eliminando registros huérfanos..."
        echo "$inconsistency3" | while read -r tag status; do
            echo "   Eliminando: $tag"
            sqlite3 "$DB_PATH" "DELETE FROM processed_tags WHERE tag_name='$tag'" 2>/dev/null || true
            echo "   ✓ $tag eliminado"
        done
        echo ""
        echo "✅ Registros huérfanos eliminados"
    fi
fi
echo ""

#===============================================================================
# 4. Deployments duplicados (no deberían existir por UNIQUE constraint)
#===============================================================================

echo "─────────────────────────────────────────────────────────────────"
echo "🔍 Buscando: tags duplicados en deployments"
echo "─────────────────────────────────────────────────────────────────"

duplicates=$(sqlite3 "$DB_PATH" "
    SELECT tag_name, COUNT(*) as count
    FROM deployments
    GROUP BY tag_name
    HAVING COUNT(*) > 1
" 2>/dev/null)

if [[ -z "$duplicates" ]]; then
    echo "✅ No se encontraron duplicados"
else
    echo "⚠️  Tags duplicados encontrados (no debería ocurrir):"
    echo "$duplicates"
    
    if [[ "$FIX_MODE" == "--fix" ]]; then
        echo ""
        echo "⚠️  Los duplicados requieren intervención manual."
        echo "   Revisa los registros con: ./scripts/db_viewer.sh"
    fi
fi
echo ""

#===============================================================================
# Resumen
#===============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo "  RESUMEN"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

if [[ "$FIX_MODE" == "--fix" ]]; then
    echo "✅ Proceso de reparación completado"
    echo ""
    echo "Recomendaciones:"
    echo "  1. Ejecuta nuevamente este script sin --fix para verificar"
    echo "  2. Revisa el estado general con: ./ci_cd.sh status"
    echo "  3. Reprocesa tags que fallaron si es necesario"
else
    echo "ℹ️  Modo de verificación (solo lectura)"
    echo ""
    if [[ -n "$inconsistency1" || -n "$inconsistency2" || -n "$inconsistency3" || -n "$duplicates" ]]; then
        echo "⚠️  Se encontraron inconsistencias."
        echo ""
        echo "Para reparar automáticamente, ejecuta:"
        echo "  $0 --fix"
    else
        echo "✅ No se encontraron inconsistencias"
        echo ""
        echo "La base de datos está en buen estado."
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
