#!/bin/bash
#
# Script para verificación visual en Chrome/Firefox
# Genera un reporte HTML con enlaces directos a todos los endpoints
#
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8080}"
BASE_URL="http://${HOST}:${PORT}"
OUTPUT_FILE="/tmp/cicd_web_ui_test_report.html"

cat > "$OUTPUT_FILE" <<EOF
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GALTTCMC CI/CD Web UI - Test Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .endpoint {
            padding: 15px;
            margin: 10px 0;
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }
        .endpoint-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .endpoint-title {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        .endpoint-method {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .endpoint-url {
            font-family: 'Courier New', monospace;
            background: #e8e8e8;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 8px 0;
            word-break: break-all;
        }
        .endpoint-url a {
            color: #667eea;
            text-decoration: none;
        }
        .endpoint-url a:hover {
            text-decoration: underline;
        }
        .endpoint-description {
            color: #666;
            font-size: 0.95em;
        }
        .quick-links {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .quick-link {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            text-decoration: none;
            transition: transform 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .quick-link:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .quick-link-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .quick-link-emoji {
            font-size: 2em;
            margin-bottom: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #10b981;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .info-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .info-box strong {
            color: #1e40af;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GALTTCMC CI/CD Web UI</h1>
        <p>Test & Verification Report</p>
        <p style="font-size: 0.9em; margin-top: 15px;">
            <span class="status-indicator"></span>
            Server: <strong>${BASE_URL}</strong>
        </p>
    </div>

    <div class="info-box">
        <strong>ℹ️ Cómo usar este reporte:</strong><br>
        Haz clic en cualquier enlace para probar el endpoint directamente en tu navegador.
        Los endpoints JSON mostrarán datos crudos. Las páginas HTML mostrarán la interfaz completa.
    </div>

    <div class="section">
        <h2>🎯 Quick Access - Main Pages</h2>
        <div class="quick-links">
            <a href="${BASE_URL}/" class="quick-link" target="_blank">
                <div class="quick-link-emoji">🏠</div>
                <div class="quick-link-title">Dashboard</div>
                <div>Vista principal con métricas</div>
            </a>
            <a href="${BASE_URL}/pipeline-runs" class="quick-link" target="_blank">
                <div class="quick-link-emoji">📋</div>
                <div class="quick-link-title">Pipeline Runs</div>
                <div>Historial de ejecuciones</div>
            </a>
            <a href="${BASE_URL}/logs" class="quick-link" target="_blank">
                <div class="quick-link-emoji">📄</div>
                <div class="quick-link-title">Logs</div>
                <div>Visor de logs</div>
            </a>
            <a href="${BASE_URL}/sonar-results" class="quick-link" target="_blank">
                <div class="quick-link-emoji">📊</div>
                <div class="quick-link-title">SonarQube</div>
                <div>Resultados de análisis</div>
            </a>
        </div>
    </div>

    <div class="section">
        <h2>📱 HTML Pages</h2>
        
        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Dashboard</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/" target="_blank">${BASE_URL}/</a></div>
            <div class="endpoint-description">Página principal con métricas, gráficos y últimos deployments</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Pipeline Runs</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/pipeline-runs" target="_blank">${BASE_URL}/pipeline-runs</a></div>
            <div class="endpoint-description">Historial completo de ejecuciones del pipeline con paginación</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Logs Viewer</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/logs" target="_blank">${BASE_URL}/logs</a></div>
            <div class="endpoint-description">Visor de archivos de log con búsqueda en tiempo real</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">SonarQube Results</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/sonar-results" target="_blank">${BASE_URL}/sonar-results</a></div>
            <div class="endpoint-description">Resultados de análisis de calidad y gráficos de tendencias</div>
        </div>
    </div>

    <div class="section">
        <h2>🔌 API Endpoints - Dashboard</h2>
        
        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Dashboard Stats</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/dashboard/stats" target="_blank">${BASE_URL}/api/dashboard/stats</a></div>
            <div class="endpoint-description">Estadísticas generales: total deployments, success rate, last 24h, avg duration</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Recent Deployments</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/dashboard/recent-deployments" target="_blank">${BASE_URL}/api/dashboard/recent-deployments</a></div>
            <div class="endpoint-description">Últimos 10 deployments con detalles</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Chart Data</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/dashboard/chart-data" target="_blank">${BASE_URL}/api/dashboard/chart-data</a></div>
            <div class="endpoint-description">Datos para gráficos (últimos 7 días) en formato Chart.js</div>
        </div>
    </div>

    <div class="section">
        <h2>🔌 API Endpoints - Deployments</h2>
        
        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Deployments List (Page 1)</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/deployments?page=1&per_page=20&status=all" target="_blank">${BASE_URL}/api/deployments?page=1&per_page=20&status=all</a></div>
            <div class="endpoint-description">Lista paginada de todos los deployments</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Success Deployments</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/deployments?status=success" target="_blank">${BASE_URL}/api/deployments?status=success</a></div>
            <div class="endpoint-description">Solo deployments exitosos</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Failed Deployments</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/deployments?status=failed" target="_blank">${BASE_URL}/api/deployments?status=failed</a></div>
            <div class="endpoint-description">Solo deployments fallidos</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Deployment Detail (ID: 1)</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/deployment/1" target="_blank">${BASE_URL}/api/deployment/1</a></div>
            <div class="endpoint-description">Detalle completo de un deployment específico (cambia el ID según sea necesario)</div>
        </div>
    </div>

    <div class="section">
        <h2>🔌 API Endpoints - Logs</h2>
        
        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Logs List</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/logs/list" target="_blank">${BASE_URL}/api/logs/list</a></div>
            <div class="endpoint-description">Lista de archivos de log disponibles con tamaños y fechas</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">View Log (last 500 lines)</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/logs/view/pipeline_$(date +%Y%m%d).log?lines=500" target="_blank">${BASE_URL}/api/logs/view/pipeline_$(date +%Y%m%d).log?lines=500</a></div>
            <div class="endpoint-description">Contenido de un log específico (últimas 500 líneas)</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">Search in Log</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/logs/view/pipeline_$(date +%Y%m%d).log?lines=1000&search=error" target="_blank">${BASE_URL}/api/logs/view/pipeline_$(date +%Y%m%d).log?lines=1000&search=error</a></div>
            <div class="endpoint-description">Buscar "error" en el log actual</div>
        </div>
    </div>

    <div class="section">
        <h2>🔌 API Endpoints - SonarQube</h2>
        
        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">SonarQube Results</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/sonar/results" target="_blank">${BASE_URL}/api/sonar/results</a></div>
            <div class="endpoint-description">Últimos 50 resultados de análisis SonarQube</div>
        </div>

        <div class="endpoint">
            <div class="endpoint-header">
                <div class="endpoint-title">SonarQube Trends</div>
                <span class="endpoint-method">GET</span>
            </div>
            <div class="endpoint-url"><a href="${BASE_URL}/api/sonar/trends" target="_blank">${BASE_URL}/api/sonar/trends</a></div>
            <div class="endpoint-description">Tendencias de métricas (últimos 10 deployments exitosos)</div>
        </div>
    </div>

    <div class="section">
        <h2>🧪 Testing Commands</h2>
        <div style="background: #f9fafb; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">
            <p><strong>Desde línea de comandos (Linux):</strong></p>
            <p># Ejecutar suite completa de tests<br>
            chmod +x test_web_ui.sh<br>
            ./test_web_ui.sh ${HOST} ${PORT}</p>
            <br>
            <p># O probar endpoints individuales con curl<br>
            curl ${BASE_URL}/api/dashboard/stats | python3 -m json.tool<br>
            curl ${BASE_URL}/api/deployments?page=1 | python3 -m json.tool</p>
            <br>
            <p><strong>Desde Chrome/Firefox:</strong></p>
            <p>1. Abre la consola de desarrollo (F12)<br>
            2. Ve a la pestaña "Network"<br>
            3. Navega por la web UI<br>
            4. Observa las peticiones HTTP y sus respuestas</p>
        </div>
    </div>

    <div style="text-align: center; padding: 40px 0; color: #999;">
        <p>GALTTCMC CI/CD Pipeline Web UI v1.0.0</p>
        <p>Generated: $(date '+%Y-%m-%d %H:%M:%S')</p>
    </div>
</body>
</html>
EOF

echo "✓ Test report generated: $OUTPUT_FILE"
echo ""
echo "Open in browser:"
echo "  firefox $OUTPUT_FILE"
echo "  google-chrome $OUTPUT_FILE"
echo ""

# Try to open in default browser if available
if command -v xdg-open &> /dev/null; then
    xdg-open "$OUTPUT_FILE" 2>/dev/null &
    echo "✓ Opening in default browser..."
elif command -v firefox &> /dev/null; then
    firefox "$OUTPUT_FILE" 2>/dev/null &
    echo "✓ Opening in Firefox..."
elif command -v google-chrome &> /dev/null; then
    google-chrome "$OUTPUT_FILE" 2>/dev/null &
    echo "✓ Opening in Chrome..."
else
    echo "Please open the file manually in your browser"
fi
