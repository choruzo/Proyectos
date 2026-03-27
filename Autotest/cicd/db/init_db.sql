-- =============================================================================
-- init_db.sql - Esquema de base de datos para auditoría CI/CD
-- =============================================================================
-- Ejecutar con: sqlite3 /home/YOUR_USER/cicd/db/pipeline.db < init_db.sql

-- Tabla principal de despliegues
CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,
    status TEXT CHECK(status IN ('pending', 'compiling', 'analyzing', 'deploying', 'success', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_seconds INTEGER,
    triggered_by TEXT DEFAULT 'daemon',  -- 'daemon' o 'manual' + usuario
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Logs de compilación por fase
CREATE TABLE IF NOT EXISTS build_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    tag TEXT NOT NULL,
    phase TEXT CHECK(phase IN ('checkout', 'prepare', 'compile', 'package')),
    start_time INTEGER,          -- Unix timestamp
    duration INTEGER,            -- Segundos
    exit_code INTEGER,
    log_file TEXT,               -- Ruta al fichero de log
    created_at TEXT DEFAULT (datetime('now'))
);

-- Resultados de análisis SonarQube
CREATE TABLE IF NOT EXISTS sonar_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    tag TEXT NOT NULL,
    timestamp TEXT,
    -- Métricas Overall Code (todo el código)
    coverage REAL,
    bugs INTEGER,
    vulnerabilities INTEGER,
    code_smells INTEGER,
    security_hotspots INTEGER,
    -- Métricas New Code (solo código nuevo - lo que evalúa el Quality Gate)
    new_coverage REAL,
    new_bugs INTEGER,
    new_vulnerabilities INTEGER,
    new_code_smells INTEGER,
    new_security_hotspots INTEGER,
    passed INTEGER CHECK(passed IN (0, 1)),
    quality_gate_status TEXT,    -- 'OK', 'WARN', 'ERROR'
    created_at TEXT DEFAULT (datetime('now'))
);

-- Log de ejecución detallado
CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id INTEGER REFERENCES deployments(id),
    phase TEXT NOT NULL,
    message TEXT,
    level TEXT CHECK(level IN ('DEBUG', 'INFO', 'WARN', 'ERROR')),
    timestamp TEXT DEFAULT (datetime('now'))
);

-- Registro de tags procesados (para evitar reprocesar)
CREATE TABLE IF NOT EXISTS processed_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,
    first_seen_at TEXT DEFAULT (datetime('now')),
    processed_at TEXT,
    status TEXT CHECK(status IN ('pending', 'processing', 'completed', 'skipped'))
);

-- =============================================================================
-- Índices para consultas frecuentes
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_deployments_tag ON deployments(tag_name);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_started ON deployments(started_at);
CREATE INDEX IF NOT EXISTS idx_sonar_tag ON sonar_results(tag);

-- =============================================================================
-- Usuarios de la Web UI (autenticacion)
-- =============================================================================
CREATE TABLE IF NOT EXISTS web_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    last_login TEXT
);

CREATE INDEX IF NOT EXISTS idx_web_users_username ON web_users(username);
CREATE INDEX IF NOT EXISTS idx_build_logs_tag ON build_logs(tag);
CREATE INDEX IF NOT EXISTS idx_execution_log_deployment ON execution_log(deployment_id);
CREATE INDEX IF NOT EXISTS idx_processed_tags_name ON processed_tags(tag_name);

-- =============================================================================
-- Vistas útiles para consultas
-- =============================================================================

-- Vista: Últimos 10 despliegues con resumen
CREATE VIEW IF NOT EXISTS v_recent_deployments AS
SELECT 
    d.id,
    d.tag_name,
    d.status,
    d.started_at,
    d.completed_at,
    d.duration_seconds,
    d.triggered_by,
    d.error_message,
    s.coverage,
    s.bugs,
    s.passed as sonar_passed
FROM deployments d
LEFT JOIN sonar_results s ON d.tag_name = s.tag
ORDER BY d.id DESC
LIMIT 10;

-- Vista: Estadísticas de despliegue
CREATE VIEW IF NOT EXISTS v_deployment_stats AS
SELECT 
    COUNT(*) as total_deployments,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(duration_seconds) as avg_duration_seconds,
    MAX(started_at) as last_deployment
FROM deployments;

-- =============================================================================
-- Datos iniciales (opcional)
-- =============================================================================

-- Insertar registro de inicio
INSERT OR IGNORE INTO execution_log (phase, message, level) 
VALUES ('init', 'Base de datos inicializada correctamente', 'INFO');
