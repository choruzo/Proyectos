# Agente vCenter — Documentación Técnica

Documentación técnica completa del agente vCenter del sistema multi-agente.

## Índice de Documentos

| Archivo | Descripción |
|---------|-------------|
| [ARQUITECTURA.md](./ARQUITECTURA.md) | Arquitectura general, componentes y flujos de datos |
| [MCP_TOOLS.md](./MCP_TOOLS.md) | Referencia completa de las 33 herramientas MCP |
| [CONNECTION_POOL.md](./CONNECTION_POOL.md) | Sistema de conexiones a vCenter y fallback vcsim |
| [SEGURIDAD.md](./SEGURIDAD.md) | Modelo de seguridad, aislamiento por usuario y patrones críticos |

## Resumen del Agente vCenter

El agente vCenter es uno de los tres componentes del sistema multi-agente. Se encarga de ejecutar operaciones sobre infraestructura VMware vCenter mediante lenguaje natural.

### Capacidades

- Gestión de VMs (desplegar, eliminar, clonar, listar, apagar/encender)
- Snapshots (crear, revertir, eliminar)
- Reconfiguración (CPU, RAM, red, NICs)
- Monitoreo ESXi en tiempo real
- Exploración de datastores
- Informes de rendimiento y obsolescencia

### Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| LLM | Ollama `gpt-oss:20b` (num_ctx=8192) |
| Orquestación | LangChain AgentExecutor |
| Herramientas | FastMCP v3.0 (33 tools) |
| vCenter API | pyvmomi |
| Sesiones | SQLite + ConversationBufferMemory |
| Logging | Structured JSON (6 categorías) |

### Archivos Clave

| Archivo | Rol |
|---------|-----|
| `src/core/agent.py` | Entry point del agente, AgentExecutor, sesiones |
| `server/mcp_tool_registry.py` | Registro centralizado de las 33 tools MCP |
| `server/mcp_tool_wrappers.py` | Adaptadores LangChain StructuredTool |
| `server/mcp_vcenter_server.py` | Servidor FastMCP v3.0 (protocolo MCP) |
| `src/utils/vcenter_tools.py` | Wrappers pyvmomi + VCenterConnectionPool |
