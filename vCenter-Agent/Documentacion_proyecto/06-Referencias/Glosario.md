---
tipo: referencia
estado: actual
tags: [glosario, terminología, referencia]
version: 1.0
ultima_actualizacion: 2026-03-24
---

# 📖 Glosario del Proyecto vCenter Multi-Agent System

> Términos clave, acrónimos y conceptos del sistema multi-agente para gestión de infraestructura VMware.

---

## A

### Agent (Agente)
Componente autónomo especializado en tareas específicas. El sistema tiene 3 agentes principales: **Orquestador**, **vCenter Agent** y **Documentation Consultant**.

### Alpha (α)
Parámetro de balance en búsqueda híbrida RAG v2.4 que controla el peso entre vector search (ChromaDB) y keyword search (BM25). Valor adaptativo: 0.35 para consultas cortas (<5 palabras), 0.70 para largas (>10 palabras).

### APScheduler
Biblioteca Python para scheduling de tareas background. Usado por `report_scheduler.py` para disparar informes diarios a las 07:00.

### Authenticated Action
Decorador `@authenticated_action` que valida sesión de usuario antes de ejecutar endpoint protegido.

---

## B

### Background Agent
Proceso autónomo que ejecuta tareas programadas sin intervención del usuario. El sistema tiene 6: report scheduler, performance report, TrueNAS collector, Cisco collector, historical collector, ESXi collector.

### Bcrypt
Algoritmo de hashing de contraseñas con salt automático. Usado en `src/utils/password_manager.py` con factor de costo 12.

### BM25
Algoritmo de ranking probabilístico para búsqueda de texto (Best Matching 25). Parte del sistema RAG v2.4 para keyword search. Parámetros: k1=1.5, b=0.75.

---

## C

### ChromaDB
Base de datos vectorial para almacenamiento de embeddings. Usada en RAG v2.4 para semantic search con modelo `nomic-embed-text`.

### Chunk
Fragmento de documento dividido para procesamiento RAG. Tamaño: 1400 caracteres con 350 de overlap (semantic chunking MD-aware).

### Classifier (Clasificador)
Sistema de 4 capas en `query_classifier.py` que determina el agente destino:
- Layer 0: Exclusive keywords (agents.yaml)
- Layer 1: Critical regex patterns  
- Layer 2: Intent detection (imperativo vs pregunta)
- Layer 3: Weighted keyword scoring
- Layer 4: LLM fallback + heurística

### ConversationBufferMemory
Clase de LangChain que almacena historial de conversación por usuario. Cada usuario tiene memoria aislada en el dict `user_memories`.

---

## D

### DRS (Distributed Resource Scheduler)
Componente de VMware que balancea carga entre hosts ESXi mediante vMotion automático.

### Datastore
Unidad de almacenamiento donde residen los archivos de VMs (VMDK, logs, configuraciones). Puede ser NFS, iSCSI, VMFS.

---

## E

### Embedding
Representación vectorial de texto. Modelo usado: `nomic-embed-text` vía Ollama (768 dimensiones).

### ESXi
Hipervisor bare-metal de VMware que ejecuta las VMs directamente sobre hardware.

---

## F

### Flask
Framework web Python usado para la API REST del sistema. Versión mínima: 2.0.

---

## G

### gpt-oss:20b
Modelo LLM local ejecutado por Ollama. Usado como executor en orquestador y agentes.

---

## H

### HA (High Availability)
Funcionalidad de vCenter que reinicia VMs automáticamente si el host ESXi falla.

### Hybrid Search
Búsqueda combinada vector+keyword en RAG v2.4: ChromaDB (semantic) + BM25 (keyword) con alpha adaptativo.

---

## I

### Intent Detection
Capa 2 del clasificador que distingue entre comandos imperativos ("despliega una VM") y preguntas de aprendizaje ("¿qué es DRS?").

---

## L

### LangChain
Framework Python para desarrollo de aplicaciones LLM. Usado para: chains, agents, memory, tool calling.

### LLM (Large Language Model)
Modelo de lenguaje grande. El sistema usa `gpt-oss:20b` ejecutado localmente vía Ollama.

### LRU Cache
Caché Least Recently Used. Usado en `embedding_cache.py` para almacenar 1000 embeddings de consultas frecuentes (~30% speedup).

---

## M

### MCP (Model Context Protocol)
Capa de abstracción entre LLM y herramientas vCenter (pyvmomi). Registro centralizado en `mcp_tool_registry.py` con 36 tools en 9 grupos (catálogo total).

### MOC (Map of Content)
Nota índice en Obsidian que enlaza y organiza documentación relacionada.

---

## N

### nomic-embed-text
Modelo de embeddings de 768 dimensiones ejecutado por Ollama. Usado en RAG v2.4 para vector search.

---

## O

### Ollama
Runtime local para ejecutar LLMs y modelos de embeddings. Endpoint por defecto: http://localhost:11434.

### Orchestrator (Orquestador)
Agente principal (`main_agent.py`) que recibe consultas del usuario, clasifica con 4 capas y delega a vCenter Agent o Documentation Consultant.

---

## P

### pyvmomi
SDK oficial de Python para VMware vSphere API. Usado por MCP tools para operaciones sobre vCenter.

### Pool de Conexiones
`VCenterConnectionPool` que mantiene máximo 5 conexiones activas con timeout de 30s para evitar agotamiento de sesiones.

---

## Q

### Query Expansion
Proceso en RAG v2.4 que expande consulta del usuario con 62 familias de términos relacionados (VMware + project tools). Implementado en `query_expander.py`.

---

## R

### RAG (Retrieval-Augmented Generation)
Técnica que augmenta LLM con recuperación de documentos relevantes. Sistema actual: RAG v2.4 (híbrido ChromaDB+BM25).

### RBAC (Role-Based Access Control)
Control de acceso basado en roles. Roles disponibles: `user`, `admin`, `superuser`.

### Reranking
Fase 6 del pipeline RAG que reordena candidatos por relevancia (40 candidatos → top 8). Heurística: 25% original + 40% term_freq + 15% length + 10% position.

---

## S

### Search Mode
Estrategia de filtrado de documentos en RAG v2.4:
- **Strict**: "SOLO vcenter" → solo carpeta vcenter
- **Boosting**: "busca en esxi" → x2 boost a carpeta esxi
- **Global**: búsqueda sin restricciones

### Session Manager
Componente que gestiona sesiones de usuario en SQLite (`data/users.db`). Timeout configurable, por defecto 3600s (1 hora).

### SmartConnect
Función de pyvmomi para establecer conexión con vCenter Server. Envuelta por connection pool.

### SNMP (Simple Network Management Protocol)
Protocolo para monitorización de dispositivos de red. Usado por TrueNAS collector (v3) y Cisco collector (v2c).

### Sticky Routing
Patrón de enrutamiento conversacional: mensajes de seguimiento se envían al último agente usado (180s timeout). Permite diálogos multi-turno naturales sin repetir contexto.

### Structured Logger
Sistema de logging categorizado en `structured_logger.py`. 6 categorías: api, audit, security, performance, business, system.

---

## T

### Tool
Función Python decorada con `@tool` que el LLM puede invocar. El sistema tiene 36 MCP tools para operaciones vCenter.

### TrueNAS
Sistema de almacenamiento NAS. Monitoreado por `truenas_snmp_collector.py` con SNMP v3.

---

## V

### vCenter
Plataforma centralizada de VMware para gestión de múltiples hosts ESXi y VMs.

### vMotion
Tecnología de VMware para migrar VMs entre hosts ESXi sin downtime.

### VMDK (Virtual Machine Disk)
Formato de archivo de disco virtual de VMware.

### vSphere
Suite completa de virtualización de VMware (incluye ESXi + vCenter).

---

## W

### Wikilink
Enlace interno de Obsidian con sintaxis `[[nombre-documento]]`. Usado para conectar documentación relacionada.

---

## Y

### YAML (YAML Ain't Markup Language)
Formato de serialización de datos. Usado en `config/agents.yaml` para keywords de enrutamiento y frontmatter de Obsidian.

---

## Acrónimos Rápidos

| Acrónimo | Significado |
|----------|-------------|
| API | Application Programming Interface |
| CRUD | Create, Read, Update, Delete |
| DRS | Distributed Resource Scheduler |
| ESXi | Elastic Sky X integrated |
| HA | High Availability |
| LLM | Large Language Model |
| MCP | Model Context Protocol |
| NFS | Network File System |
| OID | Object Identifier (SNMP) |
| PDF | Portable Document Format |
| RAG | Retrieval-Augmented Generation |
| RBAC | Role-Based Access Control |
| REST | Representational State Transfer |
| SNMP | Simple Network Management Protocol |
| SQL | Structured Query Language |
| SSL | Secure Sockets Layer |
| TLS | Transport Layer Security |
| UI/UX | User Interface / User Experience |
| VM | Virtual Machine |
| VMDK | Virtual Machine Disk |

---

## Relacionado

- [[00-MOC-Principal]] - Mapa de contenido principal
- [[Stack-Tecnologico]] - Dependencias y versiones
- [[Arquitectura-Sistema]] - Visión general del sistema

---

*Última actualización: 2026-03-24*
