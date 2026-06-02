---
tipo: propuesta
estado: actual
relacionado:
  - "[[Agente-vCenter]]"
  - "[[Arquitectura-Agente-vCenter]]"
  - "[[Sistema-MCP]]"
  - "[[Propuestas-Funcionales]]"
tags: [mejoras, roadmap, vcenter, harness, langgraph]
version: 1.0
ultima_actualizacion: 2026-05-11
---

# 🧰 Roadmap de Mejora del Arnés del Agente vCenter

> Propuesta de evolución del arnés del agente vCenter, centrada en robustez, mantenibilidad, aislamiento de estado y claridad operativa.

***
## 🎯 Objetivo

Mejorar el arnés que envuelve al agente vCenter para que:

- separe mejor runtime, web y lógica conversacional
- reduzca estado global y comportamiento dependiente del proceso
- haga el tool-calling más predecible
- endurezca las acciones destructivas
- mejore trazabilidad, testing y rendimiento

***
## 🔎 Problemas Detectados

1. **Responsabilidades mezcladas** en `src/core/agent.py`: bootstrap del agente, Flask, auth, logging, descargas y grafo conviven en el mismo módulo.
2. **Estado global fuera de LangGraph**: el `progressive disclosure` mantiene contexto de grupos en `_user_group_cache`, dependiente del proceso y no del `thread_id`.
3. **Herencia acumulativa de tools**: la unión de grupos previos puede ampliar demasiado el conjunto de herramientas activas y degradar el aislamiento por turno.
4. **Salida de tools poco estructurada**: el postproceso web depende de regex sobre texto libre para detectar adjuntos `.csv` y `.zip`.
5. **Flujo de confirmación mejorable**: ante respuestas ambiguas el estado se limpia y la intención destructiva se reanaliza, lo que puede volver el flujo menos determinista.
6. **Coste innecesario por turno**: wrappers de tools y binding al LLM se reconstruyen repetidamente.
7. **Cobertura parcial del arnés**: hay buenas unitarias del grafo y del disclosure, pero faltan pruebas más integradas del runtime completo.

***
## 🗺️ Plan por Fases

| Fase | Foco                     | Resultado esperado                            |
| ---- | ------------------------ | --------------------------------------------- |
| 0    | Línea base               | Métricas y comportamiento actual documentados |
| 1    | Separación de runtime    | Arnés vCenter desacoplado de Flask/web        |
| 2    | Estado conversacional    | Selección de tools persistida en LangGraph    |
| 3    | Contrato de herramientas | Outputs estructurados y menos regex           |
| 4    | Confirmaciones críticas  | Flujo destructivo más seguro y explícito      |
| 5    | Rendimiento y validación | Menos overhead y más cobertura E2E            |

***
## Fase 0 - Línea Base y Criterios de Éxito

**Objetivo:** fijar el punto de partida antes de reestructurar.

### Cambios propuestos

- Inventariar el flujo actual desde `process_vcenter_query()` hasta `ToolNode`.
- Registrar métricas base:
  - número de tools expuestas por query
  - latencia por turno
  - frecuencia de confirmaciones
  - porcentaje de queries que caen en fallback
- Documentar los contratos actuales entre:
  - `src/core/agent.py`
  - `src/core/agent_graph.py`
  - `server/mcp_tool_registry.py`
  - `src/api/main_agent.py`

### Criterios de cierre

- Existe una referencia clara del flujo actual.
- Se puede comparar el comportamiento antes y después de cada fase.

***
## Fase 1 - Separación de Responsabilidades del Arnés

**Objetivo:** extraer el runtime del agente vCenter a una capa propia y reducir el acoplamiento con Flask.

### Cambios propuestos

- Mover la inicialización del agente, el grafo, el prompt y el entrypoint programático a un módulo dedicado, por ejemplo:
  - `src/core/vcenter_runtime.py`
  - o `src/core/vcenter_service.py`
- Dejar `src/api/main_agent.py` como adaptador HTTP/SSE.
- Mantener en `agent.py` solo compatibilidad temporal o bootstrap mínimo.
- Centralizar helpers transversales del arnés:
  - construcción de `thread_id`
  - callback de progreso
  - extracción de respuesta final
  - adaptación entre runtime y capa web

### Beneficio principal

Menos mezcla entre UI web, sesión HTTP y ejecución conversacional del agente.

### Riesgos a vigilar

- Romper imports heredados desde `main_agent.py`
- Duplicar bootstrap si conviven módulo nuevo y antiguo demasiado tiempo

***
## Fase 2 - Mover el Estado de Selección de Tools al Grafo

**Objetivo:** eliminar dependencia del estado global de proceso y alinear el `progressive disclosure` con LangGraph.

### Cambios propuestos

- Sustituir `_user_group_cache` en `server/mcp_tool_registry.py` por campos persistidos en `VCenterAgentState`.
- Persistir en el grafo datos como:
  - grupos activos
  - timestamp de última actividad relevante
  - motivo de selección (`keyword`, `semantic`, `followup`, `fallback`)
- Calcular la selección de tools por `thread_id`, no por variable global de módulo.
- Reutilizar la infraestructura ya existente de expiración lógica de sesiones cuando tenga sentido.

### Beneficio principal

El comportamiento deja de depender del worker/proceso y pasa a depender del estado conversacional real del usuario.

### Riesgos a vigilar

- Crecer demasiado el estado del grafo
- Persistir datos redundantes si no se define bien qué se guarda y qué se recalcula

***
## Fase 3 - Rehacer la Política de Progressive Disclosure

**Objetivo:** mantener el aislamiento por turno sin perder contexto conversacional útil.

### Cambios propuestos

- Reemplazar la política de **unión acumulativa** por una política más estricta:
  1. recalcular grupos desde la query actual
  2. heredar solo en follow-ups claros
  3. limitar la herencia a uno o pocos turnos
- Añadir un modo de `decay` o reemplazo de grupos previos.
- Registrar en logs cuándo se hereda contexto y cuándo se reinicia.
- Añadir tests específicos para:
  - follow-up corto que debe heredar
  - query nueva que debe resetear grupos
  - mezcla de grupos incompatible entre turnos

### Beneficio principal

El LLM recibe menos tools irrelevantes y el arnés conserva mejor el propósito del turno actual.

### Riesgos a vigilar

- Perder demasiada continuidad en conversaciones fragmentadas
- Ser demasiado conservador y obligar al usuario a repetir contexto

***
## Fase 4 - Contrato Estructurado para Resultados de Tools

**Objetivo:** reemplazar el postproceso basado en texto libre por resultados consumibles por runtime y frontend.

### Cambios propuestos

- Definir una estructura común de salida para herramientas, por ejemplo:

```python
{
    "status": "success",
    "summary": "Informe generado",
    "details": "...",
    "artifacts": [
        {"type": "zip", "path": "reports/archivo.zip", "download_name": "archivo.zip"}
    ],
    "metadata": {...}
}
```

- Adaptar gradualmente las tools que generan adjuntos o resultados complejos.
- Hacer que el runtime traduzca esa estructura a:
  - respuesta al usuario
  - attachments seguros
  - auditoría
  - SSE/progreso
- Reducir la lógica regex en la capa web a compatibilidad transitoria.

### Beneficio principal

Menos fragilidad en descargas, errores y renderizado de respuestas complejas.

### Riesgos a vigilar

- Convivencia temporal entre tools antiguas y nuevas
- Necesidad de una capa de compatibilidad mientras se migra el catálogo

***
## Fase 5 - Endurecer Confirmaciones de Acciones Destructivas

**Objetivo:** hacer el flujo de confirmación más determinista, explícito y seguro.

### Cambios propuestos

- Mantener la acción pendiente ante respuestas ambiguas, en vez de limpiar estado inmediatamente.
- Añadir TTL de confirmación y mensaje de expiración.
- Permitir respuestas del tipo:
  - `sí`
  - `no`
  - `ver detalles`
  - `cancelar`
- Cubrir escenarios con múltiples acciones destructivas o planes encadenados.
- Registrar en auditoría:
  - acción propuesta
  - argumentos relevantes
  - aceptación/cancelación/expiración

### Beneficio principal

Menor probabilidad de reencaminar accidentalmente una acción destructiva o perder la intención a mitad del flujo.

### Riesgos a vigilar

- Hacer el flujo demasiado rígido para operaciones válidas
- Duplicar confirmaciones si el LLM insiste en reformular la misma acción

***
## Fase 6 - Rendimiento, Reutilización y Validación

**Objetivo:** reducir overhead del arnés y aumentar confianza en cambios futuros.

### Cambios propuestos

- Cachear wrappers `StructuredTool` por subconjunto activo cuando sea seguro hacerlo.
- Revisar el coste de `bind_tools()` por turno y minimizar reconstrucciones.
- Añadir tests de integración para:
  - `process_vcenter_query()`
  - confirmación persistente entre invocaciones
  - adjuntos descargables
  - follow-ups con herencia de grupos
  - fallback y errores de tool-calling JSON
- Añadir pruebas de contrato para las tools con output estructurado.

### Beneficio principal

Menos latencia y menos regresiones al tocar el arnés.

### Riesgos a vigilar

- Cachear demasiado y reutilizar herramientas con contexto incorrecto
- Tests demasiado acoplados a mensajes exactos del LLM

***
## ✅ Orden Recomendado de Implementación

1. **Fase 1** - Separación de runtime
2. **Fase 2** - Estado en LangGraph
3. **Fase 3** - Política de disclosure
4. **Fase 4** - Output estructurado de tools
5. **Fase 5** - Confirmaciones robustas
6. **Fase 6** - Rendimiento y validación

> Este orden reduce riesgo: primero se limpia la arquitectura, luego se mueve el estado, después se endurecen contratos y por último se optimiza.

***
## 📁 Archivos Más Afectados

| Archivo | Motivo |
|---------|--------|
| `src/core/agent.py` | Hoy concentra bootstrap y runtime del agente |
| `src/core/agent_graph.py` | Punto central de estado, confirmaciones y routing interno |
| `server/mcp_tool_registry.py` | Disclosure, grupos, cache actual y definición de tools |
| `server/mcp_tool_wrappers.py` | Creación de `StructuredTool` y posible caching |
| `src/api/main_agent.py` | Adaptación HTTP/SSE y postproceso de adjuntos |
| `unitary_test/test_vcenter_graph_interruption.py` | Cobertura de confirmaciones |
| `unitary_test/test_progressive_disclosure.py` | Cobertura de selección de tools |

***
## 🧭 Resultado Esperado al Final del Roadmap

Un arnés de agente vCenter:

- más modular
- menos dependiente de estado global
- más predecible en multi-turn
- más seguro para operaciones destructivas
- más fácil de observar, probar y evolucionar
