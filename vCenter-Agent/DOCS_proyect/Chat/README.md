# Documentación del Sistema de Chat

**Sistema:** vCenter & Documentation Consultant
**Última actualización:** Marzo 2026

---

## Documentos

| Archivo | Público | Contenido |
|---------|---------|-----------|
| **ARQUITECTURA_CHAT.md** | Arquitectos / Senior Devs | Diagramas Mermaid, flujo interno, enrutamiento 4-capas, timing |
| **GUIA_FUNCIONAMIENTO.md** | Usuarios / Operadores / QA | Casos de uso, palabras clave, troubleshooting, best practices, referencia rápida |
| **GUIA_IMPLEMENTACION_TECNICA.md** | Desarrolladores | Stack, patrones de código, extensión del sistema, Docker, testing |
| **API_REFERENCE.md** | Integradores | Especificación POST /chat, OpenAPI YAML, códigos de error, ejemplos curl/Python/JS |

---

## Por rol

| Rol | Empezar por |
|-----|-------------|
| Usuario final | GUIA_FUNCIONAMIENTO.md |
| QA / Tester | GUIA_FUNCIONAMIENTO.md → GUIA_IMPLEMENTACION_TECNICA.md (testing) |
| Desarrollador (integración) | API_REFERENCE.md |
| Desarrollador (extensión) | GUIA_IMPLEMENTACION_TECNICA.md |
| Arquitecto / Tech Lead | ARQUITECTURA_CHAT.md |
| DevOps | ARQUITECTURA_CHAT.md + GUIA_IMPLEMENTACION_TECNICA.md (deployment) |

---

## Preguntas frecuentes

| Pregunta | Documento | Sección |
|----------|-----------|---------|
| ¿Cómo funciona internamente? | ARQUITECTURA_CHAT.md | Arquitectura General |
| ¿Cómo uso el chat? | GUIA_FUNCIONAMIENTO.md | Interfaz de Usuario |
| ¿Qué palabras clave usar? | GUIA_FUNCIONAMIENTO.md | Palabras Clave de Enrutamiento |
| ¿Qué hacer si falla? | GUIA_FUNCIONAMIENTO.md | Troubleshooting |
| ¿Cuál es la latencia esperada? | GUIA_FUNCIONAMIENTO.md | Performance y Optimización |
| ¿Cómo llamo al endpoint? | API_REFERENCE.md | Request/Response |
| ¿Cómo agrego un nuevo agente? | GUIA_IMPLEMENTACION_TECNICA.md | Guía de Extensión |
| ¿Cuáles son los códigos de error? | API_REFERENCE.md | Error Responses |
| ¿Cómo despliego? | GUIA_IMPLEMENTACION_TECNICA.md | Deployment |

---

## Métricas de referencia

```
Latencia típica:
  Clasificación keywords:  5-10ms
  Formateo de consulta:    100-500ms (opcional, deshabilitar con ENABLE_QUERY_FORMATTING=false)
  Procesamiento agente:    1000-5000ms
  TOTAL:                   1.1s - 5.5s

Timeout sesión:   3600s (1 hora)
Mensaje máximo:   5000 caracteres
Modelos:          gpt-oss:20b (formateador y ejecutor)
```
