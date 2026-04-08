# 🚀 Portfolio de Proyectos Técnicos

Bienvenido a mi escaparate de proyectos. Aquí encontrarás la documentación detallada de mis desarrollos privados, enfocados en **Inteligencia Artificial**, **Automatización de Infraestructura** y **Sistemas de Voz**.

---

## 📂 Proyectos Destacados

| Categoría | Proyecto | Descripción | Stack Técnico | Documentación |
| :--- | :--- | :--- | :--- | :--- |
| **Cloud & AI** | **vCenter Agent System** | Plataforma multi-agente para gestión de VMware mediante lenguaje natural. Implementa RAG y MCP. | `LangChain`, `Flask`, `vSphere SDK`, `Ollama` | [📄 Ver Detalle](./vCenter-Agent) |  
| **IoT & Voice** | **Jarvis Voice Assistant** *(Completado · Proyecto de aprendizaje · 6 meses)* | Asistente de voz offline optimizado para Raspberry Pi con control domótico y reconocimiento en español. Proyecto fundacional que sentó las bases de la arquitectura de voz. | `Python`, `Raspberry Pi`, `NLP`, `Speech-to-Text` | [📄 Ver Detalle](./Jarvis-Assistant) |
| **IoT & Voice** | **Jarvis V2** *(En desarrollo)* | Evolución del asistente de voz, incorporando las lecciones aprendidas de la v1. Nueva arquitectura, mayor modularidad y capacidades extendidas. | `Python`, `Raspberry Pi`, `NLP`, `Speech-to-Text` | [📄 Ver Detalle](./Jarvis_V2) |
| **Cloud & AI** | **EXOPS** | Plataforma web para administración de entornos VMware vSphere 8.x. Dashboard con KPIs en tiempo real, gestión de VMs, Hosts y Datastores, modo oscuro y colector de métricas en background. | `FastAPI`, `pyVmomi`, `SQLite`, `Bootstrap 5`, `Uvicorn` | [📄 Ver Detalle](./EXOPS) |
| **AI & Search** | **RAG para VMware ESXi** | Sistema de Retrieval-Augmented Generation optimizado con retrieval híbrido (vectorial + BM25), chunking semántico y verificación de relevancia para consultas sobre documentación técnica. | `LangChain`, `ChromaDB`, `Ollama`, `BM25` | [📄 Ver Detalle](./RAG) |
| **DevOps & CI/CD** | **Autotest CI/CD Pipeline** | Pipeline CI/CD completo para monitorización de tags Git, compilación de DVDs/ISOs, análisis SonarQube, gestión de VMs y despliegue SSH automatizado, con Web UI de monitorización. | `Python`, `Flask`, `Alpine.js`, `Tailwind`, `SonarQube`, `vCenter` | [📄 Ver Detalle](./Autotest) |

---

## 🛠️ Especialidades Técnicas

### 🧠 Inteligencia Artificial y Agentes
- Implementación de **RAG (Retrieval-Augmented Generation)** para consulta de documentación técnica.
- Integración de **Model Context Protocol (MCP)** para estandarizar herramientas de IA.
- Orquestación de agentes con **LangChain** y modelos locales (Ollama/Claude).

### ☁️ Infraestructura y Automatización
- Gestión programática de entornos **VMware vSphere** mediante APIs.
- Desarrollo de sistemas críticos con **logging estructurado**, auditoría y seguridad enterprise.
- Plataformas web de administración con **FastAPI**, autenticación JWT y TLS.

### 🔄 DevOps y CI/CD
- Pipelines end-to-end con detección de **tags Git**, compilación, análisis de calidad y despliegue SSH.
- Integración con **SonarQube** y gestión de VMs para entornos de prueba automatizados.
- Web UI de monitorización con **Alpine.js** y **Chart.js**.

### 🎙️ Sistemas Embebidos
- Optimización de modelos de voz para hardware limitado (**Raspberry Pi 4 / Intel N95**).
- Procesamiento de lenguaje natural (NLP) completamente **offline**.

---

## 📈 Nota sobre los Repositorios
Debido a la naturaleza sensible de la configuración de infraestructura y claves de API propietarias, el código fuente se mantiene en repositorios privados. Sin embargo, en este repositorio comparto:
- 🏗️ **Arquitectura** detallada.
- 📋 **Guías de uso** y comandos disponibles.
- 📊 **Métricas de rendimiento** y pruebas de sistema.

---
