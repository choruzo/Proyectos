# 🏠 GALTTCMC CI/CD - Documentación

Bienvenido a la documentación del pipeline CI/CD de GALTTCMC.

## 📚 Estructura de la Documentación

### 🎯 Inicio Rápido
- [[00 - Visión General]] - Entendimiento del sistema completo
- [[01 - Quick Start]] - Primeros pasos y comandos esenciales

### 🏗️ Arquitectura
- [[Arquitectura del Pipeline]] - Diseño general del sistema de ejecución
- [[Arquitectura Web UI]] - Diseño del sistema de monitorización web
- [[Modelo de Datos]] - Esquema de base de datos SQLite

### 📖 Guías Técnicas

#### Pipeline de Ejecución
- [[Pipeline - Git Monitor]] - Fase 1: Detección de tags
- [[Pipeline - Compilación]] - Fase 2: Build de DVDs/ISOs
- [[Pipeline - SonarQube]] - Fase 3: Análisis de calidad
- [[Pipeline - vCenter]] - Fase 4: Gestión de VMs
- [[Pipeline - SSH Deploy]] - Fase 5: Despliegue en VM destino
- [[Pipeline - Common Functions]] - Librería compartida

#### Web UI
- [[Web - Arquitectura]] - Stack técnico y componentes
- [[Web - API Endpoints]] - REST API del backend Flask
- [[Web - Frontend Components]] - Alpine.js + Tailwind
- [[Web - Visualización de Datos]] - Chart.js y métricas

### 🛠️ Guías Operativas
- [[Operación - Instalación]] - Setup inicial del sistema
- [[Operación - Monitorización]] - Supervisión del pipeline
- [[Operación - Troubleshooting]] - Resolución de problemas comunes
- [[Operación - Mantenimiento]] - Tareas de mantenimiento

### 🔧 Referencia Técnica
- [[Referencia - Configuración]] - Archivo YAML y variables de entorno
- [[Referencia - Base de Datos]] - Esquema y queries útiles
- [[Referencia - APIs Externas]] - Git, SonarQube, vCenter
- [[Referencia - Logs]] - Sistema de logging y ubicaciones

### 📊 Diagramas
- [[Diagrama - Flujo Completo]] - Pipeline end-to-end
- [[Diagrama - Dependencias]] - Relaciones entre módulos
- [[Diagrama - Estados]] - Máquina de estados de deployments

---

**Última actualización**: 2026-03-20  
**Versión**: 1.0  
**Entorno**: SUSE Linux 15 @ YOUR_PIPELINE_HOST_IP
