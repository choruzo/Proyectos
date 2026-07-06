---
tipo: operacional
versión: 1.0
tags: [implementacion, deploy, instalacion, setup, configuracion]
última_actualización: 2026-03-24
relacionado:
  - "[[Stack-Tecnologico]]"
  - "[[Configuracion]]"
  - "[[Troubleshooting]]"
---

# Guía de Implementación — Despliegue del Sistema

Guía técnica completa para implementar el vCenter Multi-Agent System desde cero.

## Instalación Rápida (5 Pasos)

```bash
# 1. Clonar repo
git clone <repository_url>
cd Agente/vcenter_agent_system

# 2. Entorno virtual + Dependencias
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements_oficial.txt

# 3. Instalar Ollama + Modelos
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# 4. Configurar credenciales vCenter
# Editar config/config.json → vcenter_host, vcenter_user, vcenter_pass

# 5. Generar certificado SSL (self-signed)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout vcenter_agent_system/ssl/key.pem \
  -out vcenter_agent_system/ssl/cert.pem \
  -days 365 -subj "/C=ES/ST=Local/L=Local/O=NexusOps/CN=vcenter-agent"

# 6. Instalar y arrancar nginx (ver vcenter_agent_system/nginx/README.md)
# Windows: nginx -c "ruta/nginx_windows.conf"
# Ubuntu:  sudo cp nginx_ubuntu.conf /etc/nginx/sites-available/vcenter-agent && sudo systemctl reload nginx

# 7. Iniciar (start_app.bat arranca nginx + Flask automáticamente en Windows)
python run.py                   # Flask en http://127.0.0.1:5001
# → https://localhost:5000      (acceso externo vía nginx)
```

## Requisitos del Sistema

- **CPU:** 4+ cores (8 recomendado)
- **RAM:** 16 GB (32 GB con RAG v2.4)
- **Disco:** 50 GB libres (SSD)
- **Python:** 3.9-3.11
- **Ollama:** Última versión

Ver [[Stack-Tecnologico]] para dependencias completas.

## Configuración Básica

Editar `config/config.json`:

```json
{
  "vcenter_host": "172.30.188.136",
  "vcenter_user": "agent@vcenter.local",
  "vcenter_pass": "password",
  "esxi_hosts": { /* ... */ }
}
```

**⚠️ Producción:** Usar variables de entorno:
```powershell
$env:VCENTER_USER = "agent@vcenter.local"
$env:VCENTER_PASS = "password_seguro"
```

Ver [[Configuracion]] para todas las opciones.

## Verificar Instalación

```bash
# Health check (aceptar certificado self-signed con -k)
curl -k https://localhost:5000/health

# Tests
python -m pytest tests/ -v
python tests/check_system_status.py
```

## Troubleshooting

Ver [[Troubleshooting]] para soluciones detalladas.

**Problemas comunes:**
- Ollama no responde → `ollama list`
- ChromaDB error → `rm -rf data/chroma_db`
- vCenter connection failed → Verificar credenciales y ping
- Puerto en uso → `netstat -ano | findstr :5000`
- nginx no arranca → `nginx -t -c ruta/nginx_windows.conf` para ver errores de config
- Certificado no encontrado → generar con openssl (ver `ssl/README.md`)

## Checklist Pre-Producción

- [ ] Cambiar passwords por defecto
- [ ] Variables de entorno para credenciales
- [x] Habilitar HTTPS + certificado SSL (nginx proxy, puerto 5000)
- [ ] Configurar firewall (puerto 5000 TCP)
- [ ] Backup automático de `data/`
- [ ] Rotación de logs configurada

## Enlaces Relacionados

- [[Stack-Tecnologico]] — Dependencias y versiones
- [[Configuracion]] — Referencia config.json completa
- [[Troubleshooting]] — Solución de problemas
- [[Arquitectura-Sistema]] — Visión general

***
**Fuente original:** `vcenter_agent_system/DOCS_proyect/Chat/GUIA_IMPLEMENTACION_TECNICA.md`
