# Solución al Problema del Servicio systemd

## ⚠️ PROBLEMA CRÍTICO IDENTIFICADO

**Error principal**: `Failed to determine group credentials: No such process`

**Causa**: El grupo `agent` especificado en el servicio no existe en el sistema.

**Solución**: Eliminada la línea `Group=agent` del archivo de servicio. Systemd usará automáticamente el grupo primario del usuario `agent`.

---

## Problemas Identificados y Solucionados

### 0. **Grupo No Existente (CRÍTICO)** ⭐
**Problema**: `Group=agent` en el servicio causaba fallo porque el grupo no existe.

**Solución**: Eliminada la línea `Group=agent`. Systemd usa automáticamente el grupo primario del usuario especificado.

### 1. **Redirecciones de I/O Problemáticas**
**Problema**: El servicio usaba `StandardOutput=append:...` y `StandardError=append:...` que causaban conflictos cuando el script capturaba salida con `$()`.

**Solución**: Cambiado a `StandardOutput=journal` y `StandardError=journal` para usar el journal de systemd, que es más robusto.

### 2. **Captura de Salida en Daemon Loop**
**Problema**: La captura `new_tag=$(script) || true` silenciaba errores y podía capturar logs mezclados.

**Solución**: 
- Redirigir stderr explícitamente: `new_tag=$(script 2>&2) || detect_exit_code=$?`
- Validar código de salida antes de procesar
- Limpieza más agresiva de la salida capturada con `tr -d '[:space:][:cntrl:]'`

### 3. **Errores No Capturados Matando el Daemon**
**Problema**: Con `set -euo pipefail`, cualquier error mataba el daemon completo.

**Solución**:
- Trap específico para el daemon que solo loguea errores sin matar el proceso
- Uso de `set +e` / `set -e` alrededor del pipeline para capturar errores sin matar el daemon

### 4. **Buffering de Logs**
**Problema**: Logs se concatenaban sin espacios debido a buffering.

**Solución**:
- Flush inmediato después de escribir logs en `common.sh`
- Añadido `PYTHONUNBUFFERED=1` al servicio
- Llamada a `sync` antes de cada sleep en el daemon loop

---

## 🚀 Instrucciones de Aplicación Rápida (IMPORTANTE)

### Paso 1: Copiar archivo corregido
```bash
# Desde tu máquina local (Windows)
scp cicd/cicd.service YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/
```

### Paso 2: Actualizar el servicio
```bash
# Conectar al servidor
ssh YOUR_USER@YOUR_PIPELINE_HOST_IP
cd /home/YOUR_USER/cicd

# Actualizar el servicio
sudo ./update_service.sh update
```

### Paso 3: Verificar que funciona
```bash
# Ver logs en tiempo real
sudo journalctl -u cicd.service -f

# En otra terminal, verificar que NO hay errores
sudo journalctl -u cicd.service -p err --since "1 minute ago"
```

**Si todo está bien**, deberías ver:
- `[INFO] Verificando nuevos tags...` cada 5 minutos
- NO más errores de "Failed to determine group credentials"

---

## Instrucciones Detalladas (para copiar todos los archivos)

### 1. Copiar Archivos Actualizados al Servidor

```bash
# Desde tu máquina local (Windows), copiar al servidor SUSE
scp cicd/cicd.service YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/
scp cicd/ci_cd.sh YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/
scp cicd/scripts/common.sh YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/scripts/
scp cicd/update_service.sh YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/
```

### 2. Reinstalar el Servicio

```bash
# Conectar al servidor
ssh YOUR_USER@YOUR_PIPELINE_HOST_IP

# Ir al directorio de trabajo
cd /home/YOUR_USER/cicd

# Detener el servicio si está corriendo
sudo systemctl stop cicd.service

# Copiar el nuevo archivo de servicio
sudo cp cicd.service /etc/systemd/system/

# Recargar la configuración de systemd
sudo systemctl daemon-reload

# Reiniciar el servicio
sudo systemctl restart cicd.service

# Verificar estado
sudo systemctl status cicd.service
```

### 3. Monitorear los Logs

Con los cambios, los logs ahora van al journal de systemd. Para verlos:

```bash
# Ver logs en tiempo real
sudo journalctl -u cicd.service -f

# Ver últimas 100 líneas
sudo journalctl -u cicd.service -n 100

# Ver logs desde hoy
sudo journalctl -u cicd.service --since today

# Ver logs con marca de tiempo
sudo journalctl -u cicd.service --since "2026-02-26 09:00:00"

# Ver solo errores
sudo journalctl -u cicd.service -p err
```

Los logs propios del script seguirán yendo a:
- `/home/YOUR_USER/cicd/logs/pipeline_YYYYMMDD.log`

### 4. Verificar que Funciona

```bash
# 1. Ver que el servicio está activo
sudo systemctl is-active cicd.service
# Debe mostrar: active

# 2. Ver los logs en tiempo real
sudo journalctl -u cicd.service -f

# 3. Verificar que detecta tags
# Esperar unos 5 minutos (polling_interval) y ver si aparece:
# [INFO] Verificando nuevos tags...

# 4. Si hay algún error, aparecerá en el journal
```

---

## Diferencias entre Ejecución Manual vs. Servicio

| Aspecto | Manual | Servicio |
|---------|--------|----------|
| **Logs stdout/stderr** | Terminal | journal + archivos propios |
| **Variables de entorno** | Shell del usuario | Definidas en .service + .env |
| **PATH** | PATH completo del usuario | PATH limitado (ahora ampliado) |
| **Reinicio automático** | No | Sí (cada 30s si falla) |
| **Buffering** | Menos problemático | Ahora resuelto con PYTHONUNBUFFERED |

---

## Comandos Útiles

```bash
# Ver si el daemon está corriendo
ps aux | grep ci_cd.sh

# Ver el PID file
cat /home/YOUR_USER/cicd/.cicd.pid

# Matar manualmente si es necesario
kill $(cat /home/YOUR_USER/cicd/.cicd.pid)

# Ver última verificación de tags
tail -n 50 /home/YOUR_USER/cicd/logs/pipeline_$(date +%Y%m%d).log

# Reiniciar el servicio
sudo systemctl restart cicd.service

# Deshabilitar auto-reinicio temporalmente
sudo systemctl stop cicd.service

# Habilitar el servicio para que inicie con el sistema
sudo systemctl enable cicd.service
```

---

## Troubleshooting

### Si el servicio sigue muriendo:

1. **Ver errores específicos**:
   ```bash
   sudo journalctl -u cicd.service -p err --since "5 minutes ago"
   ```

2. **Verificar permisos**:
   ```bash
   ls -la /home/YOUR_USER/cicd/
   # Todo debe pertenecer a agent:agent
   ```

3. **Verificar .env**:
   ```bash
   cat /home/YOUR_USER/cicd/config/.env
   # Debe contener GIT_PASSWORD, etc.
   ```

4. **Probar manualmente primero**:
   ```bash
   # Como usuario agent (no root)
   /home/YOUR_USER/cicd/ci_cd.sh daemon
   # Ctrl+C después de un par de ciclos si funciona
   ```

5. **Ver todas las variables del servicio**:
   ```bash
   sudo systemctl show cicd.service
   ```

### Si los logs siguen concatenados:

Esto ya debería estar resuelto, pero si persiste:
- Verificar que `PYTHONUNBUFFERED=1` está en el servicio
- Verificar que `common.sh` tiene el flush de logs
- Comprobar que no hay redirecciones adicionales en otros scripts

---

## Cambios Aplicados (Resumen Técnico)

### `cicd.service` (CRÍTICO):
- ✅ **Eliminada línea `Group=agent`** - Este era el error principal
- ✅ StandardOutput/Error → journal
- ✅ PATH ampliado con /sbin
- ✅ PYTHONUNBUFFERED=1
- ✅ SyslogIdentifier para filtrar logs

### `ci_cd.sh`:
- ✅ Redirección explícita stderr: `$(cmd 2>&2)`
- ✅ Captura código de salida: `|| detect_exit_code=$?`
- ✅ Validación robusta del tag
- ✅ set +e/set -e alrededor de run_pipeline
- ✅ Trap ERR específico para daemon
- ✅ sync antes de sleep

### `common.sh`:
- ✅ mkdir -p $LOG_DIR al inicio
- ✅ Flush explícito de logs con bloque {}
- ✅ Logs siempre a stderr con >&2

---

**Nota**: Después de aplicar estos cambios, el servicio debería funcionar correctamente sin morirse. Los logs serán claros y separados correctamente.
