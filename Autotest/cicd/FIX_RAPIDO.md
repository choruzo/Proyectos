# Fix Rápido: Error "Failed to determine group credentials"

## Problema
El servicio systemd fallaba con:
```
Failed to determine group credentials: No such process
Failed at step GROUP spawning /home/YOUR_USER/cicd/ci_cd.sh: No such process
```

## Causa
El grupo `agent` especificado en `Group=agent` no existe en el sistema.

## Solución Aplicada
Eliminada la línea `Group=agent` del archivo `cicd.service`. Systemd usará automáticamente el grupo primario del usuario `agent`.

---

## Pasos para Aplicar el Fix

### 1. Verificar grupo del usuario (opcional)
```bash
# En el servidor, verificar qué grupo tiene el usuario agent
id agent

# Salida ejemplo:
# uid=1000(agent) gid=100(users) groups=100(users),...
```

### 2. Copiar el archivo corregido
```bash
# Desde Windows
scp cicd/cicd.service YOUR_USER@YOUR_PIPELINE_HOST_IP:/home/YOUR_USER/cicd/
```

### 3. Actualizar el servicio
```bash
# En el servidor
ssh YOUR_USER@YOUR_PIPELINE_HOST_IP
cd /home/YOUR_USER/cicd
sudo ./update_service.sh update
```

### 4. Verificar que funciona
```bash
# Ver estado
sudo systemctl status cicd.service

# Ver logs en tiempo real
sudo journalctl -u cicd.service -f

# Verificar que NO hay errores
sudo journalctl -u cicd.service -p err --since "1 minute ago"
```

---

## Alternativa (si se quiere usar un grupo específico)

Si prefieres que el servicio use un grupo específico:

```bash
# 1. Ver qué grupo primario tiene el usuario
id -gn agent
# Ejemplo output: users

# 2. Editar cicd.service y poner:
Group=users   # O el grupo que mostró el comando anterior

# 3. O crear el grupo 'agent' si no existe:
sudo groupadd agent
sudo usermod -g agent agent
```

**Recomendación**: Dejar el servicio SIN la línea `Group=` es lo más simple y funciona perfectamente.

---

## Verificación Final

Después de aplicar el fix:

```bash
# 1. El servicio debe estar activo
sudo systemctl is-active cicd.service
# Output: active

# 2. No debe haber errores nuevos
sudo journalctl -u cicd.service -p err --since "5 minutes ago"
# Output: (vacío o sin errores de GROUP)

# 3. Ver el daemon funcionando
sudo journalctl -u cicd.service -n 20
# Debe mostrar: [INFO] Verificando nuevos tags...
```

---

## Nota Importante

Este era el verdadero problema por el que el servicio se moría. Los cambios anteriores (de buffering, etc.) también son buenos, pero **este era el error crítico**.

Con este fix, el servicio debería funcionar perfectamente.
