# 🔧 Operación - Troubleshooting

## Visión General

Problemas comunes del pipeline CI/CD y sus soluciones.

**Relacionado con**:
- [[Operación - Monitorización]] - Detección de problemas
- [[Arquitectura del Pipeline]] - Contexto del sistema

---

## Git Monitor

### Problema: No Detecta Tags Nuevos

**Síntomas**:
```
[INFO] No new tags detected
```

**Diagnóstico**:
```bash
# Test conectividad Git
git ls-remote --tags https://YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO | head -5

# Verificar credenciales
cat config/.env | grep GIT_
```

**Soluciones**:
1. Verificar GIT_USER y GIT_PASSWORD en `.env`
2. Regenerar token de acceso
3. Verificar regex de tag_pattern en config

### Problema: Error de Autenticación Git

**Síntomas**:
```
fatal: Authentication failed
```

**Solución**:
```bash
# Actualizar credenciales en .env
nano config/.env

# Test manual
export GIT_USER="your_user"
export GIT_PASSWORD="your_token"
git ls-remote --tags https://$GIT_USER:$GIT_PASSWORD@YOUR_GIT_SERVER/YOUR_ORG/YOUR_REPO
```

---

## Compilación

### Problema: Timeout de Compilación

**Síntomas**:
```
[ERROR] Compilation timed out after 3600s
```

**Solución**:
```yaml
# Aumentar timeout en config/ci_cd_config.yaml
compilation:
  timeout: 7200  # 2 horas
```

### Problema: ISO No Generado

**Síntomas**:
```
[ERROR] ISO file not found: /home/YOUR_USER/compile/InstallationDVD.iso
```

**Diagnóstico**:
```bash
# Ver últimas líneas del log
tail -100 logs/compile_*.log | grep -i error

# Ejecutar build manualmente
cd /home/YOUR_USER/compile
./Development_TTCF/ttcf/utils/dvds/build_DVDs.sh
```

**Causas comunes**:
- Error en make
- Falta de dependencias de compilación
- Espacio en disco insuficiente

### Problema: ISO Demasiado Pequeño

**Síntomas**:
```
[ERROR] ISO too small: 524288000 bytes
```

**Solución**:
```bash
# Comparar con build anterior exitoso
ls -lh /path/to/previous/V24_02_14_01.iso

# Verificar contenido
isoinfo -l -i /home/YOUR_USER/compile/InstallationDVD.iso | less
```

---

## SonarQube

### Problema: Quality Gate Failure

**Síntomas**:
```
[ERROR] Quality gate FAILED: coverage 75.5% < 80%
```

**Opciones**:
1. **Mejorar código**: Aumentar cobertura de tests
2. **Ajustar umbrales**: Editar `config/ci_cd_config.yaml`
3. **Override temporal**: Cambiar `allow_override: true`

```yaml
sonarqube:
  thresholds:
    coverage: 75  # Reducir temporalmente
  allow_override: true  # Permitir deploy aunque falle
```

### Problema: SonarQube API Error

**Síntomas**:
```
[ERROR] Failed to query SonarQube API: 401 Unauthorized
```

**Solución**:
```bash
# Verificar token
cat config/.env | grep SONAR_TOKEN

# Test API manualmente
curl -u "$(grep SONAR_TOKEN config/.env | cut -d= -f2):"   https://YOUR_SONARQUBE_SERVER/api/system/status
```

---

## vCenter

### Problema: Error Upload ISO

**Síntomas**:
```
[ERROR] Failed to upload ISO to datastore
```

**Diagnóstico**:
```bash
# Test conectividad vCenter
python3.6 python/vcenter_api.py config/ci_cd_config.yaml get_vm_status

# Verificar credenciales
cat config/.env | grep VCENTER_
```

**Soluciones**:
- Verificar espacio en datastore
- Comprobar credenciales vCenter
- Verificar red/firewall

### Problema: VM No Arranca

**Síntomas**:
```
[ERROR] VM failed to power on
```

**Solución**:
```bash
# Verificar estado VM manualmente en vCenter
# Revisar logs de vCenter
# Intentar power on manual
```

---

## SSH Deploy

### Problema: SSH Connection Refused

**Síntomas**:
```
[ERROR] SSH connection failed: Connection refused
```

**Diagnóstico**:
```bash
# Test SSH manual
ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP 'echo OK'

# Verificar key
ls -la ~/.ssh/id_rsa
ssh-add -l
```

**Soluciones**:
```bash
# Regenerar key y copiar
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
ssh-copy-id YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP

# Verificar sshd en target VM
```

### Problema: install.sh Falla

**Síntomas**:
```
[ERROR] Installation script failed with exit code 1
```

**Diagnóstico**:
```bash
# Ver log de deploy
tail -100 logs/deploy_*.log

# Conectar a VM y revisar
ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP
cd /root/install
./install.sh "ope 1 - YES yes"  # Ejecutar manualmente
```

---

## Base de Datos

### Problema: Database Locked

**Síntomas**:
```
Error: database is locked
```

**Solución**:
```bash
# Activar WAL mode
sqlite3 db/pipeline.db "PRAGMA journal_mode = WAL;"

# Verificar
sqlite3 db/pipeline.db "PRAGMA journal_mode;"
# Output: wal
```

### Problema: Database Corrupted

**Síntomas**:
- Errores al hacer queries
- `PRAGMA integrity_check` falla

**Recuperación**:
```bash
# Backup corrupto
mv db/pipeline.db db/pipeline_corrupted.db

# Dump y recrear
sqlite3 db/pipeline_corrupted.db ".dump" | sqlite3 db/pipeline.db

# Si falla, restaurar desde backup
cp db/backups/pipeline_20260320.db db/pipeline.db
```

---

## Web UI

### Problema: Web UI No Responde

**Síntomas**:
- http://YOUR_PIPELINE_HOST_IP:8080 no carga
- Timeout

**Diagnóstico**:
```bash
# Ver estado servicio
sudo systemctl status cicd-web

# Ver logs
sudo journalctl -u cicd-web -n 50 --no-pager

# Verificar puerto
sudo lsof -i :8080
```

**Soluciones**:
```bash
# Reiniciar servicio
sudo systemctl restart cicd-web

# Si puerto ocupado
sudo lsof -i :8080
sudo kill -9 <PID>

# Cambiar puerto (editar service file)
sudo nano /etc/systemd/system/cicd-web.service
# Cambiar WEB_PORT=9090
sudo systemctl daemon-reload
sudo systemctl restart cicd-web
```

### Problema: No Se Ven Datos en Dashboard

**Causas**:
1. Base de datos vacía
2. Permisos incorrectos
3. Error en queries

**Solución**:
```bash
# Verificar datos en DB
sqlite3 db/pipeline.db "SELECT COUNT(*) FROM deployments"

# Verificar permisos
chmod 644 db/pipeline.db
chmod 755 db/

# Ver logs de Web UI
tail -f logs/web_error.log
```

---

## Servicios systemd

### Servicio No Inicia

**Diagnóstico**:
```bash
sudo systemctl status cicd
sudo journalctl -u cicd -n 50 --no-pager
```

**Soluciones comunes**:
```bash
# Verificar permisos ejecutable
chmod +x /home/YOUR_USER/cicd/ci_cd.sh

# Verificar .env existe
ls -la /home/YOUR_USER/cicd/config/.env

# Test ejecución manual
cd /home/YOUR_USER/cicd
./ci_cd.sh daemon
```

### Servicio Se Cae Continuamente

**Diagnóstico**:
```bash
# Ver logs en tiempo real
sudo journalctl -u cicd -f

# Verificar restart policy
sudo systemctl show cicd | grep Restart
```

---

## Problemas Generales

### Espacio en Disco Lleno

**Síntomas**:
```
No space left on device
```

**Solución**:
```bash
# Ver uso de disco
df -h /home/agent

# Limpiar logs antiguos
find logs/ -name "*.log" -mtime +30 -delete

# Limpiar ISOs antiguos
find /home/YOUR_USER/compile -name "*.iso" -mtime +7 -delete

# Vacuum DB
sqlite3 db/pipeline.db "VACUUM;"
```

### Deployment Stuck (Bloqueado)

**Síntomas**: Deployment en `compiling` por > 2 horas

**Solución**:
```bash
# Matar proceso
pkill -9 -f build_DVDs.sh

# Marcar como failed
sqlite3 db/pipeline.db "
UPDATE deployments 
SET status='failed', error_message='Manual termination - timeout'
WHERE tag_name='MAC_1_V24_02_15_01'
"
```

---

## Comandos de Diagnóstico Rápido

```bash
# Estado completo del sistema
./ci_cd.sh verify
./ci_cd.sh status

# Servicios
sudo systemctl status cicd cicd-web

# Logs recientes
tail -100 logs/pipeline_$(date +%Y%m%d).log

# DB check
sqlite3 db/pipeline.db "SELECT * FROM v_deployment_stats"

# Espacio
df -h /home/agent
du -sh /home/YOUR_USER/cicd/*
```

---

## Enlaces Relacionados

- [[Operación - Monitorización]]
- [[Operación - Mantenimiento]]
- [[Arquitectura del Pipeline]]
- [[01 - Quick Start]]
