#!/bin/bash
#===============================================================================
# deploy.sh - Despliegue en VM destino vía SSH
#===============================================================================
# Fase 4 del pipeline CI/CD
# Monta ISO, copia contenido y ejecuta instalación en la VM destino
#
# Uso:
#   ./deploy.sh                  # Ejecutar despliegue completo
#   ./deploy.sh wait_ssh         # Solo esperar conexión SSH
#   ./deploy.sh mount            # Solo montar ISO
#   ./deploy.sh copy             # Solo copiar contenido
#   ./deploy.sh install          # Solo ejecutar instalación
#   ./deploy.sh cleanup          # Desmontar ISO
#   ./deploy.sh status           # Ver estado de la VM
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cargar funciones comunes
source "$SCRIPT_DIR/common.sh"

# Variables de despliegue
DEPLOY_LOG_FILE="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"

#===============================================================================
# Funciones SSH
#===============================================================================

# Configurar opciones SSH
get_ssh_opts() {
    local ssh_key="${TARGET_VM_KEY:-/home/agent/.ssh/id_rsa}"
    echo "-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 -i $ssh_key"
}

# Ejecutar comando en VM destino
ssh_run() {
    local cmd=$1
    local ssh_opts
    ssh_opts=$(get_ssh_opts)
    
    log_debug "SSH ejecutando: $cmd"
    ssh $ssh_opts "${TARGET_VM_USER}@${TARGET_VM_IP}" "$cmd" 2>&1
}

# Ejecutar comando con logging
ssh_run_logged() {
    local cmd=$1
    local description=${2:-"Ejecutando comando"}
    
    log_info "$description"
    if ssh_run "$cmd" | tee -a "$DEPLOY_LOG_FILE"; then
        return 0
    else
        return 1
    fi
}

# Verificar conectividad SSH
check_ssh_connection() {
    log_debug "Verificando conexión SSH a ${TARGET_VM_IP}..."
    if ssh_run "echo ok" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

#===============================================================================
# Esperar conexión SSH
#===============================================================================

wait_for_ssh() {
    local max_attempts=${1:-30}
    local interval=${2:-10}
    local attempt=1
    
    log_info "Esperando conexión SSH a ${TARGET_VM_IP}..."
    log_info "Máximo intentos: $max_attempts, intervalo: ${interval}s"
    
    while ! check_ssh_connection; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Timeout esperando conexión SSH después de $max_attempts intentos"
            return 1
        fi
        
        log_debug "Intento $attempt/$max_attempts - Sin conexión, esperando ${interval}s..."
        sleep "$interval"
        ((attempt++))
    done
    
    log_ok "Conexión SSH establecida con ${TARGET_VM_IP}"
    
    # Mostrar info del sistema remoto
    log_info "Sistema remoto:"
    ssh_run "uname -a" 2>/dev/null || true
    
    return 0
}

#===============================================================================
# Montar ISO
#===============================================================================

mount_iso() {
    local mount_point
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "MONTAJE DE ISO"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Punto de montaje: $mount_point"
    
    # Crear punto de montaje si no existe
    ssh_run "mkdir -p $mount_point" || true
    
    # Desmontar si ya estaba montado
    log_debug "Verificando si hay montaje previo..."
    ssh_run "umount $mount_point 2>/dev/null || true"
    
    # Montar CD-ROM
    log_info "Montando CD-ROM..."
    
    # Intentar diferentes dispositivos
    local devices="/dev/cdrom /dev/sr0 /dev/sr1 /dev/dvd"
    local mounted=false
    
    for dev in $devices; do
        log_debug "Intentando montar $dev..."
        if ssh_run "mount $dev $mount_point 2>/dev/null"; then
            log_ok "ISO montado desde $dev en $mount_point"
            mounted=true
            break
        fi
    done
    
    if [[ "$mounted" != "true" ]]; then
        log_error "No se pudo montar el CD-ROM"
        log_error "Dispositivos intentados: $devices"
        return 1
    fi
    
    # Verificar contenido
    log_info "Contenido del ISO:"
    ssh_run "ls -la $mount_point" | head -20 | tee -a "$DEPLOY_LOG_FILE"
    
    return 0
}

#===============================================================================
# Copiar contenido
#===============================================================================

copy_content() {
    local mount_point
    local install_path
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
    install_path=$(config_get "target_vm.install_path" "/root/install")
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "COPIA DE CONTENIDO"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Origen: $mount_point"
    log_info "Destino: $install_path"
    
    # Verificar que el ISO está montado
    if ! ssh_run "mountpoint -q $mount_point"; then
        log_error "ISO no está montado en $mount_point"
        return 1
    fi
    
    # Limpiar directorio destino
    log_info "Limpiando directorio destino..."
    ssh_run "rm -rf $install_path" || true
    ssh_run "mkdir -p $install_path"
    
    # Copiar contenido
    log_info "Copiando contenido..."
    local start_time
    start_time=$(date +%s)
    
    if ! ssh_run "cp -r $mount_point/* $install_path/"; then
        log_error "Error copiando contenido"
        return 1
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Asignar permisos de ejecución
    log_info "Asignando permisos de ejecución..."
    ssh_run "find $install_path -name '*.sh' -exec chmod +x {} \\;" || true
    
    # Verificar contenido copiado
    local size
    size=$(ssh_run "du -sh $install_path 2>/dev/null | cut -f1" || echo "N/A")
    local file_count
    file_count=$(ssh_run "find $install_path -type f 2>/dev/null | wc -l" || echo "N/A")
    
    log_ok "Contenido copiado (${size}, ${file_count} ficheros, ${duration}s)"
    
    return 0
}

#===============================================================================
# Ejecutar instalación
#===============================================================================

run_installation() {
    local install_path
    local install_script
    local install_params
    install_path=$(config_get "target_vm.install_path" "/root/install")
    install_script=$(config_get "target_vm.install_script" "install.sh")
    install_params=$(config_get "target_vm.install_params" "ope 1 - YES yes")
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "EJECUCIÓN DE INSTALACIÓN"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Directorio: $install_path"
    log_info "Script: $install_script"
    log_info "Parámetros: $install_params"
    
    local script_path="$install_path/$install_script"
    
    # Verificar que existe el script
    if ! ssh_run "test -f $script_path"; then
        log_error "Script de instalación no encontrado: $script_path"
        log_info "Ficheros disponibles en $install_path:"
        ssh_run "ls -la $install_path" | tee -a "$DEPLOY_LOG_FILE"
        return 1
    fi
    
    # Asegurar permisos
    ssh_run "chmod +x $script_path"
    
    # Verificar que el repositorio está montado
    local mount_point
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
    if ! ssh_run "mountpoint -q $mount_point" 2>/dev/null; then
        log_warn "Advertencia: El repositorio no parece estar montado en $mount_point"
    else
        log_ok "Repositorio confirmado en $mount_point"
    fi
    
    # Ejecutar instalación con parámetros
    log_info "Ejecutando instalación..."
    log_info "Comando: ./$install_script $install_params"
    log_info "Log: $DEPLOY_LOG_FILE"
    
    local start_time
    start_time=$(date +%s)
    
    # Ejecutar con timeout largo (puede tardar).
    # El here-string (<<< "") alimenta un carácter de nueva línea al stdin de SSH,
    # lo que hace que el "read -n 1 -s -r -p ..." del install.sh lo consuma
    # automáticamente sin bloquear el pipeline.
    set +e
    local ssh_opts
    ssh_opts=$(get_ssh_opts)
    ssh $ssh_opts "${TARGET_VM_USER}@${TARGET_VM_IP}" \
        "cd $install_path && ./$install_script $install_params" \
        <<< "" 2>&1 | tee -a "$DEPLOY_LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    set -e
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        log_ok "Instalación completada exitosamente ($(format_duration $duration))"
        return 0
    else
        log_error "Instalación falló con código: $exit_code"
        log_error "Revise el log: $DEPLOY_LOG_FILE"
        return 1
    fi
}

#===============================================================================
# Limpieza post-instalación
#===============================================================================

cleanup() {
    local mount_point
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")

    log_info "Realizando limpieza post-instalación..."

    # Desmontar ISO y ejectar CD-ROM.
    # Se ignoran los errores SSH: si la VM ya reinició (install.sh hace reboot),
    # la conexión fallará, pero eso es normal y no debe interrumpir el pipeline.
    log_info "Desmontando ISO..."
    ssh_run "umount $mount_point 2>/dev/null || true" 2>/dev/null || \
        log_warn "No se pudo desmontar via SSH (VM puede estar reiniciando)"

    # Ejectar el CD-ROM para liberar el lock del kernel del SO invitado.
    # Sin esto, vCenter detecta el dispositivo como "bloqueado por el invitado"
    # y muestra el diálogo de confirmación que bloquea el pipeline.
    log_info "Eyectando CD-ROM para liberar lock del kernel..."
    ssh_run "eject /dev/sr0 2>/dev/null || eject /dev/cdrom 2>/dev/null || eject /dev/dvd 2>/dev/null || true" 2>/dev/null || \
        log_warn "No se pudo ejectar CD-ROM via SSH (VM puede estar reiniciando)"

    log_ok "Limpieza completada"
    return 0
}

#===============================================================================
# Cambiar CD-ROM a repositorio
#===============================================================================

switch_to_repository_iso() {
    log_info "═══════════════════════════════════════════════════════════"
    log_info "CAMBIO A ISO DE REPOSITORIO"
    log_info "═══════════════════════════════════════════════════════════"
    
    # IMPORTANTE: Desmontar desde el SO invitado primero para evitar bloqueo
    local mount_point
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
    
    log_info "Desmontando CD-ROM desde el SO invitado..."
    
    # Verificar si está montado
    if ssh_run "mountpoint -q $mount_point" 2>/dev/null; then
        log_info "CD-ROM está montado, desmontando..."
        ssh_run "umount $mount_point 2>/dev/null || true"
        sleep 2
        
        # Verificar nuevamente y forzar si es necesario
        if ssh_run "mountpoint -q $mount_point" 2>/dev/null; then
            log_warn "Forzando desmontaje con -f..."
            ssh_run "umount -f $mount_point 2>/dev/null || true"
            sleep 2
            
            # Último intento con -l (lazy unmount)
            if ssh_run "mountpoint -q $mount_point" 2>/dev/null; then
                log_warn "Forzando desmontaje lazy con -l..."
                ssh_run "umount -l $mount_point 2>/dev/null || true"
                sleep 3
            fi
        fi
    else
        log_info "CD-ROM no está montado"
    fi
    
    log_ok "CD-ROM desmontado desde el SO"

    # Ejectar el CD-ROM desde el SO invitado para liberar COMPLETAMENTE el lock del kernel.
    # El comando umount solo desmonta el sistema de ficheros, pero el kernel sigue
    # manteniendo el dispositivo de bloque abierto. El eject envía la señal de expulsión
    # al driver y libera el lock, evitando así el diálogo de vCenter:
    # "El sistema operativo invitado bloqueó la puerta de CD-ROM..."
    log_info "Eyectando CD-ROM para liberar el lock del kernel del SO invitado..."
    ssh_run "eject /dev/sr0 2>/dev/null || eject /dev/cdrom 2>/dev/null || eject /dev/dvd 2>/dev/null || true"
    log_ok "Lock del CD-ROM liberado correctamente"

    # Breve espera para que el hypervisor registre la liberación del dispositivo
    log_info "Esperando 3 segundos para que vCenter registre la liberación del CD-ROM..."
    sleep 3
    
    local repository_iso
    repository_iso=$(config_get "vcenter.repository_iso" "[NAS_LIBRERIA] P27/Repositorio/SLES15SP7V3P27.iso")
    
    log_info "ISO de repositorio: $repository_iso"
    
    # Llamar a vcenter_api.py para cambiar el CD-ROM
    local vcenter_script
    vcenter_script="$(dirname "$SCRIPT_DIR")/python/vcenter_api.py"
    
    if [[ ! -f "$vcenter_script" ]]; then
        log_error "Script de vCenter no encontrado: $vcenter_script"
        return 1
    fi
    
    log_info "Configurando CD-ROM con ISO de repositorio..."
    if ! python3 "$vcenter_script" "$CONFIG_FILE" configure_cdrom "$repository_iso" 2>&1 | tee -a "$DEPLOY_LOG_FILE"; then
        log_error "Error configurando CD-ROM con repositorio"
        return 1
    fi
    
    log_ok "CD-ROM configurado con repositorio"
    
    # Esperar un momento adicional para que el cambio se aplique completamente
    log_info "Esperando 3 segundos para que el cambio se aplique completamente..."
    sleep 3
    
    return 0
}

#===============================================================================
# Montar repositorio
#===============================================================================

mount_repository() {
    local mount_point
    mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
    
    log_info "═══════════════════════════════════════════════════════════"
    log_info "MONTAJE DEL REPOSITORIO"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Punto de montaje: $mount_point"
    
    # Crear punto de montaje si no existe
    ssh_run "mkdir -p $mount_point" || true
    
    # Asegurar que no hay nada montado
    log_debug "Desmontando punto de montaje previo..."
    ssh_run "umount $mount_point 2>/dev/null || true"
    
    # Esperar un momento para que el kernel reconozca el nuevo CD-ROM
    log_info "Esperando 3 segundos para que el kernel reconozca el nuevo CD-ROM..."
    sleep 3
    
    # Montar CD-ROM del repositorio
    log_info "Montando repositorio desde CD-ROM..."
    
    # Intentar diferentes dispositivos con reintentos
    local devices="/dev/cdrom /dev/sr0 /dev/sr1 /dev/dvd"
    local mounted=false
    local max_attempts=3
    
    for attempt in $(seq 1 $max_attempts); do
        log_info "Intento de montaje $attempt/$max_attempts..."
        
        for dev in $devices; do
            log_debug "Intentando montar $dev..."
            
            # Verificar que el dispositivo existe
            if ! ssh_run "test -e $dev" 2>/dev/null; then
                log_debug "Dispositivo $dev no existe"
                continue
            fi
            
            if ssh_run "mount -t iso9660 $dev $mount_point 2>/dev/null"; then
                log_ok "Repositorio montado desde $dev en $mount_point"
                mounted=true
                break 2
            else
                log_debug "Fallo al montar $dev"
            fi
        done
        
        if [[ "$mounted" != "true" && $attempt -lt $max_attempts ]]; then
            log_warn "No se pudo montar en intento $attempt, esperando 5 segundos..."
            sleep 5
        fi
    done
    
    if [[ "$mounted" != "true" ]]; then
        log_error "No se pudo montar el repositorio después de $max_attempts intentos"
        log_error "Dispositivos intentados: $devices"
        
        # Debug: Mostrar dispositivos disponibles
        log_info "Dispositivos de bloque disponibles:"
        ssh_run "ls -la /dev/sr* /dev/cd* 2>/dev/null || echo 'No se encontraron dispositivos CD'"
        
        # Debug: Verificar dmesg
        log_info "Últimas líneas de dmesg relacionadas con CD-ROM:"
        ssh_run "dmesg | grep -i 'cd\\|dvd\\|sr0' | tail -10 || echo 'Sin mensajes en dmesg'"
        
        return 1
    fi
    
    # Verificar contenido del repositorio
    log_info "Contenido del repositorio:"
    ssh_run "ls -la $mount_point" | head -20 | tee -a "$DEPLOY_LOG_FILE"
    
    return 0
}

#===============================================================================
# Pipeline de despliegue completo
#===============================================================================

run_full_deploy() {
    log_info "═══════════════════════════════════════════════════════════"
    log_info "PIPELINE DE DESPLIEGUE COMPLETO"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "VM destino: ${TARGET_VM_USER}@${TARGET_VM_IP}"
    log_info "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local total_start
    total_start=$(date +%s)
    
    # Fase 1: Esperar SSH
    log_info ""
    log_info "[1/7] Esperando conexión SSH..."
    if ! wait_for_ssh 30 10; then
        log_error "FALLO: No se pudo establecer conexión SSH"
        return 1
    fi
    
    # Fase 2: Montar ISO de instalación
    log_info ""
    log_info "[2/7] Montando ISO de instalación..."
    if ! mount_iso; then
        log_error "FALLO: No se pudo montar el ISO de instalación"
        return 1
    fi
    
    # Fase 3: Copiar contenido
    log_info ""
    log_info "[3/7] Copiando contenido del ISO..."
    if ! copy_content; then
        log_error "FALLO: No se pudo copiar el contenido"
        cleanup
        return 1
    fi
    
    # Fase 4: Desmontar ISO de instalación
    log_info ""
    log_info "[4/7] Desmontando ISO de instalación..."
    cleanup
    
    # Fase 5: Cambiar CD-ROM a repositorio
    log_info ""
    log_info "[5/7] Cambiando CD-ROM a ISO de repositorio..."
    if ! switch_to_repository_iso; then
        log_error "FALLO: No se pudo cambiar al repositorio"
        return 1
    fi
    
    # Fase 6: Montar repositorio
    log_info ""
    log_info "[6/7] Montando repositorio..."
    if ! mount_repository; then
        log_error "FALLO: No se pudo montar el repositorio"
        return 1
    fi
    
    # Fase 7: Ejecutar instalación (con repositorio montado)
    log_info ""
    log_info "[7/7] Ejecutando instalación..."
    if ! run_installation; then
        log_error "FALLO: La instalación falló"
        cleanup
        return 1
    fi

    # NO se realiza limpieza post-instalación: install.sh termina con 'reboot',
    # por lo que la VM reinicia automáticamente liberando todos los montajes y
    # locks del CD-ROM. Intentar SSH tras el reinicio falla porque el nuevo SO
    # tiene la contraseña de root expirada (cambio forzado por la instalación).
    log_info "La VM está reiniciando como parte del proceso de instalación."
    log_info "No se requiere limpieza manual: el reinicio libera todos los montajes."

    local total_end
    total_end=$(date +%s)
    local total_duration=$((total_end - total_start))
    
    log_info ""
    log_info "═══════════════════════════════════════════════════════════"
    log_ok "DESPLIEGUE COMPLETADO EXITOSAMENTE"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Duración total: $(format_duration $total_duration)"
    log_info "Log: $DEPLOY_LOG_FILE"
    log_info "Fin: $(date '+%Y-%m-%d %H:%M:%S')"
    
    return 0
}

#===============================================================================
# Estado y diagnóstico
#===============================================================================

show_status() {
    log_info "Estado de la VM destino: ${TARGET_VM_IP}"
    echo ""
    
    if check_ssh_connection; then
        log_ok "Conexión SSH: OK"
        echo ""
        
        log_info "Sistema:"
        ssh_run "uname -a" 2>/dev/null || echo "(no disponible)"
        echo ""
        
        log_info "Uptime:"
        ssh_run "uptime" 2>/dev/null || echo "(no disponible)"
        echo ""
        
        log_info "Disco:"
        ssh_run "df -h /" 2>/dev/null || echo "(no disponible)"
        echo ""
        
        local mount_point
        mount_point=$(config_get "target_vm.mount_point" "/mnt/cdrom")
        if ssh_run "mountpoint -q $mount_point" 2>/dev/null; then
            log_info "ISO montado en $mount_point:"
            ssh_run "ls -la $mount_point" 2>/dev/null | head -10
        else
            log_info "ISO: No montado"
        fi
        echo ""
        
        local install_path
        install_path=$(config_get "target_vm.install_path" "/root/install")
        if ssh_run "test -d $install_path" 2>/dev/null; then
            log_info "Directorio de instalación ($install_path):"
            ssh_run "ls -la $install_path" 2>/dev/null | head -10
        else
            log_info "Directorio de instalación: No existe"
        fi
    else
        log_error "Conexión SSH: FALLIDA"
        log_info "Verifique que la VM está encendida y accesible"
    fi
}

verify_prerequisites() {
    log_info "Verificando prerequisitos de despliegue..."
    
    local errors=0
    
    # Verificar variables
    if [[ -z "${TARGET_VM_IP:-}" ]]; then
        log_error "TARGET_VM_IP no configurado"
        ((errors++))
    else
        log_ok "TARGET_VM_IP: $TARGET_VM_IP"
    fi
    
    if [[ -z "${TARGET_VM_USER:-}" ]]; then
        log_error "TARGET_VM_USER no configurado"
        ((errors++))
    else
        log_ok "TARGET_VM_USER: $TARGET_VM_USER"
    fi
    
    local ssh_key="${TARGET_VM_KEY:-/home/agent/.ssh/id_rsa}"
    if [[ -f "$ssh_key" ]]; then
        log_ok "SSH Key: $ssh_key"
    else
        log_error "SSH Key no encontrada: $ssh_key"
        ((errors++))
    fi
    
    # Verificar comandos
    for cmd in ssh scp; do
        if command -v "$cmd" &>/dev/null; then
            log_ok "Comando disponible: $cmd"
        else
            log_error "Comando no encontrado: $cmd"
            ((errors++))
        fi
    done
    
    # Verificar conectividad
    if check_ssh_connection; then
        log_ok "Conectividad SSH: OK"
    else
        log_warn "Conectividad SSH: No disponible (VM apagada?)"
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Se encontraron $errors errores"
        return 1
    fi
    
    log_ok "Prerequisitos verificados correctamente"
    return 0
}

#===============================================================================
# Main
#===============================================================================

usage() {
    cat <<EOF
Uso: $(basename "$0") [comando]

Comandos:
  (sin args)    Ejecutar despliegue completo
  wait_ssh      Solo esperar conexión SSH
  mount         Solo montar ISO de instalación
  copy          Solo copiar contenido
  switch_repo   Solo cambiar CD-ROM a repositorio
  mount_repo    Solo montar repositorio
  install       Solo ejecutar instalación
  cleanup       Desmontar ISO
  status        Ver estado de la VM
  verify        Verificar prerequisitos
  help          Mostrar esta ayuda

Ejemplos:
  $(basename "$0")              # Despliegue completo
  $(basename "$0") status       # Ver estado
  $(basename "$0") mount        # Solo montar ISO de instalación
  $(basename "$0") switch_repo  # Cambiar a repositorio

Variables de entorno usadas:
  TARGET_VM_IP      IP de la VM destino
  TARGET_VM_USER    Usuario SSH (default: root)
  TARGET_VM_KEY     Ruta a la clave SSH

EOF
}

main() {
    local cmd="${1:-full}"
    
    # Crear directorio de logs
    mkdir -p "$LOG_DIR"
    
    case "$cmd" in
        full|"")
            run_full_deploy
            ;;
        wait_ssh|wait)
            wait_for_ssh "${2:-30}" "${3:-10}"
            ;;
        mount)
            mount_iso
            ;;
        copy)
            copy_content
            ;;
        switch_repo|switch)
            switch_to_repository_iso
            ;;
        mount_repo|repo)
            mount_repository
            ;;
        install)
            run_installation
            ;;
        cleanup|clean)
            cleanup
            ;;
        status)
            show_status
            ;;
        verify)
            verify_prerequisites
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Comando no reconocido: $cmd"
            usage
            exit 1
            ;;
    esac
}

main "$@"
