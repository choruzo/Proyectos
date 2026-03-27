<#
.SYNOPSIS
    Diagnóstico y corrección de port forwarding SSH

.DESCRIPTION
    Este script:
    1. Diagnostica por qué el port forwarding SSH falla
    2. Proporciona comandos para corregir la configuración del servidor
    3. Verifica la configuración después de aplicar los cambios

.EXAMPLE
    .\fix_ssh_forwarding.ps1
#>

# Configuración
$SSH_USER = "agent"
$SSH_HOST = "YOUR_PIPELINE_HOST_IP"
$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa"

# Colores
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# Banner
Write-Host ""
Write-ColorOutput "═══════════════════════════════════════════════" "Cyan"
Write-ColorOutput "  Diagnóstico SSH Port Forwarding" "White"
Write-ColorOutput "═══════════════════════════════════════════════" "Cyan"
Write-Host ""

# Verificar SSH
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "[FAIL] SSH no disponible" "Red"
    exit 1
}

Write-ColorOutput "[PASO 1] Comprobando configuración SSH del servidor..." "Cyan"
Write-Host ""

# Comando SSH para verificar la configuración
$sshCheckCmd = @"
ssh -i `"$SSH_KEY_PATH`" ${SSH_USER}@${SSH_HOST} 'sudo grep -E "AllowTcpForwarding|PermitOpen" /etc/ssh/sshd_config | grep -v "^#"' 2>&1
"@

Write-ColorOutput "Ejecutando: grep configuración en /etc/ssh/sshd_config..." "Gray"
$configOutput = Invoke-Expression $sshCheckCmd

if ($configOutput) {
    Write-ColorOutput "Configuración actual:" "Yellow"
    $configOutput | ForEach-Object { Write-Host "  $_" }
}
else {
    Write-ColorOutput "[INFO] No se encontraron configuraciones explícitas (se usan valores por defecto)" "Yellow"
}

Write-Host ""
Write-ColorOutput "[PASO 2] Probando port forwarding..." "Cyan"

# Intentar crear un túnel temporal
$testCmd = "ssh -v -N -L 18080:localhost:8080 -i `"$SSH_KEY_PATH`" ${SSH_USER}@${SSH_HOST}"
Write-ColorOutput "Comando de prueba: $testCmd" "Gray"
Write-Host ""

$testProcess = Start-Process -FilePath "ssh" `
    -ArgumentList @("-v", "-N", "-L", "18080:localhost:8080", "-i", $SSH_KEY_PATH, "${SSH_USER}@${SSH_HOST}") `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardError "$env:TEMP\ssh_test_error.txt"

Start-Sleep -Seconds 3

# Leer errores
$errorOutput = Get-Content "$env:TEMP\ssh_test_error.txt" -ErrorAction SilentlyContinue

# Matar proceso de prueba
Stop-Process -Id $testProcess.Id -Force -ErrorAction SilentlyContinue
Remove-Item "$env:TEMP\ssh_test_error.txt" -Force -ErrorAction SilentlyContinue

if ($errorOutput -match "administratively prohibited") {
    Write-ColorOutput "[FAIL] Port forwarding RECHAZADO por el servidor" "Red"
    Write-Host ""
    Write-ColorOutput "═══════════════════════════════════════════════" "Yellow"
    Write-ColorOutput "  SOLUCIÓN REQUERIDA EN EL SERVIDOR" "Yellow"
    Write-ColorOutput "═══════════════════════════════════════════════" "Yellow"
    Write-Host ""
    
    Write-ColorOutput "Conecte al servidor y ejecute como root:" "Cyan"
    Write-Host ""
    
    Write-ColorOutput "# 1. Editar configuración SSH" "White"
    Write-Host "   sudo vi /etc/ssh/sshd_config" -ForegroundColor Gray
    Write-Host ""
    
    Write-ColorOutput "# 2. Agregar/modificar estas líneas:" "White"
    Write-Host "   AllowTcpForwarding yes" -ForegroundColor Green
    Write-Host "   PermitOpen any" -ForegroundColor Green
    Write-Host ""
    
    Write-ColorOutput "# 3. Reiniciar SSH" "White"
    Write-Host "   sudo systemctl restart sshd" -ForegroundColor Gray
    Write-Host ""
    
    Write-ColorOutput "# 4. Verificar que no hay errores" "White"
    Write-Host "   sudo systemctl status sshd" -ForegroundColor Gray
    Write-Host ""
    
    Write-ColorOutput "═══════════════════════════════════════════════" "Yellow"
    Write-Host ""
    
    # Ofrecer ejecutar comandos automáticamente
    $response = Read-Host "¿Desea intentar aplicar la corrección automáticamente? (S/N)"
    
    if ($response -eq "S" -or $response -eq "s") {
        Write-Host ""
        Write-ColorOutput "[INFO] Intentando aplicar corrección..." "Cyan"
        
        # Script de corrección remoto
        $fixScript = @'
#!/bin/bash
set -e

echo "[INFO] Respaldando configuración actual..."
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup.$(date +%Y%m%d_%H%M%S)

echo "[INFO] Modificando sshd_config..."
sudo sed -i 's/^#*AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config
if ! grep -q "^AllowTcpForwarding" /etc/ssh/sshd_config; then
    echo "AllowTcpForwarding yes" | sudo tee -a /etc/ssh/sshd_config > /dev/null
fi

sudo sed -i 's/^#*PermitOpen.*/PermitOpen any/' /etc/ssh/sshd_config
if ! grep -q "^PermitOpen" /etc/ssh/sshd_config; then
    echo "PermitOpen any" | sudo tee -a /etc/ssh/sshd_config > /dev/null
fi

echo "[INFO] Verificando sintaxis..."
sudo sshd -t

echo "[INFO] Reiniciando SSHD..."
sudo systemctl restart sshd

echo "[OK] Configuración aplicada correctamente"
'@
        
        $fixScript | ssh -i "$SSH_KEY_PATH" "${SSH_USER}@${SSH_HOST}" 'bash -s'
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-ColorOutput "[OK] Corrección aplicada exitosamente" "Green"
            Write-Host ""
            Write-ColorOutput "Ahora puede ejecutar: .\launch_web_ui.ps1" "Cyan"
        }
        else {
            Write-ColorOutput "[FAIL] Error al aplicar la corrección. Aplíquela manualmente." "Red"
        }
    }
    else {
        Write-ColorOutput "[INFO] Aplique la corrección manualmente y luego ejecute launch_web_ui.ps1" "Yellow"
    }
}
else {
    Write-ColorOutput "[OK] Port forwarding parece estar permitido" "Green"
    Write-Host ""
    Write-ColorOutput "Si sigue teniendo problemas, verifique:" "Cyan"
    Write-ColorOutput "  1. Firewall local en el servidor" "White"
    Write-ColorOutput "  2. SELinux / AppArmor" "White"
    Write-ColorOutput "  3. Logs del servidor: sudo journalctl -u sshd -f" "White"
}

Write-Host ""
