<#
.SYNOPSIS
    Establece túnel SSH y lanza la Web UI del pipeline CI/CD en Chrome.

.DESCRIPTION
    Este script:
    1. Establece un túnel SSH hacia el servidor CI/CD (172.30.188.137:8080)
    2. Verifica que el túnel esté activo
    3. Lanza Google Chrome con la interfaz web en http://localhost:8080
    4. Mantiene el túnel abierto hasta que se cierre Chrome o se pulse Ctrl+C

.PARAMETER KeepAlive
    Mantiene el túnel SSH activo incluso después de cerrar Chrome

.EXAMPLE
    .\launch_web_ui.ps1
    Lanza el túnel SSH y abre Chrome

.EXAMPLE
    .\launch_web_ui.ps1 -KeepAlive
    Mantiene el túnel activo después de cerrar Chrome
#>

param(
    [switch]$KeepAlive
)

# Configuración
$SSH_USER = "agent"
$SSH_HOST = "172.30.188.137"
$LOCAL_PORT = 8080
$REMOTE_PORT = 8080
$WEB_URL = "http://localhost:$LOCAL_PORT"
$SSH_COMMAND = "ssh -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${SSH_USER}@${SSH_HOST}"
$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa"

# Colores para output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Verificar si SSH está disponible
function Test-SSHAvailable {
    try {
        $null = Get-Command ssh -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Verificar si el puerto local está en uso
function Test-PortInUse {
    param([int]$Port)
    
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return ($null -ne $connections)
}

# Buscar proceso SSH existente
function Get-SSHTunnelProcess {
    param([int]$Port)
    
    $sshProcesses = Get-Process | Where-Object { $_.ProcessName -eq "ssh" }
    foreach ($proc in $sshProcesses) {
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
        if ($commandLine -like "*-L*${Port}:*") {
            return $proc
        }
    }
    return $null
}

# Limpiar túneles SSH existentes
function Stop-ExistingTunnel {
    param([int]$Port)
    
    $existingTunnel = Get-SSHTunnelProcess -Port $Port
    if ($existingTunnel) {
        Write-ColorOutput "[WARN] Túnel SSH existente detectado (PID: $($existingTunnel.Id))" "Yellow"
        $response = Read-Host "¿Desea cerrarlo y crear uno nuevo? (S/N)"
        if ($response -eq "S" -or $response -eq "s") {
            Stop-Process -Id $existingTunnel.Id -Force
            Start-Sleep -Seconds 2
            Write-ColorOutput "[OK] Túnel anterior cerrado" "Green"
            return $true
        }
        else {
            return $false
        }
    }
    return $true
}

# Buscar ruta de Chrome
function Get-ChromePath {
    $chromePaths = @(
        "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe"
    )
    
    foreach ($path in $chromePaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

# Verificar conectividad al servidor remoto
function Test-RemoteConnection {
    param([string]$ServerHost, [int]$Timeout = 5)
    
    Write-ColorOutput "[INFO] Verificando conectividad con $ServerHost..." "Cyan"
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $tcpClient.BeginConnect($ServerHost, 22, $null, $null)
        $wait = $asyncResult.AsyncWaitHandle.WaitOne($Timeout * 1000)
        
        if ($wait) {
            $tcpClient.EndConnect($asyncResult)
            $tcpClient.Close()
            Write-ColorOutput "[OK] Servidor alcanzable" "Green"
            return $true
        }
        else {
            $tcpClient.Close()
            Write-ColorOutput "[FAIL] Timeout al conectar" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "[FAIL] No se puede alcanzar el servidor: $($_.Exception.Message)" "Red"
        return $false
    }
}

# Verificar autenticación SSH sin contraseña
function Test-SSHPasswordlessAuth {
    param([string]$SSHUser, [string]$SSHHost, [string]$KeyPath)
    
    Write-ColorOutput "[INFO] Verificando autenticación SSH sin contraseña..." "Cyan"
    
    # Verificar si existe la clave privada
    if (-not (Test-Path $KeyPath)) {
        Write-ColorOutput "[WARN] Clave SSH privada no encontrada: $KeyPath" "Yellow"
        return $false
    }
    
    # Probar conexión sin contraseña (BatchMode evita solicitar contraseña)
    $testCmd = "ssh -o BatchMode=yes -o ConnectTimeout=5 -i `"$KeyPath`" ${SSHUser}@${SSHHost} `"echo TEST_OK`" 2>&1"
    $testResult = Invoke-Expression $testCmd
    
    if ($testResult -like "*TEST_OK*") {
        Write-ColorOutput "[OK] Autenticación sin contraseña configurada" "Green"
        return $true
    }
    else {
        Write-ColorOutput "[WARN] Autenticación sin contraseña NO configurada" "Yellow"
        return $false
    }
}

# Banner
Write-Host ""
Write-ColorOutput "═══════════════════════════════════════════════" "Cyan"
Write-ColorOutput "  CI/CD Pipeline Web UI Launcher" "White"
Write-ColorOutput "═══════════════════════════════════════════════" "Cyan"
Write-Host ""

# Verificaciones previas
Write-ColorOutput "[INFO] Verificando requisitos..." "Cyan"

if (-not (Test-SSHAvailable)) {
    Write-ColorOutput "[FAIL] ERROR: SSH no está disponible en el PATH" "Red"
    Write-ColorOutput "  Instale OpenSSH desde: Configuración > Aplicaciones > Características opcionales" "Yellow"
    exit 1
}
Write-ColorOutput "[OK] SSH disponible" "Green"

$chromePath = Get-ChromePath
if (-not $chromePath) {
    Write-ColorOutput "[WARN] Google Chrome no encontrado, se usará el navegador predeterminado" "Yellow"
}
else {
    Write-ColorOutput "[OK] Chrome encontrado: $chromePath" "Green"
}

# Verificar conectividad
if (-not (Test-RemoteConnection -ServerHost $SSH_HOST)) {
    Write-ColorOutput "[FAIL] No se puede conectar al servidor. Verifique:" "Red"
    Write-ColorOutput "  - Conectividad de red" "Yellow"
    Write-ColorOutput "  - VPN activa si es necesaria" "Yellow"
    Write-ColorOutput "  - Firewall del servidor" "Yellow"
    exit 1
}

# Verificar autenticación sin contraseña
Write-Host ""
if (-not (Test-SSHPasswordlessAuth -SSHUser $SSH_USER -SSHHost $SSH_HOST -KeyPath $SSH_KEY_PATH)) {
    Write-Host ""
    Write-ColorOutput "[WARN] Se solicitará contraseña al establecer el túnel SSH" "Yellow"
    Write-Host ""
    Write-ColorOutput "Para evitar esto, configure autenticación por clave SSH:" "Cyan"
    Write-ColorOutput "  .\setup_ssh_key.ps1" "White"
    Write-Host ""
    
    $response = Read-Host "¿Desea configurar autenticación sin contraseña ahora? (S/N)"
    if ($response -eq "S" -or $response -eq "s") {
        Write-Host ""
        Write-ColorOutput "[INFO] Ejecutando configuración SSH..." "Cyan"
        
        $setupScript = Join-Path $PSScriptRoot "setup_ssh_key.ps1"
        if (Test-Path $setupScript) {
            & $setupScript
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-ColorOutput "[OK] Configuración completada. Reiniciando script..." "Green"
                Write-Host ""
                Start-Sleep -Seconds 2
                & $PSCommandPath @PSBoundParameters
                exit 0
            }
            else {
                Write-ColorOutput "[FAIL] Error en la configuración. Continuando con contraseña..." "Red"
                Start-Sleep -Seconds 2
            }
        }
        else {
            Write-ColorOutput "[FAIL] setup_ssh_key.ps1 no encontrado en: $setupScript" "Red"
            Start-Sleep -Seconds 2
        }
    }
    Write-Host ""
}

# Verificar y limpiar túnel existente
if (Test-PortInUse -Port $LOCAL_PORT) {
    $canContinue = Stop-ExistingTunnel -Port $LOCAL_PORT
    if (-not $canContinue) {
        Write-ColorOutput "[FAIL] Operación cancelada por el usuario" "Red"
        exit 0
    }
}

# Iniciar túnel SSH
Write-Host ""
Write-ColorOutput "[INFO] Iniciando túnel SSH..." "Cyan"

# Preparar argumentos SSH
$sshArgs = @("-L", "${LOCAL_PORT}:localhost:${REMOTE_PORT}")

# Si existe la clave privada, usarla
if (Test-Path $SSH_KEY_PATH) {
    $sshArgs += @("-i", $SSH_KEY_PATH)
    Write-ColorOutput "   Usando clave: $SSH_KEY_PATH" "Gray"
}

$sshArgs += "${SSH_USER}@${SSH_HOST}"

Write-ColorOutput "   Comando: ssh $($sshArgs -join ' ')" "Gray"

$sshProcess = Start-Process -FilePath "ssh" `
    -ArgumentList $sshArgs `
    -PassThru `
    -WindowStyle Minimized

if (-not $sshProcess) {
    Write-ColorOutput "[FAIL] Error al iniciar el proceso SSH" "Red"
    exit 1
}

Write-ColorOutput "[OK] Túnel SSH iniciado (PID: $($sshProcess.Id))" "Green"

# Esperar a que el túnel esté activo
Write-ColorOutput "[INFO] Esperando establecimiento del túnel..." "Cyan"
$maxRetries = 10
$retryCount = 0
$tunnelReady = $false

while ($retryCount -lt $maxRetries) {
    Start-Sleep -Seconds 1
    if (Test-PortInUse -Port $LOCAL_PORT) {
        $tunnelReady = $true
        break
    }
    $retryCount++
    Write-Host "." -NoNewline
}
Write-Host ""

if (-not $tunnelReady) {
    Write-ColorOutput "[FAIL] El túnel no se estableció en el tiempo esperado" "Red"
    Write-ColorOutput "  Verifique las credenciales SSH y la conexión" "Yellow"
    Stop-Process -Id $sshProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-ColorOutput "[OK] Túnel SSH establecido correctamente" "Green"

# Lanzar Chrome
Write-Host ""
Write-ColorOutput "[INFO] Abriendo Web UI en el navegador..." "Cyan"
Write-ColorOutput "   URL: $WEB_URL" "Gray"

if ($chromePath) {
    Start-Process -FilePath $chromePath -ArgumentList "--new-window", $WEB_URL
}
else {
    Start-Process $WEB_URL
}

Write-Host ""
Write-ColorOutput "═══════════════════════════════════════════════" "Green"
Write-ColorOutput "  [OK] Web UI lanzada correctamente" "Green"
Write-ColorOutput "═══════════════════════════════════════════════" "Green"
Write-Host ""
Write-ColorOutput "Información del túnel:" "Cyan"
Write-ColorOutput "  • Proceso SSH PID: $($sshProcess.Id)" "White"
Write-ColorOutput "  • Puerto local: $LOCAL_PORT" "White"
Write-ColorOutput "  • Servidor remoto: ${SSH_USER}@${SSH_HOST}:${REMOTE_PORT}" "White"
Write-Host ""

if ($KeepAlive) {
    Write-ColorOutput "[INFO] Modo KeepAlive activado" "Yellow"
    Write-ColorOutput "   Presione Ctrl+C para cerrar el túnel" "Yellow"
    Write-Host ""
    
    try {
        # Mantener el script corriendo
        while ($true) {
            if ($sshProcess.HasExited) {
                Write-ColorOutput "[WARN] El proceso SSH se ha cerrado inesperadamente" "Red"
                break
            }
            Start-Sleep -Seconds 5
        }
    }
    finally {
        Write-Host ""
        Write-ColorOutput "[INFO] Cerrando túnel SSH..." "Yellow"
        Stop-Process -Id $sshProcess.Id -Force -ErrorAction SilentlyContinue
        Write-ColorOutput "[OK] Túnel cerrado" "Green"
    }
}
else {
    Write-ColorOutput "[INFO] El túnel permanecerá activo en segundo plano" "Cyan"
    Write-ColorOutput "  Para cerrarlo, ejecute:" "Gray"
    Write-ColorOutput "    Stop-Process -Id $($sshProcess.Id)" "Gray"
    Write-Host ""
    Write-ColorOutput "  O use el parámetro -KeepAlive para gestión interactiva" "Gray"
    Write-Host ""
}

Write-ColorOutput "[OK] Script completado exitosamente" "Green"
Write-Host ""
