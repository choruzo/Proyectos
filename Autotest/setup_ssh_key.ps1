<#
.SYNOPSIS
    Configura autenticación SSH por clave pública para evitar solicitar contraseña.

.DESCRIPTION
    Este script:
    1. Verifica si existe una clave SSH (~/.ssh/id_rsa)
    2. Si no existe, genera un nuevo par de claves
    3. Copia la clave pública al servidor remoto
    4. Verifica que la autenticación sin contraseña funcione

.PARAMETER ForceNew
    Genera una nueva clave SSH incluso si ya existe una

.EXAMPLE
    .\setup_ssh_key.ps1
    Configura la autenticación SSH automáticamente

.EXAMPLE
    .\setup_ssh_key.ps1 -ForceNew
    Genera una nueva clave SSH y la configura
#>

param(
    [switch]$ForceNew
)

# Configuración
$SSH_USER = "agent"
$SSH_HOST = "172.30.188.137"
$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa"
$SSH_PUB_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa.pub"
$SSH_DIR = "$env:USERPROFILE\.ssh"

# Colores para output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Banner
Write-Host ""
Write-ColorOutput "===============================================" "Cyan"
Write-ColorOutput "  Configuración de SSH sin contraseña" "White"
Write-ColorOutput "===============================================" "Cyan"
Write-Host ""

# Verificar OpenSSH
Write-ColorOutput "[INFO] Verificando OpenSSH..." "Cyan"
try {
    $null = Get-Command ssh -ErrorAction Stop
    Write-ColorOutput "[OK] OpenSSH disponible" "Green"
}
catch {
    Write-ColorOutput "[FAIL] OpenSSH no está instalado" "Red"
    Write-ColorOutput "  Instalar desde: Configuración > Aplicaciones > Características opcionales > OpenSSH Client" "Yellow"
    exit 1
}

# Crear directorio .ssh si no existe
if (-not (Test-Path $SSH_DIR)) {
    Write-ColorOutput "[INFO] Creando directorio .ssh..." "Cyan"
    New-Item -ItemType Directory -Path $SSH_DIR -Force | Out-Null
    Write-ColorOutput "[OK] Directorio creado: $SSH_DIR" "Green"
}

# Verificar o generar clave SSH
if ((Test-Path $SSH_KEY_PATH) -and -not $ForceNew) {
    Write-ColorOutput "[INFO] Clave SSH existente encontrada: $SSH_KEY_PATH" "Cyan"
    $response = Read-Host "¿Desea usar la clave existente? (S/N)"
    if ($response -ne "S" -and $response -ne "s") {
        $ForceNew = $true
    }
}

if ($ForceNew -or -not (Test-Path $SSH_KEY_PATH)) {
    Write-Host ""
    Write-ColorOutput "[INFO] Generando nuevo par de claves SSH..." "Cyan"
    Write-ColorOutput "  Ubicación: $SSH_KEY_PATH" "Gray"
    Write-ColorOutput "  Tipo: RSA 4096 bits" "Gray"
    Write-Host ""
    
    # Generar clave sin passphrase para automatización
    $sshKeygenArgs = @(
        "-t", "rsa",
        "-b", "4096",
        "-f", $SSH_KEY_PATH,
        "-N", '""',  # Sin passphrase
        "-C", "CI/CD Web UI SSH Key - $env:USERNAME@$env:COMPUTERNAME"
    )
    
    & ssh-keygen @sshKeygenArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "[FAIL] Error al generar la clave SSH" "Red"
        exit 1
    }
    
    Write-ColorOutput "[OK] Par de claves generado exitosamente" "Green"
}

# Verificar que la clave pública existe
if (-not (Test-Path $SSH_PUB_KEY_PATH)) {
    Write-ColorOutput "[FAIL] Clave pública no encontrada: $SSH_PUB_KEY_PATH" "Red"
    exit 1
}

# Leer la clave pública
$publicKey = Get-Content $SSH_PUB_KEY_PATH -Raw

Write-Host ""
Write-ColorOutput "[INFO] Clave pública a instalar:" "Cyan"
Write-ColorOutput "  $($publicKey.Trim())" "Gray"
Write-Host ""

# Copiar clave al servidor
Write-ColorOutput "[INFO] Copiando clave pública al servidor remoto..." "Cyan"
Write-ColorOutput "  Servidor: ${SSH_USER}@${SSH_HOST}" "Gray"
Write-ColorOutput "  Se le solicitará la contraseña por ÚLTIMA vez" "Yellow"
Write-Host ""

# Comando para instalar la clave en el servidor
$installKeyCommand = @"
mkdir -p ~/.ssh && \
chmod 700 ~/.ssh && \
echo '$($publicKey.Trim())' >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
echo '[OK] Clave pública instalada correctamente'
"@

# Ejecutar el comando en el servidor remoto
$result = ssh "${SSH_USER}@${SSH_HOST}" $installKeyCommand 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "[FAIL] Error al copiar la clave al servidor" "Red"
    Write-ColorOutput "  Error: $result" "Red"
    Write-Host ""
    Write-ColorOutput "Intente manualmente:" "Yellow"
    Write-ColorOutput "  ssh-copy-id ${SSH_USER}@${SSH_HOST}" "Gray"
    Write-ColorOutput "  O copie manualmente $SSH_PUB_KEY_PATH al servidor" "Gray"
    exit 1
}

Write-ColorOutput "[OK] Clave copiada al servidor" "Green"

# Verificar autenticación sin contraseña
Write-Host ""
Write-ColorOutput "[INFO] Verificando autenticación sin contraseña..." "Cyan"

$testResult = ssh -o BatchMode=yes -o ConnectTimeout=5 "${SSH_USER}@${SSH_HOST}" "echo '[TEST_OK]'" 2>&1

if ($testResult -like "*[TEST_OK]*") {
    Write-Host ""
    Write-ColorOutput "===============================================" "Green"
    Write-ColorOutput "  [OK] Configuración completada exitosamente" "Green"
    Write-ColorOutput "===============================================" "Green"
    Write-Host ""
    Write-ColorOutput "Ya puede usar el túnel SSH sin contraseña:" "Cyan"
    Write-ColorOutput "  .\launch_web_ui.ps1" "White"
    Write-Host ""
    Write-ColorOutput "Información de la clave:" "Cyan"
    Write-ColorOutput "  Clave privada: $SSH_KEY_PATH" "White"
    Write-ColorOutput "  Clave pública: $SSH_PUB_KEY_PATH" "White"
    Write-ColorOutput "  Servidor: ${SSH_USER}@${SSH_HOST}" "White"
    Write-Host ""
}
else {
    Write-ColorOutput "[FAIL] La autenticación sin contraseña no funcionó" "Red"
    Write-ColorOutput "  Respuesta: $testResult" "Gray"
    Write-Host ""
    Write-ColorOutput "Posibles causas:" "Yellow"
    Write-ColorOutput "  - Permisos incorrectos en ~/.ssh o ~/.ssh/authorized_keys en el servidor" "Yellow"
    Write-ColorOutput "  - SELinux o configuración del servidor bloqueando la autenticación por clave" "Yellow"
    Write-ColorOutput "  - Configuración SSH del servidor (PermitRootLogin, PubkeyAuthentication)" "Yellow"
    Write-Host ""
    Write-ColorOutput "Verifique en el servidor:" "Cyan"
    Write-ColorOutput "  chmod 700 ~/.ssh" "Gray"
    Write-ColorOutput "  chmod 600 ~/.ssh/authorized_keys" "Gray"
    Write-ColorOutput "  Revisar /var/log/auth.log para errores" "Gray"
    exit 1
}
