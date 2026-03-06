# Script PowerShell para aplicar el fix rápidamente
# Ejecutar desde: c:\Users\jmartinb\Documents\Scrips_varios\Autotest

Write-Host "=== Aplicando Fix del Servicio CICD ===" -ForegroundColor Green
Write-Host ""

$server = "172.30.188.137"
$user = "agent"

# Copiar solo el archivo de servicio (el crítico)
Write-Host "1. Copiando cicd.service..." -ForegroundColor Yellow
scp cicd/cicd.service ${user}@${server}:/home/agent/cicd/

if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK - Archivo copiado" -ForegroundColor Green
} else {
    Write-Host "   ERROR - Fallo al copiar" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2. Actualizando servicio en el servidor..." -ForegroundColor Yellow
Write-Host "   (Esto ejecutará: sudo ./update_service.sh update)" -ForegroundColor Cyan
Write-Host ""

# Conectar y actualizar
ssh -t ${user}@${server} "cd /home/agent/cicd && sudo ./update_service.sh update"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Fix aplicado exitosamente ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Para verificar los logs:" -ForegroundColor Yellow
    Write-Host "  ssh $user@$server" -ForegroundColor Cyan
    Write-Host "  sudo journalctl -u cicd.service -f" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=== Hubo un problema al actualizar ===" -ForegroundColor Red
    Write-Host "Conéctate manualmente para revisar:" -ForegroundColor Yellow
    Write-Host "  ssh $user@$server" -ForegroundColor Cyan
    Write-Host "  sudo systemctl status cicd.service" -ForegroundColor Cyan
    Write-Host ""
}
