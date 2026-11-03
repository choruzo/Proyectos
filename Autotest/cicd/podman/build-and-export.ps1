#!/usr/bin/env pwsh
# =============================================================================
# build-and-export.ps1 - Construye las imagenes en local y las exporta a .tar
# =============================================================================
# Uso (desde cualquier directorio):
#   pwsh ./cicd/podman/build-and-export.ps1
#   pwsh ./cicd/podman/build-and-export.ps1 -Engine podman
#   pwsh ./cicd/podman/build-and-export.ps1 -AgentUid 1000 -AgentGid 1000
#
# Genera en cicd/podman/dist/:
#   cicd-orchestrator.tar
#   cicd-web.tar
# Copia esos dos .tar a la maquina SUSE y ejecuta podman/load-and-run.sh
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet('podman', 'docker')]
    [string]$Engine,

    [string]$BaseImage = 'opensuse/leap:15.5',

    [int]$AgentUid = 1000,

    [int]$AgentGid = 1000,

    [string]$OutDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Engine) {
    if (Get-Command podman -ErrorAction SilentlyContinue) {
        $Engine = 'podman'
    } elseif (Get-Command docker -ErrorAction SilentlyContinue) {
        $Engine = 'docker'
    } else {
        throw 'No se encontro podman ni docker en PATH.'
    }
}

$Context = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutDir) {
    $OutDir = Join-Path $PSScriptRoot 'dist'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$OrchTag = 'localhost/cicd-orchestrator:local'
$WebTag = 'localhost/cicd-web:local'
$CommonArgs = @(
    '--platform', 'linux/amd64',
    '--build-arg', "BASE_IMAGE=$BaseImage",
    '--build-arg', "AGENT_UID=$AgentUid",
    '--build-arg', "AGENT_GID=$AgentGid"
)

function Invoke-Build {
    param([string]$Containerfile, [string]$Tag)

    Write-Host "==> Construyendo $Tag con $Engine" -ForegroundColor Cyan
    & $Engine build @CommonArgs -f (Join-Path $PSScriptRoot $Containerfile) -t $Tag $Context
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo el build de $Tag (exit $LASTEXITCODE)"
    }
}

function Save-Image {
    param([string]$Tag, [string]$File)

    Write-Host "==> Exportando $Tag -> $File" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $File) {
        Remove-Item -LiteralPath $File -Force
    }
    if ($Engine -eq 'podman') {
        & $Engine save --format docker-archive -o $File $Tag
    } else {
        & $Engine save -o $File $Tag
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo el save de $Tag (exit $LASTEXITCODE)"
    }
}

Invoke-Build -Containerfile 'Containerfile.orchestrator' -Tag $OrchTag
Invoke-Build -Containerfile 'Containerfile.web' -Tag $WebTag

Save-Image -Tag $OrchTag -File (Join-Path $OutDir 'cicd-orchestrator.tar')
Save-Image -Tag $WebTag -File (Join-Path $OutDir 'cicd-web.tar')

Write-Host ''
Write-Host "Listo. Copia a la maquina SUSE:" -ForegroundColor Green
Write-Host "  $OutDir/cicd-orchestrator.tar"
Write-Host "  $OutDir/cicd-web.tar"
Write-Host "  cicd/  (codigo, config y utils)"
