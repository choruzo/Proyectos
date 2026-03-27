# 🌊 Diagrama - Flujo Completo del Pipeline

## Visión General

Visualización end-to-end del pipeline CI/CD desde detección de tag hasta despliegue final.

**Relacionado con**:
- [[Arquitectura del Pipeline]] - Documentación detallada
- [[Diagrama - Estados]] - Estados del deployment
- [[Diagrama - Dependencias]] - Módulos y dependencias

---

## Flujo Secuencial Completo

```mermaid
sequenceDiagram
    participant Dev as Desarrollador
    participant Git as Git Remote
    participant Daemon as ci_cd.sh (daemon)
    participant GM as Git Monitor
    participant C as Compilación
    participant SQ as SonarQube
    participant VC as vCenter
    participant SSH as SSH Deploy
    participant VM as Target VM
    participant DB as SQLite DB
    
    Dev->>Git: git push + tag MAC_1_V24_02_15_01
    
    loop Every 5 minutes
        Daemon->>GM: detect_new_tags()
        GM->>Git: git ls-remote --tags
        Git-->>GM: List of tags
        GM->>DB: Check processed_tags
        DB-->>GM: Already processed
        GM->>GM: Find new: MAC_1_V24_02_15_01
        GM-->>Daemon: New tag found
        
        Daemon->>DB: INSERT deployment (status=pending)
        Daemon->>GM: checkout_tag()
        GM->>Git: git clone/checkout
        Git-->>GM: Files ready
        GM->>DB: INSERT processed_tags
        
        Daemon->>DB: UPDATE status=compiling
        Daemon->>C: compile.sh
        C->>C: Prepare compile dir
        C->>C: Execute build_DVDs.sh
        Note over C: ~45 minutes
        C-->>Daemon: InstallationDVD.iso (3.5GB)
        
        Daemon->>DB: UPDATE status=analyzing
        Daemon->>SQ: sonar_check.py
        SQ->>SQ: build-wrapper + sonar-scanner
        SQ->>SQ: Query SonarQube API
        Note over SQ: ~12 minutes
        SQ-->>Daemon: Quality gate PASSED
        SQ->>DB: INSERT sonar_results
        
        Daemon->>DB: UPDATE status=deploying
        Daemon->>VC: vcenter_api.py upload_iso
        VC->>VC: Authenticate vCenter
        VC->>VC: Upload to datastore
        VC->>VC: Configure CD-ROM
        VC->>VC: Power on VM
        Note over VC: ~5 minutes
        VC-->>Daemon: VM ready
        
        Daemon->>SSH: deploy.sh
        SSH->>VM: ssh YOUR_DEPLOY_USER@YOUR_TARGET_VM_IP
        SSH->>VM: mount /dev/cdrom
        SSH->>VM: cp -r /mnt/cdrom/* /root/install/
        SSH->>VM: ./install.sh "ope 1 - YES yes"
        Note over VM: ~2 minutes
        VM-->>SSH: Installation OK
        SSH-->>Daemon: Deployment success
        
        Daemon->>DB: UPDATE status=success
        Daemon->>Dev: Notification (wall)
    end
```

---

## Fases del Pipeline

### Phase 1: Git Monitor
- **Duración**: ~30 segundos
- **Script**: `git_monitor.sh`
- **Output**: Tag checked out en `/home/YOUR_USER/compile`

### Phase 2: Compilación
- **Duración**: ~45 minutos
- **Script**: `compile.sh`
- **Output**: `InstallationDVD.iso` (3-4 GB)

### Phase 3: SonarQube
- **Duración**: ~12 minutos
- **Script**: `sonar_check.py`
- **Output**: Quality metrics en DB

### Phase 4: vCenter
- **Duración**: ~5 minutos
- **Script**: `vcenter_api.py`
- **Output**: ISO en datastore, VM encendida

### Phase 5: SSH Deploy
- **Duración**: ~3 minutos
- **Script**: `deploy.sh`
- **Output**: Software instalado en VM

**Duración Total**: ~65 minutos end-to-end

---

## Enlaces Relacionados

- [[Arquitectura del Pipeline]]
- [[Diagrama - Estados]]
- [[01 - Quick Start]]
