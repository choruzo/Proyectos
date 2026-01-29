# Guía de Instalación y Solución de Problemas - Windows

## 🪟 Instalación Paso a Paso en Windows

### Paso 1: Verificar Python

```powershell
# Abrir PowerShell y verificar versión de Python
python --version
# Debe mostrar: Python 3.9 o superior
```

### Paso 2: Crear Entorno Virtual

```powershell
# Navegar a tu carpeta del proyecto
cd D:\Archivos\Javier\Scritp_python\ollama_gpt

# Crear entorno virtual (si no existe)
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Si da error de permisos, ejecutar esto primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Paso 3: Actualizar pip

```powershell
# Con el entorno virtual activado
python -m pip install --upgrade pip
```

### Paso 4: Instalar Dependencias

**Opción A: Instalación Mínima (Recomendada)**

```powershell
pip install -r requirements.txt
```

**Opción B: Instalación Manual**

```powershell
pip install langchain==1.2.7
pip install langchain-core==1.2.7
pip install langchain-community==0.4.1
pip install langchain-text-splitters==1.1.0
pip install langchain-ollama==1.0.1
pip install langchain-chroma==1.1.0
pip install chromadb==1.4.1
pip install ollama==0.6.1
pip install pypdf==6.6.2
```

### Paso 5: Verificar Instalación

```powershell
python check_dependencies.py
```

Si ves: `✅ TODAS LAS DEPENDENCIAS ESTÁN CORRECTAMENTE INSTALADAS`, continúa al paso 6.

### Paso 6: Instalar y Configurar Ollama

1. **Descargar Ollama para Windows:**
   - Ir a: https://ollama.ai/download
   - Descargar e instalar la versión para Windows

2. **Verificar instalación:**
   ```powershell
   ollama --version
   ```

3. **Descargar modelos necesarios:**
   ```powershell
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

4. **Verificar que Ollama esté corriendo:**
   ```powershell
   ollama list
   # Debe mostrar:
   # llama3.1:8b
   # nomic-embed-text
   ```

### Paso 7: Preparar Estructura de Carpetas

```powershell
# Crear carpetas necesarias
New-Item -ItemType Directory -Force -Path docs
New-Item -ItemType Directory -Force -Path logs
New-Item -ItemType Directory -Force -Path db_esxi

# Copiar tus documentos PDF/MD/TXT a la carpeta docs
# Ejemplo:
Copy-Item "C:\mis_documentos\*.pdf" -Destination "docs\"
```

### Paso 8: Ejecutar el Sistema

```powershell
python RAG_improved.py
```

---

## 🔧 Solución de Problemas Comunes en Windows

### Error: `ModuleNotFoundError: No module named 'langchain.text_splitter'`

**Causa:** Versión incorrecta de LangChain o import antiguo

**Solución:**

```powershell
# 1. Desinstalar versiones antiguas
pip uninstall langchain langchain-community langchain-core -y

# 2. Reinstalar versiones correctas
pip install -r requirements.txt

# 3. Verificar
python check_dependencies.py
```

---

### Error: `cannot import name 'RecursiveCharacterTextSplitter'`

**Causa:** Falta el paquete `langchain-text-splitters`

**Solución:**

```powershell
pip install langchain-text-splitters==1.1.0
```

---

### Error: Permisos en PowerShell

**Síntoma:**
```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Solución:**

```powershell
# Ejecutar como Administrador o cambiar política
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Error: Ollama no responde

**Síntoma:**
```
Connection refused on localhost:11434
```

**Solución:**

1. **Verificar que Ollama está corriendo:**
   ```powershell
   # Ver procesos de Ollama
   Get-Process ollama*
   ```

2. **Si no está corriendo, iniciarlo:**
   - Buscar "Ollama" en el menú inicio y ejecutarlo
   - O reiniciar el servicio

3. **Verificar conexión:**
   ```powershell
   curl http://localhost:11434/api/tags
   ```

---

### Error: ChromaDB SQLite en Windows

**Síntoma:**
```
sqlite3.OperationalError: unable to open database file
```

**Solución:**

1. **Verificar permisos de carpeta:**
   ```powershell
   # Dar permisos completos a la carpeta db_esxi
   icacls db_esxi /grant ${env:USERNAME}:F /T
   ```

2. **Borrar y recrear base de datos:**
   ```powershell
   Remove-Item -Recurse -Force db_esxi
   python RAG_improved.py
   ```

---

### Error: Encoding en Windows

**Síntoma:**
```
UnicodeDecodeError: 'charmap' codec can't decode byte
```

**Solución:**

El código ya maneja esto con `encoding='utf-8'`, pero si persiste:

1. **Configurar variable de entorno:**
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   ```

2. **O agregar al inicio del script:**
   ```python
   import sys
   sys.stdout.reconfigure(encoding='utf-8')
   ```

---

### Error: Ruta de archivos con espacios

**Síntoma:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solución:**

```powershell
# Usar comillas en rutas con espacios
cd "D:\Archivos\Javier\Scritp python\ollama_gpt"

# O mejor, evitar espacios en nombres de carpetas
# Renombrar: "Scritp python" → "Scritp_python"
```

---

### Error: PyPDF no puede leer PDF

**Síntoma:**
```
pypdf.errors.PdfReadError: EOF marker not found
```

**Solución:**

1. **Verificar que el PDF no esté corrupto:**
   - Abrirlo en Adobe Reader
   - Si no abre, el archivo está dañado

2. **Alternativa - usar PDF2Image:**
   ```powershell
   pip install pdf2image
   # Luego OCR con tesseract
   ```

3. **Omitir PDFs problemáticos:**
   - Moverlos fuera de la carpeta `docs/`

---

### Memoria Insuficiente

**Síntoma:**
```
MemoryError: Unable to allocate array
```

**Solución:**

1. **Reducir tamaño de chunks en config.py:**
   ```python
   CHUNK_SIZE = 500  # Reducir de 1000 a 500
   TOP_K_CHUNKS = 3  # Reducir de 5 a 3
   ```

2. **Procesar menos documentos a la vez:**
   - Dividir carpeta `docs/` en subcarpetas
   - Indexar por partes

3. **Usar modelo más pequeño:**
   ```python
   MODEL_NAME = "llama3.1:8b"  # En vez de 70b
   ```

---

## 📊 Script de Diagnóstico Completo

Crear un archivo `diagnose.ps1`:

```powershell
# diagnose.ps1 - Script de diagnóstico completo

Write-Host "=== DIAGNÓSTICO DEL SISTEMA RAG ===" -ForegroundColor Cyan

Write-Host "`n1. Python Version:" -ForegroundColor Yellow
python --version

Write-Host "`n2. Entorno Virtual:" -ForegroundColor Yellow
if (Test-Path .venv) {
    Write-Host "  ✓ Entorno virtual existe" -ForegroundColor Green
} else {
    Write-Host "  ✗ Entorno virtual NO existe" -ForegroundColor Red
}

Write-Host "`n3. Ollama Status:" -ForegroundColor Yellow
try {
    ollama --version
    Write-Host "  ✓ Ollama instalado" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Ollama NO instalado" -ForegroundColor Red
}

Write-Host "`n4. Modelos Ollama:" -ForegroundColor Yellow
ollama list

Write-Host "`n5. Estructura de Carpetas:" -ForegroundColor Yellow
@("docs", "logs", "db_esxi") | ForEach-Object {
    if (Test-Path $_) {
        $count = (Get-ChildItem $_ -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "  ✓ $_ ($count archivos)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $_ NO existe" -ForegroundColor Red
    }
}

Write-Host "`n6. Dependencias Python:" -ForegroundColor Yellow
python check_dependencies.py

Write-Host "`n=== FIN DEL DIAGNÓSTICO ===" -ForegroundColor Cyan
```

**Ejecutar:**

```powershell
.\diagnose.ps1
```

---

## 🚀 Inicio Rápido (Resumen)

```powershell
# 1. Activar entorno
.venv\Scripts\Activate.ps1

# 2. Verificar dependencias
python check_dependencies.py

# 3. Verificar Ollama
ollama list

# 4. Ejecutar
python RAG_improved.py
```

---

## 📝 Checklist Pre-Ejecución

- [ ] Python 3.9+ instalado
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas (check_dependencies.py pasa)
- [ ] Ollama instalado y corriendo
- [ ] Modelos descargados (llama3.1:8b, nomic-embed-text)
- [ ] Carpeta `docs/` creada con documentos
- [ ] Permisos de escritura en carpeta del proyecto

---

## 🆘 Si Nada Funciona

1. **Borrar todo y empezar de cero:**

```powershell
# Desactivar entorno
deactivate

# Borrar entorno virtual
Remove-Item -Recurse -Force .venv

# Borrar base de datos
Remove-Item -Recurse -Force db_esxi

# Recrear entorno
python -m venv .venv
.venv\Scripts\Activate.ps1

# Reinstalar
pip install -r requirements.txt

# Verificar
python check_dependencies.py

# Ejecutar
python RAG_improved.py
```

2. **Contactar con logs:**
   - Enviar el contenido de `logs/rag_*.log`
   - Incluir output de `python check_dependencies.py`

---

## 💡 Mejores Prácticas en Windows

1. **Usar rutas sin espacios**
2. **Ejecutar PowerShell como Administrador cuando sea necesario**
3. **Mantener Ollama corriendo en segundo plano**
4. **Verificar antivirus no bloquee Ollama o Python**
5. **Usar UTF-8 para todos los archivos de texto**

---

**¿Todo listo? ¡Ejecuta tu RAG mejorado! 🚀**
