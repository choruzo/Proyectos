[⬅️ Volver al índice principal](../README.md)

# 🤖 Jarvis Voice Assistant - Asistente de Voz en Español

> **Sistema de asistente de voz completamente en español, optimizado para Raspberry Pi 4 e Intel N95, con reconocimiento offline y control inteligente del hogar.**

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Raspberry Pi 4](https://img.shields.io/badge/Raspberry%20Pi%204-Compatible-red.svg)](https://www.raspberrypi.org/)
[![Linux](https://img.shields.io/badge/Linux-Compatible-orange.svg)](https://www.linux.org/)

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación Rápida](#instalación-rápida)
- [Instalación Detallada](#instalación-detallada)
- [Configuración](#configuración)
- [Uso y Comandos](#uso-y-comandos)
- [Funciones del Asistente](#funciones-del-asistente)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Integración con Servicios Externos](#integración-con-servicios-externos)
- [Solución de Problemas](#solución-de-problemas)
- [Optimizaciones](#optimizaciones)
- [Desarrollo](#desarrollo)

---

## ✨ Características

### 🎙️ **Reconocimiento de Voz**
- ✅ Detección offline de palabra clave (Wake Word) "Jarvis"
- ✅ **Motor STT dual**: faster-whisper (OpenAI Whisper optimizado) o Vosk
- ✅ **faster-whisper**: Precisión superior (~95% WER) en español con modelos small/base
- ✅ **Vosk**: Menor latencia y uso de recursos para hardware limitado
- ✅ Procesamiento completamente local (sin enviar datos a la nube)
- ✅ Soporte para múltiples acentos y dialectos españoles
- ✅ Cancelación de ruido de fondo adaptativo
- ✅ Detección automática de hardware y selección de motor óptimo

### 🤖 **Procesamiento Inteligente de Comandos**
- ✅ **Rasa NLU con spaCy embeddings** (es_core_news_lg) para máxima precisión
- ✅ Reconocimiento semántico de sinónimos (+30% precisión)
- ✅ Robustez ante errores de transcripción (+25% mejora)
- ✅ Extracción de intención usando IA local (Ollama + Qwen3)
- ✅ Sistema de contexto conversacional
- ✅ Caché de respuestas para mejor rendimiento
- ✅ Patrones regex compilados para velocidad
- ✅ Fallback automático a Qwen cuando confianza < 0.65
- ✅ Manejo robusto de variantes de comandos

### 🔊 **Síntesis de Voz (TTS)**
- ✅ Generación de voz natural offline con Piper
- ✅ Modelos en español de alta calidad
- ✅ Velocidad de elocución adaptable
- ✅ Control de entonación y énfasis

### 🎵 **Reproductor de Música**
- ✅ Reproducción de múltiples formatos (MP3, WAV, FLAC)
- ✅ Gestión inteligente de listas de reproducción
- ✅ Control de volumen y ecualizador
- ✅ Búsqueda por género, artista o carpeta
- ✅ Cola de espera y mezcla aleatoria

### 📅 **Sistema de Recordatorios y Tareas**
- ✅ Crear recordatorios por hora y fecha
- ✅ Programación de tareas recurrentes (diarias, semanales)
- ✅ Notificaciones por voz y Telegram
- ✅ Persistencia en JSON

### 🏠 **Control del Hogar Inteligente**
- ✅ Integración con Home Assistant
- ✅ Control de luces, enchufes, termostatos
- ✅ Automatización de escenas
- ✅ Estado en tiempo real de dispositivos

### 🌤️ **Información Meteorológica**
- ✅ API oficial AEMET (España)
- ✅ Predicción del tiempo a corto y largo plazo
- ✅ Predicción de precipitaciones hora a hora
- ✅ Alertas meteorológicas
- ✅ Fallback a OpenWeather como respaldo

### 📱 **Integración Telegram**
- ✅ Notificaciones de recordatorios
- ✅ Alertas del sistema
- ✅ Mensajes de estado
- ✅ Logging remoto

### ⚡ **Optimizaciones**
- ✅ Gestión de memoria para dispositivos con recursos limitados
- ✅ Caché inteligente con TTL
- ✅ Monitoreo de rendimiento en tiempo real
- ✅ Limpieza automática de recursos
- ✅ Perfilado de CPU y memoria

---

## 📦 Contenido del Paquete

✅ **Incluido:**
- Código fuente completo de Jarvis (8000+ líneas)
- Modelos de detección de palabra clave (Porcupine)
- Modelos TTS en español (Piper)
- Modelos de reconocimiento de voz (Vosk y faster-whisper)
- Scripts de instalación automática
- Configuración pre-optimizada para RPi4, Intel N95 y AMD Ryzen
- Requirements portables y actualizados
- Documentación completa
- Ejemplos y tests

⬇️ **Se descarga automáticamente:**
- Modelos Vosk de reconocimiento de voz (~40MB / 200MB según hardware)
- Modelos Whisper (75MB-3GB según tamaño: tiny/base/small/medium/large)
- Dependencias Python desde PyPI
- Librerías del sistema necesarias

---

## 🎯 Compatibilidad

| Plataforma | Estado | Notas |
|-----------|--------|-------|
| **Raspberry Pi 4** (ARM64) | ✅ Totalmente soportado | Optimizado para 4GB RAM |
| **Intel N95** (x86_64) | ✅ Totalmente soportado | Configuración específica en `n95_config.py` |
| **Ubuntu 20.04+** (x86_64) | ✅ Soportado | Requiere librerías de audio |
| **Debian 11+** (x86_64) | ✅ Soportado | Compatible con bullseye |
| **Linux Mint** | ✅ Soportado | Basado en Ubuntu |
| **Windows** | ⚠️ Limitado | Usar WSL2 o compilar modelos |
| **macOS** | ⚠️ Experimental | No optimizado |

---

## 🚀 Instalación Rápida

```bash
# 1. Extraer paquete
tar -xzf jarvis_complete_*.tar.gz
cd jarvis_complete_*/

# 2. Ejecutar instalación automática
chmod +x install_complete.sh
./install_complete.sh

# 3. Configurar audio (ver sección siguiente)
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# 4. Editar configuración con IDs correctos
nano jarvis_modules/config.py

# 5. ¡Ejecutar Jarvis!
./run_jarvis.sh
```

---

## 📥 Instalación Detallada

### Paso 1: Requisitos Previos

#### Librerías del Sistema (Debian/Ubuntu)

```bash
# Actualizar repositorios
sudo apt-get update

# Herramientas de desarrollo
sudo apt-get install -y build-essential python3-dev python3-pip

# Audio y micrófono
sudo apt-get install -y alsa-utils pulseaudio portaudio19-dev
sudo apt-get install -y libopenblas-dev liblapack-dev libblas-dev

# Otras dependencias
sudo apt-get install -y git curl wget unzip
sudo apt-get install -y libatlas-base-dev libjasper-dev libtiff5
sudo apt-get install -y libjasper-dev libtiff5 libjasper1

# Para RPi4 específicamente
sudo apt-get install -y libhdf5-dev libharfbuzz0b libwebp6
sudo apt-get install -y libtiff5 libjasper1 libatlas-base-dev

# Permisos de audio (IMPORTANTE)
sudo usermod -a -G audio $USER
newgrp audio
# O reiniciar sesión
```

#### Verificar Dispositivos de Audio

```bash
# Listar dispositivos de audio
arecord -l      # Entrada (micrófono)
aplay -l        # Salida (altavoces)

# Alternativa con Python
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Paso 2: Clonar/Extraer Repositorio

```bash
# Opción A: Desde archivo comprimido
tar -xzf jarvis_complete_*.tar.gz
cd jarvis_complete_*/

# Opción B: Desde git
git clone https://github.com/tu_usuario/jarvis.git
cd jarvis
```

### Paso 3: Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel
```

### Paso 4: Instalar Dependencias

```bash
# Opción A: Instalación rápida (requiere venv descargado)
pip install -r requirements_portable.txt

# Opción B: Instalación completa desde PyPI
pip install -r requirements.txt

# Opción C: Instalación manual completa
pip install \
  pvporcupine \
  vosk \
  faster-whisper \
  ctranslate2 \
  sounddevice \
  PyAudio \
  pydub \
  requests \
  psutil \
  python-telegram-bot \
  ollama \
  numpy \
  scipy
```

### Paso 5: Descargar Modelos STT

#### Opción A: Modelos Vosk (Ligeros, rápidos)

```bash
# Crear directorio
mkdir -p venv/models

# Descargar modelo español pequeño (40MB)
cd venv/models
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip
rm vosk-model-small-es-0.42.zip

# Opcional: Modelo grande para mejor precisión (200MB)
# wget https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
# unzip vosk-model-es-0.42.zip
# rm vosk-model-es-0.42.zip

cd ../../
```

#### Opción B: faster-whisper (Mayor precisión)

```bash
# Los modelos Whisper se descargan automáticamente en la primera ejecución
# Se almacenan en: ~/.cache/huggingface/hub/

# Para forzar descarga anticipada:
python3 << EOF
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Modelo Whisper descargado correctamente")
EOF

# Tamaños de modelos disponibles:
# tiny   - 75 MB   (más rápido, menor precisión)
# base   - 142 MB  (balance para hardware limitado)
# small  - 461 MB  (recomendado para Ryzen/N95)
# medium - 1.45 GB (alta precisión, requiere >8GB RAM)
# large  - 2.87 GB (máxima precisión, requiere >16GB RAM)
```

#### Comparación de Modelos

| Modelo | Tamaño | RAM | Latencia | Precisión | Hardware Recomendado |
|--------|--------|-----|----------|-----------|----------------------|
| Vosk small | 40 MB | 450 MB | ~1.8s | 88% | RPi4, hardware limitado |
| Vosk large | 200 MB | 650 MB | ~2.0s | 90% | Intel N95, Ryzen |
| Whisper tiny | 75 MB | 250 MB | ~2.5s | 85% | RPi4 |
| Whisper base | 142 MB | 450 MB | ~3.0s | 92% | Intel N95 |
| Whisper small | 461 MB | 920 MB | ~3.2s | 95% | Ryzen, Intel i5+ |
| Whisper medium | 1.45 GB | 2.9 GB | ~6.0s | 97% | Workstations |

**Recomendación**:
- **Raspberry Pi 4**: Vosk small o Whisper tiny
- **Intel N95**: Whisper base o Vosk large
- **AMD Ryzen**: Whisper small (configuración por defecto)
- **Workstations**: Whisper medium/large

### Paso 6: Configuración de Audio

```bash
# Verificar dispositivos
python3 -c "import sounddevice as sd; [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(sd.query_devices())]"

# Editar configuración
nano jarvis_modules/config.py

# Buscar estas líneas y actualizar con tus IDs:
# MIC_DEVICE_ID = 1          # Tu micrófono
# SPEAKER_INDEX = 0          # Tus altavoces
```

### Paso 7: Instalar Ollama (Opcional pero Recomendado)

```bash
# Para Intel/x86_64
curl -fsSL https://ollama.ai/install.sh | sh

# Para Raspberry Pi
curl -fsSL https://ollama.ai/install.sh | sh

# Iniciar servicio
ollama serve &

# En otra terminal: descargar modelo
ollama pull qwen3:0.6b

# Verificar que funciona
curl http://localhost:11434/api/generate -d '{"model":"qwen3:0.6b","prompt":"Hola"}'
```

---

## ⚙️ Configuración

### Configuración Principal (`jarvis_modules/config.py`)

```python
# === DISPOSITIVOS DE AUDIO ===
MIC_DEVICE_ID = 1              # ID del micrófono USB
SPEAKER_INDEX = 0              # ID de los altavoces

# === RUTAS DEL PROYECTO ===
PROJECT_ROOT_DIR = "/ruta/a/jarvis"
MUSIC_DIR = "/ruta/a/jarvis/music"
DATA_DIR = "/ruta/a/jarvis/data"
MODELS_DIR = "/ruta/a/jarvis/venv/models"

# === CONFIGURACIÓN DE AUDIO ===
SAMPLE_RATE_CAPTURE = 16000    # Frecuencia de muestreo (Hz)
AUDIO_BUFFER_SIZE = 1024       # Tamaño del buffer
COMMAND_LISTEN_DURATION = 10   # Segundos de escucha

# === MODELOS DE IA ===
PICOVOICE_ACCESS_KEY = "tu_clave_aqui"
WAKE_WORD_PATH = "venv/custom_models/jarvis_es_raspberry-pi_v3_0_0.ppn"
VOSK_MODEL_PATH = "venv/models/vosk-model-small-es-0.42"

# === CONFIGURACIÓN DE TTS (PIPER) ===
PIPER_EXECUTABLE_PATH = "venv/piper/piper"
PIPER_MODEL_PATH = "venv/piper/models/es_ES-sharvard-medium.onnx"
TTS_SPEED_SCALE = 0.8          # Velocidad (0.5-2.0)

# === API KEYS Y TOKENS ===
AEMET_API_KEY = "tu_clave_aemet"
HOME_ASSISTANT_URL = "http://192.168.1.100:8123"
HOME_ASSISTANT_TOKEN = "tu_token_ha"
TELEGRAM_BOT_TOKEN = "tu_bot_token"
TELEGRAM_CHAT_ID = "tu_chat_id"

# === OPTIMIZACIONES ===
ENABLE_PROFILING = False        # Activar perfilado de rendimiento
ENABLE_MEMORY_MONITORING = True # Monitorear uso de memoria
MAX_WORKER_THREADS = 4          # Máximo de hilos de trabajo
MEMORY_LIMIT_BYTES = 2147483648 # 2GB
```

### Detectar IDs de Dispositivos

```bash
# Script Python completo
python3 << 'EOF'
import sounddevice as sd
import json

devices = sd.query_devices()
audio_info = {
    "input_devices": [],
    "output_devices": [],
    "default_input": sd.default.device[0],
    "default_output": sd.default.device[1]
}

for i, device in enumerate(devices):
    device_info = {
        "index": i,
        "name": device['name'],
        "channels": device['max_input_channels'],
    }
    
    if device['max_input_channels'] > 0:
        audio_info["input_devices"].append(device_info)
    if device['max_output_channels'] > 0:
        audio_info["output_devices"].append(device_info)

print(json.dumps(audio_info, indent=2))
EOF
```

---

## 🎤 Uso y Comandos

### Inicio Básico

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar Jarvis
python3 jarvis_main.py

# O usando script automático
./run_jarvis.sh
```

### 🤖 Uso Avanzado con Rasa NLU (Opcional)

Jarvis puede utilizar Rasa NLU como motor principal de comprensión de lenguaje natural para comandos más rápidos y precisos. Esto es completamente opcional - Jarvis funciona perfectamente sin Rasa.

**Ventajas de usar Rasa:**
- ⚡ Respuesta 10-20x más rápida para comandos comunes (50-150ms vs 2-6s)
- 🎯 Mayor precisión para intents entrenados
- 🔄 Fallback automático a Qwen para comandos complejos

**Inicio rápido con Rasa:**

```bash
# 1. Instalar Rasa (en otro terminal)
pip install rasa

# 2. Entrenar el modelo
cd rasa_config
rasa train

# 3. Iniciar servidor Rasa
rasa run --enable-api --cors "*"

# 4. Habilitar Rasa en Jarvis (en otro terminal)
export RASA_ENABLED=true
python venv/wake_word_vosk_small.py
```

**Modo sin Rasa (predeterminado):**

```bash
# Jarvis funciona sin Rasa usando Qwen directamente
python venv/wake_word_vosk_small.py
```

### Activación por Voz

Una vez iniciado, Jarvis escucha el wake word **"Jarvis"** seguido del comando.

```
Usuario: "Jarvis..."
Jarvis: [beep sonoro]
Usuario: "reproduce música clásica"
Jarvis: "Reproduciendo música clásica"
```

> 📚 **Para más información sobre Rasa NLU**: Consulta la [documentación completa de Rasa](rasa_config/README.md) que incluye configuración detallada, ejemplos y solución de problemas.

---

## 🎯 Funciones del Asistente

### 🎵 Música

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| Reproduce [género] | Inicia reproducción | "reproduce música rock" |
| Pon | Comienza la música | "pon música" |
| Para | Detiene la música | "para la música" |
| Pausa | Pausa la reproducción | "pausa" |
| Continúa | Reanuda | "continúa" |
| Siguiente | Próxima canción | "siguiente" |
| Anterior | Canción anterior | "anterior" |
| Sube volumen | Aumenta volumen | "sube el volumen" |
| Baja volumen | Disminuye volumen | "baja el volumen" |
| Volumen [0-100] | Volumen específico | "volumen 50" |

**Géneros disponibles:**
- `clásica` / `clásicos`
- `rock`
- `española` / `flamenco`
- `verano` / `playa`
- `pop`
- `jazz`

### 📅 Recordatorios

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| Recuérdame... a las [hora] | Recordatorio por hora | "recuérdame tomar medicina a las 8" |
| Recuérdame... el [fecha] | Recordatorio por fecha | "recuérdame llamar a mamá el 25 de diciembre" |
| Recuérdame en [duración] | Recordatorio relativo | "recuérdame en 30 minutos" |
| Mis recordatorios | Lista recordatorios | "mis recordatorios" |
| Borra recordatorios | Elimina recordatorios | "borra mis recordatorios" |

### 📆 Tareas Programadas

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| Programa [tarea] cada [tiempo] | Tarea recurrente | "programa riego del jardín cada día a las 6" |
| Tareas programadas | Lista tareas | "mis tareas programadas" |
| Cancela tarea [nombre] | Elimina tarea | "cancela riego" |

### 🌤️ Meteorología

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| ¿Qué tiempo hace? | Estado actual | "¿qué tiempo hace?" |
| Predicción para [ciudad] | Clima de ciudad | "predicción para Barcelona" |
| ¿Va a llover? | Probabilidad lluvia | "¿va a llover?" |
| Temperatura en [ciudad] | Solo temperatura | "temperatura en Madrid" |
| ¿Hay alerta meteorológica? | Avisos activos | "¿hay alerta meteorológica?" |
| Predicción hora a hora | Hora próximas horas | "predicción hora a hora" |
| Máxima y mínima | Temperaturas extremas | "máxima y mínima" |

### 🕐 Hora y Fecha

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| ¿Qué hora es? | Hora actual | "¿qué hora es?" |
| ¿Qué día es hoy? | Fecha actual | "¿qué día es hoy?" |
| Día de la semana | Día actual | "¿qué día de la semana es?" |
| ¿Cuántos días faltan para [fecha]? | Cuenta atrás | "¿cuántos días faltan para Navidad?" |

### 🏠 Casa Inteligente

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| Enciende [dispositivo] | Enciende | "enciende la luz del salón" |
| Apaga [dispositivo] | Apaga | "apaga el ventilador" |
| Abre [dispositivo] | Abre (cortinas, puertas) | "abre las cortinas" |
| Cierra [dispositivo] | Cierra | "cierra las puertas" |
| Estado [dispositivo] | Consulta estado | "estado del termostato" |
| Activa escena [nombre] | Escena automática | "activa escena película" |

**Dispositivos soportados:**
- Luces (luz_salón, luz_dormitorio, luz_cocina, etc.)
- Enchufes (enchufe_tv, enchufe_ventilador)
- Termostatos (termostato_casa, termostato_habitación)
- Puertas (puerta_entrada, puerta_garaje)
- Cortinas (cortinas_salón, cortinas_dormitorio)

### 💬 General

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| Hola / Jarvis | Saludo | "Hola Jarvis" |
| Hasta luego / Adiós | Cierra | "hasta luego" |
| Ayuda | Muestra funciones | "ayuda" |
| Versión | Información | "versión" |
| Estado | Sistema operativo | "estado del sistema" |

---

## 📁 Estructura del Proyecto

```
jarvis_complete_20251031_171259/
│
├── 🚀 EJECUTABLES
│   ├── jarvis_main.py                    # Punto de entrada principal
│   ├── run_jarvis.sh                     # Script de lanzamiento
│   └── install_complete.sh               # Instalador automático
│
├── 📋 CONFIGURACIÓN
│   ├── README.md                         # Esta documentación
│   ├── requirements.txt                  # Dependencias estándar
│   ├── requirements_portable.txt         # Dependencias portables
│   ├── package_info.json                 # Información del paquete
│   └── audio_config_jarvis.py            # Configuración de audio legacy
│
├── 📦 MÓDULOS PRINCIPALES (jarvis_modules/)
│   ├── config.py                         # Configuración centralizada
│   ├── command_processor_optimized.py    # Procesador de comandos (1589 líneas)
│   ├── llm_ollama.py                     # Integración LLM Ollama
│   ├── resource_manager.py               # Gestor de caché y memoria
│   ├── performance_monitor.py            # Monitor de rendimiento
│   ├── audio_processor.py                # Procesamiento de audio
│   │
│   ├── 🎵 MÚSICA
│   │   └── music_player_optimized.py     # Reproductor de música
│   │
│   ├── 📅 RECORDATORIOS Y TAREAS
│   │   ├── generic_reminders.py          # Sistema de recordatorios
│   │   └── scheduled_tasks.py            # Tareas programadas
│   │
│   ├── 🏠 SMART HOME
│   │   └── home_assistant.py             # Integración Home Assistant
│   │
│   ├── 🌤️ SERVICIOS EXTERNOS
│   │   └── aemet_client.py               # API meteorológica AEMET
│   │
│   ├── 📱 NOTIFICACIONES
│   │   └── telegram_bot.py               # Bot de Telegram
│   │
│   ├── 🖥️ OPTIMIZACIONES
│   │   ├── n95_config.py                 # Config para Intel N95
│   │   └── __init__.py
│   │
│   └── 📊 CACHÉ
│       └── reminders.json                # Recordatorios persistentes
│
├── 🎵 MÚSICA (music/)
│   ├── clasicos/                         # Música clásica
│   ├── española/                         # Música española
│   ├── rock/                             # Rock
│   ├── verano/                           # Música de verano
│   └── convert_mp3_to_wav.py             # Conversor de formatos
│
├── 🤖 MODELOS (models/ y venv/)
│   ├── custom_models/
│   │   ├── jarvis_es_raspberry-pi_v3_0_0.ppn    # Wake word RPi4
│   │   ├── jarvis_es_linux_v3_0_0.ppn           # Wake word Linux x86_64
│   │   └── porcupine_params_es.pv               # Parámetros español
│   │
│   ├── piper/                            # TTS Piper
│   │   ├── piper (binario)
│   │   ├── libpiper_phonemize.so.1
│   │   └── models/
│   │       └── es_ES-sharvard-medium.onnx        # Modelo voz español
│   │
│   └── vosk-model-small-es-0.42/         # Modelo reconocimiento voz
│       ├── mfcc.model
│       ├── model.fst
│       └── conf/...
│
├── 💾 DATOS (data/)
│   ├── reminders.json                    # Recordatorios guardados
│   └── scheduled_tasks.json              # Tareas programadas
│
├── 📖 DOCUMENTACIÓN (docs/)
│   ├── home_assistant/                   # Docs integración HA
│   ├── hora_fecha/                       # Docs funciones hora/fecha
│   ├── Meteorologia/                     # Docs capacidades meteorológicas
│   ├── recordatorios/                    # Docs sistema recordatorios
│   ├── tareas_programadas/               # Docs tareas programadas
│   └── telegram/                         # Docs integración Telegram
│
├── 🧪 PRUEBAS (test/ y tests/)
│   ├── test/home_assistant/              # Tests Home Assistant
│   ├── test/hora_fecha/                  # Tests hora/fecha
│   ├── test/Meteorologia/                # Tests meteorología
│   ├── test/recordatorios/               # Tests recordatorios
│   ├── test/tareas_programadas/          # Tests tareas programadas
│   ├── test/telegram/                    # Tests Telegram
│   └── tests/test_microphone_capture.py  # Tests micrófono
│
├── 📝 OTROS
│   ├── otros/                            # Documentos de referencia
│   ├── otros\ test/                      # Scripts de prueba legacy
│   └── logs/                             # Registros de ejecución
│
└── 📊 METADATOS
    ├── RESUMEN_FINAL.txt                 # Resumen de cambios
    ├── RESUMEN_VISUAL.txt                # Estructura visual
    └── .github/
        └── copilot-instructions.md       # Instrucciones para IA
```

---

## 🔗 Integración con Servicios Externos

### 🌤️ AEMET (Meteorología España)

```python
# En jarvis_modules/config.py
AEMET_API_KEY = "tu_clave_aemet_aqui"

# Registrarse en https://www.aemet.es/es/datos_abiertos/
```

### 🏠 Home Assistant

```python
# En jarvis_modules/config.py
HOME_ASSISTANT_URL = "http://192.168.1.100:8123"
HOME_ASSISTANT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Generar token en Home Assistant: Perfil → Tokens
```

### 📱 Telegram Bot

```python
# En jarvis_modules/config.py
TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID = "987654321"

# Crear bot: @BotFather en Telegram
# Obtener chat ID: @userinfobot
```

### 🤖 Ollama (LLM Local)

```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar modelo pequeño
ollama pull qwen3:0.6b

# Iniciar servicio
ollama serve

# Verificar en http://localhost:11434
```

---

## 🔧 Solución de Problemas

### ❌ Error "Invalid device" / "No audio input"

**Causa:** IDs de dispositivo incorrectos

**Solución:**
```bash
# 1. Listar dispositivos
python3 -c "import sounddevice as sd; [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(sd.query_devices())]"

# 2. Editar config.py con IDs correctos
nano jarvis_modules/config.py

# 3. Buscar y cambiar:
MIC_DEVICE_ID = 1          # Tu ID de micrófono
SPEAKER_INDEX = 0          # Tu ID de altavoces

# 4. Probar
python3 -c "import sounddevice as sd; sd.query_devices(MIC_DEVICE_ID)"
```

### 🔴 "ModuleNotFoundError: No module named 'vosk'"

**Causa:** Modelo Vosk no descargado

**Solución:**
```bash
# Descargar modelo
mkdir -p venv/models
cd venv/models
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip
rm vosk-model-small-es-0.42.zip
cd ../../

# O ejecutar el script automático
./install_complete.sh
```

### 🎤 Sin reconocimiento de voz (micrófono no funciona)

**Causas posibles:**
1. Micrófono desconectado o no configurado
2. Permisos de audio insuficientes
3. Pulseaudio/ALSA no funcionando

**Soluciones:**
```bash
# 1. Verificar permisos
sudo usermod -a -G audio $USER
newgrp audio
# O reiniciar sesión

# 2. Probar micrófono
arecord -D "hw:1,0" test.wav
aplay test.wav

# 3. Revisar ALSA
alsamixer

# 4. Reiniciar audio (Raspberry Pi)
sudo systemctl restart alsa-utils

# 5. Si nada funciona, ver niveles:
alsamixer -c 1
# Seleccionar entrada USB y subir volumen
```

### ⚠️ Piper TTS no genera audio

**Causa:** Librerías `.so` no encontradas

**Solución:**
```bash
# El script run_jarvis.sh configura automáticamente:
export LD_LIBRARY_PATH="./venv/piper:$LD_LIBRARY_PATH"

# Si no funciona, hacer manualmente:
cd venv/piper
ln -sf libpiper_phonemize.so libpiper_phonemize.so.1
cd ../../

# Probar:
export LD_LIBRARY_PATH="./venv/piper:$LD_LIBRARY_PATH"
./venv/piper/piper -m ./venv/piper/models/es_ES-sharvard-medium.onnx -t "Hola mundo"
```

### 🐢 Jarvis muy lento en Raspberry Pi

**Causas:** Falta de memoria, CPU al límite

**Optimizaciones:**
```python
# En config.py
SAMPLE_RATE_CAPTURE = 16000      # No aumentar
AUDIO_BUFFER_SIZE = 1024         # Mantener bajo
WAKE_WORD_SENSITIVITY = 0.5      # Bajar si tiene falsos positivos
COMMAND_LISTEN_DURATION = 8      # Reducir si es posible

# Monitorear memoria
python3 -c "from jarvis_modules.performance_monitor import resource_monitor; print(resource_monitor.get_current_metrics())"
```

### 🔌 Home Assistant no se conecta

**Verificación:**
```bash
# 1. Verificar que HA está corriendo
curl http://192.168.1.100:8123

# 2. Verificar token en config.py
HOME_ASSISTANT_URL = "http://192.168.1.100:8123"
HOME_ASSISTANT_TOKEN = "tu_token_valido"

# 3. Testear conexión
python3 << 'EOF'
import requests
url = "http://192.168.1.100:8123/api/states"
headers = {"Authorization": f"Bearer tu_token"}
r = requests.get(url, headers=headers)
print(r.status_code, r.text)
EOF

# 4. Ver logs de Jarvis
tail -f logs/jarvis_*.log
```

### 📡 Telegram bot no envía mensajes

**Verificación:**
```bash
# 1. Verificar token bot
curl https://api.telegram.org/bot123456789:ABCdefGHI/getMe

# 2. Verificar chat ID
python3 -c "from jarvis_modules.config import config; print(config.TELEGRAM_CHAT_ID)"

# 3. Probar envío manual
python3 << 'EOF'
import requests
token = "tu_bot_token"
chat_id = "tu_chat_id"
url = f"https://api.telegram.org/bot{token}/sendMessage"
data = {"chat_id": chat_id, "text": "Prueba desde Jarvis"}
r = requests.post(url, json=data)
print(r.json())
EOF
```

---

## ⚡ Optimizaciones

### Para Raspberry Pi 4

```python
# config.py ya tiene optimizaciones automáticas:
- SAMPLE_RATE_CAPTURE = 16000       # Reducido para CPU
- AUDIO_BUFFER_SIZE = 1024          # Buffer pequeño
- MAX_WORKER_THREADS = 4            # Máximo 4 hilos
- MEMORY_LIMIT_BYTES = 2GB          # Límite de RAM
```

### Para Intel N95

```python
# Se usa n95_config.py automáticamente si se detecta:
- CPU_CORES = 4                     # Aprovechar 4 cores
- MEMORY_LIMIT = 8GB                # Más memoria disponible
- Timeouts más cortos               # Mejor rendimiento
```

### Monitoreo en Tiempo Real

```bash
# Terminal 1: Ejecutar Jarvis
./run_jarvis.sh

# Terminal 2: Monitorear rendimiento
watch -n 1 'python3 -c "from jarvis_modules.performance_monitor import resource_monitor; print(resource_monitor.get_current_metrics())"'

# Ver logs
tail -f logs/jarvis_*.log
```

### Habilitar Perfilado

```bash
# Ejecutar con profiling activado
JARVIS_PROFILING=true python3 jarvis_main.py

# Ver resultados en logs/performance.log
tail -f logs/performance.log
```

---

## 🛠️ Desarrollo

### Crear un Nuevo Módulo

```bash
# 1. Crear archivo en jarvis_modules/
touch jarvis_modules/mi_modulo.py

# 2. Template básico
cat > jarvis_modules/mi_modulo.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descripción del módulo
"""

from .config import config, get_logger

logger = get_logger(__name__)

class MiModulo:
    def __init__(self):
        logger.info("Inicializando MiModulo")
    
    def mi_funcion(self):
        logger.info("Ejecutando función")
        return "resultado"

mi_modulo = MiModulo()
EOF

# 3. Importar en command_processor_optimized.py
# from . import mi_modulo

# 4. Crear tests en tests/
# python3 -m pytest tests/test_mi_modulo.py
```

### Ejecutar Tests

```bash
# Todos los tests
python3 -m pytest tests/

# Test específico
python3 -m pytest tests/test_microphone_capture.py -v

# Con cobertura
python3 -m pytest tests/ --cov=jarvis_modules --cov-report=html
```

### Debug

```bash
# Habilitar verbose logging
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG)" 
python3 jarvis_main.py

# O establecer en código
# logger.setLevel(logging.DEBUG)

# Debug interactivo
python3 -i jarvis_main.py
# >>> from jarvis_modules.command_processor_optimized import process_command
# >>> process_command("reproduce música rock")
```

---

## 📊 Estadísticas del Proyecto

- 📝 **Líneas de código**: 8000+
- 📦 **Módulos**: 15+
- 🧪 **Tests**: 30+
- 📖 **Documentación**: 50+ páginas
- 🌍 **Idiomas soportados**: Español
- 🔌 **Integraciones**: 6+ servicios
- 🎵 **Comandos**: 100+ variantes

---

## 📋 Comandos de Voz Rápida Referencia

```
MÚSICA:
  "reproduce música clásica"
  "pon rock"
  "sube el volumen"
  "pausa"

RECORDATORIOS:
  "recuérdame tomar medicina a las 8"
  "recuérdame el 25 de diciembre"
  "mis recordatorios"

CLIMA:
  "¿qué tiempo hace?"
  "predicción para Barcelona"
  "¿va a llover?"

CASA:
  "enciende la luz del salón"
  "apaga el ventilador"
  "abre las cortinas"

HORA:
  "¿qué hora es?"
  "¿qué día es hoy?"
  "¿cuántos días faltan para Navidad?"

GENERAL:
  "Hola Jarvis"
  "hasta luego"
  "ayuda"
```

---

## 📞 Soporte

- 📚 Documentación completa: `docs/`
- 📝 Logs: `logs/` (después de ejecutar)
- 🐛 Issues: Revisar `otros/CHECKLIST_FIXES.md`
- 💡 Ejemplos: `otros\ test/` (scripts de referencia)

---

## 📄 Licencia

MIT License - Libre para uso personal y comercial

---

## 🙏 Agradecimientos

- **Picovoice** - Detección de palabra clave
- **Vosk** - Reconocimiento de voz offline
- **Piper** - Síntesis de voz
- **AEMET** - Datos meteorológicos
- **Home Assistant** - Automatización del hogar
- **Telegram** - Notificaciones

---

**¡Disfruta tu asistente Jarvis! 🤖✨**

*Última actualización: 14 de noviembre de 2025*
*Versión: Complete Package 20251031*


    