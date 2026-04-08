# Jarvis V2 — Asistente de Voz Local Distribuido

Asistente de voz que corre **completamente en hardware propio**, sin APIs en la nube.
Arquitectura multi-nodo: satélites de captura de audio (Raspberry Pi 4), un Hub Central
que orquesta el pipeline, y una máquina GPU dedicada para la inferencia LLM.

---

## Arquitectura del sistema

```
Satélites (RPi4)           Hub Central (Ryzen 5700U)      Máquina GPU
─────────────────          ──────────────────────────      ────────────
reSpeaker XVF3800          reSpeaker XVF3800 (opcional)    Ollama :11434
Whisper tiny/base  →       FastAPI :8000                →  gemma4:e2b
Piper TTS          ←       LangGraph + Rasa NLU         ←  GPU dedicada
wake word Porcupine        Bot Telegram
                           hub_voz.py (nodo local)
```

| Nodo | Rol | Hardware |
|---|---|---|
| **Máquina GPU** | Motor LLM | PC con NVIDIA RTX ≥ 3060 8 GB VRAM |
| **Hub Central** | Orquestador, API REST, NLU, Telegram, voz local | AMD Ryzen 7 5700U, 16 GB RAM |
| **Satélite** | Captura/reproducción de audio por habitación | Raspberry Pi 4 (4 GB), reSpeaker XVF3800 |

---

## Idea central — Auto-descubrimiento de herramientas

El LLM principal es `gemma4:e2b` (2B parámetros). Inyectarle las 13+ herramientas
disponibles en el contexto lo satura y degrada su precisión. La solución es el
**auto-descubrimiento**: Rasa NLU clasifica la intención del usuario antes de invocar
al LLM y le pasa solo las 3-5 herramientas relevantes.

```
Voz del usuario
      │
      ▼
  [Whisper STT]           ← RPi4 / Hub (Whisper.cpp)
      │
      ▼
  [Rasa NLU + spaCy]      ← Hub Central (Docker)
      │  intención: "weather"
      ▼
  [LangGraph router]      ← Hub Central
      │  tools: [get_weather, get_forecast]
      ▼
  [Ollama → gemma4:e2b]   ← Máquina GPU (red local)
      │
      ▼
  [Piper TTS]             ← RPi4 / Hub
      │
      ▼
Respuesta de voz
```

---

## Archivos principales del repositorio

| Archivo | Descripción |
|---|---|
| `hub_server.py` | Servidor FastAPI — recibe peticiones de satélites y Telegram |
| `hub_voz.py` | Nodo de voz local del Hub — VAD + Whisper + Piper sin HTTP |
| `telegram_bot.py` | Bot bidireccional — texto y notas de voz |
| `pipeline/graph.py` | Pipeline LangGraph: clasificar → filtrar tools → invocar LLM |
| `pipeline/intent_classifier.py` | Clasificador de intenciones (Rasa o keywords) |
| `tools/definitions.py` | 13 herramientas mock en 5 grupos |
| `tools/registry.py` | Mapeo intent → herramientas relevantes |
| `config/settings.py` | Configuración singleton |
| `deploy/` | Archivos systemd listos para producción |
| `whisper.cpp/` | Submodulo con el motor STT compilado en el Hub |

---

## Fases de implementación

### Fase 0 — Preparación base ✅
Configuración común de SO, IPs estáticas y herramientas base en todos los nodos
(GPU, Hub y cada RPi). Ver [FASE_00_BASE.md](Jarvis_V2_docs/Implementacion/FASE_00_BASE.md).

### Fase 1 — Máquina GPU (Ollama) ✅
- Instalación de drivers NVIDIA y Ollama
- Descarga y configuración de `gemma4:e2b`
- Exposición en la LAN (puerto 11434)
- Latencia objetivo: P50 < 600 ms, P95 < 1 000 ms con GPU activa

Ver [FASE_01_GPU.md](Jarvis_V2_docs/Implementacion/FASE_01_GPU.md).

### Fase 2 — Hub Central (Pipeline + API REST) ✅
- Pipeline LangGraph: `classify_intent → select_tools → invoke_llm`
- API REST con FastAPI (puerto 8000) para recibir peticiones de satélites
- Rasa NLU en Docker (con fallback automático a clasificador por palabras clave)
- Suite de benchmarks: latencia LLM, precisión de tool calling, comparativa auto-discovery
- Servicio systemd en `deploy/asistente-hub.service`

Ver [FASE_02_HUB.md](Jarvis_V2_docs/Implementacion/FASE_02_HUB.md).

### Fase 2b — Hub como nodo de voz local ✅
- `hub_voz.py`: bucle VAD → Whisper → pipeline → Piper, todo en proceso
- VAD con filtro RMS + detección de frames consecutivos (evita falsos positivos)
- Whisper.cpp (`ggml-small.bin`, 14 hilos) compilado en el Hub (`whisper.cpp/`)
- Piper TTS con `paplay` (PulseAudio)
- Transcripción anticipada en hilo paralelo (ThreadPoolExecutor)

Ver [FASE_02b_HUB_VOZ.md](Jarvis_V2_docs/Implementacion/FASE_02b_HUB_VOZ.md).

### Fase 3 — Satélites Raspberry Pi 4 ✅
- Whisper.cpp (`base.es`) para STT local en cada RPi
- Piper TTS (`es_ES-davefx-medium`) para síntesis local
- VAD con `webrtcvad`, bucle de escucha continua
- Script `satelite.py` (desplegado en cada RPi) + servicio systemd `asistente-satelite`
- Identificación por habitación (`SATELLITE_ID`: salon, cocina, dormitorio…)

Ver [FASE_03_SATELITE.md](Jarvis_V2_docs/Implementacion/FASE_03_SATELITE.md).

### Fase 4 — Bot de Telegram bidireccional ✅
- Mensajes de texto → pipeline → respuesta de texto con intención y latencia
- Notas de voz → FFmpeg → Whisper STT → pipeline → respuesta de texto + audio
- Comandos `/start`, `/estado` y `/satelites`
- Lista blanca de usuarios autorizados (`TELEGRAM_ALLOWED_USERS`)
- Servicio systemd en `deploy/asistente-telegram.service`

Ver [FASE_04_TELEGRAM.md](Jarvis_V2_docs/Implementacion/FASE_04_TELEGRAM.md).

### Fase 5 — reSpeaker XVF3800
- Instalación del firmware y calibración del array de 4 micrófonos
- Beamforming y cancelación de eco (AEC) integrados en el hardware
- Aplicable tanto al Hub como a cada satélite

Ver [FASE_05_RESPEAKER.md](Jarvis_V2_docs/Implementacion/FASE_05_RESPEAKER.md).

### Fase 6 — Puesta en marcha y verificación
- Checklist de arranque de todos los servicios
- Verificación de latencias extremo a extremo
- Configuración de inicio automático con systemd (`deploy/asistente.target`)

Ver [FASE_06_PUESTA_EN_MARCHA.md](Jarvis_V2_docs/Implementacion/FASE_06_PUESTA_EN_MARCHA.md).

### Fase 7 — Wake word con Porcupine
- Sustituye el VAD continuo por detección eficiente de palabra de activación
- Modelos en español (es_ES) con soporte para wake words personalizadas (`.ppn`)
- Consumo en reposo: ~5-10 % CPU en RPi4
- Flujo: Porcupine idle → pitido → grabación VAD → STT → Hub → TTS → Porcupine idle

Ver [FASE_07_ACTIVACION.md](Jarvis_V2_docs/Implementacion/FASE_07_ACTIVACION.md).

### Fase 8 — Herramientas reales
Sustitución de las 13 herramientas mock por implementaciones reales:

| Subfase | Herramientas | Dependencias |
|---|---|---|
| 8.1 | Hora, fecha, temporizador con alarma TTS | ninguna |
| 8.2 | Tiempo actual y pronóstico 7 días | Open-Meteo (gratuita, sin API key) |
| 8.3 | Recordatorios persistentes | SQLite local |
| 8.4 | Inventario doméstico y lista de la compra | Grocy (autoalojado, LAN) |
| 8.5 | Búsqueda e información general | Wikipedia API (gratuita) |
| 8.6 | Control del hogar | Home Assistant REST API |
| 8.7 | Memoria conversacional y perfil del usuario | SQLite + ChromaDB |

Ver [FASE_08_HERRAMIENTAS.md](Jarvis_V2_docs/Implementacion/FASE_08_HERRAMIENTAS.md).

---

## Orden de configuración recomendado

```
FASE_00  →  FASE_01  →  FASE_02  →  FASE_05  →  FASE_02b
                     ↓              ↓                  ↓
                  FASE_04       FASE_03 (por cada RPi) FASE_07 (wake word)
                                    ↓
                               FASE_06 (verificación final)
                                    ↓
                               FASE_08 (herramientas reales)
```

---

## Stack tecnológico

| Capa | Tecnología | Nodo |
|---|---|---|
| LLM | Ollama + gemma4:e2b | Máquina GPU |
| NLU / Intenciones | Rasa NLU 3.6 + spaCy `es_core_news_md` | Hub (Docker) |
| Orquestación | LangGraph + LangChain | Hub |
| API REST | FastAPI + Uvicorn | Hub |
| STT | Whisper.cpp (`ggml-small.bin` hub / `base.es` RPi) | Hub y RPi4 |
| TTS | Piper TTS (`es_ES-davefx-medium`) + paplay/aplay | Hub y RPi4 |
| Wake word | Picovoice Porcupine (español) | RPi4 y Hub |
| Hardware de audio | reSpeaker XVF3800 (4 mic, AEC) | RPi4 y Hub |
| Bot remoto | python-telegram-bot v21 + httpx | Hub |
| Benchmarks | pytest + ollama-python | Hub (dev) |

---

## Comandos principales

```bash
# Instalar dependencias Python (Hub)
pip install -r requirements.txt
python -m spacy download es_core_news_md

# Entrenar y arrancar Rasa NLU
docker compose run --rm rasa-train
docker compose up -d rasa-server

# Arrancar el Hub manualmente
python hub_server.py

# Arrancar el nodo de voz local del Hub manualmente
python hub_voz.py

# Arrancar el bot de Telegram manualmente
python telegram_bot.py

# Desplegar servicios systemd (Hub)
sudo cp deploy/asistente-hub.service /etc/systemd/system/
sudo cp deploy/asistente-telegram.service /etc/systemd/system/
sudo cp deploy/asistente.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now asistente.target   # arranca hub + hub-voz + telegram

# Benchmarks (requiere Ollama activo)
python run_benchmarks.py              # suite completa
python run_benchmarks.py --unit-only  # solo tests sin Ollama
python run_benchmarks.py --skip-slow  # omite warmup y suites lentas
python run_benchmarks.py --test 03    # una suite específica (01–05)

# pytest directo
pytest tests/                                  # todos los tests
pytest tests/test_03_intent.py -v              # clasificador (sin Ollama)
pytest -m unit                                 # solo tests unitarios
pytest -m "not slow"                           # omitir tests lentos
pytest -m ollama                               # solo tests con Ollama
```

---

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama (máquina GPU) |
| `MODEL_NAME` | `gemma4:e2b` | Modelo LLM en Ollama |
| `RASA_URL` | `http://localhost:5005` | URL del servidor Rasa NLU |
| `HUB_HOST` | `0.0.0.0` | Interfaz de escucha del Hub |
| `HUB_PORT` | `8000` | Puerto de la API REST del Hub |
| `HUB_URL` | `http://localhost:8000` | URL del Hub (usado por satélites y bot) |
| `SATELLITE_ID` | `hub` / `salon` | Identificador del nodo (por habitación) |
| `TELEGRAM_BOT_TOKEN` | — | Token del bot de Telegram (obligatorio) |
| `TELEGRAM_ALLOWED_USERS` | — | IDs de Telegram autorizados (vacío = todos) |
| `WHISPER_BIN` | `whisper.cpp/build/bin/whisper-cli` | Binario de Whisper.cpp |
| `WHISPER_MODEL` | `whisper.cpp/models/ggml-small.bin` | Modelo Whisper del Hub |
| `WHISPER_THREADS` | `14` | Hilos para Whisper (ajustar al hardware) |
| `PIPER_BIN` | `venv/bin/piper` | Binario de Piper TTS |
| `PIPER_VOICE` | `piper-voices/es_ES-davefx-medium.onnx` | Voz Piper |
| `RMS_THRESHOLD` | `300` | Umbral de energía para filtrar silencio (hub_voz) |
| `SPEECH_RESET_FRAMES` | `4` | Frames consecutivos para cancelar silencio (hub_voz) |

---

## Umbrales de latencia objetivo (voz)

| Métrica | Objetivo |
|---|---|
| STT Whisper base (RPi4) | < 800 ms |
| STT Whisper small (Hub) | < 500 ms |
| Rasa NLU + LangGraph (Hub) | < 110 ms |
| LLM GPU P50 | < 600 ms |
| LLM GPU P95 | < 1 000 ms |
| TTS Piper | < 300 ms |
| **E2E total** | **< 3 000 ms** |

---

## Documentación completa

- `Jarvis_V2_docs/Implementacion/` — guías de instalación fase a fase
- `Jarvis_V2_docs/Documentacion/` — diseño, arquitectura y decisiones técnicas
