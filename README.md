# Elara - Functional Local-First AI

A multi-tier AI system designed for edge devices with a 4GB RAM target. No research fantasies, just battle-tested tools that actually ship.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone and enter directory
git clone https://github.com/Alartist40/elara.git
cd elara

# 2. Run setup (creates venv, installs deps, downloads models)
bash scripts/setup.sh

# 3. Test
python3 -m elara_core.main --text "Hello, what is 2+2?"

# 4. Interactive mode
python3 -m elara_core.main --interactive
```

### Manual setup (if script fails)
```bash
python3 -m venv elara_env
source elara_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ./OLMoASR  # Install OLMoASR from local directory
bash scripts/download_models.sh
```

## 🛠 Features

*   **4GB RAM Target:** Optimized for low-memory environments (uses ~800MB-1.5GB at runtime).
*   **Multi-Tier Routing:** Heuristic-based routing that chooses the right model for the job.
*   **Persona Manager:** Switch between AI personas with distinct voices and interaction styles.
*   **Full-Duplex Voice:** Natural conversation with VAD, streaming audio, and interruption support.
*   **Tool Integration:** Basic math/calculator support that feeds context back into the LLM.
*   **Biblical Principles:** Content filtering based on defined constitutional principles.
*   **Offline-First:** Core functionality (Tiers 1 & 2) works entirely without internet.
*   **No API Keys Required:** OLMoASR (STT) + Piper (TTS) + local LLM — fully offline.

## 🎮 Usage

### Interactive Mode
```bash
python3 -m elara_core.main --interactive
```

### Voice Mode
```bash
# Text input with voice output
python3 -m elara_core.main --interactive --voice

# Full-duplex voice conversation (microphone input)
python3 -m elara_core.main --voice-input --voice-streaming
```

### Forcing TTS Engine
```bash
# Force Piper (Default, high quality)
python3 -m elara_core.main --voice-input

# Force NeMo (requires GPU/CUDA)
python3 -m elara_core.main --voice-input --tts-nemo

# Force local fallback (pyttsx3)
python3 -m elara_core.main --voice-input --tts-cpu

# Disable Piper and use fallback
python3 -m elara_core.main --voice-input --no-tts-piper
```

### Voice Cloning
1. Record 5-10 seconds of speech as `data/voices/my_voice.wav` (22.05kHz mono)
2. Elara automatically creates an embedding on first use.
3. The voice will be available as a persona if defined in `config/personas.yaml`.

### Monitoring
```bash
# Monitor memory usage (4GB target)
python3 -m elara_core.main --interactive --monitor
```

### Building the Knowledge Base
Place `.txt` files in a directory and run:
```bash
python3 scripts/build_index.py path/to/your/docs/
```

## 🏗 Architecture

*   **Tier 1:** Qwen2.5-1.5B (default) / Gemma 3 1B IT (GGUF Q4_K_M). Chat format auto-detected.
*   **Tier 2:** RAG pipeline with FAISS and standalone extractive fallback.
*   **Tier 3:** API fallback via OpenRouter.
*   **Router:** Simple keyword and relevance-based logic.
*   **Voice:** Piper TTS for synthesis (with `sounddevice` playback); OLMoASR for STT.
*   **Safety Filter:** Multi-stage pre/post check with cleaning bypass protection.

## 📜 License
MIT
