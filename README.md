# Elara - Functional Local-First AI

A multi-tier AI system designed for edge devices with a 4GB RAM target. No research fantasies, just battle-tested tools that actually ship.

## 🚀 The Honest Assessment

The previous versions of Elara attempted to implement complex research architectures (CLaRa, TRM, TiDAR) that proved untrainable and non-functional for production.

This version (v2.0+) pivots to what actually works:
*   **Tier 1 (Local):** Gemma 3 1B (quantized) via `llama-cpp-python`.
*   **Tier 2 (RAG):** FAISS + SentenceTransformers for document Q&A.
*   **Tier 3 (API):** OpenRouter/OpenAI fallback for complex reasoning.
*   **Safety:** Multi-stage regex filtering that blocks/redacts harmful content with bypass protection.
*   **Voice:** Full-duplex conversation mode with Whisper STT and Mimi neural TTS.

## 🛠 Features

*   **4GB RAM Target:** Optimized for low-memory environments (uses ~800MB-1.5GB at runtime).
*   **Multi-Tier Routing:** Heuristic-based routing that chooses the right model for the job.
*   **Persona Manager:** Switch between AI personas with distinct voices and interaction styles.
*   **Full-Duplex Voice:** Natural conversation with VAD, streaming audio, and interruption support.
*   **Tool Integration:** Basic math/calculator support that feeds context back into the LLM.
*   **Biblical Principles:** Content filtering based on defined constitutional principles.
*   **Offline-First:** Core functionality (Tiers 1 & 2) works entirely without internet.

## 📦 Installation

### 1. Create a clean environment
```bash
python -m venv elara_env
source elara_env/bin/activate  # Linux/Mac
# elara_env\Scripts\activate  # Windows
```

### 2. Install CPU-only torch (Saves 2GB+)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download models
```bash
# Requires HF_TOKEN if downloading gated models
export HF_TOKEN=your_token_here
bash scripts/download_models.sh
```

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
# Force Mimi (Default, high quality)
python3 -m elara_core.main --voice-input

# Force NeMo (requires GPU/CUDA)
python3 -m elara_core.main --voice-input --tts-nemo

# Force local fallback (pyttsx3)
python3 -m elara_core.main --voice-input --tts-cpu

# Disable Mimi and use fallback
python3 -m elara_core.main --voice-input --no-tts-mimi
```

### Voice Cloning
1. Record 5-10 seconds of speech as `data/voices/my_voice.wav` (24kHz mono)
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

*   **Tier 1:** Gemma 3 1B IT (GGUF Q4_K_M) - The fast, local "brain".
*   **Tier 2:** RAG pipeline with FAISS and standalone extractive fallback.
*   **Tier 3:** API fallback via OpenRouter.
*   **Router:** Simple keyword and relevance-based logic.
*   **Voice:** Mimi neural codec for streaming TTS; Whisper for STT.
*   **Safety Filter:** Multi-stage pre/post check with cleaning bypass protection.

## 📜 License
MIT
