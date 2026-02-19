# Elara - Functional Local-First AI

A multi-tier AI system designed for edge devices with a 4GB RAM target. No research fantasies, just battle-tested tools that actually ship.

## 🚀 The Honest Assessment

The previous versions of Elara attempted to implement complex research architectures (CLaRa, TRM, TiDAR) that proved untrainable and non-functional for production.

This version (v2.0+) pivots to what actually works:
*   **Tier 1 (Local):** Gemma 3 1B (quantized) via `llama-cpp-python`.
*   **Tier 2 (RAG):** FAISS + SentenceTransformers for document Q&A.
*   **Tier 3 (API):** OpenRouter/OpenAI fallback for complex reasoning.
*   **Safety:** Simple, compiled regex rules that actually block/redact content.
*   **Voice:** Whisper for STT and a hybrid NeMo/pyttsx3 system for TTS.

## 🛠 Features

*   **4GB RAM Target:** Optimized for low-memory environments (uses ~800MB-1.2GB at runtime).
*   **Multi-Tier Routing:** Heuristic-based routing that chooses the right model for the job.
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
python3 -m elara_core.main --interactive --voice
```

### Forcing TTS Engine
```bash
# Force NeMo (requires GPU/CUDA)
python3 -m elara_core.main --interactive --voice --tts-nemo

# Force local fallback (pyttsx3)
python3 -m elara_core.main --interactive --voice --tts-cpu
```

### Building the Knowledge Base
Place `.txt` files in a directory and run:
```bash
python3 scripts/build_index.py path/to/your/docs/
```

## 🏗 Architecture

*   **Tier 1:** Gemma 3 1B IT (GGUF Q4_K_M) - The fast, local "brain".
*   **Tier 2:** RAG pipeline using `all-MiniLM-L6-v2` (22MB) and FAISS.
*   **Tier 3:** API fallback to GPT-4o or Claude 3.5 via OpenRouter.
*   **Router:** Simple keyword and relevance-based logic (no ML overhead).
*   **Safety Filter:** Regex-based pre- and post-generation checks.

## 📜 License
MIT
