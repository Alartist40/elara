# Elara - Comprehensive Project Guide

Welcome to the master documentation for **Elara**, a functional, local-first AI system designed for edge devices with a 4GB RAM target. This document serves as a single source of truth for understanding, setting up, running, and extending the Elara codebase.

---

## 🎯 Goal & Purpose

Elara's primary objective is to provide a **reliable and functional AI assistant** that runs entirely on local hardware (Tiers 1 & 2) with a minimal memory footprint. Unlike previous research-heavy versions, Elara v2.2+ focuses on battle-tested technologies that actually ship:
- **Local-First**: Core reasoning and knowledge retrieval happen on your machine.
- **Resource Efficient**: Targeted for devices with as little as 4GB RAM.
- **Multi-Modal**: Supports both text and high-fidelity full-duplex voice interaction.
- **Safety-First**: Implements constitutional guardrails based on defined principles.

---

## 🏗 Architecture Deep Dive

Elara uses a **Multi-Tier Routing Architecture** that balances speed, accuracy, and resource usage.

### 1. The Tiered System
- **Tier 1 (Direct Inference)**: Powering simple and fast interactions using a quantized **Gemma 3 1B** model. It uses `llama-cpp-python` for efficient execution.
- **Tier 2 (RAG - Retrieval Augmented Generation)**: Adds a "memory" layer using **FAISS** and **SentenceTransformers**. It retrieves relevant local documents and injects them into the Tier 1 prompt.
- **Tier 3 (API Fallback)**: Connects to external providers (via OpenRouter/OpenAI) for complex tasks that exceed local capacity.

### 2. The Voice Pipeline
- **STT (Speech-to-Text)**: Powered by **OpenAI Whisper** (Tiny/Base).
- **TTS (Text-to-Speech)**: Uses the **Mimi** neural codec for low-latency, high-quality audio, with fallbacks to **NeMo** and **pyttsx3**.
- **Duplex Handler**: Manages the conversation flow, allowing for natural interruptions and Voice Activity Detection (VAD).

---

## ⚙️ Setup & Installation

### Environment Setup
1. **Create VENV**: `python -m venv elara_env && source elara_env/bin/activate`
2. **Install Torch (CPU)**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. **Install Requirements**: `pip install -r requirements.txt`
4. **Mimi TTS Note**: If not found, install `moshi` manually: `pip install git+https://github.com/kyutai-labs/moshi.git`

### Model Configuration
- Run `bash scripts/download_models.sh` to fetch the default GGUF and embedding models.
- Place your knowledge base (`.txt` files) in a folder and run `python3 scripts/build_index.py path/to/docs/` to initialize RAG.

---

## 🎮 Running Elara

### Primary Modes
- **Interactive Text**: `python3 -m elara_core.main --interactive`
- **Voice Output**: Add `--voice` to text modes.
- **Full-Duplex Voice**: `python3 -m elara_core.main --voice-input --voice-streaming`

### Useful Flags
- `--monitor`: Real-time RAM monitoring.
- `--no-tts-mimi`: Disable the heavy neural TTS if memory is extremely tight.
- `--voice-debug`: See which TTS engine is currently in use.

---

## 🔍 Codebase Deep Dive

### `elara_core/main.py`
The **Orchestration Layer**. It handles CLI arguments, initializes the various engines (Tier 1, 2, 3, Voice, Safety), and routes user input through the `process_input` function.

### `elara_core/tiers/`
- `router.py`: Contains `TierRouter`, which decides whether a query needs Tier 1, 2, or 3 based on keywords and document relevance.
- `tier1.py`: Wraps `llama-cpp-python` for Gemma inference.
- `tier2.py`: Manages the FAISS index and context injection.
- `tier3.py`: Handles external API calls.

### `elara_core/voice/`
- `gateway.py`: The unified interface for audio. It lazily loads STT and TTS engines.
- `duplex_handler.py`: The complex logic that makes "talking" to Elara feel natural by managing audio buffers and interruptions.
- `mimi_tts.py`: Implementation of the high-fidelity neural speech synthesis.

### `elara_core/safety/`
- `filter.py`: Implements the `SafetyFilter`. It uses regex-based rules from `rules.yaml` to scrub input and check output for harmful content or PII.

### `elara_core/tools/`
- `router.py`: Detects and executes built-in tools (like the calculator) before the prompt reaches the LLM.

### `elara_core/persona/`
- `voice_persona.py`: Manages the "identity" of the AI, including its specific voice embedding and its system-prompt personality.

---

## 🛠 Developer Guide: How to Edit & Extend

### Adding a New Tool
1. Create a new class in `elara_core/tools/implementations/`.
2. Register the tool in `elara_core/tools/router.py`.
3. Add search keywords to `TierRouter.TOOL_KEYWORDS` in `elara_core/tiers/router.py` to ensure it routes to Tier 3.

### Modifying Safety Rules
Edit `elara_core/safety/rules.yaml`. You can add new regular expressions for blocking specific topics or redacting sensitive information.

### Changing the Visuals (TUI/CLI)
Most CLI logic is in `elara_core/main.py`. The `process_input` function is the core logic loop; you can wrap this in any UI (Rich, Textual, etc.) as needed.

---

## 📜 Credits & Licensing
Elara is built on the shoulders of giants: **Google (Gemma)**, **Meta (FAISS)**, **Kyutai Labs (Moshi)**, and the **OpenAI Whisper** team.
**License**: MIT.

*Elara v2.2 - Built for performance, guided by principles.*
