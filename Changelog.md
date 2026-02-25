# Changelog

All notable changes to the Elara project will be documented in this file.

## [2.3.1-patch] - 2026-02-25

### Fixed
- **Piper TTS**: Rewrote `piper_tts.py` — previous version had syntax corruption. Now uses `shutil.which()` for binary detection.
- **Voice Exports**: Updated `elara_core/voice/__init__.py` — removed stale `MimiTTS`/`WhisperSTT` exports, added `PiperTTS`, `StreamingPiperTTS`, `OLMoSTT`.
- **Tier 1 Model Path**: Default model changed from `gemma-3-1b-it-q4_0.gguf` to `qwen-1.5b-q4.gguf` to match download script.
- **Tier 1 Chat Format**: Auto-detects `chat_format` from model filename (`chatml` for Qwen, `gemma` for Gemma).
- **Tier 1 get_stats()**: Fixed `TypeError` — `n_ctx` is a method in newer `llama-cpp-python`, not a property. Also derives model name from actual file path instead of hardcoded string.
- **Audio Playback**: `VoiceGateway.speak()` now plays audio through speakers via `sounddevice` instead of silently returning a numpy array.
- **Quick Test**: Fixed default model path in `scripts/quick_test.sh`.

### Removed
- **`elara_core/voice/stt.py`** — Old Whisper wrapper, replaced by `olmo_stt.py`.
- **`elara_core/voice/mimi_tts.py`** — Non-functional placeholder, replaced by `piper_tts.py`.

## [2.3.0] - 2026-02-25

### Added
- **STT**: Replaced OpenAI Whisper with OLMoASR (Apache 2.0, fully open source, no API keys).
- **TTS**: Replaced broken Mimi with Piper TTS (fast, local, intelligible speech).
- **Setup**: Unified `scripts/setup.sh` for one-command environment setup.
- **Testing**: New `scripts/quick_test.sh` for validating all components.
- **Model Options**: Download script now offers Qwen2.5-1.5B (no auth, default) or Gemma-3-1B.

### Changed
- **Memory**: Fixed memory check thresholds — now uses `available_gb` (2GB to load, 1GB to process, 0.5GB critical).
- **Memory Bypasses**: Removed all `if False:` hacks that disabled memory checks in `main.py` and `tier1.py`.
- **Model Download**: Script now uses working URLs; Qwen default requires no authentication.
- **Voice Gateway**: Re-architected for OLMoASR STT and Piper TTS backends.
- **Audio Output**: Sample rate changed from 24kHz (Mimi) to 22.05kHz (Piper).
- **CLI**: `--no-tts-mimi` renamed to `--no-tts-piper`.

### Removed
- **Whisper dependency** (`openai-whisper`) — replaced by OLMoASR.
- **Mimi/Moshi dependency** — replaced by Piper TTS.
- **sphn dependency** — no longer needed.
- **pyttsx3** — removed from requirements (still works as manual fallback).

## [2.2.1-patch] - 2026-02-25

### Changed
- **Type Hint Standardization**: Updated all type hints to use Python 3.9+ lowercase built-ins (`dict`, `list`, `tuple`) across the entire `elara_core` package while maintaining `typing.Any` for compatibility with static analysis.
- **Constant Referencing**: Updated `DuplexVoiceHandler` to reference `MimiTTS.FRAME_SIZE` instead of a hardcoded value.

### Fixed
- **Import Cleanup**: Removed redundant and unused imports from multiple modules.
- **Package Integrity**: Verified `__init__.py` files across all subdirectories to ensure proper package structure.
- **Documentation**: Added comprehensive docstrings to core modules and methods across the `elara_core` package.

## [2.2.0-voice] - 2026-02-21

### Added
- **Mimi Neural TTS**: Integrated Mimi codec for high-quality, low-latency neural speech synthesis (~200MB RAM).
- **Full-Duplex Voice Mode**: New `--voice-input` mode allowing natural conversation with voice activity detection and interruptions.
- **Voice Persona Manager**: Support for multiple personas with distinct voice samples and text-generation styles.
- **Persona Conditioning**: All generation tiers now support custom system prompts based on the active persona.
- **Standalone RAG Generation**: Tier 2 now includes an extractive fallback for answering questions when Tier 1 is unavailable.
- **Environment Variable Support**: Key paths and hardware settings can now be configured via `.env` or system variables.
- **Audio Recorder**: Asynchronous microphone capture using `sounddevice`.
- **Memory Monitoring Utility**: Added `--monitor` flag and `elara_core/utils.py` for tracking RAM usage against the 4GB target.
- **Robust Model Downloads**: Added exponential backoff retry logic to `scripts/download_mimi.py`.

### Changed
- **Safety Filter**: Implemented double-check logic (pre- and post-cleaning) to prevent safety bypasses.
- **Tool Context Integration**: Refactored `main.py` to ensure tool results are correctly injected as context into the LLM prompt.
- **Voice Gateway**: Re-architected for better fallback between Mimi, NeMo, and pyttsx3.

### Fixed
- Missing `__init__.py` files in `tools/`, `safety/`, and `persona/` directories.
- Refactored `DuplexVoiceHandler` to use centralized `process_input` logic for consistency.

## [2.1.1-patch] - 2026-02-22

### Fixed
- **Tool Router Logic**: Fixed bug where tool execution caused immediate return. Tool results are now prepended to user input as context for the LLM.
- **Tier 2 Fallback**: Added better formatting and document truncation when no generator is available in Tier 2.
- **Voice Gateway**: Fixed GPU/NeMo detection logic to use `torch.cuda.is_available()`.

### Added
- **CLI Flags**: Added `--tts-nemo` and `--tts-cpu` to `main.py` for manual TTS engine selection.

### Removed
- Dead code: `ToolCallParser` and associated exports.

## [2.1.0-functional] - 2026-02-25

### Added
- **Architecture Reset**: Replaced complex research-based modules (CLaRa, TRM, TiDAR) with functional, battle-tested components.
- **New Tiered System**:
  - **Tier 1**: Gemma 3 1B via `llama-cpp-python` for fast, local inference.
  - **Tier 2**: FAISS-based RAG with `SentenceTransformers` for document-augmented generation.
  - **Tier 3**: API fallback support (OpenRouter) for high-complexity queries.
- **Safety Layer**: Simplified rule-based `SafetyFilter` replacing the old Constitutional Layer.
- **Voice Fallback**: Added `pyttsx3` as an offline, low-resource TTS fallback.
- **Tooling**:
  - `scripts/build_index.py`: Tool for building FAISS indices from text documents.
  - `scripts/download_models.sh`: One-click setup for required model weights.
  - `scripts/benchmark.py`: Latency measurement tool for different tiers.

### Changed
- **main.py**: Simplified CLI logic and routing.
- **Requirements**: Updated to reflect functional dependencies (`llama-cpp-python`, `faiss-cpu`, etc.).
- **Documentation**: Updated README and Architecture diagrams to match the new functional design.

### Removed
- Unfunctional/Untrainable research modules: `clara`, `trm`, `tidar`.
- `airllm_fallback.py` (replaced by Tier 3 API fallback).

## [2.0.0] - 2026-02-18

### Added
- **Elara v2.0 Core Architecture**: Complete implementation of the multi-tier inference engine.
- **Tiered Inference Engine**: 
  - Tier 1: Direct Mistral generation for simple queries.
  - Tier 2: CLaRa-augmented retrieval for memory-intensive queries.
  - Tier 3: Deep reasoning with TRM, TiDAR, and external tools.
- **Constitutional Layer**:
  - Code-based safety filtering using biblical principles.
  - Immutable audit logging of all safety decisions.
  - Support for YAML-defined principle rules (Truth, Harm Prevention, Dignity, Wisdom, Stewardship).
- **CLaRa (Continuous Latent Reasoning)**:
  - SCP (Salient Compressor Pretraining) for document compression into memory tokens.
  - Query Reasoner for encoding user queries into the same embedding space.
  - Persistent document store with memory-mapped file support for edge efficiency.
- **TRM (Tiny Recursive Model)**:
  - 2-layer weight-shared network for recursive reasoning.
  - Adaptive Halting mechanism with learned confidence probabilities.
  - Deep supervision support for training the reasoning trace.
- **TiDAR (Think in Diffusion, Talk in Autoregression)**:
  - Hybrid attention mechanism supporting simultaneous AR and diffusion processing.
  - Draft-Verify sampling for parallel token generation with quality guarantees.
- **Voice Gateway**:
  - Decoupled Voice-to-Text (Whisper) and Text-to-Voice (NeMo FastPitch/HiFi-GAN).
  - Audio utility tools for normalization and format conversion.
- **Tool Routing System**:
  - XML-style tool call parsing from model outputs.
  - Built-in calculator and stubs for search and vision.
- **Tier 3 Fallback (AirLLM)**:
  - Layer-wise streaming inference for large models on 4GB RAM devices.
  - Shard-based memory mapping and LRU layer caching.
- **CLI & Orchestration**:
  - Interactive chat mode CLI (`python -m elara_core.main --interactive`).
  - Integrated performance metrics tracking (latency, tier distribution).
  - Automated input multiplexing for text, voice, and multimodal data.
- **Testing & Documentation**:
  - Comprehensive unit test suite covering all modules.
  - New `Architecture.md` and `README.md`.
