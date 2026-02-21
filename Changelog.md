# Changelog

All notable changes to the Elara project will be documented in this file.

## [2.2.0-voice] - 2026-02-27

### Added
- **Mimi Neural TTS**: Integrated Mimi codec for high-quality, low-latency neural speech synthesis (~200MB RAM).
- **Full-Duplex Voice Mode**: New `--voice-input` mode allowing natural conversation with voice activity detection and interruptions.
- **Voice Persona Manager**: Support for multiple personas with distinct voice samples and text-generation styles.
- **Persona Conditioning**: All generation tiers now support custom system prompts based on the active persona.
- **Standalone RAG Generation**: Tier 2 now includes an extractive fallback for answering questions when Tier 1 is unavailable.
- **Environment Variable Support**: Key paths and hardware settings can now be configured via `.env` or system variables.
- **Audio Recorder**: Asynchronous microphone capture using `sounddevice`.

### Changed
- **Safety Filter**: Implemented double-check logic (pre- and post-cleaning) to prevent safety bypasses.
- **Tool Context Integration**: Refactored `main.py` to ensure tool results are correctly injected as context into the LLM prompt.
- **Voice Gateway**: Re-architected for better fallback between Mimi, NeMo, and pyttsx3.

### Fixed
- Missing `__init__.py` files in `tools/`, `safety/`, and `persona/` directories.
- Refactored `DuplexVoiceHandler` to use centralized `process_input` logic for consistency.

## [2.1.1-patch] - 2026-02-26

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
