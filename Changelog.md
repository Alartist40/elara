# Changelog

All notable changes to the Elara project will be documented in this file.

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
