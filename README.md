# Elara - Local-First AI Assistant

A simple, working AI system that prioritizes local inference and falls back to APIs when needed.

## Architecture

- **Tier 1:** Gemma 3 1B (4-bit quantized) for fast local inference
- **Tier 2:** FAISS + SentenceTransformers for document Q&A
- **Tier 3:** API fallback (OpenRouter) for complex queries
- **Voice:** Whisper STT + pyttsx3/NeMo TTS (separate from core model)

## Why Not [Complex Thing]?

I initially built CLaRa, TRM, and TiDAR modules based on research papers, but realized I couldn't train them effectively. This version uses battle-tested tools (llama.cpp, FAISS) that actually work.

## Installation

```bash
pip install -r requirements.txt
bash scripts/download_models.sh
```

## Usage

```bash
# Interactive mode
python3 -m elara_core.main --interactive

# Single query
python3 -m elara_core.main --text "Hello"

# Build RAG index
python3 scripts/build_index.py path/to/docs/
```

## Specs

- RAM: ~800MB for Tier 1, ~1.2GB for Tier 2
- Latency: 50-200ms local, 1-3s with API
- No training required, works out-of-box
