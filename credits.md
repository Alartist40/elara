# Credits

## Elara v2.3

### Core Architecture
- **Multi-Tier Inference** — Dynamic routing between local execution, RAG, and API fallbacks.
- **Qwen2.5 / Gemma 3 1B IT** — Primary local language models (Alibaba / Google).
- **FAISS** — Efficient semantic search for RAG (Meta).

### External Components
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — GGUF inference engine.
- **[OLMoASR](https://github.com/allenai/OLMoASR)** — Speech-to-Text (Apache 2.0 License, Allen Institute for AI).
  - Team: Huong Ngo, Matt Deitke, Martijn Bartelds, Sarah Pratt, Josh Gardner, Matt Jordan, Ludwig Schmidt.
- **[Piper TTS](https://github.com/rhasspy/piper)** — Fast, local neural text-to-speech (MIT License, Rhasspy).
- **[sentence-transformers](https://sbert.net/)** — Embedding models for RAG.

### Key Libraries
- **PyTorch** — Deep learning framework.
- **sounddevice / soundfile** — Real-time audio I/O and processing.
- **faiss-cpu** — Vector similarity search.
- **python-dotenv** — Environment management.

### Biblical Principles
Constitutional safety principles are grounded in Scripture. See `config/biblical_principles.yaml` for the full reference with scriptural citations.

---

*Built with purpose, guided by principles.*
