# Elara v2.0 Architecture (Functional)

## System Overview

```mermaid
graph TD
    A[User Input<br/>Text / Voice] --> B[Voice Gateway / Recorder]
    B --> C[Duplex Handler<br/>VAD / Interruption]
    C --> D[Safety Filter<br/>Pre-check + Cleaning]
    D -->|Blocked| X[Safety Response]
    D -->|Allowed| E[Tool Router]

    E -->|Tool Call| F[Calculator]
    E -->|No Tool| G[Tier Router + Persona]

    G -->|Tier 1| H[Gemma 1B Local]
    G -->|Tier 2| I[FAISS RAG + Gemma]
    G -->|Tier 3| J[Cloud API Fallback]

    H --> K[Safety Filter<br/>Post-check]
    I --> K
    J --> K
    F --> K

    K --> L[Voice Gateway<br/>Mimi / NeMo / pyttsx3]
    L --> M[User Output]
```

## Tier Details

### Tier 1: Direct Local Model
- Uses `llama-cpp-python` to run Gemma 3 1B IT (Q4_K_M).
- Optimized for speed and low memory (~600MB RAM).

### Tier 2: RAG with FAISS
- Uses `SentenceTransformers` (all-MiniLM-L6-v2) for embeddings.
- Uses `FAISS` for efficient vector search.
- Injects relevant context into the Tier 1 prompt.

### Tier 3: API Fallback
- Calls external LLMs (via OpenRouter/Together AI) for complex queries.
- Used when specific keywords are detected or local models are insufficient.

## Safety Layer
- Rule-based filtering using `re` and `yaml` configuration.
- Protects against harmful content and PII leaks.
- Performs both pre-generation and post-generation checks.

## Voice Pipeline
- **STT**: OpenAI Whisper (Tiny/Base).
- **Recorder**: Asynchronous capture via `sounddevice`.
- **Full-Duplex**: `DuplexVoiceHandler` manages VAD and interruptions.
- **TTS**:
  - **Mimi**: Preferred high-quality neural codec (~200MB).
  - **NeMo**: GPU-accelerated fallback.
  - **pyttsx3**: Low-resource offline fallback.

## Persona System
- **VoicePersonaManager**: Coordinates voice embeddings and text styles.
- **Voice Conditioning**: Applies persona-specific system prompts to all generation tiers.

## Monitoring & Utilities
- **Memory Monitor**: `elara_core/utils.py` provides real-time tracking of RAM usage to ensure compliance with the 4GB edge device target.
