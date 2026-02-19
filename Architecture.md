# Elara v2.0 Architecture (Functional)

## System Overview

```mermaid
graph TD
    A[User Input<br/>Text / Voice] --> B[Voice Gateway<br/>Whisper STT]
    B --> C[Safety Filter<br/>Pre-check]
    C -->|Blocked| X[Safety Response]
    C -->|Allowed| D[Tool Router]

    D -->|Tool Call| E[Calculator]
    D -->|No Tool| F[Tier Router]

    F -->|Tier 1| G[Gemma 1B Local]
    F -->|Tier 2| H[FAISS RAG + Gemma]
    F -->|Tier 3| I[Cloud API Fallback]

    G --> J[Safety Filter<br/>Post-check]
    H --> J
    I --> J
    E --> J

    J --> K[Voice Gateway<br/>TTS]
    K --> L[User Output]
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
- **TTS**: pyttsx3 (offline fallback) or NVIDIA NeMo (high quality).
