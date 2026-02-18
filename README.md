# Elara v2.0

**A Multi-Modal AI System with Biblical Constitutional Principles**

Elara is a tiered inference engine designed for edge devices (4GB RAM target). It combines multiple AI architectures for text, vision, and voice processing with an immutable safety layer.

## Architecture

```
Input → Voice STT (Whisper) → Multiplexer → Constitutional Pre-Filter
  → Tier Selection → Tier Execution → Constitutional Post-Filter
  → Voice TTS (NeMo) → Output
```

### Three Tiers

| Tier | Coverage | Components | Use Case |
|------|----------|------------|----------|
| **1** | 95% | Mistral Direct | Simple queries |
| **2** | 4% | CLaRa + Mistral | Memory/retrieval |
| **3** | 1% | TRM + TiDAR + Tools | Complex reasoning |

### Key Components

- **Constitutional Layer** — Code-enforced safety rules based on biblical principles. Immutable, auditable, no ML.
- **CLaRa** — Continuous Latent Reasoning. Compresses documents into memory tokens for efficient retrieval.
- **TRM** — Tiny Recursive Model. 2-layer weight-shared network with adaptive halting for deep reasoning.
- **TiDAR** — Think in Diffusion, Talk in Autoregression. Hybrid generation with parallel pre-drafting.
- **AirLLM** — Fallback layer-wise inference for large models on constrained hardware.
- **Voice** — Whisper (STT) + NeMo FastPitch/HiFi-GAN (TTS).

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt

# Install local packages
pip install -e ./whisper
pip install -e ./mistral-inference

# Run interactive mode
python -m elara_core.main --interactive --config config/system_config.yaml

# Single query
python -m elara_core.main --text "Hello, how are you?"

# Force a specific tier
python -m elara_core.main --text "Analyze this topic" --tier 3

# Check system status
python -m elara_core.main --status
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_constitutional.py -v
python -m pytest tests/test_clara.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_integration.py -v
```

## Project Structure

```
elara_core/
├── __init__.py
├── main.py                  # CLI entry point
├── airllm_fallback.py       # Tier 3 fallback
├── constitutional/          # Safety layer
│   ├── layer.py             # Pre/post filtering
│   ├── principles.py        # YAML principle loader
│   └── audit.py             # Audit logging
├── clara/                   # Tier 2: Document retrieval
│   ├── compressor.py        # SCP document compression
│   ├── store.py             # Persistent embeddings store
│   ├── query_reasoner.py    # Query encoding
│   └── topk.py              # Differentiable top-k
├── trm/                     # Tier 3: Recursive reasoning
│   ├── core.py              # TRM engine
│   ├── block.py             # Transformer/Mixer block
│   ├── halting.py           # Adaptive halting
│   └── state.py             # Latent state manager
├── tidar/                   # Tier 3: Hybrid generation
│   ├── generator.py         # Full TiDAR model
│   ├── attention.py         # Hybrid AR+diffusion attention
│   └── sampler.py           # Draft-verify sampler
├── tiered/                  # Orchestration
│   ├── engine.py            # TieredInferenceEngine
│   ├── multiplexer.py       # Input normalization
│   ├── router.py            # Tier selection
│   └── metrics.py           # Performance tracking
├── tools/                   # Function calling
│   ├── router.py            # Tool execution
│   ├── schema.py            # Schema definitions
│   └── parser.py            # XML tool call parser
└── voice/                   # Voice I/O
    ├── gateway.py           # Unified voice interface
    ├── stt.py               # Whisper wrapper
    ├── tts.py               # NeMo wrapper
    └── audio_utils.py       # Audio helpers
```

## Configuration

All settings are in `config/system_config.yaml`. Key sections:

- `voice.stt` — Whisper model size and device
- `voice.tts` — NeMo TTS model paths
- `constitutional` — Principles file path and strict mode
- `clara` — Store path and compression settings
- `trm` — Recursion depth and confidence threshold
- `tidar` — Block size and AR/diffusion mix ratio
- `tiering` — Tier selection thresholds

## License

See individual component licenses: Mistral (Apache 2.0), Whisper (MIT), NeMo (Apache 2.0).
