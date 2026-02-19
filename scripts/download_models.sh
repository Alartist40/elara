#!/bin/bash
set -euo pipefail

# Download models for Elara v2.0

mkdir -p models data

# Requires a HuggingFace token: https://huggingface.co/settings/tokens
# Accept Gemma terms at: https://huggingface.co/google/gemma-3-1b-it
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable is not set." >&2
    echo "Please set it with: export HF_TOKEN=your_token" >&2
    echo "And ensure you have accepted the model terms on Hugging Face." >&2
    exit 1
fi

echo "Downloading Gemma 3 1B quantized (600MB)..."
wget --header="Authorization: Bearer ${HF_TOKEN}" \
    https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf \
    -O models/gemma-3-1b-it-q4_0.gguf

echo "Pre-downloading embedding model..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Setup complete."
