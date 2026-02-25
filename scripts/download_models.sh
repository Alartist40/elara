#!/bin/bash
set -euo pipefail

echo "=== Elara Model Downloader ==="
echo "Choose model to download:"
echo "1) Qwen2.5-1.5B-Instruct (1.1GB, recommended, no auth required)"
echo "2) Gemma-3-1B (requires HuggingFace token)"
echo "3) Skip model download (manual setup)"
read -p "Selection [1-3]: " choice

mkdir -p models

case $choice in
  1)
    echo "Downloading Qwen2.5-1.5B-Instruct..."
    curl -L --progress-bar \
      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
      -o models/qwen-1.5b-q4.gguf
    echo "ELARA_MODEL_PATH=models/qwen-1.5b-q4.gguf" > .env
    echo "✓ Downloaded and configured"
    ;;
  2)
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "Error: HF_TOKEN not set. Get token from huggingface.co/settings/tokens"
      exit 1
    fi
    echo "Downloading Gemma-3-1B..."
    curl -L --progress-bar \
      -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf" \
      -o models/gemma-3-1b-it-q4_0.gguf
    echo "ELARA_MODEL_PATH=models/gemma-3-1b-it-q4_0.gguf" > .env
    echo "✓ Downloaded and configured"
    ;;
  3)
    echo "Skipping download. Set ELARA_MODEL_PATH in .env manually"
    ;;
  *)
    echo "Invalid selection"
    exit 1
    ;;
esac

echo ""
echo "=== Pre-downloading Embedding Model ==="
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('✓ Embedding model cached')"

echo ""
echo "=== Downloading Piper TTS Voice ==="
mkdir -p models/piper

if [ ! -f models/piper/en_US-lessac-medium.onnx ]; then
  echo "Downloading Piper voice (medium quality, ~60MB)..."
  curl -L --progress-bar \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
    -o models/piper/en_US-lessac-medium.onnx
  curl -L --progress-bar \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
    -o models/piper/en_US-lessac-medium.onnx.json
  echo "✓ Piper voice downloaded"
else
  echo "✓ Piper voice already exists"
fi

echo ""
echo "=== Setup Complete ==="
ls -lh models/ 2>/dev/null || echo "No models directory"
echo ""
echo "Test with: python3 -m elara_core.main --text 'Hello'"