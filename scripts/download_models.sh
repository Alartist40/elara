#!/bin/bash
# Download models for Elara v2.0

mkdir -p models data

echo "Downloading Gemma 3 1B quantized (600MB)..."
# Using wget for direct download
wget https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf -O models/gemma-3-1b-it-q4_0.gguf

echo "Pre-downloading embedding model..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Setup complete."
