#!/bin/bash
set -e

echo "=== Elara v2.3 Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "elara_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv elara_env
fi

# Activate
source elara_env/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Download models
echo ""
read -p "Download models now? [Y/n]: " download_models
if [[ $download_models =~ ^[Yy]$ ]] || [[ -z $download_models ]]; then
    bash scripts/download_models.sh
else
    echo "Skipping model download. Run 'bash scripts/download_models.sh' later."
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate environment: source elara_env/bin/activate"
echo "Test: python3 -m elara_core.main --text 'Hello'"
echo "Interactive: python3 -m elara_core.main --interactive"
