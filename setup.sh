#!/bin/bash
"""One-command environment setup for AI Image Detection project."""

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== AI Image Detection — Environment Setup ==="

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.12 virtual environment..."
    python3.12 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed/{train/{real,fake},val/{real,fake},test/{sdv14,sdv15,midjourney,adm,glide,biggan,wukong,vqdm}},forensynths}
mkdir -p src/models scripts backend frontend/src/components notebooks
mkdir -p checkpoints/backups results/{metrics,plots,gradcam_samples,tables}
mkdir -p report/{figures,presentation} tests

# Create __init__.py files
touch src/__init__.py src/models/__init__.py tests/__init__.py

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Verify with: python scripts/verify_setup.py"
