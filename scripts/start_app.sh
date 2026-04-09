#!/bin/bash
# Start the AI Image Detector web application
# Usage: bash scripts/start_app.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== AI Image Detector ==="
echo "Project: $PROJECT_ROOT"

# Activate venv
source .venv/bin/activate

# Start backend
echo ""
echo "Starting FastAPI backend on http://localhost:8000"
echo "Frontend: Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

PYTHONPATH="$PROJECT_ROOT" uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
