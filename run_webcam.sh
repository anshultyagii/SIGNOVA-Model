#!/usr/bin/env bash
# ===================================================================
# SIGNOVA - Real-Time Indian Sign Language & NLP Translator
# macOS / Linux Live Webcam Launcher
# ===================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " Starting SIGNOVA Live ISL & NLP Translator (macOS/Linux)"
echo "========================================================"

# Determine Python command
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

echo "Using Python: $($PY_CMD --version)"

# Check macOS Camera Permission Notice
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS Note: If camera doesn't start, verify camera permissions in:"
    echo "  System Settings -> Privacy & Security -> Camera -> Terminal / IDE"
    echo ""
fi

$PY_CMD scripts/realtime_isl_translator.py
