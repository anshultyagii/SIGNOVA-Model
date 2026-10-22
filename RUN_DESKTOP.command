#!/usr/bin/env bash
# ===================================================================
# SIGNOVA - Instant Desktop Translator Launcher (macOS Double-Clickable)
# ===================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Set terminal title
echo -ne "\033]0;SIGNOVA Desktop ISL Translator\007"

clear 2>/dev/null || true

echo "========================================================"
echo " Starting SIGNOVA Desktop Translator..."
echo "========================================================"

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Error: Python 3 not found in PATH."
    exit 1
fi

$PY_CMD scripts/realtime_isl_translator.py "$@"
