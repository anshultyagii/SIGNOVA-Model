#!/usr/bin/env bash
# ===================================================================
# SIGNOVA - Interactive Web Dashboard Launcher (macOS / Linux)
# ===================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " Starting SIGNOVA Web Dashboard GUI (macOS/Linux)"
echo " Opening browser at http://localhost:8501 ..."
echo "========================================================"

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

$PY_CMD -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
