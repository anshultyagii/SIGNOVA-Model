#!/usr/bin/env bash
# ===================================================================
# SIGNOVA - Master Cross-Platform Launcher (macOS / Linux)
# ===================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

clear 2>/dev/null || true

echo "==================================================================="
echo "              SIGNOVA - INDIAN SIGN LANGUAGE TRANSLATOR             "
echo "                      macOS / Linux Edition                        "
echo "==================================================================="
echo ""
echo "  [1] Launch High-Performance Desktop Window (Recommended)"
echo "      - Fast 60 FPS OpenCV HUD window with Hotkey [Spacebar]"
echo ""
echo "  [2] Launch Interactive Web Dashboard GUI"
echo "      - Streamlit browser interface and Sign Dictionary"
echo ""
echo "==================================================================="
read -p "Select an option [1 or 2] (Default is 1): " choice

choice=${choice:-1}

if [ "$choice" = "2" ]; then
    bash "$SCRIPT_DIR/run_dashboard.sh"
else
    bash "$SCRIPT_DIR/run_webcam.sh"
fi
