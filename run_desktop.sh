#!/usr/bin/env bash
# ===================================================================
# SIGNOVA - Instant Desktop Shortcut (CLI)
# ===================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_webcam.sh" "$@"
