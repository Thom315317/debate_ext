#!/usr/bin/env bash
# setup_venv.sh — Create and configure the Python virtual environment for auxiliary tools.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$TOOLS_DIR/.venv"

echo "=== Setting up Python venv for Debate Orchestrator tools ==="

# Check Python3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install it first (e.g., sudo apt install python3 python3-venv)."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Upgrade pip
echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo ""
echo "=== Python venv ready ==="
echo "  Activate: source $VENV_DIR/bin/activate"
echo "  Python:   $VENV_DIR/bin/python3"
