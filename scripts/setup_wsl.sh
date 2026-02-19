#!/usr/bin/env bash
# setup_wsl.sh — Full setup for debate_ext on WSL.
# Installs Node.js, npm, Python venv, and extension deps.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$ROOT_DIR/vscode-extension"

echo "============================================"
echo "  DEBATE EXT — WSL Setup"
echo "============================================"
echo ""
echo "Root: $ROOT_DIR"
echo ""

# --- Node.js & npm ---
echo "--- Checking Node.js & npm ---"
if ! command -v node &>/dev/null; then
    echo "Node.js not found. Installing via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq nodejs npm
    echo "Node.js installed: $(node --version)"
else
    echo "Node.js found: $(node --version)"
fi

if ! command -v npm &>/dev/null; then
    echo "npm not found. Installing..."
    sudo apt-get install -y -qq npm
fi
echo "npm: $(npm --version)"
echo ""

# --- Python 3 ---
echo "--- Checking Python 3 ---"
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Installing..."
    sudo apt-get install -y -qq python3 python3-venv python3-pip
fi
echo "Python3: $(python3 --version)"
echo ""

# --- VS Code Extension: npm install ---
echo "--- Installing VS Code extension dependencies ---"
cd "$EXT_DIR"
npm install
echo ""

# --- Compile TypeScript ---
echo "--- Compiling extension ---"
npm run compile
echo ""

# --- Make scripts executable ---
echo "--- Making scripts executable ---"
chmod +x "$ROOT_DIR/scripts/"*.sh 2>/dev/null || true
echo ""

echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set environment variables:"
echo "     export ANTHROPIC_API_KEY=\"...\""
echo "     export OPENAI_API_KEY=\"...\""
echo "     export GEMINI_API_KEY=\"...\"    # for tie-breaker"
echo ""
echo "  2. Open VS Code on this folder:"
echo "     code $ROOT_DIR/vscode-extension"
echo ""
echo "  3. Press F5 to launch Extension Development Host"
echo ""
echo "  4. For benchmarks:"
echo "     cd vscode-extension"
echo "     node out/benchmark_paper.js --dry-run --limit 5"
echo ""
