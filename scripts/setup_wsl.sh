#!/usr/bin/env bash
# setup_wsl.sh — Full setup for CRISTAL CODE on WSL.
# Installs Node.js, npm, Claude CLI, Codex CLI, Python venv, and extension deps.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$ROOT_DIR/vscode-extension"
TOOLS_DIR="$ROOT_DIR/tools"

echo "============================================"
echo "  CRISTAL CODE — WSL Setup"
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

# --- Claude CLI ---
echo "--- Checking Claude CLI ---"
if command -v claude &>/dev/null; then
    echo "Claude CLI found: $(claude --version 2>&1 || echo 'installed')"
else
    echo "Claude CLI not found. Installing..."
    npm install -g @anthropic-ai/claude-code
    echo "Claude CLI installed: $(claude --version 2>&1 || echo 'installed')"
fi
echo ""

# --- Codex CLI ---
echo "--- Checking Codex CLI ---"
if command -v codex &>/dev/null; then
    echo "Codex CLI found: $(codex --version 2>&1 || echo 'installed')"
else
    echo "Codex CLI not found. Installing..."
    npm install -g @openai/codex
    echo "Codex CLI installed: $(codex --version 2>&1 || echo 'installed')"
fi
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

# --- Python venv ---
echo "--- Setting up Python venv ---"
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Installing..."
    sudo apt-get install -y -qq python3 python3-venv python3-pip
fi

bash "$TOOLS_DIR/scripts/setup_venv.sh"
echo ""

# --- Make scripts executable ---
echo "--- Making scripts executable ---"
chmod +x "$ROOT_DIR/scripts/"*.sh
chmod +x "$TOOLS_DIR/scripts/"*.sh
chmod +x "$TOOLS_DIR/scripts/"*.py 2>/dev/null || true
echo ""

echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Claude : $(command -v claude || echo 'NOT FOUND')"
echo "  Codex  : $(command -v codex || echo 'NOT FOUND')"
echo ""
echo "Next steps:"
echo "  1. Open VS Code on this folder (Remote - WSL):"
echo "     code $ROOT_DIR/vscode-extension"
echo ""
echo "  2. Press F5 to launch Extension Development Host"
echo ""
echo "  NOTE: Claude CLI needs auth first if not done:"
echo "     claude login"
echo ""
echo "  NOTE: Codex CLI needs an OpenAI API key:"
echo "     export OPENAI_API_KEY=\"sk-...\""
echo "     (add to ~/.bashrc to persist)"
echo ""
