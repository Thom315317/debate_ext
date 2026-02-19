#!/usr/bin/env bash
# run_extension_dev.sh — Start the TypeScript watch compiler for extension development.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$ROOT_DIR/vscode-extension"

echo "=== Debate Orchestrator — Development Mode ==="
echo ""
echo "Starting TypeScript watch compiler..."
echo "Once running, press F5 in VS Code to launch the Extension Development Host."
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$EXT_DIR"
npm run watch
