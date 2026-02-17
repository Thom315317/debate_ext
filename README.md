# Debate Orchestrator

A VS Code extension that orchestrates adaptive debates between **Claude Code** (coder) and **Codex/GPT** (reviewer/QA) to produce robust code changes.

## Architecture

```
debate_ext/
  vscode-extension/    ← VS Code extension (TypeScript)
  tools/               ← Python auxiliary scripts
    .venv/             ← Python virtual environment
    scripts/           ← patch_utils.py, setup_venv.sh
  scripts/             ← WSL setup & dev scripts
```

## Prerequisites

- **WSL** (Ubuntu recommended)
- **VS Code** with the **Remote - WSL** extension
- **Node.js** >= 18 and **npm**
- **Python 3** with `venv` module
- **Claude CLI** — must be logged in (`claude --version` to verify)
- **Codex CLI** or **OpenAI CLI** — at least one must be available (`codex --version` or `openai --version`)

## Quick Start

### 1. Setup (run once in WSL)

```bash
cd /home/thom315/debate_ext
bash scripts/setup_wsl.sh
```

This installs Node.js dependencies, creates the Python venv, and makes scripts executable.

### 2. Open in VS Code (Remote - WSL)

```bash
code /home/thom315/debate_ext/vscode-extension
```

Or from VS Code: **Remote-WSL: Open Folder** → `/home/thom315/debate_ext/vscode-extension`

### 3. Launch in development mode

- Open a terminal in VS Code and run:
  ```bash
  npm run watch
  ```
  Or from WSL:
  ```bash
  bash /home/thom315/debate_ext/scripts/run_extension_dev.sh
  ```

- Press **F5** to open the Extension Development Host.

### 4. Use the extension

Open the Command Palette (`Ctrl+Shift+P`) and search for:

| Command | Description |
|---------|-------------|
| `Debate Orchestrator: Run Debate` | Auto-detect mode (SIMPLE/MOYEN/COMPLEXE) |
| `Debate Orchestrator: Run Simple` | Force SIMPLE mode (Claude only) |
| `Debate Orchestrator: Run Complex` | Force COMPLEXE mode (full debate) |
| `Debate Orchestrator: Show Logs` | Show the output channel with logs |
| `Debate Orchestrator: Configure CLIs` | Set CLI paths and timeouts |

## Debate Modes

| Mode | Flow | Max Iterations |
|------|------|----------------|
| **SIMPLE** | Claude → apply patch | 1 |
| **MOYEN** | Claude → Codex review → Claude correction | 2 |
| **COMPLEXE** | Claude → Codex review → Claude correction → Codex validation | 3 |

Mode is auto-detected based on:
- Prompt length and complexity keywords
- Number of files involved
- Whether tests are mentioned
- Selection size

## Memory System

The extension stores a lightweight memory in `.debate_memory/` in your workspace:

- `decisions.jsonl` — past decisions and constraints
- `snippets.jsonl` — code snippets cited in debates
- `runs/` — full logs for each debate run

Relevant memories are retrieved by keyword matching and injected as context.

## CLI Requirements

Both CLIs must be authenticated **before** using the extension. The extension calls them as local processes — it does **not** use API keys directly.

```bash
# Verify Claude CLI
claude --version

# Verify Codex CLI (or OpenAI as fallback)
codex --version
# or
openai --version
```

## Configuration

Settings are under `debateOrchestrator.*` in VS Code:

| Setting | Default | Description |
|---------|---------|-------------|
| `claudePath` | `claude` | Path to Claude CLI |
| `codexPath` | `codex` | Path to Codex CLI |
| `openaiPath` | `openai` | Path to OpenAI CLI (fallback) |
| `claudeTimeout` | `120000` | Claude CLI timeout (ms) |
| `codexTimeout` | `120000` | Codex/OpenAI CLI timeout (ms) |
| `testCommand` | *(empty)* | Optional test command between iterations |
| `pythonPath` | *(empty)* | Python path for auxiliary scripts (auto-detected) |

## Security

- Zero API keys hardcoded
- No secrets logged
- CLIs use their own authentication (login once, use everywhere)
