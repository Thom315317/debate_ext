# CRISTAL CODE

AI debate orchestrator — Claude + OpenAI collaborate via API keys to produce robust code.
Includes a benchmark suite that evaluates inter-LLM collaboration quality.

## Architecture

```
debate_ext/
  vscode-extension/    ← VS Code extension (TypeScript)
    src/               ← Extension source + benchmark suites
    scripts/           ← smoke.js (CI smoke tests)
    data/              ← HumanEval.jsonl (benchmark v2)
  tools/               ← Python auxiliary scripts
    .venv/             ← Python virtual environment
    scripts/           ← patch_utils.py, setup_venv.sh
```

### Extension (VS Code sidebar)

Claude (Anthropic API) and GPT (OpenAI API) collaborate in a debate loop
to generate, review, and refine code changes. The extension auto-detects
task complexity and adapts the debate depth accordingly.

### Benchmark Suite

Two benchmark harnesses compare 8 collaboration configurations:

| Config | Description |
|--------|-------------|
| `gen1-solo` | Qwen3-Coder solo |
| `gen2-solo` | MiniMax M2 solo |
| `gen1-lead` / `gen2-lead` | One leads, the other consults |
| `gen1-orch` / `gen2-orch` | One orchestrates, the other codes |
| `gen1-selfrefine` / `gen2-selfrefine` | Single agent self-reviews |

**Generators** are mid-tier models via Ollama (Qwen3-Coder 480B, MiniMax M2).
**Judges** are frontier models from distinct providers:

| Role | Model | Provider |
|------|-------|----------|
| Judge 1 | Claude Sonnet 4.5 | Anthropic API |
| Judge 2 | GPT-4.1 | OpenAI API |
| Tie-breaker | Gemini 2.5 Pro | Google AI API |

Three distinct providers = zero conflict of interest.

#### Judge Escalation

1. **Round 1** — Both judges score independently (correctness, completeness, edge cases, code quality, readability)
2. **Divergence check** — If scores diverge >20%, judges debate
3. **Round 2** — Judges see each other's scores + arguments, re-score
4. **Tie-breaker** — If still divergent, Gemini 2.5 Pro makes the final call

## Prerequisites

- **Node.js** >= 18 and **npm**
- **Python 3** with `venv` module (for patch utilities)
- **Ollama** with mid-tier models pulled (for benchmarks)

## Environment Variables

Copy `.env.example` and fill in your keys:

```bash
cp vscode-extension/.env.example vscode-extension/.env
```

| Variable | Required for | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Extension + Benchmarks | Anthropic API key (sk-ant-...) |
| `OPENAI_API_KEY` | Extension + Benchmarks | OpenAI API key (sk-proj-...) |
| `GEMINI_API_KEY` | Benchmarks (tie-breaker) | Google AI API key (AIza...) |
| `OLLAMA_HOST` | Benchmarks | Ollama host (default: localhost) |
| `OLLAMA_PORT` | Benchmarks | Ollama port (default: 11434) |
| `GEN1_MODEL` | Benchmarks | Generator 1 model name |
| `GEN2_MODEL` | Benchmarks | Generator 2 model name |

API keys for the extension are stored securely via VS Code SecretStorage
(configured from the sidebar gear icon).

## Quick Start

### Extension Development

```bash
cd vscode-extension
npm install
npm run watch    # auto-compile on save
# Press F5 in VS Code to launch Extension Development Host
```

### Run Benchmarks

```bash
cd vscode-extension

# Benchmark v1 — 100 hand-crafted cases, 9 categories
npm run benchmark -- --cases algo-fibonacci --configs gen1-solo,gen2-solo

# Benchmark v2 — HumanEval (164 Python problems + execution)
npm run bench2 -- --limit 10 --runs 1

# Dry-run (mock scores, no API calls, validate pipeline)
npm run benchmark -- --dry-run
npm run bench2 -- --dry-run --limit 5
```

### Smoke Tests

```bash
npm run smoke    # checks build artifacts, --help, security patterns
```

## VS Code Commands

| Command | Description |
|---------|-------------|
| `CRISTAL CODE: Run Debate` | Auto-detect mode and run debate |
| `CRISTAL CODE: Run Simple` | Force simple mode (single pass) |
| `CRISTAL CODE: Run Complex` | Force complex mode (full debate) |
| `CRISTAL CODE: Stop Debate` | Stop the current debate |
| `CRISTAL CODE: Clear Chat` | Clear chat history |
| `CRISTAL CODE: Configuration` | Configure models and settings |
| `CRISTAL CODE: Configure Anthropic Key` | Set Anthropic API key (SecretStorage) |
| `CRISTAL CODE: Configure OpenAI Key` | Set OpenAI API key (SecretStorage) |
| `CRISTAL CODE: Show Logs` | Show the output channel |

## Extension Settings

Settings are under `cristalCode.*` in VS Code:

| Setting | Default | Description |
|---------|---------|-------------|
| `claudeModel` | `claude-sonnet-4-20250514` | Claude model for the extension |
| `claudeTimeout` | `300000` | Claude API timeout (ms) |
| `openaiModel` | `gpt-4o` | OpenAI model for the extension |
| `openaiTimeout` | `300000` | OpenAI API timeout (ms) |
| `testCommand` | *(empty)* | Test command between iterations |
| `pythonPath` | *(empty)* | Python path (auto-detected) |

## Security

- Zero API keys in source code — enforced by smoke tests
- Extension keys stored via VS Code SecretStorage (OS keychain)
- Benchmark keys via environment variables only
- `.env` excluded by `.gitignore`
- Forbidden credential patterns checked on every build
