# Debate Orchestrator — VS Code Extension

Adaptive debate orchestration between Claude Code and Codex/GPT for robust code generation.

## Development

```bash
# Install dependencies
npm install

# Watch mode (auto-compile on save)
npm run watch

# One-time compile
npm run compile
```

Press **F5** in VS Code to launch the Extension Development Host.

## Module Overview

| File | Purpose |
|------|---------|
| `extension.ts` | Entry point, command registration, CLI checks |
| `orchestrator.ts` | Main debate loop — coordinates all modules |
| `cliRunner.ts` | Robust CLI execution wrapper (spawn, timeout, detection) |
| `modes.ts` | Mode detection heuristics (SIMPLE/MOYEN/COMPLEXE) |
| `contextCollector.ts` | Gathers VS Code context + workspace memory |
| `consensus.ts` | Structured review prompt builder + parser |
| `patchApplier.ts` | Diff extraction, parsing, preview, WorkspaceEdit apply |
| `logger.ts` | Output channel logging + persistent run logs |

## Commands

- `debateOrchestrator.runDebate` — Auto mode
- `debateOrchestrator.runSimple` — Force SIMPLE
- `debateOrchestrator.runComplex` — Force COMPLEXE
- `debateOrchestrator.showLogs` — Show output channel
- `debateOrchestrator.configureCLIs` — Interactive CLI configuration
