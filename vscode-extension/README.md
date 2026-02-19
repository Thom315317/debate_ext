# DEBATE EXT — VS Code Extension

AI debate orchestrator: Claude + OpenAI collaborate via API keys to produce robust code.

## Development

```bash
npm install
npm run watch     # auto-compile on save
npm run compile   # one-time compile
npm run smoke     # smoke tests (build + security)
```

Press **F5** in VS Code to launch the Extension Development Host.

## Module Overview

| File | Purpose |
|------|---------|
| `extension.ts` | Entry point, command registration, SecretStorage key management |
| `orchestrator.ts` | Main debate loop — coordinates all modules |
| `cliRunner.ts` | API callers (Anthropic, OpenAI) + SecretStorage helpers |
| `modes.ts` | Mode detection heuristics (SIMPLE/MOYEN/COMPLEXE) |
| `contextCollector.ts` | Gathers VS Code context + workspace memory |
| `consensus.ts` | Structured review prompt builder + parser |
| `patchApplier.ts` | Diff extraction, parsing, preview, WorkspaceEdit apply |
| `chatPanel.ts` | Webview sidebar panel for chat UI |
| `logger.ts` | Output channel logging + persistent run logs |
| `benchmark.ts` | Benchmark v1 — 100 cases, 9 categories, 8 configs |
| `benchmark_v2.ts` | Benchmark v2 — HumanEval 164 problems + code execution |
| `benchmark_paper.ts` | Benchmark v3 — MBPP+ paper-grade (pass@k, variance, cost-efficiency) |

## Commands

| ID | Title |
|----|-------|
| `debateExt.runDebate` | DEBATE EXT: Run Debate (auto) |
| `debateExt.runSimple` | DEBATE EXT: Run Simple |
| `debateExt.runComplex` | DEBATE EXT: Run Complex |
| `debateExt.stopDebate` | DEBATE EXT: Stop Debate |
| `debateExt.clearChat` | DEBATE EXT: Clear Chat |
| `debateExt.configureCLIs` | DEBATE EXT: Configuration |
| `debateExt.configureAnthropicKey` | DEBATE EXT: Configure Anthropic Key |
| `debateExt.configureOpenAIKey` | DEBATE EXT: Configure OpenAI Key |
| `debateExt.showLogs` | DEBATE EXT: Show Logs |

## Benchmark Usage

```bash
# v1 — 100 hand-crafted cases
npm run benchmark -- --help
npm run benchmark -- --dry-run
npm run benchmark -- --cases algo-fibonacci --configs gen1-solo,gen2-solo

# v2 — HumanEval + execution
npm run bench2 -- --help
npm run bench2 -- --dry-run --limit 5
npm run bench2 -- --limit 10 --runs 1 --configs gen1-solo,gen2-solo

# v3 — MBPP+ Paper-Grade Benchmark
npm run bench:mbppplus -- --help
npm run bench:mbppplus -- --dry-run --limit 5
npm run bench:mbppplus -- --limit 30 --runs 3

# Requires: python scripts/setup_mbppplus.py first (needs evalplus or datasets).
```

Key flags: `--dry-run`, `--seed N`, `--configs`, `--judge-threshold N`,
`--gen1-model`, `--gen2-model`, `--claude-judge-model`, `--openai-judge-model`,
`--tiebreaker-model`.
