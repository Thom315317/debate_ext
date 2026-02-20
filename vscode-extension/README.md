# DEBATE EXT — VS Code Extension

AI debate orchestrator: two LLM generators collaborate via structured debate, evaluated by frontier judges.

## Features

- **Multi-Agent Debate:** Orchestrates collaboration between two generators (solo, lead, orch, selfrefine modes).
- **3-Level Judge Protocol:** R1 (independent) → R2 (debate on divergence) → Tie-breaker (Gemini).
- **Paper-Grade Benchmark:** MBPP+ 378 tasks, pass@k, McNemar, Spearman, Cohen's kappa.
- **VS Code Integration:** Native chat panel, diff previews, and workspace context awareness.

## Prerequisites

- **Node.js**: v16 or higher.
- **Python 3.8+**: Required for benchmark data setup.

## Development

```bash
npm install
npm run compile   # one-time compile
npm run smoke     # smoke tests (build + security)
npm test          # unit tests
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
| `benchmark_paper.ts` | **Paper benchmark** — MBPP+ 378 tasks, 8 configs, blind judges, EvalPlus+ |
| `benchmark.ts` | *(deprecated)* v1 benchmark with hardcoded tasks |
| `benchmark_v2.ts` | *(deprecated)* v2 benchmark using HumanEval |

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
# Paper-Grade Benchmark (MBPP+)
npm run bench:mbppplus -- --help
npm run bench:mbppplus -- --dry-run --limit 5
npm run bench:mbppplus -- --limit 30 --runs 3
npm run bench:mbppplus -- --resume              # resume from checkpoint
```

Key flags: `--dry-run`, `--resume`, `--seed N`, `--limit N`, `--runs N`,
`--configs`, `--judge-threshold N`, `--checkpoint-file <path>`, `--dataset mbpp|humaneval`,
`--gen1-model`, `--gen2-model`,
`--claude-judge-model`, `--openai-judge-model`, `--tiebreaker-model`.

Data setup (requires Python):
```bash
pip install evalplus datasets
python scripts/setup_mbppplus.py      # MBPP+ (378 tasks)
python scripts/setup_humanevalplus.py  # HumanEval+ (164 tasks)
```
