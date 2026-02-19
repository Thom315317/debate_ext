# Changelog

## [1.0.0] — 2025-02-19

### Added
- Paper-grade benchmark suite (`benchmark_paper.ts`)
  - 8 collaboration configurations (solo, lead, orch, selfrefine × 2 generators)
  - 3-level judge evaluation (R1 → R2 debate → Tie-breaker)
  - Blind judge mode (no exec status leakage)
  - EvalPlus+ extended test evaluation
  - Statistical metrics: pass@k, McNemar, AUC, Spearman, Cohen's κ
- Workflow features: --rejudge-from, --merge, --offset, --resume
- Circuit-breaker on judge API failures
- Smoke test suite
- MBPP+ data loader (378 tasks)

### VS Code Extension
- Multi-agent debate sidebar (Claude + GPT)
- Auto-detect task complexity
- Streaming output with think-block detection
